# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Batched inverse dynamics in Newton's public generalized-coordinate basis."""

from __future__ import annotations

import numpy as np
import warp as wp

from ..math import transform_twist
from .articulation import (
    compute_2d_rotational_dofs,
    compute_3d_rotational_dofs,
    eval_fk_batched,
    transform_spatial_inertia,
)
from .enums import JointType
from .model import Model


@wp.func
def _spatial_cross(a: wp.spatial_vector, b: wp.spatial_vector):
    linear_a = wp.spatial_top(a)
    angular_a = wp.spatial_bottom(a)
    linear_b = wp.spatial_top(b)
    angular_b = wp.spatial_bottom(b)
    return wp.spatial_vector(
        wp.cross(angular_a, linear_b) + wp.cross(linear_a, angular_b),
        wp.cross(angular_a, angular_b),
    )


@wp.func
def _spatial_cross_dual(a: wp.spatial_vector, b: wp.spatial_vector):
    linear_a = wp.spatial_top(a)
    angular_a = wp.spatial_bottom(a)
    linear_b = wp.spatial_top(b)
    angular_b = wp.spatial_bottom(b)
    return wp.spatial_vector(
        wp.cross(angular_a, linear_b),
        wp.cross(angular_a, angular_b) + wp.cross(linear_a, linear_b),
    )


@wp.func
def _body_inertia(mass: float, inertia: wp.mat33):
    return wp.spatial_matrix(
        mass,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        mass,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        mass,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        inertia[0, 0],
        inertia[0, 1],
        inertia[0, 2],
        0.0,
        0.0,
        0.0,
        inertia[1, 0],
        inertia[1, 1],
        inertia[1, 2],
        0.0,
        0.0,
        0.0,
        inertia[2, 0],
        inertia[2, 1],
        inertia[2, 2],
    )


@wp.func
def _write_motion_subspaces(
    joint_idx: int,
    joint_type: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_dof_dim: wp.array2d[wp.int32],
    joint_X_p: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    body_com: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    joint_q: wp.array[wp.float32],
    joint_S: wp.array[wp.spatial_vector],
):
    parent = joint_parent[joint_idx]
    child = joint_child[joint_idx]
    dof_start = joint_qd_start[joint_idx]
    coord_start = joint_q_start[joint_idx]
    linear_count = joint_dof_dim[joint_idx, 0]
    angular_count = joint_dof_dim[joint_idx, 1]
    joint_kind = joint_type[joint_idx]

    X_wpj = joint_X_p[joint_idx]
    if parent >= 0:
        X_wpj = body_q[parent] * X_wpj

    if joint_kind == JointType.PRISMATIC:
        joint_S[dof_start] = transform_twist(X_wpj, wp.spatial_vector(joint_axis[dof_start], wp.vec3()))
        return

    if joint_kind == JointType.REVOLUTE:
        joint_S[dof_start] = transform_twist(X_wpj, wp.spatial_vector(wp.vec3(), joint_axis[dof_start]))
        return

    if joint_kind == JointType.BALL:
        joint_S[dof_start + 0] = transform_twist(X_wpj, wp.spatial_vector(wp.vec3(), wp.vec3(1.0, 0.0, 0.0)))
        joint_S[dof_start + 1] = transform_twist(X_wpj, wp.spatial_vector(wp.vec3(), wp.vec3(0.0, 1.0, 0.0)))
        joint_S[dof_start + 2] = transform_twist(X_wpj, wp.spatial_vector(wp.vec3(), wp.vec3(0.0, 0.0, 1.0)))
        return

    if joint_kind == JointType.D6:
        position = wp.vec3()
        for axis_idx in range(linear_count):
            axis = joint_axis[dof_start + axis_idx]
            position += axis * joint_q[coord_start + axis_idx]
            joint_S[dof_start + axis_idx] = transform_twist(X_wpj, wp.spatial_vector(axis, wp.vec3()))

        angular_start = dof_start + linear_count
        coordinate_angular_start = coord_start + linear_count
        axis_0 = wp.vec3()
        axis_1 = wp.vec3()
        axis_2 = wp.vec3()
        if angular_count == 1:
            axis_0 = joint_axis[angular_start]
        elif angular_count == 2:
            _, axis_0 = compute_2d_rotational_dofs(
                joint_axis[angular_start],
                joint_axis[angular_start + 1],
                joint_q[coordinate_angular_start],
                joint_q[coordinate_angular_start + 1],
                1.0,
                0.0,
            )
            _, axis_1 = compute_2d_rotational_dofs(
                joint_axis[angular_start],
                joint_axis[angular_start + 1],
                joint_q[coordinate_angular_start],
                joint_q[coordinate_angular_start + 1],
                0.0,
                1.0,
            )
        elif angular_count == 3:
            _, axis_0 = compute_3d_rotational_dofs(
                joint_axis[angular_start],
                joint_axis[angular_start + 1],
                joint_axis[angular_start + 2],
                joint_q[coordinate_angular_start],
                joint_q[coordinate_angular_start + 1],
                joint_q[coordinate_angular_start + 2],
                1.0,
                0.0,
                0.0,
            )
            _, axis_1 = compute_3d_rotational_dofs(
                joint_axis[angular_start],
                joint_axis[angular_start + 1],
                joint_axis[angular_start + 2],
                joint_q[coordinate_angular_start],
                joint_q[coordinate_angular_start + 1],
                joint_q[coordinate_angular_start + 2],
                0.0,
                1.0,
                0.0,
            )
            _, axis_2 = compute_3d_rotational_dofs(
                joint_axis[angular_start],
                joint_axis[angular_start + 1],
                joint_axis[angular_start + 2],
                joint_q[coordinate_angular_start],
                joint_q[coordinate_angular_start + 1],
                joint_q[coordinate_angular_start + 2],
                0.0,
                0.0,
                1.0,
            )

        X_wja = X_wpj * wp.transform(position, wp.quat_identity())
        if angular_count > 0:
            joint_S[angular_start] = transform_twist(X_wja, wp.spatial_vector(wp.vec3(), axis_0))
        if angular_count > 1:
            joint_S[angular_start + 1] = transform_twist(X_wja, wp.spatial_vector(wp.vec3(), axis_1))
        if angular_count > 2:
            joint_S[angular_start + 2] = transform_twist(X_wja, wp.spatial_vector(wp.vec3(), axis_2))
        return

    if joint_kind == JointType.FREE or joint_kind == JointType.DISTANCE:
        rotation = wp.transform_get_rotation(X_wpj)
        axis_x = wp.quat_rotate(rotation, wp.vec3(1.0, 0.0, 0.0))
        axis_y = wp.quat_rotate(rotation, wp.vec3(0.0, 1.0, 0.0))
        axis_z = wp.quat_rotate(rotation, wp.vec3(0.0, 0.0, 1.0))
        position_com = wp.transform_point(body_q[child], body_com[child])
        joint_S[dof_start + 0] = wp.spatial_vector(axis_x, wp.vec3())
        joint_S[dof_start + 1] = wp.spatial_vector(axis_y, wp.vec3())
        joint_S[dof_start + 2] = wp.spatial_vector(axis_z, wp.vec3())
        joint_S[dof_start + 3] = wp.spatial_vector(wp.cross(position_com, axis_x), axis_x)
        joint_S[dof_start + 4] = wp.spatial_vector(wp.cross(position_com, axis_y), axis_y)
        joint_S[dof_start + 5] = wp.spatial_vector(wp.cross(position_com, axis_z), axis_z)


@wp.func
def _d6_acceleration_bias(
    joint_idx: int,
    joint_parent: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_dof_dim: wp.array2d[wp.int32],
    joint_X_p: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
    joint_S: wp.array[wp.spatial_vector],
):
    parent = joint_parent[joint_idx]
    coord_start = joint_q_start[joint_idx]
    dof_start = joint_qd_start[joint_idx]
    linear_count = joint_dof_dim[joint_idx, 0]
    angular_count = joint_dof_dim[joint_idx, 1]

    X_wpj = joint_X_p[joint_idx]
    if parent >= 0:
        X_wpj = body_q[parent] * X_wpj
    position = wp.vec3()
    velocity_linear = wp.vec3()
    for axis_idx in range(linear_count):
        position += joint_axis[dof_start + axis_idx] * joint_q[coord_start + axis_idx]
        velocity_linear += wp.spatial_top(joint_S[dof_start + axis_idx]) * joint_qd[dof_start + axis_idx]

    angular_start = dof_start + linear_count
    velocity_angular = wp.vec3()
    velocity_angular_prefix = wp.vec3()
    acceleration_angular_bias = wp.vec3()
    for axis_idx in range(angular_count):
        axis_world = wp.spatial_bottom(joint_S[angular_start + axis_idx])
        axis_velocity = axis_world * joint_qd[angular_start + axis_idx]
        acceleration_angular_bias += wp.cross(velocity_angular_prefix, axis_world) * joint_qd[angular_start + axis_idx]
        velocity_angular_prefix += axis_velocity
        velocity_angular += axis_velocity

    position_anchor = wp.transform_get_translation(X_wpj * wp.transform(position, wp.quat_identity()))
    acceleration_linear_bias = wp.cross(velocity_linear, velocity_angular) + wp.cross(
        position_anchor, acceleration_angular_bias
    )
    return wp.spatial_vector(acceleration_linear_bias, acceleration_angular_bias)


@wp.kernel
def _inverse_dynamics_forward(
    articulation_start: wp.array[wp.int32],
    articulation_end: wp.array[wp.int32],
    joint_type: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_dof_dim: wp.array2d[wp.int32],
    joint_X_p: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    body_com: wp.array[wp.vec3],
    body_inertia: wp.array[wp.mat33],
    body_mass: wp.array[wp.float32],
    body_world: wp.array[wp.int32],
    gravity: wp.array[wp.vec3],
    joint_q: wp.array2d[wp.float32],
    joint_qd: wp.array2d[wp.float32],
    joint_qdd: wp.array2d[wp.float32],
    body_q: wp.array2d[wp.transform],
    body_f_external: wp.array2d[wp.spatial_vector],
    joint_S: wp.array2d[wp.spatial_vector],
    body_v: wp.array2d[wp.spatial_vector],
    body_a: wp.array2d[wp.spatial_vector],
    body_f: wp.array2d[wp.spatial_vector],
):
    problem_idx, articulation_idx = wp.tid()
    joint_start = articulation_start[articulation_idx]
    joint_end = articulation_end[articulation_idx]

    for joint_idx in range(joint_start, joint_end):
        parent = joint_parent[joint_idx]
        child = joint_child[joint_idx]
        dof_start = joint_qd_start[joint_idx]
        dof_end = joint_qd_start[joint_idx + 1]
        _write_motion_subspaces(
            joint_idx,
            joint_type,
            joint_parent,
            joint_child,
            joint_q_start,
            joint_qd_start,
            joint_dof_dim,
            joint_X_p,
            joint_axis,
            body_com,
            body_q[problem_idx],
            joint_q[problem_idx],
            joint_S[problem_idx],
        )

        velocity_joint = wp.spatial_vector()
        acceleration_joint = wp.spatial_vector()
        for dof_idx in range(dof_start, dof_end):
            motion = joint_S[problem_idx, dof_idx]
            velocity_joint += motion * joint_qd[problem_idx, dof_idx]
            acceleration_joint += motion * joint_qdd[problem_idx, dof_idx]
        joint_kind = joint_type[joint_idx]
        if (joint_kind == JointType.FREE or joint_kind == JointType.DISTANCE) and parent < 0:
            velocity_com = wp.vec3()
            velocity_angular = wp.vec3()
            for axis_idx in range(3):
                velocity_com += (
                    wp.spatial_top(joint_S[problem_idx, dof_start + axis_idx])
                    * joint_qd[problem_idx, dof_start + axis_idx]
                )
                velocity_angular += (
                    wp.spatial_bottom(joint_S[problem_idx, dof_start + 3 + axis_idx])
                    * joint_qd[problem_idx, dof_start + 3 + axis_idx]
                )
            acceleration_joint += wp.spatial_vector(-wp.cross(velocity_angular, velocity_com), wp.vec3())
        elif joint_kind == JointType.D6:
            acceleration_joint += _d6_acceleration_bias(
                joint_idx,
                joint_parent,
                joint_q_start,
                joint_qd_start,
                joint_dof_dim,
                joint_X_p,
                joint_axis,
                body_q[problem_idx],
                joint_q[problem_idx],
                joint_qd[problem_idx],
                joint_S[problem_idx],
            )

        velocity_parent = wp.spatial_vector()
        acceleration_parent = wp.spatial_vector()
        if parent >= 0:
            velocity_parent = body_v[problem_idx, parent]
            acceleration_parent = body_a[problem_idx, parent]

        velocity = velocity_parent + velocity_joint
        acceleration = acceleration_parent + acceleration_joint + _spatial_cross(velocity, velocity_joint)
        body_v[problem_idx, child] = velocity
        body_a[problem_idx, child] = acceleration

        position_com = wp.transform_point(body_q[problem_idx, child], body_com[child])
        X_world_com = wp.transform(position_com, wp.transform_get_rotation(body_q[problem_idx, child]))
        inertia_world = transform_spatial_inertia(X_world_com, _body_inertia(body_mass[child], body_inertia[child]))
        momentum = inertia_world * velocity
        force = inertia_world * acceleration + _spatial_cross_dual(velocity, momentum)

        world_idx = wp.max(body_world[child], 0)
        force_gravity = body_mass[child] * gravity[world_idx]
        force -= wp.spatial_vector(force_gravity, wp.cross(position_com, force_gravity))
        if body_f_external:
            external = body_f_external[problem_idx, child]
            external_force = wp.spatial_top(external)
            external_torque = wp.spatial_bottom(external) + wp.cross(position_com, external_force)
            force -= wp.spatial_vector(external_force, external_torque)
        body_f[problem_idx, child] = force


@wp.kernel
def _inverse_dynamics_backward(
    articulation_start: wp.array[wp.int32],
    articulation_end: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_armature: wp.array[wp.float32],
    joint_qdd: wp.array2d[wp.float32],
    joint_S: wp.array2d[wp.spatial_vector],
    body_f: wp.array2d[wp.spatial_vector],
    joint_f_out: wp.array2d[wp.float32],
):
    problem_idx, articulation_idx = wp.tid()
    joint_start = articulation_start[articulation_idx]
    joint_end = articulation_end[articulation_idx]

    for offset in range(joint_end - joint_start):
        joint_idx = joint_end - offset - 1
        parent = joint_parent[joint_idx]
        child = joint_child[joint_idx]
        force = body_f[problem_idx, child]
        dof_start = joint_qd_start[joint_idx]
        dof_end = joint_qd_start[joint_idx + 1]
        for dof_idx in range(dof_start, dof_end):
            joint_f_out[problem_idx, dof_idx] = (
                wp.dot(joint_S[problem_idx, dof_idx], force) + joint_armature[dof_idx] * joint_qdd[problem_idx, dof_idx]
            )
        if parent >= 0:
            body_f[problem_idx, parent] += force


class DynamicsInverse:
    """Evaluate batched inverse dynamics with fixed, preallocated storage.

    The evaluator uses Newton's public generalized-coordinate convention. In
    particular, root ``FREE`` joint linear velocities and accelerations refer to
    the child center of mass and are expressed in the parent joint frame. Body
    wrenches use the :attr:`~newton.State.body_f` convention: world-frame force
    and torque at the body center of mass.

    ``CABLE`` joints and descendant ``FREE``/``DISTANCE`` joints are rejected
    at construction because their public acceleration bias is not yet
    implemented. Rejecting such models prevents a plausible but incorrect
    inverse-dynamics result.

    Args:
        model: Fixed articulation model. Its topology and inertial properties
            must remain unchanged while this evaluator is used.
        batch_size: Number of generalized states evaluated per call.
    """

    def __init__(self, model: Model, batch_size: int):
        if model.articulation_count < 1:
            raise ValueError("Inverse dynamics requires at least one articulation")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        joint_type = model.joint_type.numpy()
        unsupported = joint_type == int(JointType.CABLE)
        if np.any(unsupported):
            names = sorted({JointType(int(value)).name for value in joint_type[unsupported]})
            raise NotImplementedError(f"Inverse dynamics does not support joint types: {', '.join(names)}")
        free_descendant = np.isin(joint_type, (int(JointType.FREE), int(JointType.DISTANCE))) & (
            model.joint_parent.numpy() >= 0
        )
        if np.any(free_descendant):
            raise NotImplementedError("Inverse dynamics does not support descendant FREE or DISTANCE joints")

        self.model = model
        self.batch_size = batch_size
        self._body_q = wp.empty((batch_size, model.body_count), dtype=wp.transform, device=model.device)
        self._body_qd = wp.empty((batch_size, model.body_count), dtype=wp.spatial_vector, device=model.device)
        self._joint_S = wp.empty((batch_size, model.joint_dof_count), dtype=wp.spatial_vector, device=model.device)
        self._body_v = wp.empty((batch_size, model.body_count), dtype=wp.spatial_vector, device=model.device)
        self._body_a = wp.empty((batch_size, model.body_count), dtype=wp.spatial_vector, device=model.device)
        self._body_f = wp.empty((batch_size, model.body_count), dtype=wp.spatial_vector, device=model.device)

    def compute(
        self,
        joint_q: wp.array2d[wp.float32],
        joint_qd: wp.array2d[wp.float32],
        joint_qdd: wp.array2d[wp.float32],
        joint_f_out: wp.array2d[wp.float32],
        body_f: wp.array2d[wp.spatial_vector] | None = None,
    ) -> None:
        """Compute generalized forces for prescribed motion and body wrenches.

        The result satisfies ``joint_f_out = M(q) joint_qdd + h(q, joint_qd)
        - J(q)^T body_f`` and includes :attr:`~newton.Model.joint_armature`.
        The call performs no allocation or synchronization.

        Args:
            joint_q: Generalized positions [m or rad], shape
                ``[batch_size, joint_coord_count]``.
            joint_qd: Generalized velocities [m/s or rad/s], shape
                ``[batch_size, joint_dof_count]``.
            joint_qdd: Generalized accelerations [m/s^2 or rad/s^2], shape
                ``[batch_size, joint_dof_count]``.
            joint_f_out: Output generalized forces [N or N·m], shape
                ``[batch_size, joint_dof_count]``.
            body_f: Optional external world-frame body wrenches [N, N·m] at
                each body center of mass. The spatial-vector shape is
                ``[batch_size, body_count]``; scalar storage is
                ``[batch_size, body_count, 6]``.
        """
        expected_shapes = (
            ("joint_q", joint_q.shape, (self.batch_size, self.model.joint_coord_count)),
            ("joint_qd", joint_qd.shape, (self.batch_size, self.model.joint_dof_count)),
            ("joint_qdd", joint_qdd.shape, (self.batch_size, self.model.joint_dof_count)),
            ("joint_f_out", joint_f_out.shape, (self.batch_size, self.model.joint_dof_count)),
        )
        for name, shape, expected in expected_shapes:
            if shape != expected:
                raise ValueError(f"{name} must have shape {expected}, received {shape}")
        arrays = (joint_q, joint_qd, joint_qdd, joint_f_out)
        if any(array.device != self.model.device for array in arrays):
            raise ValueError("Inverse-dynamics arrays must reside on model.device")
        if body_f is not None:
            expected = (self.batch_size, self.model.body_count)
            if body_f.shape != expected:
                raise ValueError(f"body_f must have shape {expected}, received {body_f.shape}")
            if body_f.device != self.model.device:
                raise ValueError("body_f must reside on model.device")

        joint_f_out.zero_()
        eval_fk_batched(self.model, joint_q, joint_qd, self._body_q, self._body_qd)
        wp.launch(
            kernel=_inverse_dynamics_forward,
            dim=(self.batch_size, self.model.articulation_count),
            inputs=[
                self.model.articulation_start,
                self.model.articulation_end,
                self.model.joint_type,
                self.model.joint_parent,
                self.model.joint_child,
                self.model.joint_q_start,
                self.model.joint_qd_start,
                self.model.joint_dof_dim,
                self.model.joint_X_p,
                self.model.joint_axis,
                self.model.body_com,
                self.model.body_inertia,
                self.model.body_mass,
                self.model.body_world,
                self.model.gravity,
                joint_q,
                joint_qd,
                joint_qdd,
                self._body_q,
                body_f,
            ],
            outputs=[self._joint_S, self._body_v, self._body_a, self._body_f],
            device=self.model.device,
        )
        wp.launch(
            kernel=_inverse_dynamics_backward,
            dim=(self.batch_size, self.model.articulation_count),
            inputs=[
                self.model.articulation_start,
                self.model.articulation_end,
                self.model.joint_parent,
                self.model.joint_child,
                self.model.joint_qd_start,
                self.model.joint_armature,
                joint_qdd,
                self._joint_S,
                self._body_f,
            ],
            outputs=[joint_f_out],
            device=self.model.device,
        )
