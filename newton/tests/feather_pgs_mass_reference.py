# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""An independent float64 evaluation of an articulation's joint-space mass matrix, the conformance reference's reference.

:class:`MassReference` shares no code with the solver: it composes the world poses from the joint coordinates, writes
each coordinate's motion column about the articulation's origin (its first root's center of mass) and sums
``H = sum_b J_b^T M_b J_b`` over the bodies, ``J_b`` the columns of the body's ancestors and ``M_b`` its spatial inertia
at the origin, in Newton's twist layout ``(v, w)``. Numpy only, in float64.
"""

import numpy as np

import newton


def _rotation(quaternion) -> np.ndarray:
    """The rotation matrix of a unit quaternion ``(x, y, z, w)``."""
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _turn(axis, angle) -> np.ndarray:
    """The rotation of ``angle`` about the unit ``axis`` (Rodrigues)."""
    k = _skew(axis)
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * k @ k


def _skew(v) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _column(point, axis, angular) -> np.ndarray:
    """The unit twist about the origin of ``axis`` through ``point``: a turn, or a slide along it."""
    return np.r_[np.cross(point, axis), axis] if angular else np.r_[axis, 0.0, 0.0, 0.0]


class MassReference:
    """A model's tables, read once, and the mass matrices of its articulations at given coordinates."""

    def __init__(self, model):
        self.start = model.articulation_start.numpy()
        self.member = model.joint_articulation.numpy()
        self.types = model.joint_type.numpy()
        self.parent = model.joint_parent.numpy()
        self.child = model.joint_child.numpy()
        self.q_start = model.joint_q_start.numpy()
        self.qd_start = model.joint_qd_start.numpy()
        self.dim = model.joint_dof_dim.numpy()
        self.axis = model.joint_axis.numpy().astype(np.float64)
        self.X_p = model.joint_X_p.numpy().astype(np.float64)
        self.X_c = model.joint_X_c.numpy().astype(np.float64)
        self.mass = model.body_mass.numpy().astype(np.float64)
        self.inertia = model.body_inertia.numpy().astype(np.float64)
        self.com = model.body_com.numpy().astype(np.float64)

    def _joint(self, joint, q, R, point):
        """The joint's motion from the frame ``(R, point)`` at coordinates ``q``: its rotation, its translation, and
        its columns about the origin (``point`` relative to it)."""
        kind, (lin, ang) = int(self.types[joint]), (int(v) for v in self.dim[joint])
        axes = self.axis[self.qd_start[joint] : self.qd_start[joint] + lin + ang]
        if kind in (int(newton.JointType.PRISMATIC), int(newton.JointType.REVOLUTE)):
            angular = kind == int(newton.JointType.REVOLUTE)
            column = _column(point, R @ axes[0], angular)
            return (_turn(axes[0], q[0]), np.zeros(3)) if angular else (np.eye(3), axes[0] * q[0]), column[None]
        if kind == int(newton.JointType.BALL):
            return (_rotation(q[:4]), np.zeros(3)), np.stack([_column(point, R @ e, True) for e in np.eye(3)])
        if kind == int(newton.JointType.FIXED):
            return (np.eye(3), np.zeros(3)), np.zeros((0, 6))
        if kind in (int(newton.JointType.FREE), int(newton.JointType.DISTANCE)):
            columns = [_column(point, R @ e, False) for e in np.eye(3)]
            columns += [np.r_[0.0, 0.0, 0.0, R @ e] for e in np.eye(3)]
            return (_rotation(q[3:7]), q[:3]), np.stack(columns)
        if kind == int(newton.JointType.D6):
            # The chain of its axes: the slides, then each turn about the origin they moved, carried through the
            # turns before it.
            slide = sum((axes[k] * q[k] for k in range(lin)), np.zeros(3))
            columns = [_column(point, R @ axes[k], False) for k in range(lin)]
            pivot = point + R @ slide
            rotation = np.eye(3)
            for k in range(ang):
                axis = rotation @ axes[lin + k]
                columns.append(_column(pivot, R @ axis, True))
                rotation = _turn(axis, q[lin + k]) @ rotation
            return (rotation, slide), np.stack(columns) if columns else np.zeros((0, 6))
        raise ValueError(f"No reference law of joint type {kind}")

    def mass_matrix(self, joint_q, articulation) -> np.ndarray:
        """The mass matrix of ``articulation`` at the coordinates ``joint_q``, without armature or drives."""
        tree = np.flatnonzero(self.member == articulation)
        first_dof = int(self.qd_start[tree[0]])
        size = int(sum(self.qd_start[joint + 1] - self.qd_start[joint] for joint in tree))
        H = np.zeros((size, size))
        pose, supports, origin = {}, {}, None
        columns = np.zeros((size, 6))
        for joint in tree:
            parent, child = int(self.parent[joint]), int(self.child[joint])
            R_p, t_p = pose.get(parent, (np.eye(3), np.zeros(3)))
            R_a, t_a = R_p @ _rotation(self.X_p[joint, 3:]), t_p + R_p @ self.X_p[joint, :3]
            q = joint_q[self.q_start[joint] : self.q_start[joint + 1]].astype(np.float64)
            # The first root gives the origin, which its columns need: its body's pose comes first.
            (R_j, t_j), _ = self._joint(joint, q, R_a, np.zeros(3))
            R_c = R_a @ R_j @ _rotation(self.X_c[joint, 3:]).T
            t_c = t_a + R_a @ t_j - R_c @ self.X_c[joint, :3]
            pose[child] = (R_c, t_c)
            if origin is None:
                origin = t_c + R_c @ self.com[child]
            first = int(self.qd_start[joint]) - first_dof
            own = self._joint(joint, q, R_a, t_a - origin)[1]
            columns[first : first + own.shape[0]] = own
            supports[child] = [*supports.get(parent, []), *range(first, first + own.shape[0])]
            r = t_c + R_c @ self.com[child] - origin
            cross = _skew(r)
            mass = self.mass[child]
            M = np.zeros((6, 6))
            M[:3, :3] = mass * np.eye(3)
            M[:3, 3:] = -mass * cross
            M[3:, :3] = mass * cross
            M[3:, 3:] = R_c @ self.inertia[child] @ R_c.T - mass * cross @ cross
            support = supports[child]
            J = columns[support]
            H[np.ix_(support, support)] += J @ M @ J.T
        return H
