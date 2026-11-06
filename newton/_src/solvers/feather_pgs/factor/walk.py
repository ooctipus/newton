# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The walk (Z_UNION §1.2): the tree factor as a six-number state's steps along the tree, in local frames.

:class:`TreeWalk` is the walk's reference on the host: ``up`` (a row's footprints ``Z = L^-1 P J^T``), ``down`` (the
leave ``P^T L^-T u``) and ``meet``. On the device (FL3.1) a lane walks a seed up and a lane group a tree down, the path
table in constant memory: the walk's preludes (:func:`walk_prelude`, :func:`walk_row_prelude`, :func:`walk_down_prelude`, :func:`walk_loads_prelude`), the leave
(:func:`walk_leave_kernel`), the walk's device struct (:class:`WalkTrees`) and its refresh, the inward pass
(:func:`walk_step_kernel`). The factor (:class:`.tree_factor.TreeFactor`) holds the walk's data.
"""

from __future__ import annotations

import functools
import hashlib
from string import Template

import numpy as np
import warp as wp

from ....core import memory
from ....core.execution import Use, Work
from ....sim import JointType
from .. import tree_scan as ts
from ..device_preludes import DevicePrelude, memory_words, substitute
from ..tree_scan import WALK_AHEAD, WALK_BOTH, WALK_TOUCH, WALK_UNROLL, WalkClass, indent_text
from .kernels import Clock, clock_advance

__all__ = [
    "FREE_BODY_WALK",
    "WALK_COORDINATE",
    "WALK_FRAME",
    "WALK_LINK",
    "WALK_NOW",
    "WALK_PIVOT",
    "WALK_UNPACKED",
    "leave_work",
    "walk_down_prelude",
    "walk_leave_kernel",
    "walk_prelude",
    "walk_row_prelude",
    "walk_tables",
    "walk_work",
]


def _cross(d: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``d x v`` for a 3-vector ``d`` and ``v`` of shape ``(3, ...)``, in their dtype."""
    return np.stack((d[1] * v[2] - d[2] * v[1], d[2] * v[0] - d[0] * v[2], d[0] * v[1] - d[1] * v[0]))


def _rotation(quaternion) -> np.ndarray:
    """The rotation matrix of a unit quaternion ``(x, y, z, w)``, in float64."""
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _turn(axis, angle) -> np.ndarray:
    """The rotation of ``angle`` about the unit ``axis``, in float64."""
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * k @ k


def local_frames(model, joint_q: np.ndarray, articulation: int, poses: np.ndarray | None = None) -> tuple:
    """One articulation's tree in local frames at the coordinates ``joint_q``, in float64 from the model's data.

    A body's frame sits at its joint's anchor (the parent's joint frame), with world axes; a free body's at its centre
    of mass. Its coordinates' columns are the factor's (:func:`.spatial_laws.joint_motion`: a free joint's turn about
    the articulation's origin, its first root's centre of mass) moved to that frame, and its spatial inertia is
    taken there, in Newton's ``(v, w)`` layout. Nothing is held farther than one link from its frame.

    Args:
        model: The model.
        joint_q: The coordinates.
        articulation: The articulation.
        poses: The bodies' world poses ``[bodies, 7]`` to read instead of composing them in float64 from ``joint_q``
            (a float32 forward kinematics', say).

    Returns:
        ``(parent, first, columns, inertia, offset)`` for :class:`TreeWalk`: each body's parent (-1: the world), each
        body's first coordinate (and the count after the last), each coordinate's column ``[n, 6]``, each body's
        inertia ``[bodies, 6, 6]`` and each body's parent's frame less its own ``[bodies, 3]`` (zero at a root).
    """
    start = model.articulation_start.numpy()
    joints = np.arange(start[articulation], start[articulation + 1])
    kinds, dims = model.joint_type.numpy()[joints], model.joint_dof_dim.numpy()[joints]
    parent_body, child_body = model.joint_parent.numpy()[joints], model.joint_child.numpy()[joints]
    q_start, qd_start = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
    X_p, X_c = model.joint_X_p.numpy().astype(np.float64), model.joint_X_c.numpy().astype(np.float64)
    axes = model.joint_axis.numpy().astype(np.float64)
    mass, com = model.body_mass.numpy().astype(np.float64), model.body_com.numpy().astype(np.float64)
    body_inertia = model.body_inertia.numpy().astype(np.float64)
    local = {int(body): index for index, body in enumerate(child_body)}
    if len(local) < len(joints) or any(body >= 0 and int(body) not in local for body in parent_body):
        raise ValueError("The articulation's joints are not a tree: walk the solver's model, whose loops are cut")
    parent = np.array([local[int(body)] if body >= 0 else -1 for body in parent_body], dtype=np.int64)
    first = np.append(qd_start[joints], qd_start[joints[-1] + 1]) - qd_start[joints[0]]
    count = len(joints)
    rotation, position, frame = np.zeros((count, 3, 3)), np.zeros((count, 3)), np.zeros((count, 3))
    columns, inertia = np.zeros((int(first[-1]), 6)), np.zeros((count, 6, 6))
    origin = None
    for b, joint in enumerate(joints):
        p = parent[b]
        R_p, t_p = (rotation[p], position[p]) if p >= 0 else (np.eye(3), np.zeros(3))
        R_a, t_a = R_p @ _rotation(X_p[joint, 3:]), t_p + R_p @ X_p[joint, :3]
        q = np.asarray(joint_q[q_start[joint] : q_start[joint + 1]], dtype=np.float64)
        lin, ang = int(dims[b, 0]), int(dims[b, 1])
        axis = axes[qd_start[joint] : qd_start[joint + 1]]
        kind = int(kinds[b])
        # The joint's motion (rotation, translation) and its columns at the anchor, world axes.
        own = []
        if kind == int(JointType.REVOLUTE):
            motion = (_turn(axis[0], q[0]), np.zeros(3))
            own.append(np.r_[0.0, 0.0, 0.0, R_a @ axis[0]])
        elif kind == int(JointType.PRISMATIC):
            motion = (np.eye(3), axis[0] * q[0])
            own.append(np.r_[R_a @ axis[0], 0.0, 0.0, 0.0])
        elif kind == int(JointType.BALL):
            motion = (_rotation(q[:4]), np.zeros(3))
            own += [np.r_[0.0, 0.0, 0.0, R_a @ e] for e in np.eye(3)]
        elif kind in (int(JointType.FREE), int(JointType.DISTANCE)):
            motion = (_rotation(q[3:7]), q[:3])
            own += [np.r_[R_a @ e, 0.0, 0.0, 0.0] for e in np.eye(3)]
            own += [np.r_[0.0, 0.0, 0.0, R_a @ e] for e in np.eye(3)]
        elif kind == int(JointType.D6):
            slide = sum((axis[k] * q[k] for k in range(lin)), np.zeros(3))
            own += [np.r_[R_a @ axis[k], 0.0, 0.0, 0.0] for k in range(lin)]
            turned = np.eye(3)
            for k in range(ang):
                w = R_a @ (turned @ axis[lin + k])
                # A turn about the pivot the slides moved, at the anchor: v = (R_a slide) x w.
                own.append(np.r_[np.cross(R_a @ slide, w), w])
                turned = _turn(turned @ axis[lin + k], q[lin + k]) @ turned
            motion = (turned, slide)
        elif kind == int(JointType.FIXED):
            motion = (np.eye(3), np.zeros(3))
        else:
            raise ValueError(f"No walk of joint type {kind}")
        R_c = R_a @ motion[0] @ _rotation(X_c[joint, 3:]).T
        t_c = t_a + R_a @ motion[1] - R_c @ X_c[joint, :3]
        if poses is not None:
            pose = np.asarray(poses[child_body[b]], dtype=np.float64)
            R_c, t_c = _rotation(pose[3:]), pose[:3]
        rotation[b], position[b] = R_c, t_c
        centre = t_c + R_c @ com[child_body[b]]
        if origin is None:
            origin = centre
        free = kind in (int(JointType.FREE), int(JointType.DISTANCE))
        frame[b] = centre if free else t_a
        if free:
            # The factor turns a free joint about the articulation's origin: the same twists at the body's frame.
            for column in own[3:]:
                column[:3] += np.cross(column[3:], frame[b] - origin)
        if own:
            columns[first[b] : first[b + 1]] = np.stack(own)
        r = centre - frame[b]
        cross = np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])
        m = mass[child_body[b]]
        inertia[b, :3, :3] = m * np.eye(3)
        inertia[b, :3, 3:] = -m * cross
        inertia[b, 3:, :3] = m * cross
        inertia[b, 3:, 3:] = R_c @ body_inertia[child_body[b]] @ R_c.T - m * cross @ cross
    offset = np.where(parent[:, None] >= 0, frame[np.maximum(parent, 0)] - frame, 0.0)
    return parent, first, columns, inertia, offset


class TreeWalk:
    """One articulation's tree factor as a walk (Z_UNION §1.2, Theorem A), in local frames (:func:`local_frames`).

    The inward pass eliminates the coordinates leaf first, one at a time, from each body's own inertia plus what its
    children pass up,

        ell_c = (s_c^T I s_c + D_c)^1/2,    ut_c = I s_c / ell_c,    I <- I - ut_c ut_c^T,

    ``D`` the coordinates' armature and drive stiffness, and then moves the body's remaining inertia one link up. Its
    data ``(s, ut, ell)`` is the leaf-first factor ``P (H + D) P^T = L L^T`` (``L_cc = ell_c``, and ``L`` is
    ``s_a^T ut_c`` for ``a`` an ancestor coordinate of ``c``), so:

    - :meth:`up` is ``Z = L^-1 P J^T``, a row's footprints: its seed, a wrench at a body or a covector on the
      coordinates, walked up to the root, each coordinate taking ``e_c^T p`` of the state ``p`` and passing
      ``p - ut_c z_c``, the wrench it cannot feel;
    - :meth:`down` is the leave ``P^T L^-T u``, the walk run back down; its state at a body is the body's twist;
    - :meth:`meet` is ``Z_k^T Z_l``: two walks carried to their lowest common ancestor, met through its Gram.

    A footprint ``z_c`` sits at its coordinate ``c`` (the kit's entry ``c`` of a side). The arithmetic runs in
    ``dtype``; a negative pivot leaves NaN, as the factor would. The step calls none of it.

    Args:
        parent: Each body's parent (-1: the world), every parent before its children.
        first: Each body's first coordinate, and the count after the last.
        columns: Each coordinate's motion column at its body's frame, ``[n, 6]``.
        inertia: Each body's spatial inertia at its frame, ``[bodies, 6, 6]``.
        offset: Each body's parent's frame less its own, ``[bodies, 3]``.
        diagonal: Each coordinate's armature and drive stiffness.
        dtype: The walk's arithmetic.
    """

    def __init__(self, parent, first, columns, inertia, offset, diagonal, *, dtype=np.float64):
        self.dtype = np.dtype(dtype)
        self.parent = np.asarray(parent, dtype=np.int64)
        self.first = np.asarray(first, dtype=np.int64)
        if np.any(self.parent >= np.arange(self.parent.size)):
            raise ValueError("The walk needs every parent before its children")
        self.s = np.asarray(columns, dtype=self.dtype)
        self.offset = np.asarray(offset, dtype=self.dtype)
        held = np.asarray(inertia, dtype=self.dtype).copy()
        added = np.asarray(diagonal, dtype=self.dtype)
        self.ell = np.zeros(self.s.shape[0], dtype=self.dtype)
        self.ut = np.zeros_like(self.s)
        self.own = np.zeros(self.s.shape[0], dtype=self.dtype)
        with np.errstate(invalid="ignore", divide="ignore"):
            for b in range(self.parent.size - 1, -1, -1):
                I = held[b]
                for c in range(self.first[b + 1] - 1, self.first[b] - 1, -1):
                    Is = I @ self.s[c]
                    self.own[c] = self.s[c] @ Is
                    self.ell[c] = np.sqrt(self.own[c] + added[c])
                    self.ut[c] = Is / self.ell[c]
                    I = I - np.outer(self.ut[c], self.ut[c])
                if self.parent[b] >= 0:
                    held[self.parent[b]] += self._lift(I, self.offset[b])

    @property
    def pivots(self) -> np.ndarray:
        """Each coordinate's pivot ``ell_c^2``."""
        return self.ell * self.ell

    @property
    def own_pivots(self) -> np.ndarray:
        """Each coordinate's pivot before its own added diagonal, ``s_c^T I s_c`` (the split pivot's, SAT_PHYS §1.2:
        formed before a drive's ``K`` is added, so no cancellation takes it back out)."""
        return self.own

    def _lift(self, inertia: np.ndarray, d: np.ndarray) -> np.ndarray:
        """``inertia`` at a body's frame, taken at its parent's (``d`` = the parent's frame less the body's)."""
        D = np.array([[0.0, -d[2], d[1]], [d[2], 0.0, -d[0]], [-d[1], d[0], 0.0]], dtype=self.dtype)
        to = np.eye(6, dtype=self.dtype)
        to[:3, 3:] = D  # a twist at the parent's frame, taken at the body's: v + d x w
        wrench = np.eye(6, dtype=self.dtype)
        wrench[3:, :3] = -D  # a wrench at the body's frame, taken at the parent's: t - d x f
        return wrench @ inertia @ to

    def _step(self, b: int, p: np.ndarray, x: np.ndarray | None, z: np.ndarray | None) -> np.ndarray:
        """Walk body ``b``'s coordinates, leaf first, from the state ``p`` at its frame: store the footprints in
        ``z`` (when given), with ``x`` a covector's own entries, and return the state its joint passes."""
        for c in range(self.first[b + 1] - 1, self.first[b] - 1, -1):
            footprint = self.s[c] @ p if x is None else x[c] + self.s[c] @ p
            footprint = footprint / self.ell[c]
            if z is not None:
                z[c] = footprint
            p = p - np.multiply.outer(self.ut[c], footprint)
        return p

    def _carry(self, p: np.ndarray, b: int) -> np.ndarray:
        """The state ``p`` passed by body ``b``'s joint, at the parent's frame: ``t - d x f``."""
        d = self.offset[b]
        return np.concatenate((p[:3], p[3:] - _cross(d, p[:3])))

    def up(self, body=None, wrench=None, covector=None) -> tuple[np.ndarray, np.ndarray]:
        """Footprints ``Z = L^-1 P J^T`` of rows seeded by ``wrench`` at ``body`` (a wrench ``(f, t)`` at the body's
        frame, ``[6]`` or one column per row ``[6, m]``, with ``body`` one body or one per row) and by ``covector`` on
        the coordinates (``[n]`` or ``[n, m]``): a limit's, a mimic's, or any row's ``J``.

        Returns:
            The footprints ``[n]`` or ``[n, m]`` by coordinate, and the state each root passes to the world
            ``[bodies, 6, ...]`` (zero above a free root).
        """
        batch = () if wrench is None and covector is None else None
        if wrench is not None:
            wrench = np.asarray(wrench, dtype=self.dtype)
            batch = wrench.shape[1:]
        if covector is not None:
            covector = np.asarray(covector, dtype=self.dtype)
            batch = covector.shape[1:]
        count = self.parent.size
        state = np.zeros((count, 6, *batch), dtype=self.dtype)
        active = np.zeros(count, dtype=bool)
        if wrench is not None:
            if np.ndim(body) == 0:
                state[int(body)] += wrench
                active[int(body)] = True
            else:
                for row, b in enumerate(np.asarray(body).tolist()):
                    state[b, :, row] += wrench[:, row]
                    active[b] = True
        if covector is not None:
            touched = np.flatnonzero(np.any(covector.reshape(covector.shape[0], -1) != 0.0, axis=1))
            for c in touched:
                active[np.searchsorted(self.first, c, side="right") - 1] = True
        z = np.zeros((self.s.shape[0], *batch), dtype=self.dtype)
        for b in range(count - 1, -1, -1):
            if not active[b]:
                continue
            state[b] = self._step(b, state[b], covector, z)
            if self.parent[b] >= 0:
                state[self.parent[b]] += self._carry(state[b], b)
                active[self.parent[b]] = True
                state[b] = 0.0
        return z, state

    def down(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The leave ``P^T L^-T u`` of ``u`` (``[n]`` or ``[n, m]``, by coordinate as :meth:`up` returns them).

        Returns:
            The joint velocities ``[n, ...]`` and each body's twist ``(v, w)`` at its frame ``[bodies, 6, ...]``.
        """
        u = np.asarray(u, dtype=self.dtype)
        count = self.parent.size
        twist = np.zeros((count, 6, *u.shape[1:]), dtype=self.dtype)
        qd = np.zeros_like(u)
        for b in range(count):
            p = self.parent[b]
            if p >= 0:
                # The parent's twist at this body's frame: v + d x w, d = the parent's frame less this one.
                V = twist[p]
                V = np.concatenate((V[:3] + _cross(self.offset[b], V[3:]), V[3:]))
            else:
                V = twist[b]
            for c in range(self.first[b], self.first[b + 1]):
                qd[c] = (u[c] - self.ut[c] @ V) / self.ell[c]
                V = V + np.multiply.outer(self.s[c], qd[c])
            twist[b] = V
        return qd, twist

    def ancestors(self, b: int) -> list[int]:
        """``b`` and its ancestors, ``b`` first."""
        path = []
        while b >= 0:
            path.append(int(b))
            b = int(self.parent[b])
        return path

    def carry(self, body: int, wrench: np.ndarray, to: int) -> np.ndarray:
        """``wrench`` at ``body`` walked up to its ancestor ``to``: the state that reaches ``to``'s frame, before its
        coordinates (``Psi`` of Z_UNION §1.2)."""
        p = np.asarray(wrench, dtype=self.dtype)
        b = body
        while b != to:
            if b < 0:
                raise ValueError(f"Body {to} is not an ancestor of body {body}")
            p = self._carry(self._step(b, p, None, None), b)
            b = int(self.parent[b])
        return p

    def meet(self, k: int, w_k: np.ndarray, l: int, w_l: np.ndarray) -> np.ndarray:
        """``Z_k^T Z_l`` of the rows seeded by ``w_k`` at ``k`` and ``w_l`` at ``l`` (``[6]`` or ``[6, m]``): both
        walks carried to their lowest common ancestor ``c``, met through ``Gamma_c^T Gamma_c``, the Gram of the
        footprints of ``c``'s six unit wrenches (never the 6x6 recurrence, which loses float32 on deep chains)."""
        path = set(self.ancestors(l))
        c = next((b for b in self.ancestors(k) if b in path), -1)
        w_k, w_l = np.asarray(w_k, dtype=self.dtype), np.asarray(w_l, dtype=self.dtype)
        if c < 0:
            return np.zeros((*w_k.shape[1:], *w_l.shape[1:]), dtype=self.dtype)
        gamma, _ = self.up(c, np.eye(6, dtype=self.dtype))
        return self.carry(k, w_k, c).T @ (gamma.T @ gamma) @ self.carry(l, w_l, c)

    def device_data(self) -> tuple[np.ndarray, np.ndarray]:
        """The walk's data as the device walks it (:func:`walk_prelude`), float32: each coordinate's record
        (:func:`pack_walk_records` of ``s``, ``ut`` and ``1 / ell``), and each body's link (:data:`WALK_LINK`), its
        frame point (:meth:`frame_points`)."""
        data = pack_walk_records(self.s, self.ut, 1.0 / self.ell, np.zeros(self.s.shape[0]))
        link = np.zeros((self.parent.size, WALK_LINK), dtype=np.float32)
        link[:, 0:3] = self.frame_points()
        return data, link

    def frame_points(self) -> np.ndarray:
        """Each body's frame point with the roots at the origin: its parent's less its offset (``[bodies, 3]``)."""
        frames = np.zeros((self.parent.size, 3))
        for b, up in enumerate(self.parent):
            if up >= 0:
                frames[b] = frames[up] - np.asarray(self.offset[b], dtype=np.float64)
        return frames

    @property
    def path_table(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """The tree's path table (:func:`walk_prelude`): each body's parent, and each body's first coordinate with the
        count after the last."""
        return tuple(int(b) for b in self.parent), tuple(int(c) for c in self.first)


# ---------------------------------------------------------------------------------------------------------------------
# The walk on the device (FL3.1): a lane per seed up, a lane per tree down, the path table in constant memory
# ---------------------------------------------------------------------------------------------------------------------

WALK_COORDINATE = 12
"""Floats per coordinate of the device walk's record, three aligned ``float4`` (48 B): its column's non-zero half and
``1 / ell``; ``ut`` 0..3; ``ut`` 4..5, the pivot without any drive's ``K`` (``P0 = s^T I s + R``, the split pivot's,
SAT_PHYS §1.2) and the side offset. A column at its body's frame has one non-zero half for every joint but a D6 joint's
turns after its slides and a free joint's turns below a root: the record names its half by ``1 / ell``'s sign bit (set:
the angular half), and a column with both halves keeps its linear half in a side ``float4`` whose offset in floats from
the record its last word holds (zero: none; :func:`pack_walk_records`). Readers take it unpacked, :data:`WALK_UNPACKED`
floats (``s`` 6, ``ut`` 6, ``1 / ell``, ``P0``), through ``WALK_READ``."""

WALK_UNPACKED = 14
"""Floats of a coordinate's record unpacked (``WALK_READ``): ``s`` 6, ``ut`` 6, ``1 / ell``, ``P0``."""

WALK_LINK = 4
"""Floats per body of the device walk's data: its frame's world point and a pad, one aligned ``float4``. A walk forms a
body's offset to its parent's frame from the two frame points (``WALK_READ_LINK``, ``WALK_LINK_IN``), the float32
difference the refresh takes (``WALK_OFFSET``); the step's links (:attr:`WalkTrees.now_link`) take the same layout."""

WALK_PIVOT = 13
"""Where a coordinate's unpacked record holds its pivot without ``K`` (:data:`WALK_UNPACKED`)."""


def pack_walk_records(s: np.ndarray, ut: np.ndarray, inverse: np.ndarray, pivot: np.ndarray) -> np.ndarray:
    """Coordinates' records as the refresh stores them (:data:`WALK_COORDINATE`), float32: each column ``s`` (``[n,
    6]``) by its non-zero half, ``ut``, ``1 / ell`` and ``P0``; with any column of two halves the rows double, the side
    ``float4`` of coordinate ``c`` at row ``n + c`` (its offset ``12 n`` floats)."""
    s = np.asarray(s, dtype=np.float32).reshape(-1, 6)
    n = s.shape[0]
    turns = np.any(s[:, 3:6] != 0.0, axis=1)
    both = turns & np.any(s[:, 0:3] != 0.0, axis=1)
    data = np.zeros((2 * n if both.any() else n, WALK_COORDINATE), dtype=np.float32)
    data[:n, 0:3] = np.where(turns[:, None], s[:, 3:6], s[:, 0:3])
    data[:n, 3] = np.where(turns, -1.0, 1.0) * np.asarray(inverse, dtype=np.float32).reshape(n)
    data[:n, 4:10] = np.asarray(ut, dtype=np.float32).reshape(n, 6)
    data[:n, 10] = np.asarray(pivot, dtype=np.float32).reshape(n)
    if both.any():
        data[:n, 11] = np.where(both, np.int32(WALK_COORDINATE * n), np.int32(0)).view(np.float32)
        data[n:, 0:3] = np.where(both[:, None], s[:, 0:3], 0.0)
    return data


def unpack_walk_records(data: np.ndarray, n: int) -> np.ndarray:
    """The first ``n`` records of ``data`` (:func:`pack_walk_records`) unpacked, :data:`WALK_UNPACKED` floats a
    coordinate (``s``, ``ut``, ``1 / ell``, ``P0``), as ``WALK_READ`` reads them."""
    data = np.asarray(data, dtype=np.float32)
    flat = data.reshape(-1)
    out = np.zeros((n, WALK_UNPACKED), dtype=np.float32)
    turns = np.signbit(data[:n, 3])
    side = data[:n, 11].view(np.int32)
    lin = np.zeros((n, 3), dtype=np.float32)
    for c in np.flatnonzero(side):
        at = WALK_COORDINATE * c + int(side[c])
        lin[c] = flat[at : at + 3]
    out[:, 0:3] = np.where(turns[:, None], lin, data[:n, 0:3])
    out[:, 3:6] = np.where(turns[:, None], data[:n, 0:3], 0.0)
    out[:, 6:12] = data[:n, 4:10]
    out[:, 12] = np.abs(data[:n, 3])
    out[:, 13] = data[:n, 10]
    return out


WALK_SLIP_BATCH = 8
"""The steps of a slip's transposed release a walk loads together (``WALK_SLIP_BACK``: the leave's, the audit's): one
global round trip a batch of its path, in place of one a step."""

WALK_FRAME = 0
"""Where a body's link holds its frame's world point (:data:`WALK_LINK`), at the refresh's poses: the factor's
inertias are taken there."""

WALK_NOW = 6
"""Floats per coordinate of the step's columns (:attr:`WalkTrees.now`, a ``wp.spatial_vector`` each, ``S``'s layout):
in a model without a column of two halves ``S`` itself (``joint_S_s``, which the finish stores anyway), whose half at the
body's frame the record names (its other half the lever about the origin a turn carries, which the frame drops); with
one, the finish's columns at each body's frame, both halves (``WALK_READ_COLUMN``). Its links,
:attr:`WalkTrees.now_link`, take :data:`WALK_LINK`'s layout."""


_WALK_TABLE = """    // The classes' path tables in constant memory, class after class: each body's parent (WALK_PARENT), its first
    // coordinate and the count after the last (WALK_FIRST, a class's rows one more), each class's first body
    // (WALK_CLASS). WALK_TREE(cls): the class of the tree a lane walks next (class 0 until set).
    $parent_table
    $first_table
    $class_table
    int WALK_AT = 0, WALK_AT_FIRST = 0, WALK_BODIES = WALK_CLASS[1];
    const auto WALK_TREE = [&](const int cls) {
        WALK_AT = WALK_CLASS[cls];
        WALK_AT_FIRST = WALK_AT + cls;
        WALK_BODIES = WALK_CLASS[cls + 1] - WALK_AT;
    };
    const auto WALK_PARENT_OF = [&](const int b) -> int { return WALK_PARENT[WALK_AT + b]; };
    const auto WALK_FIRST_OF = [&](const int b) -> int { return WALK_FIRST[WALK_AT_FIRST + b]; };$joins"""

_WALK_JOINS = """
    // Each body's join slot (WALK_JOIN, WalkClass.slots: -1 where no child other than the next body meets it).
    $join_table
    const auto WALK_JOIN_OF = [&](const int b) -> int { return WALK_JOIN[WALK_AT + b]; };"""


def walk_table(tables: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...], joins: bool = False) -> str:
    """The classes' path tables in constant memory (:data:`_WALK_TABLE`): ``WALK_TREE(cls)``, ``WALK_PARENT_OF``,
    ``WALK_FIRST_OF`` and ``WALK_BODIES``, with ``joins`` each body's join slot (``WALK_JOIN_OF``,
    :attr:`.tree_scan.WalkClass.slots`). No walking tree: one class of no body.

    Args:
        tables: Each class's path table: each body's parent (-1: the world, every parent before its children), and
            each body's first coordinate with the count after the last.
        joins: Whether to declare each body's join slot.
    """
    tables = tables or (((), (0,)),)
    if any(len(first) != len(parent) + 1 or any(p >= b for b, p in enumerate(parent)) for parent, first in tables):
        raise ValueError("a walk's path table needs every parent before its children and a first coordinate per body")
    starts = [0]
    for parent, _ in tables:
        starts.append(starts[-1] + len(parent))
    slots = [s for parent, first in tables for s in ts.WalkClass(parent, first).slots[0]]
    return substitute(
        _WALK_TABLE,
        parent_table=ts.constant_table("WALK_PARENT", [p for parent, _ in tables for p in parent]),
        first_table=ts.constant_table("WALK_FIRST", [f for _, first in tables for f in first]),
        class_table=ts.constant_table("WALK_CLASS", starts),
        joins=substitute(_WALK_JOINS, join_table=ts.constant_table("WALK_JOIN", slots)) if joins else "",
    )


_WALK = """    // The walk of a tree factor (Z_UNION section 1.2, tree_factor.TreeWalk) over the classes' path tables above
    // (WALK_PARENT_OF, WALK_FIRST_OF, WALK_BODIES): each body's parent (-1: the world, every parent before its
    // children) and first coordinate. data holds each coordinate's s, ut ((v, w), six each) and 1 / ell
    // (WALK_COORDINATE floats), link each body's offset to its parent's frame (WALK_LINK floats), as
    // TreeWalk.device_data and TreeFactor.walk_step lay them out.
    //
    // WALK_UP(data, link, now, now_link, shift, body, top, p, w, sink): the states p of $rows rows, p[6 r + k] the wrench
    // (f, t) of row r at body's frame, walked up to the root from coordinate top of body (its last for a wrench seed):
    // each coordinate c takes the footprint z = (s_c . p) / ell_c, given to sink(r, c, z, j), and passes p - ut_c z; a
    // joint passes t - d x f to its parent. now (null for rows seeded by covectors alone): the step's geometry the
    // finish published, each coordinate's column N_c at its body's frame ($now floats a coordinate) and each body's
    // frame point in now_link (WALK_LINK floats a body, the point at WALK_FRAME), with w[6 r + k] row r's seed wrench at
    // the current body's frame (link's frame point, the factor's). The frame points are the articulation's, relative
    // to its origin: shift is the factor's origin less the step's (TreeFactor's walk_origin, zero on a refresh step).
    // A seed enters as the step's J and the factor whitens
    // it (a held factor between refreshes, L^-1 P J^T as the dense rows do): the step's column of c at the factor's
    // frame is s_now_c = (N_v + N_w x (frame - frame_now), N_w), the J entry j = s_now_c . w (the point is any one: w
    // is at the same), and p is the seed's wrench less what the factor has taken, so the footprint takes
    // (s_now_c - s_c) . w more; w passes each joint as p does. The shift is the body's motion since the refresh, small,
    // and zero on a refresh step, where the two geometries are one: one residual state keeps FL3.1's precision with no
    // lever about the articulation's origin. The sink's j is the step's J entry (zero without now), so a row's incident
    // J v is the sum of j v_c along its walk.
    const auto WALK_UP = [&](const float* data, const float* link, const float* now, const float* now_link,
                             const float* shift, int body, int top, float* p, float* w, auto&& sink) {
$up_head
$record_c
                // The step's column at the factor's frame: its linear part shifted there from the step's frame, its
                // angular part as is (the frames read at each coordinate, so no register holds them across the walk;
                // once for every row).
$column_c
                float e0 = 0.0f, e1 = 0.0f, e2 = 0.0f;
                if (now != nullptr) {
$point_b
                    const float o0 = (l[$frame] - f[0]) + shift[0], o1 = (l[$frame + 1] - f[1]) + shift[1],
                                o2 = (l[$frame + 2] - f[2]) + shift[2];
                    e0 = S[0] + (S[4] * o2 - S[5] * o1);
                    e1 = S[1] + (S[5] * o0 - S[3] * o2);
                    e2 = S[2] + (S[3] * o1 - S[4] * o0);
                }
                for (int r = 0; r < $rows; ++r) {
                    float* q = p + 6 * r;
                    float x = $dot_q;
                    float j = 0.0f;
                    if (now != nullptr) {
                        const float* v = w + 6 * r;
                        x += $dot_pose;
                        j = $dot_j;
                    }
                    $step_q
                    sink(r, c, z, j);
                }
$up_tail
    };
    // WALK_SEED(data, c, x, p, sink): the first step of rows seeded by the covector x[r] at coordinate c (a limit's, a
    // velocity limit's), on the states p: z = (x + s_c . p) / ell_c, its J entry x[r]. WALK_UP continues from c - 1 of
    // c's body.
    const auto WALK_SEED = [&](const float* data, int c, const float* x, float* p, auto&& sink) {
$record_seed
        for (int r = 0; r < $rows; ++r) {
            float* q = p + 6 * r;
            $seed_q
            sink(r, c, z, x[r]);
        }
    };
    // WALK_SWEEP(data, link, at, body, seed, sink): the walk with entry points, up to $rows covector walks of one tree
    // in one sweep of the union of their root paths, leaf to root: walk r enters at coordinate at[r] of body[r] (-1:
    // none) with seed[r] (WALK_SEED's step), then passes up its root path as WALK_UP does without the step's geometry.
    // The deepest body still to visit goes first: its link and each of its coordinates' records are loaded once for
    // every walk that passes them, and the walks' steps are independent, so their chains run side by side. A walk's
    // operations are WALK_SEED's and WALK_UP's, in its own order: the same footprints. sink(r, c, z); body is spent.
    const auto WALK_SWEEP = [&](const float* data, const float* link, const int* at, int* body, const float* seed,
                                auto&& sink) {
        float p[6 * $rows];
        int top[$rows];
WALK_UNROLLED
        for (int r = 0; r < $rows; ++r) {
            top[r] = at[r];
            for (int k = 0; k < 6; ++k) p[6 * r + k] = 0.0f;
        }
#pragma unroll 1
        for (;;) {
            int b = -1, c = -1;
WALK_UNROLLED
            for (int r = 0; r < $rows; ++r) b = body[r] > b ? body[r] : b;
            if (b < 0) break;
WALK_UNROLLED
            for (int r = 0; r < $rows; ++r) c = body[r] == b && top[r] > c ? top[r] : c;
            $link_b
#pragma unroll 1
            for (; c >= WALK_FIRST_OF(b); --c) {
$record_c
WALK_UNROLLED
                for (int r = 0; r < $rows; ++r) {
                    if (body[r] != b || c > top[r]) continue;
                    float* q = p + 6 * r;
                    if (c == at[r]) {
                        $sweep_seed
                        sink(r, c, z);
                    } else {
                        float x = $dot_q;
                        $sweep_step
                        sink(r, c, z);
                    }
                }
            }
            const int up = WALK_PARENT_OF(b);
WALK_UNROLLED
            for (int r = 0; r < $rows; ++r) {
                if (body[r] != b) continue;
                if (up >= 0) {
                    float* q = p + 6 * r;
                    $sweep_edge
                }
                body[r] = up;
                top[r] = up >= 0 ? WALK_FIRST_OF(up + 1) - 1 : -1;
            }
        }
    };"""

_WALK_RELEASE = """    // The slips' release of tree art (WalkRelease, written by the predictor's border; W9's verdict: R = L_F^-1 L_K,
    // tree-lower, so every row keeps its support). Slip j's steps along its root path, leaf first, start at
    // at = (art * cap + j) * steps of release.step (WALK_STEP's records); a coordinate of -1 ends the path.
    // WALK_SLIP_ON(release, at, z, rows, stride, place): z <- R_j z on rows rows (at most $rows) of a footprint whose
    // coordinate c sits at z[place(c) + r * stride] (place(c) < 0 where the footprint's support does not hold c: zero
    // there, and every coordinate of the path below the support's is off it), leaf first, a step at a time: its
    // readers are row kernels at their register line (the support writer, the local owners' packets), where a batch's
    // registers cost more than its round trips save (measured, HANDOFF_ZW2 §3).
    // The transposed forms (WALK_SLIP_BACK: the leave, a thread a tree, the audit) walk a slip's steps in batches of
    // $batch: the batch's coordinates and coefficients loaded together, then its entries, then the recurrence in
    // registers, then its stores. A path's coordinates are distinct, so no step's load waits on another step's store
    // (one global round trip a batch, not a step).
    // WALK_SLIP_T(release, at, u): u <- R_j^T u, root first. WALK_SLIP_IT(release, at, u): u <- R_j^-T u, root first.
    // WALK_RELEASE_ON(release, art, z, rows, stride, place): a row's footprints z <- R z, the first slip first;
    // WALK_RELEASE(release, art, z, stride) the same on one row whose coordinate c sits at z[c * stride].
    // WALK_RELEASE_T(release, art, u): the leave's u <- R^T u, the last slip first.
    // WALK_RELEASE_IT(release, art, u): the audit's coordinates u <- R^-T u, the first slip first.
    const auto WALK_SLIP_ON = [&](const auto& release, const size_t at, float* z, const int rows, const int stride,
                                  auto&& place) {
        const int steps = release.step.shape[2];
        const auto* records = release.step.data + at;
        float s[$rows];
        for (int r = 0; r < $rows; ++r) s[r] = 0.0f;
        for (int k = 0; k < steps; ++k) {
            int c;
            float b, a, g;
            WALK_STEP(records + k, c, b, a, g);
            if (c < 0) break;
            const int e = place(c);
            if (e < 0) continue;
            for (int r = 0; r < rows; ++r) {
                $slip_on
            }
        }
    };
    // The transposed forms, root first: a batch of the steps above the last one taken, the path's end (-1) skipped.
    const auto WALK_SLIP_BACK = [&](const auto& release, const size_t at, float* u, auto&& step) {
        const int steps = release.step.shape[2];
        const auto* records = release.step.data + at;
        float t = 0.0f;
        for (int k1 = steps - 1; k1 >= 0; k1 -= $batch) {
            int cq[$batch];
            float bk[$batch], ak[$batch], gk[$batch], x[$batch];
#pragma unroll
            for (int q = 0; q < $batch; ++q) {
                const int k = k1 - q;
                cq[q] = -1;
                bk[q] = ak[q] = gk[q] = 0.0f;
                if (k >= 0) WALK_STEP(records + k, cq[q], bk[q], ak[q], gk[q]);
            }
#pragma unroll
            for (int q = 0; q < $batch; ++q) x[q] = cq[q] >= 0 ? u[cq[q]] : 0.0f;
#pragma unroll
            for (int q = 0; q < $batch; ++q)
                if (cq[q] >= 0) x[q] = step(x[q], t, bk[q], ak[q], gk[q]);
#pragma unroll
            for (int q = 0; q < $batch; ++q)
                if (cq[q] >= 0) u[cq[q]] = x[q];
        }
    };
    const auto WALK_SLIP_T = [&](const auto& release, const size_t at, float* u) {
        WALK_SLIP_BACK(release, at, u, [](const float v, float& t, const float b, const float a, const float g) {
            $slip_t
        });
    };
    const auto WALK_SLIP_IT = [&](const auto& release, const size_t at, float* u) {
        WALK_SLIP_BACK(release, at, u, [](const float v, float& t, const float b, const float a, const float g) {
            $slip_it
        });
    };
    const auto WALK_RELEASE_ON = [&](const auto& release, const int art, float* z, const int rows, const int stride,
                                     auto&& place) {
        const int cap = release.step.shape[1], steps = release.step.shape[2];
        for (int j = 0; j < release.count.data[art]; ++j)
            WALK_SLIP_ON(release, ((size_t)art * cap + j) * steps, z, rows, stride, place);
    };
    const auto WALK_RELEASE = [&](const auto& release, const int art, float* z, const int stride = 1) {
        WALK_RELEASE_ON(release, art, z, 1, 0, [&](const int c) { return c * stride; });
    };
    const auto WALK_RELEASE_T = [&](const auto& release, const int art, float* u) {
        const int cap = release.step.shape[1], steps = release.step.shape[2];
        for (int j = release.count.data[art] - 1; j >= 0; --j) WALK_SLIP_T(release, ((size_t)art * cap + j) * steps, u);
    };
    const auto WALK_RELEASE_IT = [&](const auto& release, const int art, float* u) {
        const int cap = release.step.shape[1], steps = release.step.shape[2];
        for (int j = 0; j < release.count.data[art]; ++j) WALK_SLIP_IT(release, ((size_t)art * cap + j) * steps, u);
    };"""


_UP_HEAD, _UP_TAIL = ts.root_path(
    "body",
    "top",
    begin=["$link_b"],
    joint=[
        "for (int r = 0; r < $rows; ++r) {",
        "    float* q = p + 6 * r;",
        "    $edge_q",
        "}",
        "if (now != nullptr) {",
        "    for (int r = 0; r < $rows; ++r) {",
        "        float* v = w + 6 * r;",
        "        $edge_v",
        "    }",
        "}",
    ],
)
_WALK = Template(_WALK).safe_substitute(up_head=ts.indent_text(_UP_HEAD, 8), up_tail=ts.indent_text(_UP_TAIL, 8))
"""``WALK_UP``'s root path, leaf first (:func:`.tree_scan.root_path`): each body's link where the path opens it, its
coordinates, the rows' and the seed wrenches' joint to its parent."""

_POSE = ("(e0 - d[0])", "(e1 - d[1])", "(e2 - d[2])", "(S[3] - d[3])", "(S[4] - d[4])", "(S[5] - d[5])")
_WALK_TEXTS = {
    "dot_q": ts.dot(ts.LOADED.s, ts.channels("q")),
    "dot_pose": ts.dot(_POSE, ts.channels("v")),
    "dot_j": ts.dot(("e0", "e1", "e2", "S[3]", "S[4]", "S[5]"), ts.channels("v")),
    "step_q": ts.statements(ts.WHITEN.step(ts.LOADED, "q", value="x"), 20),
    "seed_q": ts.statements(ts.WHITEN.step(ts.LOADED, "q", source="x[r]"), 12),
    "sweep_seed": ts.statements(ts.WHITEN.step(ts.LOADED, "q", source="seed[r]"), 24),
    "sweep_step": ts.statements(ts.WHITEN.step(ts.LOADED, "q", value="x"), 24),
    "sweep_edge": ts.statements(ts.WHITEN.joint(ts.LOADED.lever, "q"), 20),
    "edge_q": ts.statements(ts.WHITEN.joint(ts.LOADED.lever, "q"), 20),
    "edge_v": ts.statements(ts.WHITEN.joint(ts.LOADED.lever, "v"), 24),
}
"""The path walk's arithmetic (:data:`_WALK`): the rows' steps up (the held pose a source of statements, ``x``), a
covector seed's first step, and the joints of the rows' and the seed wrenches' up."""


def _release_text(rows: int) -> str:
    """The slips' release (:data:`_WALK_RELEASE`) on up to ``rows`` rows a footprint: R, R^T and R^-T along each
    slip's root path, read from :class:`WalkRelease`'s records alone (no tree's path table)."""
    return substitute(
        _WALK_RELEASE,
        rows=rows,
        batch=WALK_SLIP_BATCH,
        slip_on=ts.statements(ts.slip("R", "z[e + r * stride]", "s[r]", "b", "a", "g", "z[e + r * stride] = {};"), 16),
        slip_t=ts.statements(ts.slip("RT", "v", "t", "b", "a", "g", "return {};"), 12),
        slip_it=ts.statements(ts.slip("RIT", "v", "t", "b", "a", "g", "return {};"), 12),
    )


@functools.cache
def walk_release_prelude() -> DevicePrelude:
    """The slips' release alone (:func:`_release_text`, ``WALK_RELEASE_T`` and the rest), after the walk's one access,
    for both devices: a reader that releases without walking a path (the leave)."""
    text = {cuda: _walk_loads_text(cuda) + "\n" + _release_text(1) for cuda in (True, False)}
    return DevicePrelude("walk_release", text[True], cpu=text[False])


_WALK_LOADS = """#ifndef WALK_READ
// The walk's field groups and their one access, the one place the device layout lives (WALK_COORDINATE, WALK_LINK,
// WALK_NOW), one text for both devices: a coordinate's record, three aligned quads (its column's non-zero half and
// 1 / ell; ut 0..3; ut 4..5, P0 and the side offset); a body's link, its frame point, one quad, read with its parent's;
// a step column, six floats (WALK_NOW, S's layout); a step link's frame point, one quad. Each read
// takes its load kind, the reader's data: WALK_LDG the read-only path (no walk writes them while a reader runs; ptxas
// may move it down to its use), WALK_PLAIN a coherent load issued where the text places it (the class emitter's, a
// store of the chain after it keeps it there).
// WALK_READ(load, d, p, half): record p as the walk's WALK_UNPACKED floats d: s, ut, 1 / ell, P0. half is the
// column's half as the reader knows it, its class's data: 0 linear, 1 angular (compiled: no flag, no side load),
// else named by the record (1 / ell's sign bit set: the angular half), 3 (WALK_EITHER) in a model without a column
// of two halves, 2 (WALK_BOTH) with the side quad of one (the linear half; its offset in floats the record's last
// word). The half not stored is zero: the same products as a stored zero's, up to the sign of a zero.
#define WALK_READ(load, d, p, half)                                                                             \
    {                                                                                                          \
        const float* walk_p_ = (p);                                                                            \
        const WALK_QUAD walk_a_ = load(walk_p_), walk_b_ = load(walk_p_ + 4), walk_e_ = load(walk_p_ + 8);    \
        WALK_QUAD walk_v_ = walk_a_;                                                                           \
        walk_v_.x = walk_v_.y = walk_v_.z = 0.0f;                                                              \
        bool walk_t_ = (half) == 1;                                                                            \
        if ((half) > 1) {                                                                                      \
            const int walk_s_ = WALK_BITS(walk_e_.w);                                                          \
            if ((half) == 2 && walk_s_ != 0) walk_v_ = load(walk_p_ + walk_s_);                                \
            walk_t_ = WALK_BITS(walk_a_.w) < 0;                                                                \
        }                                                                                                      \
        d[0] = walk_t_ ? walk_v_.x : walk_a_.x; d[1] = walk_t_ ? walk_v_.y : walk_a_.y;                        \
        d[2] = walk_t_ ? walk_v_.z : walk_a_.z; d[3] = walk_t_ ? walk_a_.x : 0.0f;                             \
        d[4] = walk_t_ ? walk_a_.y : 0.0f; d[5] = walk_t_ ? walk_a_.z : 0.0f;                                  \
        d[6] = walk_b_.x; d[7] = walk_b_.y; d[8] = walk_b_.z; d[9] = walk_b_.w; d[10] = walk_e_.x;             \
        d[11] = walk_e_.y; d[12] = WALK_ABS(walk_a_.w); d[13] = walk_e_.z;                                     \
    }
// WALK_READ_COLUMN(load, S, p, r, half): a coordinate's step column at its body's frame as six floats S, from its row p
// of the step's columns (WALK_NOW: S's own, or the finish's at the frame) and its record r: the half WALK_READ's names
// (its pair loads, load_PAIR), the other zero; with WALK_BOTH a column of two halves (the record's side) whole.
#define WALK_READ_COLUMN(load, S, p, r, half)                                                                   \
    {                                                                                                          \
        const float* walk_p_ = (p);                                                                            \
        const auto walk_x_ = load##_PAIR(walk_p_), walk_y_ = load##_PAIR(walk_p_ + 2),                         \
                   walk_z_ = load##_PAIR(walk_p_ + 4);                                                         \
        bool walk_l_ = (half) == 0, walk_t_ = (half) == 1;                                                     \
        if ((half) > 1) {                                                                                      \
            walk_t_ = WALK_BITS(load(r).w) < 0;                                                                \
            walk_l_ = !walk_t_ || ((half) == 2 && WALK_BITS(load((r) + 8).w) != 0);                            \
        }                                                                                                      \
        S[0] = walk_l_ ? walk_x_.x : 0.0f; S[1] = walk_l_ ? walk_x_.y : 0.0f; S[2] = walk_l_ ? walk_y_.x : 0.0f; \
        S[3] = walk_t_ ? walk_y_.y : 0.0f; S[4] = walk_t_ ? walk_z_.x : 0.0f; S[5] = walk_t_ ? walk_z_.y : 0.0f; \
    }
// WALK_READ_LINK(load, l, p, b, up): body b's link as l, its offset to its parent up's frame (zero at a root, up < 0),
// the difference of the two frame points, the refresh's WALK_OFFSET, the same float32 subtraction; then its frame
// point.
#define WALK_READ_LINK(load, l, p, b, up)                                                                       \
    {                                                                                                          \
        const WALK_QUAD walk_f_ = load((p) + $link * (b));                                                     \
        l[0] = l[1] = l[2] = 0.0f;                                                                             \
        if ((up) >= 0) {                                                                                       \
            const WALK_QUAD walk_u_ = load((p) + $link * (up));                                                \
            l[0] = walk_u_.x - walk_f_.x; l[1] = walk_u_.y - walk_f_.y; l[2] = walk_u_.z - walk_f_.z;          \
        }                                                                                                      \
        l[3] = walk_f_.x; l[4] = walk_f_.y; l[5] = walk_f_.z;                                                  \
    }
// WALK_READ_POINT(load, f, p): a step link's frame point and pad as f.
#define WALK_READ_POINT(load, f, p)                                                                             \
    {                                                                                                          \
        const WALK_QUAD walk_a_ = load(p);                                                                     \
        f[0] = walk_a_.x; f[1] = walk_a_.y; f[2] = walk_a_.z; f[3] = walk_a_.w;                                \
    }
// WALK_SHIFT(w, a, s): articulation a's factor origin less its step origin (WalkTrees.origin, .now_origin), the shift
// a factor's frame point takes to meet the step's (both the articulation's, relative to its origin).
#define WALK_SHIFT(w, a, s)                                                                                     \
    {                                                                                                          \
        const wp::vec3 walk_o_ = (w).origin.data[a], walk_n_ = (w).now_origin.data[a];                         \
        for (int walk_k_ = 0; walk_k_ < 3; ++walk_k_) s[walk_k_] = walk_o_[walk_k_] - walk_n_[walk_k_];        \
    }
// WALK_STEP(p, c, b, a, g): a slip's step record at p (WalkRelease.step: the step's coordinate as its bit pattern, -1
// past the path, then b, a, g), one quad: c, b, a and g arrive in one load, on the line the slip's next steps share.
#define WALK_STEP(p, c, b, a, g)                                                                                \
    {                                                                                                          \
        const WALK_QUAD walk_s_ = WALK_PLAIN(reinterpret_cast<const float*>(p));                               \
        c = WALK_BITS(walk_s_.x); b = walk_s_.y; a = walk_s_.z; g = walk_s_.w;                                 \
    }
// WALK_MUL(a, b): a walk step's output, a product rounded on its own: the chain's coordinate is a float32 value before
// any consumer adds it, whatever block the consumer sits in.
$device
#endif"""

_WALK_DEVICE = {
    True: """#define WALK_QUAD float4
#define WALK_LDG(p) __ldg(reinterpret_cast<const float4*>(p))
#define WALK_PLAIN(p) (*reinterpret_cast<const float4*>(p))
#define WALK_LDG_PAIR(p) __ldg(reinterpret_cast<const float2*>(p))
#define WALK_PLAIN_PAIR(p) (*reinterpret_cast<const float2*>(p))
#define WALK_LOAD_PIVOT(p) __ldg((p) + 10)
#define WALK_BITS(x) __float_as_int(x)
#define WALK_FLOAT(i) __int_as_float(i)
#define WALK_ABS(x) fabsf(x)
#define WALK_MUL(a, b) __fmul_rn((a), (b))
#define WALK_CONSTANT __constant__
// WALK_UNROLLED: a loop over a walk's rows unrolled, so their states stay registers (none on the CPU).
#define WALK_UNROLLED _Pragma("unroll")
// WALK_TOUCH(p): ask L1 for the line of p, a hint ahead of the load that reads it (no register, no scoreboard).
#define WALK_TOUCH(p) asm volatile("prefetch.global.L1 [%0];" ::"l"(p))""",
    False: """// The CPU's quad is Warp's (four named floats); its loads are plain. Its WALK_MUL is a function boundary, as CUDA's
// __fmul_rn: a product inside the consumer's expression could be contracted into it.
#define WALK_QUAD wp::quatf
#define WALK_LDG(p) (wp::quatf((p)[0], (p)[1], (p)[2], (p)[3]))
#define WALK_PLAIN(p) WALK_LDG(p)
#define WALK_LDG_PAIR(p) (wp::quatf((p)[0], (p)[1], 0.0f, 0.0f))
#define WALK_PLAIN_PAIR(p) WALK_LDG_PAIR(p)
#define WALK_LOAD_PIVOT(p) ((p)[10])
#define WALK_BITS(x) __builtin_bit_cast(int, x)
#define WALK_FLOAT(i) __builtin_bit_cast(float, i)
#define WALK_ABS(x) __builtin_fabsf(x)
#define WALK_MUL(a, b) ([](const float walk_a_, const float walk_b_) -> float { return walk_a_ * walk_b_; }((a), (b)))
#define WALK_CONSTANT const
#define WALK_UNROLLED
#define WALK_TOUCH(p) ((void)(p))""",
}
"""The device's words of :data:`_WALK_LOADS`: its quad, its load kinds, the record's pivot, the bit and absolute
value, the step's pin, the tables' constant memory and the L1 hint (none on the CPU). Everything else of the access is
one text."""

WALK_LINK_FRAME = 3
"""Where a body's link, as a walk reads it (``WALK_READ_LINK``'s registers), holds its frame point: after its offset to
its parent's frame."""


def _walk_loads_text(cuda: bool) -> str:
    """The walk's one access (:data:`_WALK_LOADS`) and the step's pin ``WALK_MUL`` at the device layout, for a device,
    under its one guard: every walk prelude carries it."""
    return substitute(_WALK_LOADS, link=WALK_LINK, device=_WALK_DEVICE[cuda])


@functools.cache
def walk_loads_prelude() -> DevicePrelude:
    """The walk's one access alone (:func:`_walk_loads_text`), for both devices: a reader of the records without a
    walk (the predictor's border) declares it."""
    return DevicePrelude("walk_loads", _walk_loads_text(True), cpu=_walk_loads_text(False))


@functools.cache
def walk_prelude(
    tables: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...], rows: int, half: int = WALK_BOTH
) -> DevicePrelude:
    """The device walk of the walking classes, for both devices: a prelude declaring

    - the classes' path tables in constant memory (``WALK_PARENT``, ``WALK_FIRST``, ``WALK_CLASS``): topology fixed at
      build, so a lane walks a root path without a global load in front of its steps (GPU_WALK §8, variant C); a lane
      names its tree's class (``WALK_TREE(cls)``, the factor's :attr:`WalkTrees.cls`) where the tables hold more than
      one;
    - ``WALK_UP``, ``WALK_SEED`` and ``WALK_SWEEP``, the rows: a lane per seed walks its ``rows`` rows up the seed's
      root path, every load shared (Z_UNION Theorem B), and gives each footprint, with the step's ``J`` entry it
      whitens, to the caller's sink, the placement's store;
    - the slips' release (:func:`_release_text`).

    Its reads (``_WALK``'s ``$record_*``, ``$link_b``, ``$column_c`` and ``$point_b``) are the one access's
    (:data:`_WALK_LOADS`, the read-only load kind, its records and columns of ``half``). No step launches it (FL3.1).
    The data is :meth:`TreeWalk.device_data`'s.

    Args:
        tables: Each class's path table (a :class:`.tree_scan.WalkClass`'s ``parent`` and ``first``, :func:`walk_tables`):
            each body's parent (-1: the world, every parent before its children), and each body's first coordinate with
            the count after the last.
        rows: The rows a lane walks together: the seed's unit (1 a limit, 2 a velocity limit, 3 a point contact, 4
            with spin).
        half: The classes' records' half as their walk reads them (``WALK_READ``, :func:`.tree_scan.classes_half`).
    """
    if rows < 1:
        raise ValueError(f"a lane walks at least one row, got {rows}")
    table = walk_table(tables)
    # No walking tree: one class of no body.
    tables = tables or (((), (0,)),)
    tag = hashlib.sha256(repr(tables).encode()).hexdigest()[:8] if len(tables) > 1 else str(len(tables[0][0]))
    name = f"walk_{rows}_{tag}" + ("" if half == WALK_BOTH else f"_{half}")
    record = f"float d[{WALK_UNPACKED}];\nWALK_READ(WALK_LDG, d, data + {WALK_COORDINATE} * c, {half});"
    column = (
        "float S[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};\nif (now != nullptr)\n"
        f"    WALK_READ_COLUMN(WALK_LDG, S, now + {WALK_NOW} * c, data + {WALK_COORDINATE} * c, {half});"
    )
    walk = substitute(
        _WALK,
        rows=rows,
        **_WALK_TEXTS,
        now=WALK_NOW,
        frame=WALK_LINK_FRAME,
        record_c=ts.indent_text(record, 16),
        record_seed=ts.indent_text(record, 8),
        # A body's link where the root path opens it (ts.root_path's begin: one line, the path's indent).
        link_b="float l[6]; WALK_READ_LINK(WALK_LDG, l, link, b, WALK_PARENT_OF(b));",
        column_c=ts.indent_text(column, 16),
        point_b=ts.indent_text(
            f"float f[4];\nWALK_READ_POINT(WALK_LDG, f, now_link + {WALK_LINK} * b + {WALK_FRAME});", 20
        ),
    )
    walk += "\n" + _release_text(rows)
    text = {cuda: _walk_loads_text(cuda) + "\n" + table + "\n" + walk for cuda in (True, False)}
    return DevicePrelude(name, text[True], cpu=text[False])


_WALK_ROW = """    // WALK_WRENCH(q, sign, d, at, origin, o, torque): a row's seed at a body's frame (the native twin of
    // contact_laws.contact_seed and spin_seed): the force f = sign d at the world point at, its wrench (f, l x f) with
    // l = (at - origin) - o the lever from the body's frame point o (the articulation's, relative to its origin), or
    // the torque (0, f).
    const auto WALK_WRENCH = [](float* q, const float sign, const float* d, const float* at, const float* origin,
                                const float* o, const bool torque) {
        const float f[3] = {sign * d[0], sign * d[1], sign * d[2]};
        const float l[3] = {(at[0] - origin[0]) - o[0], (at[1] - origin[1]) - o[1], (at[2] - origin[2]) - o[2]};
        q[0] = torque ? 0.0f : f[0];
        q[1] = torque ? 0.0f : f[1];
        q[2] = torque ? 0.0f : f[2];
        q[3] = torque ? f[0] : l[1] * f[2] - l[2] * f[1];
        q[4] = torque ? f[1] : l[2] * f[0] - l[0] * f[2];
        q[5] = torque ? f[2] : l[0] * f[1] - l[1] * f[0];
    };
    // WALK_ROW(data, link, now, now_link, shift, body, k, seed, sink): every reader's seed walk, after the walk
    // prelude (walk_prelude). Rows seeded at body walked up its root path into sink: the wrench
    // seed[6 r + i] at body's frame (k < 0; now, now_link and shift its step's geometry, as WALK_UP takes them), or the
    // covector seed[r] e_k at body's coordinate k (WALK_SEED, then WALK_UP from k - 1, without the step's geometry).
    const auto WALK_ROW = [&](const float* data, const float* link, const float* now, const float* now_link,
                              const float* shift, const int body, const int k, const float* seed, auto&& sink) {
        float p[6 * $rows], w[6 * $rows];
        int top = WALK_FIRST_OF(body + 1) - 1;
        for (int e = 0; e < 6 * $rows; ++e) p[e] = w[e] = k < 0 ? seed[e] : 0.0f;
        if (k >= 0) {
            WALK_SEED(data, k, seed, p, sink);
            top = k - 1;
            now = nullptr;
        }
        WALK_UP(data, link, now, now_link, shift, body, top, p, w, sink);
    };"""


@functools.cache
def walk_row_prelude(rows: int) -> DevicePrelude:
    """The seed walk of ``rows`` rows (``WALK_ROW``) and a wrench seed's formation (``WALK_WRENCH``), for both devices:
    every row reader's (the one writer's units and natives, the parallel tiers' rows, the local blocks' seeds) binds its
    sink to it. It follows the walk prelude whose
    ``WALK_UP`` and ``WALK_SEED`` it calls, and only readers that walk seeds declare it."""
    text = substitute(_WALK_ROW, rows=rows)
    return DevicePrelude(f"walk_row_{rows}", text, cpu=text)


FREE_BODY_WALK = WalkClass((-1,), (0, 6), (0, 0, 0, 1, 1, 1))
"""A free rigid body's walking class: one body of six coordinates."""

_WALK_ACCESS = """    // The class emitter's reads, ahead of its chain, by the one access (WALK_READ, WALK_READ_LINK) and the plain load
    // kind: the walk issues each where its text places it, WALK_AHEAD items ahead of the step that reads it
    // (WalkClass.walk), and a store of the chain after it keeps it there; ptxas moves a read-only load down to its use.
    // WALK_IN(d, u, p, half): coordinate record d's step input, its record unpacked (half its class's, WALK_READ's),
    // and the caller's two values: the step's input (u) and the prior value its output updates (p). WALK_HAND(x, u, p):
    // a held input x with the caller's values. WALK_LINK_IN(link, b, up): body b's link (WALK_READ_LINK's).
    struct WalkIn {
        float d[$unpacked];
        float u, p;
    };
    struct WalkLinkIn {
        float l[6];
    };
    const auto WALK_IN = [&](const float* d, const float u, const float p, const int half) -> WalkIn {
        WalkIn in;
        WALK_READ(WALK_PLAIN, in.d, d, half);
        in.u = u;
        in.p = p;
        return in;
    };
    const auto WALK_HAND = [](WalkIn x, const float u, const float p) -> WalkIn {
        x.u = u;
        x.p = p;
        return x;
    };
    const auto WALK_LINK_IN = [&](const float* link, const int b, const int up) -> WalkLinkIn {
        WalkLinkIn in;
        WALK_READ_LINK(WALK_PLAIN, in.l, link, b, up);
        return in;
    };
    const auto WALK_LOAD = [&](const float* p) -> float { return *p; };"""


# The class emitter's walks down, one text a combine (:data:`.tree_scan.LEAVE`, :data:`.tree_scan.ENTER`): the macro's
# description, its name and parameters after the class, the records and the links, and its step's sink, input and
# prior output and a body's end (:func:`.tree_scan.walk_texts`).
_WALK_DOWNS = {
    ts.LEAVE: (
        """    // WALK_DOWN(cls, data, link, u, prior, qd, twist): the chain of the leave P^T L^-T u of one tree of class cls (an
    // index into the classes the prelude was emitted for), on one lane, root first: each coordinate c takes
    // q = (u_c - ut_c . V) / ell_c, then V += s_c q, V its body's twist (v, w) at its frame, a register, which
    // twist(b, V) receives after the body's coordinates. u(c) reads coordinate c, prior(c) the value its output
    // updates (or 0), and qd(c, q, prior) writes the output. Every coordinate's record, link, u and prior are read
    // WALK_AHEAD items ahead of the chain (WalkClass.walk), so qd may write in place, but not into a coordinate after c.""",
        "WALK_DOWN",
        "auto&& u, auto&& prior, auto&& qd,\n                               auto&& twist",
        {"sink": "qd($c, q, $X.p);", "value": "u($c)", "prior": "prior($c)", "end": "twist($b, $V);"},
    ),
    ts.ENTER: (
        """    // WALK_ENTER(cls, data, link, qd, u): the walk down inverted, u = L^T P qd, of one tree of class cls on one lane,
    // root first: each coordinate c of known velocity qd_c takes u_c = ell_c qd_c + ut_c . V, then V += s_c qd_c, V its
    // body's twist at its frame, a register (the audit's coordinates, :func:`walk_enter_kernel`).""",
        "WALK_ENTER",
        "const float* qd, float* u",
        {"sink": "u[$c]", "value": "qd[$c]", "prior": "0.0f"},
    ),
}


def _walk_down(classes: tuple[WalkClass, ...], unroll: int, ahead: int | None, combine: ts.WalkCombine) -> str:
    """The class emitter's walk down of ``classes`` by ``combine`` (:data:`_WALK_DOWNS`): ``WALK_DOWN``, the leave, or
    ``WALK_ENTER``."""
    description, name, parameters, ends = _WALK_DOWNS[combine]
    texts = ts.walk_texts(combine, source="$X.u", coordinate=WALK_COORDINATE, link=WALK_LINK, **ends)
    walk = indent_text(ts.by_class([walk.walk(**texts, unroll=unroll, ahead=ahead) for walk in classes]), 8)
    return f"""{description}
    const auto {name} = [&](const int cls, const float* data, const float* link, {parameters}) {{
        (void)cls;
{walk}
    }};"""


@functools.cache
def walk_down_prelude(
    classes: tuple[WalkClass, ...],
    unroll: int = WALK_UNROLL,
    ahead: int | None = None,
    *,
    down: bool = True,
    enter: bool = False,
) -> DevicePrelude:
    """The one walk down of the walking classes ``classes``, for both devices: with ``down`` a prelude declaring
    ``WALK_DOWN(cls, data, link, u, prior, qd, twist)``, the leave ``P^T L^-T u``, the chain on one lane: each class's
    walk emitted by :meth:`WalkClass.walk` (straight-line code up to ``unroll`` coordinates), its links, records, ``u``
    and prior outputs read :data:`WALK_AHEAD` items ahead of the chain (``WALK_IN``), every twist a register, handed to
    ``twist(b, V)`` after its body; with ``enter`` the same walk inverted, ``WALK_ENTER(cls, data, link, qd, u)``, the
    walk's coordinates of known velocities (``u = L^T P qd``, :data:`.tree_scan.ENTER`), the same walk with another
    combine.

    No shared memory and no scratch. The step's one leave takes the leave (:func:`walk_leave_kernel`), the audit the
    enter (:func:`walk_enter_kernel`), and the general sweep's free bodies both, on their one-body class
    (:data:`FREE_BODY_WALK`): a speed row reads its body's velocity by the leave and pushes its change back by the
    enter.

    Args:
        classes: The walking classes, indexed by ``cls``.
        unroll: Coordinates up to which a class's walk is straight-line code (:data:`WALK_UNROLL`).
        ahead: Items whose inputs the chain's lane holds in registers ahead of it (:meth:`WalkClass.walk`).
        down: Declare the leave, ``WALK_DOWN``.
        enter: Declare the enter, ``WALK_ENTER``.
    """
    if not classes:
        raise ValueError("a walk down needs a class")
    if not (down or enter):
        raise ValueError("a walk down declares the leave, the enter or both")
    tag = hashlib.sha256(
        repr((classes, unroll, WALK_AHEAD if ahead is None else ahead, WALK_TOUCH)).encode()
    ).hexdigest()[:8]
    walks = [_walk_down(classes, unroll, ahead, combine) for combine, on in ((ts.LEAVE, down), (ts.ENTER, enter)) if on]
    # The loads package carries the step's pin (WALK_MUL), under its one guard.
    walk = "\n".join((substitute(_WALK_ACCESS, unpacked=WALK_UNPACKED), *walks))
    texts = {cuda: _walk_loads_text(cuda) + "\n" + walk for cuda in (True, False)}
    kind = "_".join(name for name, on in (("down", down), ("enter", enter)) if on)
    return DevicePrelude(f"walk_{kind}_{tag}", texts[True], cpu=texts[False])


def leave_work(trees: int) -> Work:
    """The step's leave (:func:`walk_leave_kernel`): a lane group a tree, of a measured 4 lanes on CUDA. They fetch the
    tree's changes and velocities into L1 in one parallel trip and one of them walks, its records and links asked of
    L1 :data:`WALK_TOUCH` items ahead (WALK_MAPPING §4, B5: 4 lanes ran at or under 8 on g1, franka and allegro)."""
    return Work(trees, lanes=4)


_LEAVE = """    const int tree = tid / lanes, lane = tid - tree * lanes;
    if (tree >= arts.shape[0]) return;
    const int art = arts.data[tree];
    const int cls = walk_class.data[art];
    if (cls < 0) return;
    const int j0 = walk.joint_start.data[art], d0 = walk.dof_start.data[art], n = walk.dofs.data[art];
    // Coordinate c's change is change[node(c)]: the walk's du at the tree's coordinates (node(c) = c), or a sweep's
    // own layout (the tree's base there, and its class's node of each coordinate).
    const float* const change = du.data + du_base.data[art];
    float* const v = v_out.data + d0;
    const float* const data = walk.data.data + (size_t)d0 * $coordinate;
    const float* const link = walk.link.data + (size_t)j0 * $link;
    const auto prior = [&](const int c) -> float { return v[c]; };
    const auto qd = [&](const int c, const float q, const float p) { v[c] = p + q; };
    const auto twist = [](const int, const float*) {};
$classes"""

_LEAVE_CLASS = """    // The group's lanes fetch the tree's changes and velocity into L1; lane 0 prepares the change (ready) and walks the
    // tree down (WALK_DOWN: its records and links asked of L1 WALK_TOUCH items ahead, loaded WALK_AHEAD items ahead;
    // a whole tree fetched at the start overflows L1, chain_106's 22 trees an SM hold 230 KB).
    FETCH(change, $span, lane, lanes);
    FETCH(v, n * 4, lane, lanes);
$release    if (lane == 0) {
        ready();
        WALK_DOWN($cls, data, link, read, prior, qd, twist);
    }
"""

_LEAVE_PLAIN = """    const auto ready = []() {};
    const auto read = [&](const int c) -> float { return WALK_LOAD(change + $node); };
"""

_LEAVE_RELEASE = """    // The slips' release transposed, R^T du, on a copy where the tree slips (the predictor's border's R): slip j's
    // R_j^T along its root path, root first, the last slip first (WALK_RELEASE_T). $how
    float r[$coordinates];
    bool slipped = false;
    const auto read = [&](const int c) -> float { return slipped ? r[c] : WALK_LOAD(change + $node); };
    const auto ready = [&]() {
        if (walk.release.count.data[art] == 0) return;
$copy
        slipped = true;
    };
"""

_LEAVE_SLIPS = """        const int cap = walk.release.step.shape[1], steps = walk.release.step.shape[2];
        for (int j = walk.release.count.data[art] - 1; j >= 0; --j) {
            // Slip j's steps, one record each (WALK_STEP): its first holds the slip's coordinate.
            const auto* records = walk.release.step.data + ((size_t)art * cap + j) * steps;
            int slip;
            float t = 0.0f, b0, a0, g0;
            WALK_STEP(records, slip, b0, a0, g0);
            switch (slip) {
$paths
            }
        }"""


def _leave_class(k: int, walk: WalkClass, node: tuple[int, ...] | None, release: bool, unroll: int) -> str:
    """The leave of a tree of class ``k``: its read of the change, the release where the class can slip, the unit."""
    n = walk.coordinates
    at = "c" if node is None else f"LEAVE_NODE_{k}[c]"
    table = "" if node is None else f"    {ts.constant_table(f'LEAVE_NODE_{k}', node)}\n"
    span = "n * 4" if node is None else str(4 * (max(node, default=-1) + 1))
    if not release:
        head = substitute(_LEAVE_PLAIN, node=at)
    elif n <= unroll:
        # Every index into the copy is a constant (the class's coordinates, each slip's path by a switch on its
        # coordinate), so the copy stays in registers.
        steps = []
        for c in range(n):
            path = walk.root_path(c)
            # Step t's record (WALK_STEP); step 0's arrived with the slip's coordinate.
            body = " ".join(
                "{ "
                + ("" if t == 0 else f"int c_; float b_, a_, g_; WALK_STEP(records + {t}, c_, b_, a_, g_); ")
                + " ".join(
                    ts.slip(
                        "RT",
                        f"r[{d}]",
                        "t",
                        "b0" if t == 0 else "b_",
                        "a0" if t == 0 else "a_",
                        "g0" if t == 0 else "g_",
                        f"r[{d}] = {{}};",
                    )
                )
                + " }"
                for t, d in reversed(list(enumerate(path)))
            )
            steps.append(f"case {c}: {{ {body} }} break;")
        copy = f"        #pragma unroll\n        for (int c = 0; c < {n}; ++c) r[c] = WALK_LOAD(change + {at});\n"
        copy += substitute(_LEAVE_SLIPS, paths=indent_text("\n".join(steps), 12))
        head = substitute(
            _LEAVE_RELEASE,
            how="Each index a constant: the copy is registers.",
            coordinates=max(n, 1),
            node=at,
            copy=copy,
        )
    else:
        copy = f"        for (int c = 0; c < {n}; ++c) r[c] = WALK_LOAD(change + {at});\n"
        copy += "        WALK_RELEASE_T(walk.release, art, r);"
        head = substitute(
            _LEAVE_RELEASE, how="A loop-form class: the copy on the stack.", coordinates=max(n, 1), node=at, copy=copy
        )
    return table + substitute(_LEAVE_CLASS, span=span, release=head, cls=k)


@functools.cache
def walk_leave_kernel(
    classes: tuple[WalkClass, ...], release: bool, nodes: tuple[tuple[int, ...], ...] | None = None
) -> wp.Kernel:
    """The step's one leave, ``v += P^T L^-T R^T du``, of every walking tree (``arts``), a lane group a tree (``lanes``:
    :func:`leave_work`'s; ``dim`` trees times lanes), by the walk down (``WALK_DOWN`` of
    :func:`walk_down_prelude`, the tree's class at ``walk_class``). Coordinate ``c``'s change is
    ``du[du_base[art] + node(c)]``: every placement's sweep adds its change in the walk's coordinates into
    :attr:`WalkTrees.du` (the base the tree's first coordinate, ``node(c) = c``), the support placement's sparse sweep
    publishes it in its factor coordinates (the base the tree's offset in its world's, ``node`` its class's factor
    coordinates, ``nodes``). Where the class can slip (``release``) the leave takes the slips' release transposed on a
    copy of the tree's changes, in registers for a straight-line class (each slip's root path by a switch on its
    coordinate), and walks each tree down onto ``v_out``, which holds the predictor and the changes of the placements
    that do not walk.

    Args:
        classes: The walking classes, indexed by ``walk_class``.
        release: Whether a tree of the classes can slip (:attr:`.tree_factor.TreeFactor.walk_slip_capacity`).
        nodes: Each class's node of each coordinate, or ``None`` for the walk's own coordinates.
    """
    from ..kernel_template import KernelTemplate, items  # noqa: PLC0415

    if nodes is not None and (
        len(nodes) != len(classes) or any(len(k) != walk.coordinates for k, walk in zip(nodes, classes, strict=True))
    ):
        raise ValueError("the leave needs a node for each coordinate of each class")
    bodies = [
        _leave_class(k, walk, None if nodes is None else nodes[k], release, WALK_UNROLL)
        for k, walk in enumerate(classes)
    ]
    body = substitute(_LEAVE, classes=indent_text(ts.by_class(bodies), 4), coordinate=WALK_COORDINATE, link=WALK_LINK)
    tag = hashlib.sha256(repr((classes, nodes)).encode()).hexdigest()[:8]
    return KernelTemplate(
        f"walk_leave_{len(classes)}_{tag}" + ("_release" if release else ""),
        (
            ("arts", wp.array[int]),
            ("lanes", int),
            ("walk", WalkTrees),
            ("walk_class", wp.array[int]),
            ("du", wp.array[float]),
            ("du_base", wp.array[int]),
            ("v_out", wp.array[float]),
        ),
        body,
        items("tid", flat=True),
        preludes=(
            memory_words(memory.Place.STREAMED),
            *((walk_release_prelude(),) if release else ()),
            walk_down_prelude(classes, WALK_UNROLL),
        ),
        dual=True,
        native_name="walk_leave_native",
    ).kernel()


# ---------------------------------------------------------------------------------------------------------------------
# The walk's view (FL3.3): what the one writer, the leave and the audit read of the converted class's trees
# ---------------------------------------------------------------------------------------------------------------------


@wp.struct
class WalkRelease:
    """The release of each tree's slipping drives this step, which the predictor's border writes
    (:func:`.tree_kinetic_state.border_solve`, W9): its slips (``count``, 0 for none), each one's coordinate within its
    tree (``dof``, root-most first) and its root (``root``, its path's root-most coordinate), and at each step of each
    one's root path, leaf first, one 16 B record (``step``, ``[tree, slip, step]``, as deep as the class's deepest
    root path): the step's coordinate as its bit pattern (-1 past the path), the column ``b`` of the released inverse and
    the recurrence's ``a`` (``beta d^1/2``) and ``g`` (``(d / dbar)^1/2``), in the order the walks read them
    (``WALK_STEP``): a slip's coordinate and its steps arrive together. Slip ``j``'s ``R_j`` maps a footprint ``z`` along its path, leaf first,
    ``u = z_c - b s``, ``s += a u``, ``z_c = g u``; ``R = R_(s-1) ... R_0 = L_F^-1 L_K``, the decided factor's whitening on
    the all-``K`` walk, which keeps every row's support (``WALK_RELEASE``, ``WALK_RELEASE_T``, ``WALK_RELEASE_IT``). Two slips' paths meet if and only if their roots are equal (their ``R_j`` touch
    disjoint coordinates otherwise, and commute), and a footprint whose support holds every ancestor of its coordinates
    (a row's) meets slip ``j``'s path if and only if it holds ``root[j]``."""

    count: wp.array[int]
    dof: wp.array2d[int]
    root: wp.array2d[int]
    step: wp.array3d[wp.vec4]


def release_view(articulations: int, capacity: int, depth: int, device) -> WalkRelease:
    """A :class:`WalkRelease` of ``capacity`` slips a tree and ``depth`` steps a slip; no slip room without a capacity."""
    view = WalkRelease()
    trees = articulations if capacity else 0
    view.count = wp.zeros(articulations, dtype=wp.int32, device=device)
    view.dof = wp.zeros((trees, max(capacity, 1)), dtype=wp.int32, device=device)
    view.root = wp.zeros((trees, max(capacity, 1)), dtype=wp.int32, device=device)
    view.step = release_steps((trees, max(capacity, 1), depth), device)
    return view


def release_steps(shape, device) -> wp.array:
    """:attr:`WalkRelease.step` records of ``shape``, every step past its path (coordinate -1, its bit pattern)."""
    records = np.zeros((*shape, 4), dtype=np.float32)
    records.view(np.int32)[..., 0] = -1
    return wp.array(records, dtype=wp.vec4, device=device)


@wp.struct
class WalkTrees:
    """The walk of every held tree of the converted class (:meth:`TreeFactor.plan_walk`), by global coordinate, body
    (its joint) and articulation:

    - ``data``, ``link``: :data:`WALK_COORDINATE` floats a coordinate and :data:`WALK_LINK` floats a body (each tree's
      path is its class's, ``cls``);
    - ``now``, ``now_link``, ``now_com``: each coordinate's column at its body's frame, each body's link
      (:data:`WALK_LINK`'s layout: its frame point) and its centre of mass (a ``vec3`` a joint), at the step's poses,
      which the held tree's finish publishes with its motion columns: a seed enters the walk as ``J`` there (each column
      shifted to its body's frame point in ``link`` by the body's motion since the refresh), so a held factor (a step
      between refreshes) whitens the step's ``J``, and a row's incident ``J v_hat`` is summed along its walk; a refresh
      takes them as the factor's frames;
    - ``origin``, ``now_origin``: each articulation's world origin at the factor's frames and at the step's (the
      finish's ``articulation_origin``). Every frame point is the articulation's own, relative to its origin, so its
      precision does not depend on where the articulation stands; a world point meets one through the origin, and a
      factor's frame meets the step's through the two origins' difference (``WALK_UP``'s ``shift``, zero on a refresh
      step); ``joint_art``: each joint's articulation;
    - each articulation's first joint, joint end, first coordinate and coordinates (a free rigid body is a one-body
      tree of six coordinates);
    - ``dof_body``: each coordinate's body within its tree; ``body_joint``: each body's joint;
    - ``du``: each coordinate's velocity change in the walk's coordinates, which the sweeps add into from zero each
      step and the step's one leave walks down (:func:`walk_leave_kernel`);
    - ``release``: the release of each tree's slips (the block primitive's first phase, :class:`WalkRelease`), which the
      predictor's border writes (:func:`.tree_kinetic_state.border_solve`, FL3.9): the rows take ``R``, the leave
      ``R^T`` and the audit's coordinates ``R^-T``;
    - ``body_art``: each body's articulation; ``first_art``: the responding articulation whose coordinates start at
      each global coordinate (-1 elsewhere), by which a matrix-free sweep finds a free body's tree
      (:class:`.sweep_routes.FreeBodyCoordinates`);
    - ``cls``: each walking articulation's class (:attr:`.tree_factor.TreeFactor.walk_classes`), by which a lane reads
      its tree's path from the classes' tables in constant memory (``WALK_TREE``, :func:`walk_prelude`).
    """

    data: wp.array2d[float]
    link: wp.array2d[float]
    now: wp.array[wp.spatial_vector]
    now_link: wp.array2d[float]
    now_com: wp.array[wp.vec3]
    origin: wp.array[wp.vec3]
    now_origin: wp.array[wp.vec3]
    joint_art: wp.array[int]
    joint_start: wp.array[int]
    joint_end: wp.array[int]
    dof_start: wp.array[int]
    dofs: wp.array[int]
    dof_body: wp.array[int]
    body_joint: wp.array[int]
    release: WalkRelease
    du: wp.array[float]
    body_art: wp.array[int]
    first_art: wp.array[int]
    cls: wp.array[int]


_ENTER = """    if (tid >= arts.shape[0]) return;
    const int art = arts.data[tid];
    const int world = art_to_world.data[art], offset = world_offset.data[art];
    if (world < 0 || offset < 0) return;
    const int j0 = walk.joint_start.data[art], d0 = walk.dof_start.data[art];
    float* u = coordinates.data + (size_t)world * coordinates.shape[1] + offset;
    // The class's walk down inverted (its path compiled), from the tree's velocity.
    WALK_ENTER(walk.cls.data[art], walk.data.data + (size_t)d0 * $coordinate, walk.link.data + (size_t)j0 * $link,
               velocity.data + d0, u);
    // The rows' release: u <- R^-T u (the predictor's border's R).
    WALK_RELEASE_IT(walk.release, art, u);
"""


@functools.cache
def walk_enter_kernel(classes: tuple[WalkClass, ...]) -> wp.Kernel:
    """Each held tree's coordinates of a velocity in the walk's frame, ``u = L^T P v``, a thread per tree: its class's
    walk down inverted (``WALK_ENTER`` of :func:`walk_down_prelude`, the class at :attr:`WalkTrees.cls`), then the
    slips' ``R^-T``, at its world coordinates: what the audit reads the walked rows against (:mod:`.audit`)."""
    from ..kernel_template import KernelTemplate, items  # noqa: PLC0415

    tag = hashlib.sha256(repr(classes).encode()).hexdigest()[:8]
    return KernelTemplate(
        f"walk_enter_{tag}",
        (
            ("arts", wp.array[int]),
            ("walk", WalkTrees),
            ("art_to_world", wp.array[int]),
            ("world_offset", wp.array[int]),
            ("velocity", wp.array[float]),
            ("coordinates", wp.array2d[float]),
        ),
        substitute(_ENTER, coordinate=WALK_COORDINATE, link=WALK_LINK),
        items("tid", flat=True),
        preludes=(walk_release_prelude(), walk_down_prelude(classes, WALK_UNROLL, down=False, enter=True)),
        dual=True,
        native_name="walk_enter_native",
    ).kernel()


RELEASE_SLIPS = 16
"""The most slipping drives a tree's release holds (:class:`WalkRelease`, three floats and a coordinate a slip and path
step): a tree past it keeps the rest's ``K`` and ``constraint_capacity_status()["slip_frames"]`` flags it. It bounds the
release's memory until a data-sized arena holds it (DECISIONS 10-24); nothing else reads it."""


# ---------------------------------------------------------------------------------------------------------------------
# The walk's step on the device (FL3.3): the step's geometry every step, the inward pass on the factor's refreshes,
# one launch, a lane group per tree
# ---------------------------------------------------------------------------------------------------------------------

WALK_JOIN_SLOTS = 32
"""The most joins one tree's inward pass holds at once: a parent that waits for the lifted inertia of a child other
than the joint after it (:meth:`.tree_factor.TreeFactor.plan_walk` assigns the slots). On CUDA lane ``k`` of the tree's group
holds slot ``k`` in registers, so a group has at least as many lanes as its tree's joins in flight; the CPU's one
thread holds them all."""


def walk_work(trees: int, bodies: int, joins: int) -> Work:
    """The refresh's work (:func:`walk_step_kernel`): a lane group a tree feeding the tree's inward chain (FEED), a
    lane a body of a batch, never fewer lanes than the joins a tree holds at once (one a lane), in blocks of 64
    threads (whole groups, so a group never straddles a warp). Its line is the kernel's 125-127 registers
    (``kernel_sass``, sm_120 and sm_103), so 512 threads an SM and the group widens while the trees still fit one wave.

    Measured on both GPUs (``SP/core/FL/HANDOFF_ZR.md``): with more trees than the device holds a group of four
    wins (g1, kuka, franka-reach at 16,384 trees: 4 lanes), with fewer the batch widens (chain_106 at 4,096: 16).

    Args:
        trees: The trees the launch walks.
        bodies: The most bodies of one of the class's walked trees.
        joins: The most joins one of its trees holds at once (:meth:`.tree_factor.TreeFactor.plan_walk`).
    """
    return Work(trees, Use.FEED, width=bodies, least=1 << max(joins - 1, 0).bit_length(), block=64, registers=128)


_INWARD = """    // The refresh by the tree's lane group (walk_group): the lane feed (FEED, leaves to root) hands the chain one body
    // at a time; a lane builds each body of a batch (BODY, its path entry, its join slots and its last coordinate's
    // column), every lane runs the chain on it, read by shuffle. A body's lifted inertia is carried when its parent is
    // the next joint inward, else added into the parent's join slot (joins[parent], lane k's registers on CUDA, the
    // CPU's one thread every slot) until the chain reaches it. On the CPU one thread is the group. The tree's path and
    // join slots are its class's (WALK_TREE(cls): the classes' tables in constant memory).
    SYNC();
    WALK_TREE(cls);
    // The refresh's arithmetic on one tree, float32 (TreeWalk's inward pass, on the factor's diagonal), at the step's
    // geometry the finish published, each coordinate's column in now and each body's link and frame point in
    // now_link: on a refresh they become the factor's frames. The lane feed's batches run it leaf first; the chain's
    // state is the articulated inertia h,
    // symmetric (A 00 01 02 11 12 22, B row-major 3x3, D 00 01 02 11 12 22), in registers.
    //   BODY(j, up, r): body j's inertia at its frame [m 1, -m X; m X, R I R^T - m X X] (X the skew of c, its centre of
    //     mass less its frame, both as the finish published them in now_com and now_link) as words m, m c, rot, and
    //     its link offset d
    //     (WALK_OFFSET from its parent's frame point, up's, and its own); its frame point becomes the factor's link
    //     (every walk forms d from the two points). Nothing here reads a world position: the frame points are the
    //     articulation's, relative to its origin.
    //   ADD(h, r): h += body r's inertia.
    //   STEP(h, s, c): coordinate c of column s (s 6, R, R + K) eliminated (tree_scan.inward_step); its record
    //     (s by its half, ut, 1/ell, P0 = own + R) leaves whole, once.
    //   LIFT(h, d): h to the parent's frame (tree_scan.inward_edge).
    const float* Lk = now_link.data;
    const float* Lc = reinterpret_cast<const float*>(now_com.data);
    // A column of two halves keeps its linear half WALK_SIDE floats past its record, in data's second half (its rows
    // doubled where a joint's column can have two halves: TreeFactor's walk_data).
    const int WALK_SIDE = data.shape[0] > armature.shape[0] ? $coordinate * armature.shape[0] : 0;
    const float* Nw = reinterpret_cast<const float*>(now.data);
    float* Lo = link.data;
    float* Dt = data.data;
    const float* Ra = armature.data;
    const int* Dr = drive_row_by_dof.data;
    const float* Kr = row_K.data;
    const int* Ct = joint_child.data;
    const float* Qb = reinterpret_cast<const float*>(body_q.data);
    const float* Mb = body_mass.data;
    const float* Ib = reinterpret_cast<const float*>(body_inertia.data);
    const auto SYM = [](int a, int b) -> int { return a <= b ? (a == 0 ? b : (a == 1 ? 2 + b : 5)) : (b == 0 ? a : (b == 1 ? 2 + a : 5)); };
    enum { M = 0, MC = 1, ROT = 4, OFFSET = 10, INERTIA = 13 };
    // A coordinate's column at the step's poses and what the factor adds to its diagonal, R and R + K.
    const auto COLUMN = [&](int c, float* x) {
        // The step column (WALK_NOW) at its body's frame, its half its own words, the record's rule (STEP's): a column
        // that turns keeps its angular half and its linear half only as a side (WALK_SIDE, non-zero), one that slides
        // its linear half; the other zero. Its six words load together, ahead of any choice.
        const float* n = Nw + $now * c;
        float v[6];
        for (int k = 0; k < 6; ++k) v[k] = n[k];
        const bool turns = v[3] != 0.0f || v[4] != 0.0f || v[5] != 0.0f;
        const bool slides = !turns || (WALK_SIDE != 0 && (v[0] != 0.0f || v[1] != 0.0f || v[2] != 0.0f));
        for (int k = 0; k < 3; ++k) {
            x[k] = slides ? v[k] : 0.0f;
            x[3 + k] = turns ? v[3 + k] : 0.0f;
        }
        float added = Ra[c];
        x[6] = added;
        const int row = Dr[c];
        if (row >= 0) {
            if (Kr[row] > 0.0f) added += Kr[row];
        }
        x[7] = added;
    };
    // Every line one body's inertia and link read.
    const auto AHEAD = [&](int j) {
        const int b = Ct[j];
        WALK_TOUCH(Qb + 7 * b + 3);
        WALK_TOUCH(Qb + 7 * b + 6);
        WALK_TOUCH(Mb + b);
        WALK_TOUCH(Ib + 9 * b);
        WALK_TOUCH(Ib + 9 * b + 8);
        WALK_TOUCH(Lk + $link * j);
        WALK_TOUCH(Lc + 3 * j);
    };
    // WALK_OFFSET(p, f, d): a body's link offset d, its parent's frame point p (null at a root) less its own f: the
    // one place a walk forms it from the frame points the finish published.
    const auto WALK_OFFSET = [](const float* p, const float* f, float* d) {
        for (int k = 0; k < 3; ++k) d[k] = p != nullptr ? p[k] - f[k] : 0.0f;
    };
    const auto BODY = [&](int j, int up, float* r) {
        const float* Lr = Lk + $link * j;
        float offset[3];
        WALK_OFFSET(up >= 0 ? Lk + $link * up + $frame : nullptr, Lr + $frame, offset);
        for (int k = 0; k < 3; ++k) r[OFFSET + k] = offset[k];
        *reinterpret_cast<WALK_QUAD*>(Lo + $link * j) = WALK_PLAIN(Lr);
        const int b = Ct[j];
        const float* q = Qb + 7 * b;
        const float* I = Ib + 9 * b;
        const float m = Mb[b];
        // The body's rotation and its centre of mass less its frame, c.
        const float x = q[3], y = q[4], z = q[5], w = q[6];
        const float R[3][3] = {
            {1.0f - 2.0f * (y * y + z * z), 2.0f * (x * y - z * w), 2.0f * (x * z + y * w)},
            {2.0f * (x * y + z * w), 1.0f - 2.0f * (x * x + z * z), 2.0f * (y * z - x * w)},
            {2.0f * (x * z - y * w), 2.0f * (y * z + x * w), 1.0f - 2.0f * (x * x + y * y)}};
        float c[3], mc[3];
        for (int a = 0; a < 3; ++a) c[a] = Lc[3 * j + a] - Lr[$frame + a];
        for (int a = 0; a < 3; ++a) mc[a] = m * c[a];
        // R I R^T - m X X, X the skew of c: -m X X = m (|c|^2 1 - c c^T).
        float RI[3][3];
        for (int a = 0; a < 3; ++a)
            for (int e = 0; e < 3; ++e) RI[a][e] = R[a][0] * I[e] + R[a][1] * I[3 + e] + R[a][2] * I[6 + e];
        const float cc = c[0] * c[0] + c[1] * c[1] + c[2] * c[2];
        r[M] = m;
        for (int a = 0; a < 3; ++a) r[MC + a] = mc[a];
        for (int a = 0; a < 3; ++a)
            for (int e = a; e < 3; ++e)
                r[ROT + SYM(a, e)] = (RI[a][0] * R[e][0] + RI[a][1] * R[e][1] + RI[a][2] * R[e][2]) + mc[a] * -c[e] + (a == e ? m * cc : 0.0f);
    };
    // h = body + h (carrying) or h = body.
    const auto ADD = [&](float* h, const float* r, bool carrying) {
        const float bm = r[M];
        const float* bc = r + MC;
        const float* br = r + ROT;
        const float body[21] = {bm, 0.0f, 0.0f, bm, 0.0f, bm,
                                0.0f, bc[2], -bc[1], -bc[2], 0.0f, bc[0], bc[1], -bc[0], 0.0f,
                                br[0], br[1], br[2], br[3], br[4], br[5]};
        for (int k = 0; k < 21; ++k) h[k] = carrying ? body[k] + h[k] : body[k];
    };
    const auto STEP = [&](float* h, const float* s, int dof) {
$step_head
        if (LANE == 0) {
            // The record (WALK_COORDINATE): the column's non-zero half (the angular one where it turns, 1 / ell's
            // sign bit set), and a column of two halves its linear half in the side float4 (WALK_SIDE floats on).
            float* o = Dt + $coordinate * dof;
            const bool turns = s[3] != 0.0f || s[4] != 0.0f || s[5] != 0.0f;
            const bool both = WALK_SIDE != 0 && turns && (s[0] != 0.0f || s[1] != 0.0f || s[2] != 0.0f);
            const int side = both ? WALK_SIDE : 0;
            const float half[3] = {turns ? s[3] : s[0], turns ? s[4] : s[1], turns ? s[5] : s[2]};
#if defined(__CUDA_ARCH__)
            float4* o4 = reinterpret_cast<float4*>(o);
            o4[0] = make_float4(half[0], half[1], half[2], turns ? -inv : inv);
            o4[1] = make_float4(ut[0], ut[1], ut[2], ut[3]);
            o4[2] = make_float4(ut[4], ut[5], own + s[6], __int_as_float(side));
            if (both) reinterpret_cast<float4*>(o + side)[0] = make_float4(s[0], s[1], s[2], 0.0f);
#else
            for (int k = 0; k < 3; ++k) o[k] = half[k];
            o[3] = turns ? -inv : inv;
            for (int k = 0; k < 6; ++k) o[4 + k] = ut[k];
            o[10] = own + s[6];
            o[11] = __builtin_bit_cast(float, side);
            if (both) {
                for (int k = 0; k < 3; ++k) o[side + k] = s[k];
                o[side + 3] = 0.0f;
            }
#endif
        }
$step_update
    };
    const auto LIFT = [&](float* h, const float* bd) {
$lift
    };
#if defined(__CUDA_ARCH__)
    constexpr int HELD = 1;
#else
    constexpr int HELD = $slots;
#endif
    for (int c = dof0 + LANE; c < dof1; c += LANES) {
        WALK_TOUCH(Nw + $now * c);
        WALK_TOUCH(Ra + c);
        WALK_TOUCH(Dr + c);
    }
    float held[HELD][21];
    for (int s = 0; s < HELD; ++s)
        for (int k = 0; k < 21; ++k) held[s][k] = 0.0f;
    const auto OWNS = [&](int slot) -> bool { return HELD > 1 || LANE == slot; };
    const auto AT = [&](int slot) -> int { return HELD > 1 ? slot : 0; };
    float h[21];
    for (int k = 0; k < 21; ++k) h[k] = 0.0f;
    bool carrying = false;
    // A body as the feed hands it to the chain: BODY's words, its last coordinate's column (s 6, R, R + K), and as
    // bits its parent, coordinates [lo, hi), its own join slot and its parent's.
    struct Feed { float w[26]; };
    enum { RECORD = INERTIA, UP = RECORD + 8, LO = UP + 1, HI = LO + 1, SLOT = HI + 1, INTO = SLOT + 1 };
    FEED(j0, j1 - 1, -1,
        [&](int j) { AHEAD(j); },
        [&](int mine, bool valid) -> Feed {
            Feed r;
            for (int k = 0; k < 26; ++k) r.w[k] = 0.0f;
            int up = -1, lo = 0, hi = 0, slot = -1, into = -1;
            if (valid) {
                const int b = mine - j0;
                up = WALK_PARENT_OF(b);
                lo = dof0 + WALK_FIRST_OF(b);
                hi = dof0 + WALK_FIRST_OF(b + 1);
                slot = WALK_JOIN_OF(b);
                into = up >= 0 ? WALK_JOIN_OF(up) : -1;
                BODY(mine, up >= 0 ? j0 + up : -1, r.w);
                if (hi > lo) COLUMN(hi - 1, r.w + RECORD);
            }
            r.w[UP] = __int_as_float(up);
            r.w[LO] = __int_as_float(lo);
            r.w[HI] = __int_as_float(hi);
            r.w[SLOT] = __int_as_float(slot);
            r.w[INTO] = __int_as_float(into);
            return r;
        },
        [&](int b, const Feed& r) {
            float s[8];
            for (int k = 0; k < 8; ++k) s[k] = r.w[RECORD + k];
            const int b_up = __float_as_int(r.w[UP]), b_lo = __float_as_int(r.w[LO]), b_hi = __float_as_int(r.w[HI]);
            const int b_slot = __float_as_int(r.w[SLOT]), b_into = __float_as_int(r.w[INTO]);
            ADD(h, r.w, carrying);
            if (b_slot >= 0) {
                for (int k = 0; k < 21; ++k) h[k] += READ(held[AT(b_slot)][k], b_slot);
                if (OWNS(b_slot))
                    for (int k = 0; k < 21; ++k) held[AT(b_slot)][k] = 0.0f;
            }
            for (int dof = b_hi - 1; dof >= b_lo; --dof) {
                // The next coordinate's column, loaded before this one stores (a body with several coordinates).
                float next[8];
                if (dof > b_lo) COLUMN(dof - 1, next);
                STEP(h, s, dof);
                if (dof > b_lo)
                    for (int k = 0; k < 8; ++k) s[k] = next[k];
            }
            carrying = false;
            if (b_up >= 0) {
                LIFT(h, r.w + OFFSET);
                if (b_up == b - j0 - 1) {
                    carrying = true;
                } else if (OWNS(b_into)) {
                    for (int k = 0; k < 21; ++k) held[AT(b_into)][k] += h[k];
                }
            }
        });
"""


def _walk_inward_snippet(classes: tuple[WalkClass, ...]) -> str:
    from ..device_preludes import CPU_MATH, LANE_FEED, dual_native, lane_group  # noqa: PLC0415

    step_head, step_update = ts.inward_step("h", "s")
    table = walk_table(walk_tables(classes), joins=True)
    tables = DevicePrelude("walk_tables", table, cpu=table)
    return dual_native(
        substitute(
            _INWARD,
            step_head=ts.indent_text(step_head, 8),
            step_update=ts.indent_text(step_update, 8),
            lift=ts.indent_text(ts.inward_edge(("bd[0]", "bd[1]", "bd[2]"), "h"), 8),
            coordinate=WALK_COORDINATE,
            link=WALK_LINK,
            now=WALK_NOW,
            frame=WALK_FRAME,
            slots=WALK_JOIN_SLOTS,
        ),
        preludes=(walk_loads_prelude(), tables, lane_group("lanes"), CPU_MATH, LANE_FEED),
    )


def walk_tables(classes) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """The walking classes' path tables (:func:`walk_table`'s, :func:`walk_prelude`'s), each class's parent and first
    coordinate."""
    return tuple((walk.parent, walk.first) for walk in classes)


@functools.cache
def walk_step_kernel(classes: tuple[WalkClass, ...]) -> wp.Kernel:
    """The walk's refresh of the walking classes ``classes``, a group of ``lanes`` threads per tree
    (:meth:`TreeFactor.walk_step`), float32, at the poses ``body_q`` holds, on the trees the refresh just factored
    (``mass_update_mask``; every other tree returns): the step's geometry the finish published (``walk_now``,
    ``walk_now_link``, ``walk_now_com``) becomes the factor's frames, with the articulation's origin they are relative
    to, and the group runs the refresh (the inward pass, the lane feed's batches; each tree's path and join slots its
    class's in constant memory, ``walk_class``), so the walk and ``L`` are one factor. Its first thread advances the
    step's ``clock`` past this step (``dt``): the kernel reads no clock, the step's repair and mask read it before, its
    finish and warm start after (:class:`.factor.kernels.Clock`)."""
    from ..kernel_template import python_kernel  # noqa: PLC0415

    @wp.func_native(_walk_inward_snippet(classes))
    def walk_inward(
        lanes: int,
        cls: int,
        j0: int,
        j1: int,
        dof0: int,
        dof1: int,
        now: wp.array[wp.spatial_vector],
        now_link: wp.array2d[float],
        now_com: wp.array[wp.vec3],
        armature: wp.array[float],
        drive_row_by_dof: wp.array[int],
        row_K: wp.array[float],
        joint_child: wp.array[int],
        body_q: wp.array[wp.transform],
        body_mass: wp.array[float],
        body_inertia: wp.array[wp.mat33],
        link: wp.array2d[float],
        data: wp.array2d[float],
    ):
        """The refresh's inward pass of the tree of joints ``[j0, j1)`` and coordinates ``[dof0, dof1)``, of class
        ``cls``, by its group of ``lanes`` (one thread on the CPU), at the step's geometry the finish published
        (``now``, ``now_link``): the factor's frames (each body's ``link``) and each coordinate's whole record in
        ``data`` (:data:`WALK_COORDINATE`), with each body's spatial inertia built in the pass from its pose and mass
        properties."""
        ...

    def walk_step(
        arts: wp.array[int],
        lanes: int,
        mass_update_mask: wp.array[int],
        articulation_start: wp.array[int],
        articulation_joint_end: wp.array[int],
        walk_class: wp.array[int],
        joint_child: wp.array[int],
        joint_qd_start: wp.array[int],
        body_q: wp.array[wp.transform],
        body_mass: wp.array[float],
        body_inertia: wp.array[wp.mat33],
        armature: wp.array[float],
        drive_row_by_dof: wp.array[int],
        row_K: wp.array[float],
        walk_now: wp.array[wp.spatial_vector],
        walk_now_link: wp.array2d[float],
        walk_now_com: wp.array[wp.vec3],
        articulation_origin: wp.array[wp.vec3],
        clock: wp.array[Clock],
        dt: float,
        # outputs
        walk_data: wp.array2d[float],
        walk_link: wp.array2d[float],
        walk_origin: wp.array[wp.vec3],
    ):
        tree, lane = wp.tid()
        if tree == 0 and lane == 0:
            clock_advance(clock, dt)
        art = arts[tree]
        if mass_update_mask[art] == 0:
            return
        j0 = articulation_start[art]
        j1 = articulation_joint_end[art]
        walk_inward(
            lanes,
            walk_class[art],
            j0,
            j1,
            joint_qd_start[j0],
            joint_qd_start[j1],
            walk_now,
            walk_now_link,
            walk_now_com,
            armature,
            drive_row_by_dof,
            row_K,
            joint_child,
            body_q,
            body_mass,
            body_inertia,
            walk_link,
            walk_data,
        )
        if lane == 0:
            walk_origin[art] = articulation_origin[art]

    tag = hashlib.sha256(repr(walk_tables(classes)).encode()).hexdigest()[:8]
    return python_kernel(walk_step, f"walk_step_{tag}")
