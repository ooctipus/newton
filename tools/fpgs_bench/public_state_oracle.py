# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""FP64 oracles of an articulation's public state and physical mass matrix, for cache tests."""

import numpy as np
import warp as wp

import newton
from tools.fpgs_bench.fp64_transforms import compose, inverse, rotate


def physical_mass(model, state):
    """Assemble full physical H from public COM twists, including every weld."""
    count = model.joint_dof_count
    probe, twists = model.state(), []
    for dof in range(count):
        unit = np.zeros(count, np.float32)
        unit[dof] = 1.0
        newton.eval_fk(model, state.joint_q, wp.array(unit, dtype=float, device=model.device), probe)
        twists.append(probe.body_qd.numpy().astype(float))
    twists = np.asarray(twists).transpose(1, 2, 0)
    mass, inertia, poses = model.body_mass.numpy(), model.body_inertia.numpy(), probe.body_q.numpy()
    result = np.zeros((count, count))
    for body in range(model.body_count):
        rotation = np.asarray(wp.quat_to_matrix(wp.quat(*poses[body, 3:])), float).reshape(3, 3)
        linear, angular = twists[body, :3], twists[body, 3:]
        # Use all authored inertia entries: do not symmetrize the only oracle.
        result += mass[body] * linear.T @ linear + angular.T @ (rotation @ inertia[body] @ rotation.T) @ angular
    return result


def public_reference(model, state):
    """Evaluate current public poses/COM twists in FP64, including free-root anchors.

    Reuse the independent publication reference's exact transform primitives,
    not the candidate's scan or its current S. Generic FP32 FK subtracts world
    positions at each joint: saved translations of 35--76 m make that a less
    accurate velocity oracle. No quaternion/input renormalization is introduced.
    """
    names = (
        "joint_type",
        "joint_parent",
        "joint_child",
        "joint_q_start",
        "joint_qd_start",
        "joint_X_p",
        "joint_X_c",
        "joint_axis",
        "body_com",
    )
    values = {name: getattr(model, name).numpy() for name in names}
    q, qd = (getattr(state, name).numpy().astype(np.float64) for name in ("joint_q", "joint_qd"))
    poses = np.zeros((model.body_count, 7), np.float64)
    anchors = np.zeros((model.joint_count, 7), np.float64)
    roots = np.zeros(model.body_count, int)
    for joint in range(model.joint_count):
        kind, parent, body, qs, ds = (int(values[name][joint]) for name in names[:5])
        anchor = values["joint_X_p"][joint].astype(np.float64)
        if parent >= 0:
            anchor = compose(poses[parent], anchor)
        transform = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        if kind == int(newton.JointType.REVOLUTE):
            transform[3:6] = values["joint_axis"][ds].astype(np.float64) * np.sin(q[qs] / 2)
            transform[6] = np.cos(q[qs] / 2)
        elif kind == int(newton.JointType.PRISMATIC):
            transform[:3] = values["joint_axis"][ds].astype(np.float64) * q[qs]
        elif kind == int(newton.JointType.FREE):
            transform = q[qs : qs + 7]
            if parent >= 0:
                raise ValueError("Public oracle admits free articulation roots only")
        elif kind != int(newton.JointType.FIXED):
            raise ValueError(f"Unsupported public oracle joint type: {kind}")
        anchors[joint] = anchor
        poses[body] = compose(compose(anchor, transform), inverse(values["joint_X_c"][joint].astype(np.float64)))
        roots[body] = roots[parent] if parent >= 0 else body
    com = poses[:, :3] + rotate(poses[:, 3:], values["body_com"].astype(np.float64))
    motion = np.zeros((model.body_count, 6), np.float64)
    for joint in range(model.joint_count):
        kind, parent, body, _, ds = (int(values[name][joint]) for name in names[:5])
        anchor, local = anchors[joint], np.zeros(6)
        if kind in (int(newton.JointType.PRISMATIC), int(newton.JointType.REVOLUTE)):
            axis = rotate(anchor[3:], values["joint_axis"][ds].astype(np.float64))
            if kind == int(newton.JointType.PRISMATIC):
                local[:3] = axis * qd[ds]
            else:
                local[3:] = axis * qd[ds]
                local[:3] = np.cross(anchor[:3] - com[roots[body]], local[3:])
        elif kind == int(newton.JointType.FREE):
            # Free-root linear DOFs describe its COM, in the parent-anchor basis.
            local[:3] = rotate(anchor[3:], qd[ds : ds + 3])
            local[3:] = rotate(anchor[3:], qd[ds + 3 : ds + 6])
        motion[body] = local + (motion[parent] if parent >= 0 else 0)
    public = motion.copy()
    public[:, :3] += np.cross(motion[:, 3:], com - com[roots])
    return {"body_q": poses, "body_qd": public}


def public_state(test, model, state):
    """Check FP64 poses and coordinate-invariant linear/angular velocity errors."""
    expected = public_reference(model, state)
    for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
        test.assertTrue(np.isfinite(getattr(state, name).numpy()).all(), name)
    np.testing.assert_allclose(state.body_q.numpy(), expected["body_q"], rtol=2e-5, atol=3e-6, err_msg="body_q")
    # A near-zero Cartesian component can cancel large ancestor contributions;
    # an elementwise relative gate depends on the arbitrary world orientation.
    # Keep linear and angular units separate, and use the stricter 3e-6
    # absolute-plus-relative physical vector bound against independent FP64.
    actual = state.body_qd.numpy().astype(np.float64).reshape(-1, 2, 3)
    reference = expected["body_qd"].reshape(-1, 2, 3)
    error = np.linalg.norm(actual - reference, axis=2)
    scale = 1.0 + np.linalg.norm(reference, axis=2)
    test.assertLess(float(np.max(error / scale)), 3e-6, "public linear/angular velocity")


def force_response(owner, solver, state, control, velocity, dt) -> np.ndarray:
    """The joint acceleration ``control.joint_f`` alone causes through the held factor: the difference of the predictor
    with and without it, which cancels every other term. Solves ``(H + R) x = joint_f`` for an independent ``H``."""
    owner.predict(state, solver.factor, solver.model.control(), velocity, dt)
    free = solver.factor.joint_qdd.numpy().astype(float).copy()
    owner.predict(state, solver.factor, control, velocity, dt)
    return solver.factor.joint_qdd.numpy().astype(float) - free
