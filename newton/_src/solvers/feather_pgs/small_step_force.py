# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Private current-force and held-W prediction for the bounded world owner."""

from types import SimpleNamespace

import numpy as np
import warp as wp

from ...sim import BodyFlags
from . import sparse_factor


@wp.struct
class ForceInput:
    descendant_offsets: wp.array[int]
    descendant_bodies: wp.array[int]
    q_index: wp.array[int]
    q_start: wp.array[int]
    body_f_s: wp.array[wp.spatial_vector]
    joint_S_s: wp.array[wp.spatial_vector]
    external: wp.array[wp.spatial_vector]
    body_q: wp.array[wp.transform]
    body_com: wp.array[wp.vec3]
    origin: wp.array[wp.vec3]
    body_flags: wp.array[int]
    actuation: wp.array[float]
    stiffness: wp.array[float]
    reference: wp.array[float]
    damping: wp.array[float]
    q: wp.array[float]
    current_qd: wp.array[float]
    u0: wp.array[float]
    stage3_qd: wp.array[float]
    kinematic_dof_mask: wp.array[int]
    kinematic_joint_mask: wp.array[int]
    dt: float


def build_force_plan(model):
    """Validate the existing G1 layout and store its scalar descendant lists."""
    host = sparse_factor.make_plan(model)
    worlds = int(model.world_count)
    starts = model.joint_q_start.numpy()[:-1].reshape(worlds, 44)
    relative = starts - starts[:, :1]
    if not np.all(relative == relative[0]):
        raise ValueError("Small-step force requires identical G1 coordinate order")
    q_index = relative[0, host["dof_joint"]].astype(np.int32)
    q_index[:6] = -1
    if not np.array_equal(q_index[6:], np.arange(7, 44)):
        raise ValueError("Small-step force requires the checked free-root coordinate layout")
    masks = host["body_mask"][:44]
    offsets, bodies = [0], []
    for natural in range(6, 43):
        bodies.extend(np.flatnonzero((masks & np.uint64(1 << natural)) != 0).tolist())
        offsets.append(len(bodies))
    if len(bodies) != 206:
        raise ValueError("Small-step force descendant work differs from the checked tree")
    values = {
        "descendant_offsets": np.asarray(offsets, dtype=np.int32),
        "descendant_bodies": np.asarray(bodies, dtype=np.int32),
        "q_index": q_index,
    }
    return SimpleNamespace(
        **{name: wp.array(value, dtype=int, device=model.device) for name, value in values.items()},
        host=values,
        scalar_body_visits=206,
        root_body_visits=44,
    )


def make_force_input(solver, state_in, state_aug, control, stage3_qd, dt, *, plan):
    """Bind ready current force and clamped drive inputs without launching work."""
    result = ForceInput()
    for name in ("descendant_offsets", "descendant_bodies", "q_index"):
        setattr(result, name, getattr(plan, name))
    result.q_start = solver.model.joint_q_start
    result.body_f_s = state_aug.body_f_s
    result.joint_S_s = state_aug.joint_S_s
    result.external = state_in.body_f
    result.body_q = state_in.body_q
    result.body_com = solver.model.body_com
    result.origin = solver.articulation_origin
    result.body_flags = solver.model.body_flags
    result.actuation = control.joint_f
    result.stiffness = solver._passive_spring_stiffness
    result.reference = solver._passive_spring_ref
    result.damping = solver._passive_joint_damping
    result.q = state_in.joint_q
    result.current_qd = state_in.joint_qd
    result.u0 = state_aug.joint_tau
    result.stage3_qd = stage3_qd
    result.kinematic_dof_mask = solver._kinematic_dof_mask
    result.kinematic_joint_mask = solver._kinematic_joint_mask
    result.dt = dt
    return result


def get_force_source():
    """Return a CUDA fragment producing shared natural-order ``small_vhat``.

    The containing32-thread world kernel supplies p, d, f and integer locals
    group, art, world, start, lane. Current body forces and current screws are
    projected before the held inverse whitener. All scratch remains private;
    canonical force, acceleration and predictor arrays are never written.
    """
    return r"""
    __shared__ float small_body_force[44*6], small_root_force[6];
    __shared__ float small_tau[43], small_whitened[43], small_vhat[43];
    const int small_js = p.art_joint_start.data[art];
    const int small_qs = f.q_start.data[small_js];
    for (int local = lane; local < 44; local += 32) {
        const int body = p.joint_child.data[small_js+local];
        auto external = f.external.data[body];
        if ((f.body_flags.data[body] & SMALL_KINEMATIC_FLAG) != 0)
            external = wp::spatial_vector_t<float>();
        const auto linear = wp::spatial_top(external);
        const auto center = wp::transform_point(f.body_q.data[body], f.body_com.data[body]) - f.origin.data[art];
        const auto angular = wp::spatial_bottom(external) + wp::cross(center, linear);
        const float* bias = reinterpret_cast<const float*>(&f.body_f_s.data[body]);
        for (int c = 0; c < 3; ++c) {
            small_body_force[local*6+c] = bias[c] - linear[c];
            small_body_force[local*6+c+3] = bias[c+3] - angular[c];
        }
    }
    __syncwarp();
    // Reuse one complete root wrench for all six floating coordinates.
    if (lane < 6) {
        float value = 0.0f;
        for (int body = 0; body < 44; ++body) value += small_body_force[body*6+lane];
        small_root_force[lane] = value;
    }
    __syncwarp();
    for (int natural = lane; natural < 43; natural += 32) {
        const float* screw = reinterpret_cast<const float*>(&f.joint_S_s.data[start+natural]);
        float projection = 0.0f;
        if (natural < 6) {
            for (int c = 0; c < 6; ++c) projection += screw[c]*small_root_force[c];
        } else {
            float force[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            const int lo = f.descendant_offsets.data[natural-6];
            const int hi = f.descendant_offsets.data[natural-5];
            for (int item = lo; item < hi; ++item) {
                const int body = f.descendant_bodies.data[item];
                #pragma unroll
                for (int c = 0; c < 6; ++c) force[c] += small_body_force[body*6+c];
            }
            #pragma unroll
            for (int c = 0; c < 6; ++c) projection += screw[c]*force[c];
        }
        float value = -projection + f.actuation.data[start+natural];
        if (natural >= 6) {
            float passive = f.stiffness.data[start+natural] *
                (f.reference.data[start+natural] - f.q.data[small_qs+f.q_index.data[natural]]);
            passive -= f.damping.data[start+natural]*f.current_qd.data[start+natural];
            value += passive;
        }
        small_tau[natural] = value + f.u0.data[start+natural];
    }
    __syncwarp();
    for (int row = lane; row < 43; row += 32) {
        float value = 0.0f;
        for (int col = 0; col <= row; ++col) {
            const int entry = p.index.data[row*43+col];
            if (entry >= 0) value += d.W.data[group*434+entry]*small_tau[42-col];
        }
        small_whitened[row] = value;
    }
    __syncwarp();
    for (int col = lane; col < 43; col += 32) {
        float value = 0.0f;
        for (int item = 0; item < p.inverse_count.data[col]; ++item) {
            const int row = p.inverse_nodes.data[col*18+item];
            value += d.W.data[group*434+p.index.data[row*43+col]]*small_whitened[row];
        }
        const int natural = 42-col;
        float acceleration = d.valid.data[world] && !d.status.data[world] ? value : NAN;
        if (f.kinematic_dof_mask.data[start+natural] != 0) acceleration = 0.0f;
        small_vhat[natural] = f.stage3_qd.data[start+natural] + acceleration*f.dt;
    }
    __syncwarp();
    if (lane < 3 && f.kinematic_joint_mask.data[small_js] == 0) {
        const int a = (lane+1)%3, b = (lane+2)%3;
        const float transport = f.stage3_qd.data[start+3+a]*f.stage3_qd.data[start+b]
                              - f.stage3_qd.data[start+3+b]*f.stage3_qd.data[start+a];
        small_vhat[lane] += transport*f.dt;
    }
    __syncwarp();
""".replace("SMALL_KINEMATIC_FLAG", str(int(BodyFlags.KINEMATIC)))
