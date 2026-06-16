# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp kernels for the standalone Position Anymal-C Newton repro MDP."""

from __future__ import annotations

import warp as wp
from envs.math import warp as math_warp

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _update_command_kernel(
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    jc_per_env: wp.int32,
    jd_per_env: wp.int32,
    cmd_mask: wp.array2d(dtype=wp.bool),
    reward_scales: wp.array(dtype=wp.float32),
    step_dt: wp.float32,
    advance_success: wp.int32,
    cmd_buf: wp.array3d(dtype=wp.float32),
    command_error: wp.array2d(dtype=wp.float32),
):
    i = wp.tid()

    q0 = i * jc_per_env
    qd0 = i * jd_per_env
    cur_pos = wp.vec3f(joint_q[q0 + 0], joint_q[q0 + 1], joint_q[q0 + 2])
    cur_quat = wp.quatf(joint_q[q0 + 3], joint_q[q0 + 4], joint_q[q0 + 5], joint_q[q0 + 6])
    cur_linvel = wp.vec3f(joint_qd[qd0 + 0], joint_qd[qd0 + 1], joint_qd[qd0 + 2])
    cur_angvel = wp.vec3f(joint_qd[qd0 + 3], joint_qd[qd0 + 4], joint_qd[qd0 + 5])

    rpy = math_warp.euler_xyz_from_quat(cur_quat)

    cmd_buf[i, 2, 0] = cur_pos[0]
    cmd_buf[i, 2, 1] = cur_pos[1]
    cmd_buf[i, 2, 2] = cur_pos[2]
    cmd_buf[i, 2, 3] = rpy[0]
    cmd_buf[i, 2, 4] = rpy[1]
    cmd_buf[i, 2, 5] = rpy[2]
    cmd_buf[i, 2, 6] = cur_linvel[0]
    cmd_buf[i, 2, 7] = cur_linvel[1]
    cmd_buf[i, 2, 8] = cur_linvel[2]
    cmd_buf[i, 2, 9] = cur_angvel[0]
    cmd_buf[i, 2, 10] = cur_angvel[1]
    cmd_buf[i, 2, 11] = cur_angvel[2]

    target_pos = wp.vec3f(cmd_buf[i, 0, 0], cmd_buf[i, 0, 1], cmd_buf[i, 0, 2])
    target_roll = cmd_buf[i, 0, 3]
    target_pitch = cmd_buf[i, 0, 4]
    target_yaw = cmd_buf[i, 0, 5]
    target_linvel = wp.vec3f(cmd_buf[i, 0, 6], cmd_buf[i, 0, 7], cmd_buf[i, 0, 8])
    target_angvel = wp.vec3f(cmd_buf[i, 0, 9], cmd_buf[i, 0, 10], cmd_buf[i, 0, 11])
    target_time = cmd_buf[i, 0, 12]

    m0 = wp.float32(cmd_mask[i, 0])
    m1 = wp.float32(cmd_mask[i, 1])
    m2 = wp.float32(cmd_mask[i, 2])
    m3 = wp.float32(cmd_mask[i, 3])
    m4 = wp.float32(cmd_mask[i, 4])
    m5 = wp.float32(cmd_mask[i, 5])
    m6 = wp.float32(cmd_mask[i, 6])
    m7 = wp.float32(cmd_mask[i, 7])
    m8 = wp.float32(cmd_mask[i, 8])
    m9 = wp.float32(cmd_mask[i, 9])
    m10 = wp.float32(cmd_mask[i, 10])
    m11 = wp.float32(cmd_mask[i, 11])

    pos_w = wp.vec3f(
        (target_pos[0] - cur_pos[0]) * m0,
        (target_pos[1] - cur_pos[1]) * m1,
        (target_pos[2] - cur_pos[2]) * m2,
    )
    pos_err = wp.quat_rotate_inv(cur_quat, pos_w)
    cmd_buf[i, 1, 0] = pos_err[0]
    cmd_buf[i, 1, 1] = pos_err[1]
    cmd_buf[i, 1, 2] = pos_err[2]

    target_quat = math_warp.quat_from_euler_xyz(target_roll, target_pitch, target_yaw)
    quat_err = math_warp.quat_mul(math_warp.quat_conjugate(cur_quat), target_quat)
    rot_err = math_warp.axis_angle_from_quat(quat_err, 1.0e-6)
    cmd_buf[i, 1, 3] = rot_err[0] * m3
    cmd_buf[i, 1, 4] = rot_err[1] * m4
    cmd_buf[i, 1, 5] = rot_err[2] * m5

    linvel_w = wp.vec3f(
        (target_linvel[0] - cur_linvel[0]) * m6,
        (target_linvel[1] - cur_linvel[1]) * m7,
        (target_linvel[2] - cur_linvel[2]) * m8,
    )
    linvel_err = wp.quat_rotate_inv(cur_quat, linvel_w)
    cmd_buf[i, 1, 6] = linvel_err[0]
    cmd_buf[i, 1, 7] = linvel_err[1]
    cmd_buf[i, 1, 8] = linvel_err[2]

    angvel_w = wp.vec3f(
        (target_angvel[0] - cur_angvel[0]) * m9,
        (target_angvel[1] - cur_angvel[1]) * m10,
        (target_angvel[2] - cur_angvel[2]) * m11,
    )
    angvel_err = wp.quat_rotate_inv(cur_quat, angvel_w)
    cmd_buf[i, 1, 9] = angvel_err[0]
    cmd_buf[i, 1, 10] = angvel_err[1]
    cmd_buf[i, 1, 11] = angvel_err[2]

    err_pos = wp.length(pos_err)
    err_rot = wp.length(wp.vec3f(rot_err[0] * m3, rot_err[1] * m4, rot_err[2] * m5))
    err_lin = wp.length(linvel_err)
    err_ang = wp.length(angvel_err)
    command_error[i, 0] = err_pos
    command_error[i, 1] = err_rot
    command_error[i, 2] = err_lin
    command_error[i, 3] = err_ang

    if advance_success != 0:
        if (
            err_pos < reward_scales[0]
            and err_rot < reward_scales[1]
            and err_lin < reward_scales[2]
            and err_ang < reward_scales[3]
        ):
            cmd_buf[i, 2, 12] = cmd_buf[i, 2, 12] + step_dt

    cmd_buf[i, 1, 12] = target_time - cmd_buf[i, 2, 12]


@wp.kernel
def _build_observations_kernel(
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    jc_per_env: wp.int32,
    jd_per_env: wp.int32,
    last_action: wp.array2d(dtype=wp.float32),
    cmd_buf: wp.array3d(dtype=wp.float32),
    height_scan: wp.array2d(dtype=wp.float32),
    observations: wp.array2d(dtype=wp.float32),
):
    i = wp.tid()

    q0 = i * jc_per_env
    qd0 = i * jd_per_env
    root_quat = wp.quatf(joint_q[q0 + 3], joint_q[q0 + 4], joint_q[q0 + 5], joint_q[q0 + 6])
    base_lin_vel = wp.quat_rotate_inv(root_quat, wp.vec3f(joint_qd[qd0 + 0], joint_qd[qd0 + 1], joint_qd[qd0 + 2]))
    base_ang_vel = wp.quat_rotate_inv(root_quat, wp.vec3f(joint_qd[qd0 + 3], joint_qd[qd0 + 4], joint_qd[qd0 + 5]))
    projected_gravity = wp.quat_rotate_inv(root_quat, wp.vec3f(0.0, 0.0, -1.0))

    observations[i, 0] = base_lin_vel[0]
    observations[i, 1] = base_lin_vel[1]
    observations[i, 2] = base_lin_vel[2]
    observations[i, 3] = base_ang_vel[0]
    observations[i, 4] = base_ang_vel[1]
    observations[i, 5] = base_ang_vel[2]
    observations[i, 6] = projected_gravity[0]
    observations[i, 7] = projected_gravity[1]
    observations[i, 8] = projected_gravity[2]

    for j in range(12):
        observations[i, 9 + j] = joint_q[q0 + 7 + j]
        observations[i, 21 + j] = joint_qd[qd0 + 6 + j]
        observations[i, 33 + j] = last_action[i, j]
        observations[i, 45 + j] = cmd_buf[i, 1, j]

    for j in range(416):
        observations[i, 57 + j] = height_scan[i, j]


@wp.kernel
def _update_task_outputs_kernel(
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    jc_per_env: wp.int32,
    jd_per_env: wp.int32,
    default_joint_pos: wp.array2d(dtype=wp.float32),
    cmd_buf: wp.array3d(dtype=wp.float32),
    max_episode_length: wp.int32,
    episode_length: wp.array(dtype=wp.int32),
    reward: wp.array(dtype=wp.float32),
    terminated: wp.array(dtype=wp.bool),
    truncated: wp.array(dtype=wp.bool),
):
    i = wp.tid()

    episode_length[i] = episode_length[i] + 1

    q0 = i * jc_per_env
    qd0 = i * jd_per_env
    root_quat = wp.quatf(joint_q[q0 + 3], joint_q[q0 + 4], joint_q[q0 + 5], joint_q[q0 + 6])
    projected_gravity = wp.quat_rotate_inv(root_quat, wp.vec3f(0.0, 0.0, -1.0))

    joint_pos_diff = wp.float32(0.0)
    for j in range(12):
        joint_pos_diff = wp.max(joint_pos_diff, wp.abs(joint_q[q0 + 7 + j] - default_joint_pos[i, j]))

    if cmd_buf[i, 1, 12] <= 0.0 and joint_pos_diff < 0.25:
        reward[i] = 1.0
    else:
        reward[i] = 0.0

    lin_speed = wp.length(wp.vec3f(joint_qd[qd0 + 0], joint_qd[qd0 + 1], joint_qd[qd0 + 2]))
    ang_speed = wp.length(wp.vec3f(joint_qd[qd0 + 3], joint_qd[qd0 + 4], joint_qd[qd0 + 5]))
    terminated[i] = projected_gravity[2] > -0.1736 or lin_speed > 20.0 or ang_speed > 100.0
    truncated[i] = episode_length[i] >= max_episode_length


@wp.kernel
def _set_joint_dof_property_kernel(
    values: wp.array(dtype=wp.float32),
    jd_per_env: wp.int32,
    value: wp.float32,
):
    env_id, dof_id = wp.tid()
    values[env_id * jd_per_env + 6 + dof_id] = value


@wp.kernel
def _zero_actions_kernel(
    default_joint_pos: wp.array2d(dtype=wp.float32),
    last_action: wp.array2d(dtype=wp.float32),
    targets: wp.array2d(dtype=wp.float32),
):
    env_id, joint_id = wp.tid()
    last_action[env_id, joint_id] = 0.0
    targets[env_id, joint_id] = default_joint_pos[env_id, joint_id]


@wp.kernel
def _build_goal_arrow_instances_kernel(
    cmd_buf: wp.array3d(dtype=wp.float32),
    cmd_mask: wp.array2d(dtype=wp.bool),
    xforms: wp.array(dtype=wp.transformf),
    scales: wp.array(dtype=wp.vec3),
    colors: wp.array(dtype=wp.vec3),
    materials: wp.array(dtype=wp.vec4),
):
    env_id = wp.tid()

    target_pos = wp.vec3f(cmd_buf[env_id, 0, 0], cmd_buf[env_id, 0, 1], cmd_buf[env_id, 0, 2] + 0.5)
    target_quat = math_warp.quat_from_euler_xyz(
        cmd_buf[env_id, 0, 3],
        cmd_buf[env_id, 0, 4],
        cmd_buf[env_id, 0, 5],
    )
    xforms[env_id] = wp.transformf(target_pos, target_quat)

    active = (
        cmd_mask[env_id, 0]
        or cmd_mask[env_id, 1]
        or cmd_mask[env_id, 2]
        or cmd_mask[env_id, 3]
        or cmd_mask[env_id, 4]
        or cmd_mask[env_id, 5]
    )
    if active:
        scales[env_id] = wp.vec3f(0.6, 1.0, 1.0)
    else:
        scales[env_id] = wp.vec3f(0.0, 0.0, 0.0)

    colors[env_id] = wp.vec3f(1.0, 0.0, 0.0)
    materials[env_id] = wp.vec4f(0.0, 0.0, 0.0, 0.0)


@wp.kernel
def _clear_counter_kernel(counter: wp.array(dtype=wp.int32)):
    counter[0] = 0


@wp.kernel
def _collect_reset_ids_kernel(
    terminated: wp.array(dtype=wp.bool),
    truncated: wp.array(dtype=wp.bool),
    reset_ids: wp.array(dtype=wp.int32),
    reset_count: wp.array(dtype=wp.int32),
):
    env_id = wp.tid()
    if terminated[env_id] or truncated[env_id]:
        write_id = wp.atomic_add(reset_count, 0, 1)
        reset_ids[write_id] = env_id


@wp.kernel
def _reset_envs_kernel(
    reset_ids: wp.array(dtype=wp.int32),
    source_ids: wp.array(dtype=wp.int32),
    reset_count: wp.array(dtype=wp.int32),
    use_source_ids: wp.int32,
    reset_states: wp.array2d(dtype=wp.float32),
    cmd_buf_table: wp.array3d(dtype=wp.float32),
    cmd_mask_table: wp.array2d(dtype=wp.bool),
    height_scan_table: wp.array2d(dtype=wp.float32),
    terrain_env_origins_table: wp.array2d(dtype=wp.float32),
    reset_state_count: wp.int32,
    cmd_count: wp.int32,
    height_count: wp.int32,
    terrain_count: wp.int32,
    jc_per_env: wp.int32,
    jd_per_env: wp.int32,
    num_dofs: wp.int32,
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    control_f: wp.array(dtype=wp.float32),
    cmd_buf: wp.array3d(dtype=wp.float32),
    cmd_mask: wp.array2d(dtype=wp.bool),
    height_scan: wp.array2d(dtype=wp.float32),
    terrain_env_origins: wp.array2d(dtype=wp.float32),
    default_joint_pos: wp.array2d(dtype=wp.float32),
    targets: wp.array2d(dtype=wp.float32),
    last_action: wp.array2d(dtype=wp.float32),
    episode_length: wp.array(dtype=wp.int32),
):
    idx = wp.tid()
    if idx >= reset_count[0]:
        return

    env_id = reset_ids[idx]
    source_id = env_id % reset_state_count
    if use_source_ids != 0:
        source_id = source_ids[idx] % reset_state_count
    cmd_id = source_id % cmd_count
    height_id = source_id % height_count
    terrain_id = source_id % terrain_count
    q0 = env_id * jc_per_env
    qd0 = env_id * jd_per_env

    for j in range(7):
        joint_q[q0 + j] = reset_states[source_id, j]
    for j in range(6):
        joint_qd[qd0 + j] = reset_states[source_id, 7 + j]
    for j in range(12):
        joint_q[q0 + 7 + j] = reset_states[source_id, 13 + j]
        joint_qd[qd0 + 6 + j] = reset_states[source_id, 13 + num_dofs + j]
        targets[env_id, j] = default_joint_pos[env_id, j]
        last_action[env_id, j] = 0.0

    for j in range(18):
        control_f[qd0 + j] = 0.0
    for row in range(3):
        for col in range(13):
            cmd_buf[env_id, row, col] = cmd_buf_table[cmd_id, row, col]
    for j in range(12):
        cmd_mask[env_id, j] = cmd_mask_table[cmd_id, j]
    for j in range(416):
        height_scan[env_id, j] = height_scan_table[height_id, j]
    for j in range(3):
        terrain_env_origins[env_id, j] = terrain_env_origins_table[terrain_id, j]
    episode_length[env_id] = 0
