# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-free replay MDP for the captured factory nut-thread bundle.

Pure-Torch obs/action/reset against the live newton.Model (xyzw quats). The franka
runs on the solver's implicit position PD; gains/armature/friction come from the
captured model state.
"""

from __future__ import annotations

import math
import os

import newton
import numpy as np
import torch
import warp as wp
from envs.math.torch import (
    combine_frame_transforms,
    euler_xyz_from_quat,
    quat_apply,
    subtract_frame_transforms,
    wrap_to_pi,
)

# Per-env body indices (robot 0-12, table 13, board 14, bolt 15, nut 16).
ROBOT_ROOT_BODY = 0  # panda_link0
EE_BODY = 12  # panda_fingertip_centered
BOLT_BODY = 15  # fixed_asset
NUT_BODY = 16  # held_asset

NUM_ARM_JOINTS = 7
NUM_FINGER_JOINTS = 2
NUM_DOFS = NUM_ARM_JOINTS + NUM_FINGER_JOINTS  # 9

# Per-env free-joint coord blocks (pos3 + quat4 xyzw): nut q[9:16], table q[16:23],
# board q[23:30], bolt q[30:37]. Board, bolt and nut are captured rigidly together.
NUT_FREE_Q = 9
BOARD_FREE_Q = 23
BOLT_FREE_Q = 30

TERM_SIZES = (7, 7, 6, 7, 9, 8)  # held, fixed, ee_vel, ee_pose, joint_pos, prev_action (held-first)
OBS_DIM = 220
HISTORY_LENGTH = 5
ACTION_DIM = 8

TIP_OFFSET = (0.0, 0.0, 0.035)  # FixedAssetTip == HeldAssetTip
ARM_ACTION_SCALE = 0.02  # RelativeJointPositionAction scale
GRIPPER_OPEN = 0.04  # BinaryJointPositionAction open / close targets
GRIPPER_CLOSE = 0.0

# Success termination (nut_thread_m16 progress_context): nut center-axis-bottom vs the
# bolt fully-screwed offset (both identity quat), then env thresholds. oob = held-asset
# env-relative AABB (a too-low or off-board nut instant-fails like the real env).
HELD_ALIGN_OFFSET = (0.0, 0.0, 0.01)  # NUT_M16.center_axis_bottom
ASSEMBLED_OFFSET = (0.0, 0.0, 0.022)  # BOLT_M16.fully_screwed_nut_offset
SUCCESS_THRESHOLD = 0.001  # held-in-fixed z [m]
SUCCESS_EULER_TOL = 0.025  # |euler_x| + |euler_y| [rad]
SUCCESS_XY_TOL = 0.0025  # held-in-fixed xy norm [m]
OOB_LO = (0.0, -0.675, -0.05)  # held-asset env-relative bound min [m]
OOB_HI = (1.0, 0.675, 1.0)  # held-asset env-relative bound max [m]


class MDP:
    """Factory nut-thread task logic for a Newton-only replay."""

    def __init__(
        self,
        sim,
        env_origins,
        num_envs: int,
        physics_dt: float,
        decimation: int,
        episode_length_s: float,
        device: str,
        extras_dir: str,
        policy: str = "tap",
    ) -> None:
        if not getattr(sim, "model_state_applied", False):
            raise RuntimeError("factory_nut_thread needs a bundle with model_state.npz; re-capture.")
        self.sim = sim
        self.num_envs = num_envs
        self.device = device
        self.extras_dir = extras_dir
        self.env_origins = env_origins.to(device).view(num_envs, 3).contiguous()

        model = sim.model
        self.bodies_per_env = model.body_count // num_envs
        self.jc_per = int(model.joint_coord_world_start.numpy()[1])
        self.jd_per = int(model.joint_dof_world_start.numpy()[1])
        self.max_episode_length = max(1, int(math.ceil(episode_length_s / (physics_dt * decimation))))

        # Live state/control views (shared memory); franka driven via implicit PD (joint_target_pos).
        self.joint_q = wp.to_torch(sim.state.joint_q)
        self.joint_qd = wp.to_torch(sim.state.joint_qd)
        self.joint_f = wp.to_torch(sim.control.joint_f)
        self.joint_target_pos = wp.to_torch(sim.control.joint_target_pos)
        self.joint_target_vel = wp.to_torch(sim.control.joint_target_vel)

        self._q_base = torch.arange(num_envs, device=device).view(-1, 1) * self.jc_per
        self._d_base = torch.arange(num_envs, device=device).view(-1, 1) * self.jd_per
        self._dof_off = torch.arange(NUM_DOFS, device=device).view(1, -1)
        self._free_off = torch.arange(7, device=device).view(1, -1)
        self._dof_q_idx = (self._q_base + self._dof_off).reshape(-1)
        self._dof_d_idx = (self._d_base + self._dof_off).reshape(-1)
        # abnormal-robot watchdog: snapshot the franka per-dof velocity limits (static).
        self._robot_vel_limit = wp.to_torch(model.joint_velocity_limit)[self._dof_d_idx].view(num_envs, NUM_DOFS)

        self.tip_offset = torch.tensor(TIP_OFFSET, device=device).view(1, 3)
        self._held_off = torch.tensor(HELD_ALIGN_OFFSET, device=device).view(1, 3)
        self._assembled_off = torch.tensor(ASSEMBLED_OFFSET, device=device).view(1, 3)
        self._ident_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).view(1, 4)
        self._oob_lo = torch.tensor(OOB_LO, device=device)
        self._oob_hi = torch.tensor(OOB_HI, device=device)

        self._load_captured_state()
        self._load_policy(policy)

        # arm = relative delta (re-added to live q each step); gripper = absolute target.
        self._arm_delta = torch.zeros(num_envs, NUM_ARM_JOINTS, device=device)
        self._finger_target = self._reset_dof_q[:, NUM_ARM_JOINTS:NUM_DOFS].clone()
        self.prev_action = torch.zeros(num_envs, ACTION_DIM, device=device)
        self.history = [torch.zeros(num_envs, HISTORY_LENGTH, s, device=device) for s in TERM_SIZES]
        self.episode_length = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.terminated = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.truncated = torch.zeros(num_envs, dtype=torch.bool, device=device)

        self.reset(torch.arange(num_envs, device=device))

    def _load_captured_state(self) -> None:
        """Read the captured robot / board / bolt / nut reset state."""
        reset_state = np.load(os.path.join(self.extras_dir, "reset_state.npy")).astype(np.float32)
        with np.load(os.path.join(self.extras_dir, "initial_observations.npz")) as data:
            success = data["success"].astype(np.float32)
        n = self.num_envs

        # reset_state row: root(7), rootvel(6), dof_q(9), dof_v(9).
        self._reset_dof_q = torch.tensor(reset_state[:n, 13:22], device=self.device)
        self._reset_dof_v = torch.tensor(reset_state[:n, 22:31], device=self.device)

        # success row: robot(31), board(13), bolt(13), nut(13), time(1); each asset
        # block is env-relative root_state pos(3) + quat xyzw(4) + linvel(3) + angvel(3).
        def asset_pose(col: int):
            pos = torch.tensor(success[:n, col : col + 3], device=self.device) + self.env_origins
            return pos, torch.tensor(success[:n, col + 3 : col + 7], device=self.device)

        self._reset_board_pos, self._reset_board_quat = asset_pose(31)
        self._reset_bolt_pos, self._reset_bolt_quat = asset_pose(44)
        self._reset_nut_pos, self._reset_nut_quat = asset_pose(57)

    def _load_policy(self, policy: str) -> None:
        path = os.path.join(self.extras_dir, f"policy_{policy}.pt")
        self.policy = torch.jit.load(path, map_location=self.device)
        self.policy.eval()

    def _body_pose(self, body: int):
        view = wp.to_torch(self.sim.state.body_q).view(self.num_envs, self.bodies_per_env, 7)
        return view[:, body, :3], view[:, body, 3:7]

    def _compute_terms(self) -> list[torch.Tensor]:
        """The six policy obs terms (held-first order)."""
        root_pos, root_quat = self._body_pose(ROBOT_ROOT_BODY)
        ee_pos, ee_quat = self._body_pose(EE_BODY)
        bolt_pos, bolt_quat = self._body_pose(BOLT_BODY)
        nut_pos, nut_quat = self._body_pose(NUT_BODY)

        tip = self.tip_offset.expand(self.num_envs, 3)
        fixed_tip_pos = bolt_pos + quat_apply(bolt_quat, tip)
        ee_tip_pos = ee_pos + quat_apply(ee_quat, tip)

        held_p, held_q = subtract_frame_transforms(fixed_tip_pos, bolt_quat, nut_pos, nut_quat)
        fixed_p, fixed_q = subtract_frame_transforms(ee_pos, ee_quat, fixed_tip_pos, bolt_quat)
        pose_p, pose_q = subtract_frame_transforms(root_pos, root_quat, ee_tip_pos, ee_quat)

        # ee_vel is zero in the captured env (Newton leaves body velocities unset).
        term_vel = torch.zeros(self.num_envs, TERM_SIZES[2], device=self.device)

        return [
            torch.cat([held_p, held_q], dim=-1),
            torch.cat([fixed_p, fixed_q], dim=-1),
            term_vel,
            torch.cat([pose_p, pose_q], dim=-1),
            self.joint_q[self._dof_q_idx].view(self.num_envs, NUM_DOFS),
            self.prev_action,
        ]

    def _build_observation(self) -> torch.Tensor:
        return torch.cat([h.reshape(self.num_envs, -1) for h in self.history], dim=-1)

    def act(self) -> None:
        """Run the policy on a fresh observation and store the joint targets."""
        terms = self._compute_terms()
        for buf, term in zip(self.history, terms):
            buf[:, :-1] = buf[:, 1:].clone()
            buf[:, -1] = term

        with torch.no_grad():
            action = self.policy(self._build_observation()).view(self.num_envs, ACTION_DIM)

        self._arm_delta = ARM_ACTION_SCALE * action[:, :NUM_ARM_JOINTS]
        self._finger_target = torch.where(
            action[:, 7:8] > 0.0,
            torch.full((self.num_envs, NUM_FINGER_JOINTS), GRIPPER_OPEN, device=self.device),
            torch.full((self.num_envs, NUM_FINGER_JOINTS), GRIPPER_CLOSE, device=self.device),
        )
        self.prev_action = action.clone()

    def apply_actuator(self) -> None:
        """Write the franka targets to the solver's implicit PD each physics step."""
        cur_q = self.joint_q[self._dof_q_idx].view(self.num_envs, NUM_DOFS)
        arm_target = cur_q[:, :NUM_ARM_JOINTS] + self._arm_delta
        targets = torch.cat([arm_target, self._finger_target], dim=-1)
        self.joint_target_pos[self._dof_d_idx] = targets.reshape(-1)
        self.joint_target_vel[self._dof_d_idx] = 0.0

    def _success(self) -> torch.Tensor:
        """progress_context success: nut threaded onto the bolt within tolerance."""
        bolt_pos, bolt_quat = self._body_pose(BOLT_BODY)
        nut_pos, nut_quat = self._body_pose(NUT_BODY)
        n = self.num_envs
        fixed_p, fixed_q = combine_frame_transforms(
            bolt_pos, bolt_quat, self._assembled_off.expand(n, 3), self._ident_quat.expand(n, 4)
        )
        held_p, held_q = combine_frame_transforms(
            nut_pos, nut_quat, self._held_off.expand(n, 3), self._ident_quat.expand(n, 4)
        )
        rel_p, rel_q = subtract_frame_transforms(fixed_p, fixed_q, held_p, held_q)
        roll, pitch, _ = euler_xyz_from_quat(rel_q)
        euler_xy = wrap_to_pi(roll).abs() + wrap_to_pi(pitch).abs()
        xy = torch.norm(rel_p[:, 0:2], dim=1)
        return (euler_xy < SUCCESS_EULER_TOL) & (xy < SUCCESS_XY_TOL) & (rel_p[:, 2] < SUCCESS_THRESHOLD)

    def _out_of_bound(self) -> torch.Tensor:
        """Held asset (nut) env-relative root position outside the AABB."""
        nut_pos, _ = self._body_pose(NUT_BODY)
        local = nut_pos - self.env_origins
        return ((local < self._oob_lo) | (local > self._oob_hi)).any(dim=1)

    def _abnormal_robot(self) -> torch.Tensor:
        """Joint-velocity watchdog: any franka joint speed over twice its limit (diverged solve)."""
        qd = self.joint_qd[self._dof_d_idx].view(self.num_envs, NUM_DOFS)
        return (qd.abs() > 2.0 * self._robot_vel_limit).any(dim=1)

    def forward(self):
        self.episode_length += 1
        self.terminated = self._success() | self._out_of_bound() | self._abnormal_robot()
        self.truncated = self.episode_length >= self.max_episode_length
        return self.terminated, self.truncated

    def reset_done(self) -> None:
        env_ids = torch.nonzero(self.terminated | self.truncated, as_tuple=False).flatten()
        if env_ids.numel() > 0:
            self.reset(env_ids)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Restore the captured robot / board / bolt / nut state for the given envs."""
        if env_ids.numel() == 0:
            return

        q_base = env_ids.view(-1, 1) * self.jc_per
        d_base = env_ids.view(-1, 1) * self.jd_per
        dof_q_idx = (q_base + self._dof_off).reshape(-1)
        dof_d_idx = (d_base + self._dof_off).reshape(-1)

        self.joint_q[dof_q_idx] = self._reset_dof_q[env_ids].reshape(-1)
        self.joint_qd[dof_d_idx] = self._reset_dof_v[env_ids].reshape(-1)
        self.joint_f[dof_d_idx] = 0.0

        for free_q, pos, quat in (
            (BOARD_FREE_Q, self._reset_board_pos, self._reset_board_quat),
            (BOLT_FREE_Q, self._reset_bolt_pos, self._reset_bolt_quat),
            (NUT_FREE_Q, self._reset_nut_pos, self._reset_nut_quat),
        ):
            idx = (q_base + (free_q + self._free_off)).reshape(-1)
            self.joint_q[idx] = torch.cat([pos[env_ids], quat[env_ids]], dim=-1).reshape(-1)

        newton.eval_fk(self.sim.model, self.sim.state.joint_q, self.sim.state.joint_qd, self.sim.state, None)

        self.prev_action[env_ids] = 0.0
        self._arm_delta[env_ids] = 0.0
        self._finger_target[env_ids] = self._reset_dof_q[env_ids, NUM_ARM_JOINTS:NUM_DOFS]
        self.episode_length[env_ids] = 0

        for buf, term in zip(self.history, self._compute_terms()):
            buf[env_ids] = term[env_ids].unsqueeze(1).expand(-1, HISTORY_LENGTH, -1)

    def log_visuals(self, viewer) -> None:
        return
