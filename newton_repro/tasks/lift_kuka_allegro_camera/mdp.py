# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone MDP for the captured ``Isaac-Lift-KukaAllegro-Camera`` Newton bundle.

Thin orchestration over the reusable ``envs`` building blocks:

* :class:`~envs.newton_entities.RobotEntities` -- per-env joint/body index tables.
* :class:`~envs.mdp.actions.ImplicitJointActuator` -- implicit PD + relative-position action.
* :class:`~envs.sensors.newton_camera.NewtonCamera` -- 64x64 base depth image.
* :class:`~envs.sensors.contact_force.FingerContactForce` -- fingertip object-contact forces.
* :class:`~envs.mdp.commands.ObjectUniformPoseCommand` -- target object pose in the robot frame.
* :class:`~envs.policy.JitPolicy` -- exported rsl_rl CNN actor.

Observation layout (no history; matches the captured initial observations):

* ``policy`` (34): ``object_quat_b``(4) + ``target_object_pose_b``(7) + ``last_action``(23)
* ``proprio`` (123): ``contact``(12) + ``joint_pos``(23, absolute) + ``joint_vel``(23) +
  ``hand_tips_state_b``(65). The hand-tip body velocities read back as zero on the live Newton
  backend, so they are zeroed here too.
* ``obs_1d`` = ``cat(policy, proprio)`` (157), fed to the actor with the ``(1,64,64)`` depth image.
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch
import warp as wp
from envs.mdp import observations as obs_utils
from envs.mdp.actions import ImplicitJointActuator
from envs.mdp.commands import ObjectUniformPoseCommand, PoseCommandRanges
from envs.mdp.terminations import abnormal_robot_state, object_out_of_bound, time_out
from envs.newton_entities import RobotEntities
from envs.policy import JitPolicy
from envs.sensors.contact_force import FingerContactForce
from envs.sensors.newton_camera import NewtonCamera
from newton import eval_fk

# Actuated robot joints in the policy (Isaac Lab articulation) order.
ACTUATED_JOINTS = [
    "iiwa7_joint_1", "iiwa7_joint_2", "iiwa7_joint_3", "iiwa7_joint_4",
    "iiwa7_joint_5", "iiwa7_joint_6", "iiwa7_joint_7",
    "index_joint_0", "index_joint_1", "index_joint_2", "index_joint_3",
    "middle_joint_0", "middle_joint_1", "middle_joint_2", "middle_joint_3",
    "ring_joint_0", "ring_joint_1", "ring_joint_2", "ring_joint_3",
    "thumb_joint_0", "thumb_joint_1", "thumb_joint_2", "thumb_joint_3",
]  # fmt: skip
NUM_DOFS = 23
TIP_BODY_PATTERNS = [r"palm_link", r".*_biotac_tip"]  # regex (resolved by RobotEntities)
# Glob patterns (Newton ``match_labels`` uses ``fnmatch``, not regex).
FINGER_CONTACT_EXPR = ["*/index_link_3", "*/middle_link_3", "*/ring_link_3", "*/thumb_link_3"]
OBJECT_BODY_EXPR = "*/Object"
ACTION_SCALE = 0.1

# Command sampling (dexsuite lift, robot root frame); orientation is sampled but lift ignores it.
CMD_RANGES = PoseCommandRanges(
    pos_x=(-0.7, -0.3), pos_y=(-0.25, 0.25), pos_z=(0.55, 0.95), roll=(-3.14, 3.14), pitch=(-3.14, 3.14), yaw=(0.0, 0.0)
)
RESAMPLE_TIME_RANGE = (2.0, 3.0)

# Reset randomization (dexsuite events): robot joints +/-0.5 rad, wrist joint7 +/-3 rad,
# object position +/-0.2 m in xy and +[0, 0.4] m in z, random orientation.
JOINT_RESET_OFFSET = 0.5
WRIST_JOINT_NAME = "iiwa7_joint_7"
WRIST_RESET_OFFSET = 3.0
OBJ_RESET_POS_XY = 0.2
OBJ_RESET_POS_Z = (0.0, 0.4)

# Debug: when True, print the offending joint speed for envs that reset due to ``abnormal_robot_state``
# (a joint exceeding 2x its rated velocity limit). Set to False to mute.
REPORT_RESET_JOINT_VEL: bool = True


class MDP:
    """Lift-KukaAllegro-Camera task logic for a Newton-only replay."""

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
    ) -> None:
        self.sim = sim
        self.num_envs = int(num_envs)
        self.device = device
        self.decimation = int(decimation)
        self.step_dt = float(physics_dt) * self.decimation
        self.max_episode_length = max(1, int(math.ceil(episode_length_s / self.step_dt)))
        self.env_origins = torch.as_tensor(env_origins, dtype=torch.float32, device=device)[: self.num_envs]
        self.extras_dir = extras_dir

        self.entities = RobotEntities(
            sim.model,
            num_envs=self.num_envs,
            actuated_joint_names=ACTUATED_JOINTS,
            tip_body_patterns=TIP_BODY_PATTERNS,
            device=device,
        )
        self._load_extras()
        self._build_actuator()
        self.camera = self._build_camera()
        self.contact = FingerContactForce(
            sim,
            num_envs=self.num_envs,
            num_fingers=len(FINGER_CONTACT_EXPR),
            sensing_body_expr=FINGER_CONTACT_EXPR,
            counterpart_body_expr=OBJECT_BODY_EXPR,
            device=device,
        )
        self.command = ObjectUniformPoseCommand(
            self.num_envs, CMD_RANGES, RESAMPLE_TIME_RANGE, self.step_dt, make_quat_unique=True, device=device
        )
        self.policy = JitPolicy(os.path.join(extras_dir, "policy.pt"), device)

        self.last_action = torch.zeros(self.num_envs, NUM_DOFS, device=device)
        self.episode_length = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self._joint_q = wp.to_torch(sim.state.joint_q)
        self._joint_qd = wp.to_torch(sim.state.joint_qd)

        self._all_env_ids = torch.arange(self.num_envs, dtype=torch.long, device=device)
        self.reset(self._all_env_ids, to_captured=True)

    # ------------------------------------------------------------------ setup
    def _load_extras(self) -> None:
        with np.load(os.path.join(self.extras_dir, "robot_state.npz"), allow_pickle=True) as data:
            self._default_joint_pos = torch.as_tensor(
                data["default_joint_pos"][: self.num_envs], dtype=torch.float32, device=self.device
            )
            vel_limits = np.asarray(data["joint_vel_limits"])
            self._joint_vel_limits = torch.as_tensor(
                vel_limits[: self.num_envs], dtype=torch.float32, device=self.device
            )
        with np.load(os.path.join(self.extras_dir, "object_state.npz"), allow_pickle=True) as data:
            self._object_default_pose = torch.as_tensor(
                data["object__default_root_pose"][: self.num_envs], dtype=torch.float32, device=self.device
            )
            self._object_captured_state = torch.as_tensor(
                data["object__root_state"][: self.num_envs], dtype=torch.float32, device=self.device
            )
        self._actuator_state = np.load(os.path.join(self.extras_dir, "actuator_state.npz"), allow_pickle=True)
        self._camera_state = np.load(os.path.join(self.extras_dir, "camera_state.npz"), allow_pickle=True)
        self._wrist_col = ACTUATED_JOINTS.index(WRIST_JOINT_NAME)

    def _build_actuator(self) -> None:
        dof0 = self.entities.joint_dof_idx[0].cpu().numpy()
        ke = torch.as_tensor(self._actuator_state["joint_target_ke"][dof0], dtype=torch.float32)
        kd = torch.as_tensor(self._actuator_state["joint_target_kd"][dof0], dtype=torch.float32)
        armature = None
        if "joint_armature" in self._actuator_state.files:
            armature = torch.as_tensor(self._actuator_state["joint_armature"][dof0], dtype=torch.float32)
        effort_limit = None
        if "joint_effort_limit" in self._actuator_state.files:
            effort_limit = torch.as_tensor(self._actuator_state["joint_effort_limit"][dof0], dtype=torch.float32)
        self.actuator = ImplicitJointActuator(
            self.sim,
            self.entities,
            target_ke=ke,
            target_kd=kd,
            armature=armature,
            effort_limit=effort_limit,
            scale=ACTION_SCALE,
            device=self.device,
        )

    def _build_camera(self) -> NewtonCamera:
        cs = self._camera_state
        prefix = "base_camera__"
        return NewtonCamera(
            self.sim.model,
            positions_w=cs[prefix + "pos_w"][: self.num_envs],
            quats_w_world=cs[prefix + "quat_w_world"][: self.num_envs],
            intrinsics=cs[prefix + "intrinsic_matrices"][: self.num_envs],
            width=int(cs[prefix + "width"]),
            height=int(cs[prefix + "height"]),
            device=self.device,
        )

    # ------------------------------------------------------------- observation
    def _state_tensors(self):
        return (
            self._joint_q,
            self._joint_qd,
            wp.to_torch(self.sim.state.body_q),
            wp.to_torch(self.sim.state.body_qd),
        )

    def get_observations(self):
        joint_q, joint_qd, body_q, body_qd = self._state_tensors()
        # dexsuite proprio uses absolute joint positions (``mdp.joint_pos``), not relative-to-default.
        joint_pos = self.entities.joint_pos(joint_q)
        joint_vel = self.entities.joint_vel(joint_qd)
        root_pos_w, root_quat_w = obs_utils.body_pose_w(body_q, self.entities.robot_root_body_idx)
        _, object_quat_w = obs_utils.body_pose_w(body_q, self.entities.object_body_idx)

        object_quat = obs_utils.object_quat_b(root_quat_w, object_quat_w)
        target_pose = self.command.command
        tips = obs_utils.body_state_b(
            body_q, body_qd, self.entities.tip_body_idx, root_pos_w, root_quat_w, zero_velocity=True
        )
        contact = self.contact.forces_b(root_quat_w)

        # Per-term observation clips mirror the live env's ``ObsTerm(clip=...)`` saturation. These
        # bound fast-motion values (tip velocities routinely exceed 2 m/s, rad/s) so the policy
        # never sees out-of-distribution magnitudes that would otherwise drive runaway actions.
        tips = tips.clamp(-2.0, 2.0)
        contact = contact.clamp(-20.0, 20.0)

        # Group/term order must match the live actor obs exactly: the ``policy`` group, then the
        # ``proprio`` group laid out as ``[contact, joint_pos, joint_vel, hand_tips]``.
        policy_grp = torch.cat((object_quat, target_pose, self.last_action), dim=-1)
        proprio_grp = torch.cat((contact, joint_pos, joint_vel, tips), dim=-1)
        obs_1d = torch.cat((policy_grp, proprio_grp), dim=-1)
        depth = self.camera.depth_normalized(self.sim.state)
        return obs_1d, depth

    # --------------------------------------------------------------- stepping
    def act(self) -> None:
        obs_1d, depth = self.get_observations()
        action = self.policy.act(obs_1d, [depth])
        self.last_action = action.detach().clone().to(self.device)
        self.actuator.set_action(self.last_action)

    def apply_actuator(self) -> None:
        self.actuator.apply()

    def forward(self):
        self.command.step()
        self.episode_length += 1
        joint_vel = self.entities.joint_vel(self._joint_qd)
        body_q = wp.to_torch(self.sim.state.body_q)
        object_pos_w, _ = obs_utils.body_pose_w(body_q, self.entities.object_body_idx)
        abnormal = abnormal_robot_state(joint_vel, self._joint_vel_limits)
        self._report_reset_joint_vel(joint_vel, abnormal)
        terminated = abnormal | object_out_of_bound(object_pos_w, self.env_origins)
        truncated = time_out(self.episode_length, self.max_episode_length)
        self._terminated = terminated
        self._truncated = truncated
        return terminated, truncated

    def _report_reset_joint_vel(self, joint_vel: torch.Tensor, abnormal: torch.Tensor) -> None:
        """Print the offending joint speed for envs that reset due to ``abnormal_robot_state``.

        Only fires for envs whose fastest joint exceeded 2x its rated velocity limit (the reset
        condition), reporting the joint name, its speed [rad/s], and that reset threshold.
        """
        if not REPORT_RESET_JOINT_VEL or not bool(abnormal.any()):
            return
        abs_vel = joint_vel.abs()
        vmax, jdof = abs_vel.max(dim=1)
        lim = torch.gather(self._joint_vel_limits, 1, jdof.unsqueeze(1)).squeeze(1)
        for e in abnormal.nonzero(as_tuple=False).flatten().tolist():
            print(
                f"[mdp] env{e} reset: abnormal joint vel {ACTUATED_JOINTS[int(jdof[e])]}="
                f"{float(vmax[e]):.2f} rad/s (reset threshold {2.0 * float(lim[e]):.2f})"
            )

    def reset_done(self) -> None:
        done = self._terminated | self._truncated
        env_ids = done.nonzero(as_tuple=False).flatten()
        if env_ids.numel() > 0:
            self.reset(env_ids, to_captured=False)

    # ----------------------------------------------------------------- resets
    def reset(self, env_ids: torch.Tensor, to_captured: bool = False) -> None:
        if env_ids.numel() == 0:
            return
        if to_captured:
            # The policy's reset observation corresponds to joints at the default pose
            # (relative joint-pos obs == 0), so reset to default rather than to the later
            # offset state recorded in robot_state.
            joint_targets = self._default_joint_pos[env_ids]
            object_pose = self._object_captured_state[env_ids, :7]
        else:
            joint_targets = self._sample_joint_pos(env_ids)
            object_pose = self._sample_object_pose(env_ids)

        coord_idx = self.entities.joint_coord_idx[env_ids]
        self._joint_q[coord_idx.reshape(-1)] = joint_targets.reshape(-1)
        self._joint_qd[self.entities.joint_dof_idx[env_ids].reshape(-1)] = 0.0
        self._write_object_pose(env_ids, object_pose)
        # Zero the object's free-joint velocity (6 DOFs) so a reset object does not retain the
        # velocity it had while diverging. Mirrors the live ``reset_object`` event's zero
        # ``velocity_range``; without it a tumbling object survives the reset and re-diverges.
        obj_dof = self.entities.object_dof_start[env_ids]
        obj_dof = obj_dof[obj_dof >= 0]
        if obj_dof.numel() > 0:
            obj_vel_idx = obj_dof.unsqueeze(1) + torch.arange(6, device=self.device)
            self._joint_qd[obj_vel_idx.reshape(-1)] = 0.0

        eval_fk(self.sim.model, self.sim.state.joint_q, self.sim.state.joint_qd, self.sim.state, None)
        self.last_action[env_ids] = 0.0
        self.episode_length[env_ids] = 0
        self.command.reset(env_ids)
        self.policy.reset()

    def _sample_joint_pos(self, env_ids: torch.Tensor) -> torch.Tensor:
        n = env_ids.numel()
        base = self._default_joint_pos[env_ids]
        offset = torch.empty(n, NUM_DOFS, device=self.device).uniform_(-JOINT_RESET_OFFSET, JOINT_RESET_OFFSET)
        offset[:, self._wrist_col] = torch.empty(n, device=self.device).uniform_(
            -WRIST_RESET_OFFSET, WRIST_RESET_OFFSET
        )
        return base + offset

    def _sample_object_pose(self, env_ids: torch.Tensor) -> torch.Tensor:
        n = env_ids.numel()
        pose = self._object_default_pose[env_ids].clone()  # [pos(3) env-local, quat_xyzw(4)]
        # ``default_root_pose`` is env-local; place it in the world by adding the env origin.
        pose[:, :3] = pose[:, :3] + self.env_origins[env_ids]
        pose[:, 0] += torch.empty(n, device=self.device).uniform_(-OBJ_RESET_POS_XY, OBJ_RESET_POS_XY)
        pose[:, 1] += torch.empty(n, device=self.device).uniform_(-OBJ_RESET_POS_XY, OBJ_RESET_POS_XY)
        pose[:, 2] += torch.empty(n, device=self.device).uniform_(*OBJ_RESET_POS_Z)
        rand_quat = torch.randn(n, 4, device=self.device)
        pose[:, 3:] = rand_quat / rand_quat.norm(dim=-1, keepdim=True)
        return pose

    def _write_object_pose(self, env_ids: torch.Tensor, pose: torch.Tensor) -> None:
        """Write per-env object world pose ``[pos(3), quat_xyzw(4)]`` into the free-joint coords.

        The captured object pose and Newton free-joint coords both use the xyzw quaternion layout,
        so the 7-vector is written directly.
        """
        start = self.entities.object_coord_start[env_ids]
        if (start < 0).any():
            return
        idx = (start.unsqueeze(1) + torch.arange(7, device=self.device).unsqueeze(0)).reshape(-1)
        self._joint_q[idx] = pose[:, :7].reshape(-1)

    def log_visuals(self, viewer) -> None:
        """No extra debug visuals for the headless repro (depth obs is produced in :meth:`act`)."""
        return
