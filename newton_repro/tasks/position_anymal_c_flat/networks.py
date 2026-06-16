# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Torch network boundaries for the standalone Position Anymal-C Newton repro MDP."""

from __future__ import annotations

import torch
import torch.nn as nn
import warp as wp

ACTION_SCALE = 0.2
EFFORT_LIMIT = 80.0
SATURATION_EFFORT = 120.0
VELOCITY_LIMIT = 7.5


class CNNEncoderPolicy(nn.Module):
    """Torch policy boundary for Warp-owned MDP buffers."""

    def __init__(
        self,
        num_envs: int,
        num_dofs: int,
        device: str,
        default_joint_pos_wp,
        last_action_wp,
        targets_wp,
        actor_state_dict: dict[str, torch.Tensor] | None = None,
        actor: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.num_envs = num_envs
        self.num_dofs = num_dofs
        self.device = device
        self.default_joint_pos = wp.to_torch(default_joint_pos_wp)
        self.last_action = wp.to_torch(last_action_wp)
        self.targets = wp.to_torch(targets_wp)

        if actor_state_dict is not None:
            self.obs_mean = actor_state_dict["obs_normalizer._mean"].to(device=device)
            self.obs_std = actor_state_dict["obs_normalizer._std"].to(device=device)
            self.height_mean = actor_state_dict["encoder_normalizers.height_scan._mean"].to(device=device)
            self.height_std = actor_state_dict["encoder_normalizers.height_scan._std"].to(device=device)

            self.height_encoder = nn.Sequential(
                nn.Linear(416, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 64)
            ).to(device=device)
            self.head_norm = nn.LayerNorm(121).to(device=device)
            self.mlp = nn.Sequential(
                nn.Linear(121, 256),
                nn.ELU(),
                nn.Linear(256, 256),
                nn.ELU(),
                nn.Linear(256, 128),
                nn.ELU(),
                nn.Linear(128, num_dofs),
            ).to(device=device)

            self._copy_linear(actor_state_dict, "encoders.height_scan", self.height_encoder)
            self._copy_linear(actor_state_dict, "mlp", self.mlp)
            self.head_norm.weight.data.copy_(actor_state_dict["head_norm.weight"].to(device=device))
            self.head_norm.bias.data.copy_(actor_state_dict["head_norm.bias"].to(device=device))

        self.eval()

    @classmethod
    def from_path(
        cls,
        policy_path: str,
        num_envs: int,
        num_dofs: int,
        device: str,
        default_joint_pos_wp,
        last_action_wp,
        targets_wp,
    ) -> CNNEncoderPolicy:
        """Load a JIT actor or reconstruct the checkpoint actor."""
        actor = None
        actor_state_dict = None
        try:
            actor = torch.jit.load(policy_path, map_location=device).eval()
        except RuntimeError:
            checkpoint = torch.load(policy_path, map_location=device, weights_only=False)
            if not isinstance(checkpoint, dict) or "actor_state_dict" not in checkpoint:
                raise RuntimeError(f"Unsupported policy artifact: {policy_path}")
            actor_state_dict = checkpoint["actor_state_dict"]
        return cls(
            num_envs,
            num_dofs,
            device,
            default_joint_pos_wp,
            last_action_wp,
            targets_wp,
            actor_state_dict=actor_state_dict,
            actor=actor,
        )

    @staticmethod
    def _copy_linear(state_dict: dict[str, torch.Tensor], prefix: str, module: nn.Sequential) -> None:
        for layer_idx, layer in enumerate(module):
            if not isinstance(layer, nn.Linear):
                continue
            layer.weight.data.copy_(state_dict[f"{prefix}.{layer_idx}.weight"].to(device=layer.weight.device))
            layer.bias.data.copy_(state_dict[f"{prefix}.{layer_idx}.bias"].to(device=layer.bias.device))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if self.actor is not None:
            return self.actor(observations)
        passthrough = observations[:, :57]
        height_scan = observations[:, 57:]
        passthrough = (passthrough - self.obs_mean) / (self.obs_std + 1.0e-2)
        height_scan = (height_scan - self.height_mean) / (self.height_std + 1.0e-2)
        height_latent = self.height_encoder(height_scan)
        latent = self.head_norm(torch.cat((passthrough, height_latent), dim=-1))
        return self.mlp(latent)

    def act(self, observations_wp) -> None:
        observations = wp.to_torch(observations_wp)
        with torch.inference_mode():
            actions = self(observations)
        self.last_action[:] = actions[:, : self.num_dofs]
        self.targets[:] = self.default_joint_pos + ACTION_SCALE * self.last_action

    def reset(self, reset_ids_wp, count: int) -> None:
        if count == 0 or self.actor is None or not hasattr(self.actor, "reset"):
            return
        reset_ids = wp.to_torch(reset_ids_wp)[:count].to(dtype=torch.long)
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        dones[reset_ids] = True
        self.actor.reset(dones)


class ActuatorNetLSTM:
    """Torch LSTM actuator boundary for Warp-owned Newton buffers."""

    def __init__(
        self,
        path: str,
        num_envs: int,
        num_dofs: int,
        jc_per_env: int,
        jd_per_env: int,
        device: str,
        joint_q_wp,
        joint_qd_wp,
        control_f_wp,
        targets_wp,
    ) -> None:
        self.num_envs = num_envs
        self.num_dofs = num_dofs
        self.device = device
        self.joint_q = wp.to_torch(joint_q_wp).view(num_envs, jc_per_env)
        self.joint_qd = wp.to_torch(joint_qd_wp).view(num_envs, jd_per_env)
        self.control_f = wp.to_torch(control_f_wp).view(num_envs, jd_per_env)
        self.targets = wp.to_torch(targets_wp)

        self.model = torch.jit.load(path, map_location=device).eval()
        state_dict = self.model.lstm.state_dict()
        num_layers = len(state_dict) // 4
        hidden_dim = state_dict["weight_hh_l0"].shape[1]
        self.hidden = torch.zeros(num_layers, num_envs * num_dofs, hidden_dim, device=device)
        self.cell = torch.zeros_like(self.hidden)

    def apply(self) -> None:
        joint_vel = self.joint_qd[:, 6:]
        sea_input = torch.stack(((self.targets - self.joint_q[:, 7:]).flatten(), joint_vel.flatten()), dim=-1)
        torques, (self.hidden[:], self.cell[:]) = self.model(sea_input.unsqueeze(1), (self.hidden, self.cell))
        torques = self._clip_effort(torques.reshape(self.num_envs, self.num_dofs), joint_vel)
        self.control_f[:, :6] = 0.0
        self.control_f[:, 6:] = torques

    def reset(self, reset_ids_wp, count: int) -> None:
        if count == 0:
            return
        env_ids = wp.to_torch(reset_ids_wp)[:count].to(dtype=torch.long)
        offsets = (env_ids.unsqueeze(1) * self.num_dofs + torch.arange(self.num_dofs, device=self.device)).flatten()
        self.hidden[:, offsets] = 0.0
        self.cell[:, offsets] = 0.0

    @staticmethod
    def _clip_effort(effort: torch.Tensor, joint_vel: torch.Tensor) -> torch.Tensor:
        vel_at_effort_limit = VELOCITY_LIMIT * (1.0 + EFFORT_LIMIT / SATURATION_EFFORT)
        joint_vel = joint_vel.clamp(min=-vel_at_effort_limit, max=vel_at_effort_limit)
        torque_speed_top = SATURATION_EFFORT * (1.0 - joint_vel / VELOCITY_LIMIT)
        torque_speed_bottom = SATURATION_EFFORT * (-1.0 - joint_vel / VELOCITY_LIMIT)
        max_effort = torch.clamp(torque_speed_top, max=EFFORT_LIMIT)
        min_effort = torch.clamp(torque_speed_bottom, min=-EFFORT_LIMIT)
        return torch.clamp(effort, min=min_effort, max=max_effort)
