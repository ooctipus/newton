# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resolve per-environment entity indices from a multi-entity Newton model.

Isaac Lab's manager MDP addresses joints/bodies through ``SceneEntityCfg`` views on a single
articulation. A standalone repro instead works directly on the finalized :class:`newton.Model`,
which packs every environment's robot + object(s) + table into flat joint/body arrays. This
module recovers the per-environment index tensors a manipulation MDP needs:

* actuated robot joint coord/DOF indices, in a caller-provided canonical order (the captured
  Isaac Lab ``joint_names`` order) so observations match the trained policy;
* selected robot body indices (e.g. ``palm_link`` + fingertips), sorted by body index to mirror
  Isaac Lab's ``SceneEntityCfg`` resolution order;
* the object root-body index per environment.

It uses ``model.joint_q_start`` / ``model.joint_qd_start`` (per-joint coord/DOF starts) and the
``/World/envs/env_<e>/`` label prefix written by the cloner's label renaming.
"""

from __future__ import annotations

import re

import numpy as np
import torch

_ENV_RE = re.compile(r"/env_(\d+)/")


def _env_id(label: str) -> int | None:
    match = _ENV_RE.search(label)
    return int(match.group(1)) if match else None


class RobotEntities:
    """Per-environment joint/body index tables for a robot + object in a Newton model.

    Args:
        model: Finalized :class:`newton.Model`.
        num_envs: Number of environments (worlds) to resolve.
        actuated_joint_names: Actuated joint names in the canonical (policy) order, e.g.
            ``["iiwa7_joint_1", ..., "thumb_joint_3"]``.
        tip_body_patterns: Regexes (full-match against the leaf body name) selecting the bodies
            for the hand-tip state observation; resolved indices are sorted ascending.
        object_body_name: Leaf name of the object root body (default ``"Object"``).
        robot_segment: Path segment identifying robot joints/bodies (default ``"/Robot/"``).
        device: Torch device for the produced index tensors.
    """

    def __init__(
        self,
        model,
        num_envs: int,
        actuated_joint_names: list[str],
        tip_body_patterns: list[str],
        object_body_name: str = "Object",
        robot_root_body_name: str = "iiwa7_link_0",
        robot_segment: str = "/Robot/",
        device: str = "cuda:0",
    ) -> None:
        self.num_envs = int(num_envs)
        self.device = device
        self.num_dofs = len(actuated_joint_names)
        self.robot_root_body_name = robot_root_body_name

        joint_label = list(model.joint_label)
        body_label = list(model.body_label)
        q_start = np.asarray(model.joint_q_start.numpy())
        qd_start = np.asarray(model.joint_qd_start.numpy())
        joint_child = np.asarray(model.joint_child.numpy()) if getattr(model, "joint_child", None) is not None else None
        tip_res = [re.compile(p) for p in tip_body_patterns]

        # Per-env name -> joint index (robot joints only).
        joints_by_env: dict[int, dict[str, int]] = {e: {} for e in range(self.num_envs)}
        for j, label in enumerate(joint_label):
            env = _env_id(label)
            if env is None or env >= self.num_envs or robot_segment not in label:
                continue
            joints_by_env[env][label.rsplit("/", 1)[-1]] = j

        coord_idx = np.zeros((self.num_envs, self.num_dofs), dtype=np.int64)
        dof_idx = np.zeros((self.num_envs, self.num_dofs), dtype=np.int64)
        for env in range(self.num_envs):
            for k, name in enumerate(actuated_joint_names):
                j = joints_by_env[env].get(name)
                if j is None:
                    raise KeyError(f"Actuated joint {name!r} not found for env {env}.")
                coord_idx[env, k] = int(q_start[j])
                dof_idx[env, k] = int(qd_start[j])

        # Per-env tip/object/robot-root body indices.
        tips_by_env: dict[int, list[int]] = {e: [] for e in range(self.num_envs)}
        object_idx = np.full(self.num_envs, -1, dtype=np.int64)
        root_idx = np.full(self.num_envs, -1, dtype=np.int64)
        for b, label in enumerate(body_label):
            env = _env_id(label)
            if env is None or env >= self.num_envs:
                continue
            leaf = label.rsplit("/", 1)[-1]
            if robot_segment in label and any(rx.fullmatch(leaf) for rx in tip_res):
                tips_by_env[env].append(b)
            if robot_segment in label and leaf == robot_root_body_name:
                root_idx[env] = b
            elif leaf == object_body_name:
                object_idx[env] = b

        num_tips = len(tips_by_env[0])
        tip_idx = np.zeros((self.num_envs, num_tips), dtype=np.int64)
        for env in range(self.num_envs):
            sorted_tips = sorted(tips_by_env[env])
            if len(sorted_tips) != num_tips:
                raise ValueError(f"env {env} resolved {len(sorted_tips)} tip bodies, expected {num_tips}.")
            tip_idx[env] = sorted_tips
        if (object_idx < 0).any():
            raise KeyError(f"Object body {object_body_name!r} not found for some envs.")
        if (root_idx < 0).any():
            raise KeyError(f"Robot root body {robot_root_body_name!r} not found for some envs.")

        # Object free-joint coord/DOF start per env (the joint whose child is the object body).
        # ``coord`` indexes ``joint_q`` (7 = pos3 + quat4); ``dof`` indexes ``joint_qd``
        # (6 = linvel3 + angvel3) and is needed to zero the object velocity on reset.
        object_coord_start = np.full(self.num_envs, -1, dtype=np.int64)
        object_dof_start = np.full(self.num_envs, -1, dtype=np.int64)
        if joint_child is not None:
            child_to_joint = {int(c): j for j, c in enumerate(joint_child)}
            for env in range(self.num_envs):
                j = child_to_joint.get(int(object_idx[env]))
                if j is not None:
                    object_coord_start[env] = int(q_start[j])
                    object_dof_start[env] = int(qd_start[j])

        self.num_tips = num_tips
        self.joint_coord_idx = torch.as_tensor(coord_idx, dtype=torch.long, device=device)
        self.joint_dof_idx = torch.as_tensor(dof_idx, dtype=torch.long, device=device)
        self.tip_body_idx = torch.as_tensor(tip_idx, dtype=torch.long, device=device)
        self.object_body_idx = torch.as_tensor(object_idx, dtype=torch.long, device=device)
        self.robot_root_body_idx = torch.as_tensor(root_idx, dtype=torch.long, device=device)
        self.object_coord_start = torch.as_tensor(object_coord_start, dtype=torch.long, device=device)
        self.object_dof_start = torch.as_tensor(object_dof_start, dtype=torch.long, device=device)
        # Flat views for scatter/gather into 1-D state arrays.
        self._coord_flat = self.joint_coord_idx.reshape(-1)
        self._dof_flat = self.joint_dof_idx.reshape(-1)

    def joint_pos(self, joint_q: torch.Tensor) -> torch.Tensor:
        """Gather actuated joint positions ``(num_envs, num_dofs)`` from flat ``joint_q`` [rad]."""
        return joint_q[self._coord_flat].view(self.num_envs, self.num_dofs)

    def joint_vel(self, joint_qd: torch.Tensor) -> torch.Tensor:
        """Gather actuated joint velocities ``(num_envs, num_dofs)`` from flat ``joint_qd`` [rad/s]."""
        return joint_qd[self._dof_flat].view(self.num_envs, self.num_dofs)
