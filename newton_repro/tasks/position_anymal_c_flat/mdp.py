# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone MDP for the captured Isaac-Position-Anymal-C-v0 Newton bundle."""

from __future__ import annotations

import math
import os

import newton
import numpy as np
import warp as wp
from envs.mdp.events import find_body_indices, randomize_rigid_body_mass, randomize_rigid_body_material
from envs.sensors.height_scan import NewtonHeightScanner
from newton.solvers import SolverNotifyFlags
from tasks.position_anymal_c_flat.kernels import (
    _build_goal_arrow_instances_kernel,
    _build_observations_kernel,
    _clear_counter_kernel,
    _collect_reset_ids_kernel,
    _reset_envs_kernel,
    _set_joint_dof_property_kernel,
    _update_command_kernel,
    _update_task_outputs_kernel,
)
from tasks.position_anymal_c_flat.networks import ActuatorNetLSTM, CNNEncoderPolicy

NUM_DOFS = 12
ARMATURE = 0.001
# Mirrors ``EventsCfg`` in
# ``source/isaaclab_tasks/.../locomotion/position/position_env_cfg.py``.
# These are ``mode="startup"`` randomizations; the bundle stops capture at the
# cloner hook (before EventManager runs), so we replay them here.
RANDOMIZATION_SEED = 42
STATIC_FRICTION_RANGE = (1.0, 1.5)
RESTITUTION_RANGE = (0.0, 0.0)
BASE_MASS_RANGE = (-5.0, 5.0)


class MDP:
    """Position Anymal-C task logic for a Newton-only replay."""

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
        self.env_origins = env_origins
        self.num_envs = num_envs
        self.physics_dt = physics_dt
        self.decimation = decimation
        self.episode_length_s = episode_length_s
        self.device = device
        self.extras_dir = extras_dir

        jc_starts = self.sim.model.joint_coord_world_start.numpy()
        jd_starts = self.sim.model.joint_dof_world_start.numpy()
        self.jc_per = int(jc_starts[1]) - int(jc_starts[0])
        self.jd_per = int(jd_starts[1]) - int(jd_starts[0])
        self.num_dofs = self.jc_per - 7
        if self.num_dofs != NUM_DOFS:
            raise ValueError(f"Position Anymal-C expects {NUM_DOFS} joints, got {self.num_dofs}.")

        self.step_dt = self.physics_dt * self.decimation
        self.max_episode_length = max(1, int(math.ceil(self.episode_length_s / self.step_dt)))
        self._last_action_wp = wp.zeros((self.num_envs, self.num_dofs), dtype=wp.float32, device=self.device)
        self._targets_wp = wp.zeros((self.num_envs, self.num_dofs), dtype=wp.float32, device=self.device)
        self._goal_arrow_mesh_viewer_ids: set[int] = set()
        self._goal_arrow_xforms_wp = wp.empty(self.num_envs, dtype=wp.transformf, device=self.device)
        self._goal_arrow_scales_wp = wp.empty(self.num_envs, dtype=wp.vec3, device=self.device)
        self._goal_arrow_colors_wp = wp.empty(self.num_envs, dtype=wp.vec3, device=self.device)
        self._goal_arrow_materials_wp = wp.empty(self.num_envs, dtype=wp.vec4, device=self.device)

        self._load_captured_buffers()
        self.height_scanner = NewtonHeightScanner(
            self.sim.model,
            num_envs=self.num_envs,
            jc_per_env=self.jc_per,
            device=self.device,
            expected_num_rays=self.num_height_rays,
        )
        self._configure_actuator_model()
        self._apply_startup_randomization()
        self._load_policy()
        self._load_lstm()

        self.reset(
            self._all_env_ids_wp, source_ids=self._all_source_ids_wp, count_wp=self._all_count_wp, count=self.num_envs
        )

    def _apply_startup_randomization(self) -> None:
        """Replay the ``mode="startup"`` randomization EventTerms from
        ``position_env_cfg.EventsCfg``: per-shape friction over all bodies, and
        an additive perturbation to the ``base`` body mass.

        The seed is hard-coded here for run-to-run reproducibility of the
        replayed bundle; pass a different ``RANDOMIZATION_SEED`` to explore
        other random draws.
        """
        randomize_rigid_body_material(
            self.sim,
            static_friction_range=STATIC_FRICTION_RANGE,
            restitution_range=RESTITUTION_RANGE,
            seed=RANDOMIZATION_SEED,
        )
        randomize_rigid_body_mass(
            self.sim,
            body_indices=find_body_indices(self.sim.model, r".*/base"),
            mass_distribution_params=BASE_MASS_RANGE,
            operation="add",
            seed=RANDOMIZATION_SEED + 1,
        )

    def _load_captured_buffers(self) -> None:
        reset_states = np.load(os.path.join(self.extras_dir, "reset_state.npy")).astype(np.float32, copy=False)
        self.reset_state_count = int(reset_states.shape[0])
        self._reset_states_wp = wp.array(reset_states, dtype=wp.float32, device=self.device)

        with np.load(os.path.join(self.extras_dir, "robot_state.npz")) as data:
            default_joint_pos = data["default_joint_pos"][: self.num_envs].astype(np.float32, copy=True)
        self._default_joint_pos_wp = wp.array(default_joint_pos, dtype=wp.float32, device=self.device)

        with np.load(os.path.join(self.extras_dir, "command_state.npz")) as data:
            cmd_buf = data["cmd_buf"].astype(np.float32, copy=False)
            cmd_mask = data["cmd_mask"].astype(np.bool_, copy=False)
            terrain_env_origins = data["terrain_env_origins"].astype(np.float32, copy=False)
        self.cmd_count = int(cmd_buf.shape[0])
        self.terrain_count = int(terrain_env_origins.shape[0])
        self._cmd_buf_table_wp = wp.array(cmd_buf, dtype=wp.float32, device=self.device)
        self._cmd_mask_table_wp = wp.array(cmd_mask, dtype=wp.bool, device=self.device)
        self._terrain_env_origins_table_wp = wp.array(terrain_env_origins, dtype=wp.float32, device=self.device)

        with np.load(os.path.join(self.extras_dir, "initial_observations.npz")) as data:
            if "height_scan" in data.files:
                height_scan = (
                    data["height_scan"].reshape(data["height_scan"].shape[0], -1).astype(np.float32, copy=False)
                )
            else:
                policy_obs = data["policy"].reshape(data["policy"].shape[0], -1).astype(np.float32, copy=False)
                if policy_obs.shape[1] <= 45:
                    raise KeyError(
                        "initial_observations.npz has no height_scan key and policy observations do not "
                        "contain the expected 416 height rays after the first 45 entries."
                    )
                height_scan = policy_obs[:, 45:]
        self.height_count = int(height_scan.shape[0])
        self.num_height_rays = int(height_scan.shape[1])
        self._height_scan_table_wp = wp.array(height_scan, dtype=wp.float32, device=self.device)

        self._cmd_buf_wp = wp.zeros((self.num_envs, 3, 13), dtype=wp.float32, device=self.device)
        self._cmd_mask_wp = wp.zeros((self.num_envs, 12), dtype=wp.bool, device=self.device)
        self._observations_wp = wp.empty(
            (self.num_envs, 57 + self.num_height_rays), dtype=wp.float32, device=self.device
        )
        self._episode_length_wp = wp.zeros(self.num_envs, dtype=wp.int32, device=self.device)
        self._reward_wp = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self._terminated_wp = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
        self._truncated_wp = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
        self._terrain_env_origins_wp = wp.zeros((self.num_envs, 3), dtype=wp.float32, device=self.device)
        self.command_error = wp.zeros(shape=(self.num_envs, 4), dtype=wp.float32, device=self.device)
        self.reward_scales = wp.array([0.5, 0.5, 0.3, 0.3], dtype=wp.float32, device=self.device)

        all_env_ids = np.arange(self.num_envs, dtype=np.int32)
        all_source_ids = all_env_ids % self.reset_state_count
        self._all_env_ids_wp = wp.array(all_env_ids, dtype=wp.int32, device=self.device)
        self._all_source_ids_wp = wp.array(all_source_ids, dtype=wp.int32, device=self.device)
        self._all_count_wp = wp.array(np.asarray([self.num_envs], dtype=np.int32), dtype=wp.int32, device=self.device)
        self._reset_ids_wp = wp.empty(self.num_envs, dtype=wp.int32, device=self.device)
        self._reset_source_ids_wp = wp.empty(self.num_envs, dtype=wp.int32, device=self.device)
        self._reset_count_wp = wp.zeros(1, dtype=wp.int32, device=self.device)

    def _load_policy(self) -> None:
        self.policy = CNNEncoderPolicy.from_path(
            os.path.join(self.extras_dir, "policy.pt"),
            self.num_envs,
            self.num_dofs,
            self.device,
            self._default_joint_pos_wp,
            self._last_action_wp,
            self._targets_wp,
        )

    def _load_lstm(self) -> None:
        self.actuator = ActuatorNetLSTM(
            os.path.join(self.extras_dir, "anydrive_3_lstm_jit.pt"),
            self.num_envs,
            self.num_dofs,
            self.jc_per,
            self.jd_per,
            self.device,
            self.sim.state.joint_q,
            self.sim.state.joint_qd,
            self.sim.control.joint_f,
            self._targets_wp,
        )

    def _configure_actuator_model(self) -> None:
        for name, value in (
            ("joint_armature", ARMATURE),
            ("joint_target_ke", 0.0),
            ("joint_target_kd", 0.0),
            ("joint_friction", 0.0),
            ("joint_effort_limit", 1.0e9),
        ):
            array = getattr(self.sim.model, name, None)
            if array is None:
                continue
            wp.launch(
                _set_joint_dof_property_kernel,
                dim=(self.num_envs, self.num_dofs),
                inputs=[array, wp.int32(self.jd_per), wp.float32(value)],
                device=self.device,
            )
        self.sim.notify_model_changed(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

    def _update_command(self, advance_success_time: bool = False) -> None:
        """Update ``cmd_buf`` and ``command_error`` via a single fused Warp kernel."""
        wp.launch(
            _update_command_kernel,
            dim=self.num_envs,
            inputs=[
                self.sim.state.joint_q,
                self.sim.state.joint_qd,
                wp.int32(self.jc_per),
                wp.int32(self.jd_per),
                self._cmd_mask_wp,
                self.reward_scales,
                wp.float32(self.step_dt),
                wp.int32(1 if advance_success_time else 0),
                self._cmd_buf_wp,
                self.command_error,
            ],
            device=self.device,
        )

    def get_observations(self):
        self.height_scanner.update(self.sim.state.joint_q, self.sim.state.body_q)
        wp.launch(
            _build_observations_kernel,
            dim=self.num_envs,
            inputs=[
                self.sim.state.joint_q,
                self.sim.state.joint_qd,
                wp.int32(self.jc_per),
                wp.int32(self.jd_per),
                self._last_action_wp,
                self._cmd_buf_wp,
                self.height_scanner.height_scan,
                self._observations_wp,
            ],
            device=self.device,
        )
        return self._observations_wp

    def act(self) -> None:
        observations = self.get_observations()
        self.policy.act(observations)

    def apply_actuator(self) -> None:
        self.actuator.apply()

    def log_visuals(self, viewer) -> None:
        viewer.log_points("/Visuals/Command/goal_point", None, hidden=True)

        if id(viewer) not in self._goal_arrow_mesh_viewer_ids:
            mesh = newton.Mesh.create_arrow(
                0.04,
                0.75,
                cap_radius=0.10,
                cap_height=0.25,
                up_axis=newton.Axis.X,
                segments=24,
            )
            viewer.log_mesh(
                "/Visuals/Command/goal_arrow_mesh",
                wp.array(np.asarray(mesh.vertices, dtype=np.float32), dtype=wp.vec3),
                wp.array(np.asarray(mesh.indices, dtype=np.int32), dtype=wp.int32),
                normals=wp.array(np.asarray(mesh.normals, dtype=np.float32), dtype=wp.vec3),
                uvs=wp.array(np.asarray(mesh.uvs, dtype=np.float32), dtype=wp.vec2),
                hidden=True,
            )
            self._goal_arrow_mesh_viewer_ids.add(id(viewer))

        wp.launch(
            _build_goal_arrow_instances_kernel,
            dim=self.num_envs,
            inputs=[
                self._cmd_buf_wp,
                self._cmd_mask_wp,
                self._goal_arrow_xforms_wp,
                self._goal_arrow_scales_wp,
                self._goal_arrow_colors_wp,
                self._goal_arrow_materials_wp,
            ],
            device=self.device,
        )
        viewer.log_instances(
            "/Visuals/Command/goal_arrow",
            "/Visuals/Command/goal_arrow_mesh",
            self._goal_arrow_xforms_wp,
            self._goal_arrow_scales_wp,
            self._goal_arrow_colors_wp,
            self._goal_arrow_materials_wp,
            hidden=False,
        )

    def forward(self):
        self._update_command(advance_success_time=True)
        wp.launch(
            _update_task_outputs_kernel,
            dim=self.num_envs,
            inputs=[
                self.sim.state.joint_q,
                self.sim.state.joint_qd,
                wp.int32(self.jc_per),
                wp.int32(self.jd_per),
                self._default_joint_pos_wp,
                self._cmd_buf_wp,
                wp.int32(self.max_episode_length),
                self._episode_length_wp,
                self._reward_wp,
                self._terminated_wp,
                self._truncated_wp,
            ],
            device=self.device,
        )
        return self._reward_wp, self._terminated_wp, self._truncated_wp

    def reset_done(self) -> None:
        wp.launch(_clear_counter_kernel, dim=1, inputs=[self._reset_count_wp], device=self.device)
        wp.launch(
            _collect_reset_ids_kernel,
            dim=self.num_envs,
            inputs=[self._terminated_wp, self._truncated_wp, self._reset_ids_wp, self._reset_count_wp],
            device=self.device,
        )
        count = int(self._reset_count_wp.numpy()[0])
        if count > 0:
            self.reset(self._reset_ids_wp, count_wp=self._reset_count_wp, count=count)

    def reset(self, env_ids, source_ids=None, count_wp=None, count: int | None = None) -> None:
        if count_wp is None:
            count_wp = self._all_count_wp
        if source_ids is None:
            source_ids = self._reset_source_ids_wp
            use_source_ids = 0
        else:
            use_source_ids = 1
        wp.launch(
            _reset_envs_kernel,
            dim=self.num_envs,
            inputs=[
                env_ids,
                source_ids,
                count_wp,
                wp.int32(use_source_ids),
                self._reset_states_wp,
                self._cmd_buf_table_wp,
                self._cmd_mask_table_wp,
                self._height_scan_table_wp,
                self._terrain_env_origins_table_wp,
                wp.int32(self.reset_state_count),
                wp.int32(self.cmd_count),
                wp.int32(self.height_count),
                wp.int32(self.terrain_count),
                wp.int32(self.jc_per),
                wp.int32(self.jd_per),
                wp.int32(self.num_dofs),
                self.sim.state.joint_q,
                self.sim.state.joint_qd,
                self.sim.control.joint_f,
                self._cmd_buf_wp,
                self._cmd_mask_wp,
                self.height_scanner.height_scan,
                self._terrain_env_origins_wp,
                self._default_joint_pos_wp,
                self._targets_wp,
                self._last_action_wp,
                self._episode_length_wp,
            ],
            device=self.device,
        )
        if count is None:
            count = int(count_wp.numpy()[0])
        if count == 0:
            return
        self.actuator.reset(env_ids, count)
        self.policy.reset(env_ids, count)
        newton.eval_fk(self.sim.model, self.sim.state.joint_q, self.sim.state.joint_qd, self.sim.state, None)
        self._update_command()
        self.height_scanner.update(self.sim.state.joint_q, self.sim.state.body_q)
