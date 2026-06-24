# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fast terrain height scanner for standalone Newton repro MDPs.

This mirrors the IsaacLab ``FastTerrainScanner`` data flow while staying free of
IsaacLab imports: a shared local ray pattern is built once, static terrain shapes
are resolved once from the Newton model, and a fused Warp kernel updates the
live clipped height-scan observation from the current floating-base pose.
"""

from __future__ import annotations

import numpy as np
import warp as wp
from newton._src.geometry.raycast import ray_intersect_geom
from newton._src.geometry.types import GeoType
from newton._src.utils.heightfield import HeightfieldData

ALIGNMENT_WORLD = wp.constant(0)
ALIGNMENT_YAW = wp.constant(1)
ALIGNMENT_BASE = wp.constant(2)
INF_F = wp.constant(float("inf"))


@wp.func
def _quat_yaw_only(q: wp.quatf) -> wp.quatf:
    qx = q[0]
    qy = q[1]
    qz = q[2]
    qw = q[3]
    yaw = wp.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    half_yaw = yaw * 0.5
    return wp.quatf(0.0, 0.0, wp.sin(half_yaw), wp.cos(half_yaw))


@wp.func
def _compute_world_frame_ray(
    sensor_pose: wp.transformf,
    drift: wp.vec3f,
    ray_cast_drift: wp.vec3f,
    local_start: wp.vec3f,
    local_dir: wp.vec3f,
    alignment_mode: int,
):
    combined_pos = wp.transform_get_translation(sensor_pose) + drift
    combined_quat = wp.transform_get_rotation(sensor_pose)

    if alignment_mode == ALIGNMENT_WORLD:
        pos_drifted = wp.vec3f(
            combined_pos[0] + ray_cast_drift[0], combined_pos[1] + ray_cast_drift[1], combined_pos[2]
        )
        ray_start_w = local_start + pos_drifted
        ray_direction_w = local_dir
    elif alignment_mode == ALIGNMENT_YAW:
        yaw_q = _quat_yaw_only(combined_quat)
        rot_drift = wp.quat_rotate(yaw_q, ray_cast_drift)
        pos_drifted = wp.vec3f(combined_pos[0] + rot_drift[0], combined_pos[1] + rot_drift[1], combined_pos[2])
        ray_start_w = wp.quat_rotate(yaw_q, local_start) + pos_drifted
        ray_direction_w = local_dir
    else:
        rot_drift = wp.quat_rotate(combined_quat, ray_cast_drift)
        pos_drifted = wp.vec3f(combined_pos[0] + rot_drift[0], combined_pos[1] + rot_drift[1], combined_pos[2])
        ray_start_w = wp.quat_rotate(combined_quat, local_start) + pos_drifted
        ray_direction_w = wp.quat_rotate(combined_quat, local_dir)

    return combined_pos, combined_quat, ray_start_w, ray_direction_w


@wp.kernel(enable_backward=False)
def _height_scan_kernel(
    joint_q: wp.array(dtype=wp.float32),
    body_q: wp.array(dtype=wp.transformf),
    jc_per_env: int,
    drift: wp.array(dtype=wp.vec3f),
    ray_cast_drift: wp.array(dtype=wp.vec3f),
    ray_starts_local: wp.array(dtype=wp.vec3f),
    ray_directions_local: wp.array(dtype=wp.vec3f),
    alignment_mode: int,
    terrain_shape_indices: wp.array(dtype=wp.int32),
    terrain_shape_count: int,
    shape_body: wp.array(dtype=wp.int32),
    shape_transform: wp.array(dtype=wp.transformf),
    shape_type: wp.array(dtype=wp.int32),
    shape_scale: wp.array(dtype=wp.vec3f),
    shape_source_ptr: wp.array(dtype=wp.uint64),
    shape_heightfield_index: wp.array(dtype=wp.int32),
    heightfield_data: wp.array(dtype=HeightfieldData),
    heightfield_elevations: wp.array(dtype=wp.float32),
    max_distance: float,
    observation_offset: float,
    ray_hits_w: wp.array2d(dtype=wp.vec3f),
    height_scan: wp.array2d(dtype=wp.float32),
):
    env_id, ray_id = wp.tid()

    q_start = env_id * jc_per_env
    sensor_pose = wp.transform(
        wp.vec3f(joint_q[q_start + 0], joint_q[q_start + 1], joint_q[q_start + 2]),
        wp.quatf(joint_q[q_start + 3], joint_q[q_start + 4], joint_q[q_start + 5], joint_q[q_start + 6]),
    )

    combined_pos, _combined_quat, ray_start_w, ray_direction_w = _compute_world_frame_ray(
        sensor_pose,
        drift[env_id],
        ray_cast_drift[env_id],
        ray_starts_local[ray_id],
        ray_directions_local[ray_id],
        alignment_mode,
    )

    min_t = max_distance
    hit_z = INF_F

    for terrain_slot in range(terrain_shape_count):
        shape_idx = terrain_shape_indices[terrain_slot]
        body_idx = shape_body[shape_idx]

        body_to_world = wp.transform_identity()
        if body_idx >= 0:
            body_to_world = body_q[body_idx]
        geom_to_world = wp.mul(body_to_world, shape_transform[shape_idx])

        geom_type = shape_type[shape_idx]
        mesh_id = wp.uint64(0)
        if geom_type == int(GeoType.MESH) or geom_type == int(GeoType.CONVEX_MESH):
            mesh_id = shape_source_ptr[shape_idx]

        t_hit, _normal = ray_intersect_geom(
            geom_to_world,
            shape_scale[shape_idx],
            geom_type,
            ray_start_w,
            ray_direction_w,
            mesh_id,
            shape_idx,
            shape_heightfield_index,
            heightfield_data,
            heightfield_elevations,
        )
        if t_hit >= 0.0 and t_hit < min_t:
            min_t = t_hit
            hit_z = ray_start_w[2] + t_hit * ray_direction_w[2]

    if min_t < max_distance:
        ray_hits_w[env_id, ray_id] = ray_start_w + min_t * ray_direction_w
        height_scan[env_id, ray_id] = wp.clamp(combined_pos[2] - hit_z - observation_offset, -1.0, 1.0)
    else:
        ray_hits_w[env_id, ray_id] = wp.vec3f(INF_F, INF_F, INF_F)
        height_scan[env_id, ray_id] = -1.0


class NewtonHeightScanner:
    """Fast height scanner matching the position task's FastTerrainScanner setup."""

    def __init__(
        self,
        model,
        num_envs: int,
        jc_per_env: int,
        device: str,
        expected_num_rays: int = 416,
        resolution: float = 0.1,
        size: tuple[float, float] = (2.5, 1.5),
        offset_pos: tuple[float, float, float] = (0.5, 0.0, 20.0),
        direction: tuple[float, float, float] = (0.0, 0.0, -1.0),
        ray_alignment: str = "yaw",
        max_distance: float = 1.0e6,
        observation_offset: float = 0.5,
    ) -> None:
        self.model = model
        self.num_envs = int(num_envs)
        self.jc_per_env = int(jc_per_env)
        self.device = device
        self.max_distance = float(max_distance)
        self.observation_offset = float(observation_offset)
        self.alignment_mode = {"world": 0, "yaw": 1, "base": 2}[ray_alignment]

        self._initialize_terrain_shapes()
        self._initialize_rays(resolution, size, offset_pos, direction, expected_num_rays)

        self._drift = wp.zeros(self.num_envs, dtype=wp.vec3f, device=self.device)
        self._ray_cast_drift = wp.zeros(self.num_envs, dtype=wp.vec3f, device=self.device)
        self._ray_hits_w = wp.zeros((self.num_envs, self.num_rays), dtype=wp.vec3f, device=self.device)
        self.height_scan = wp.zeros((self.num_envs, self.num_rays), dtype=wp.float32, device=self.device)

    def _initialize_terrain_shapes(self) -> None:
        shape_types = self.model.shape_type.numpy()
        shape_bodies = self.model.shape_body.numpy()
        terrain_types = {int(GeoType.MESH), int(GeoType.CONVEX_MESH), int(GeoType.HFIELD), int(GeoType.PLANE)}

        labels = getattr(self.model, "shape_label", None)
        if labels is not None:
            indices = [
                shape_id
                for shape_id, label in enumerate(labels)
                if "/World/ground" in str(label) and int(shape_types[shape_id]) in terrain_types
            ]
        else:
            indices = []

        if not indices:
            indices = [
                shape_id
                for shape_id, (body_id, shape_type) in enumerate(zip(shape_bodies, shape_types, strict=True))
                if int(body_id) < 0 and int(shape_type) in terrain_types
            ]
        if not indices:
            raise RuntimeError("NewtonHeightScanner could not find static terrain shapes in the Newton model.")

        self.terrain_shape_indices = tuple(int(index) for index in indices)
        self._terrain_shape_indices_wp = wp.array(
            np.asarray(self.terrain_shape_indices, dtype=np.int32), dtype=wp.int32, device=self.device
        )

    def _initialize_rays(
        self,
        resolution: float,
        size: tuple[float, float],
        offset_pos: tuple[float, float, float],
        direction: tuple[float, float, float],
        expected_num_rays: int,
    ) -> None:
        x = np.arange(-size[0] / 2, size[0] / 2 + 1.0e-9, step=resolution, dtype=np.float32)
        y = np.arange(-size[1] / 2, size[1] / 2 + 1.0e-9, step=resolution, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(x, y, indexing="xy")

        ray_starts = np.zeros((grid_x.size, 3), dtype=np.float32)
        ray_starts[:, 0] = grid_x.flatten()
        ray_starts[:, 1] = grid_y.flatten()
        ray_starts += np.asarray(offset_pos, dtype=np.float32).reshape(1, 3)

        direction_arr = np.asarray(direction, dtype=np.float32)
        direction_arr = direction_arr / max(float(np.linalg.norm(direction_arr)), 1.0e-8)
        ray_directions = np.broadcast_to(direction_arr.reshape(1, 3), ray_starts.shape).copy()

        self.num_rays = int(ray_starts.shape[0])
        if self.num_rays != int(expected_num_rays):
            raise RuntimeError(f"Expected {expected_num_rays} height-scan rays, built {self.num_rays}.")

        self._ray_starts_local = wp.array(ray_starts, dtype=wp.vec3f, device=self.device)
        self._ray_directions_local = wp.array(ray_directions, dtype=wp.vec3f, device=self.device)

    def update(self, joint_q: wp.array, body_q: wp.array) -> None:
        """Refresh the clipped live height-scan observation in :attr:`height_scan`."""
        wp.launch(
            _height_scan_kernel,
            dim=(self.num_envs, self.num_rays),
            inputs=[
                joint_q,
                body_q,
                self.jc_per_env,
                self._drift,
                self._ray_cast_drift,
                self._ray_starts_local,
                self._ray_directions_local,
                self.alignment_mode,
                self._terrain_shape_indices_wp,
                len(self.terrain_shape_indices),
                self.model.shape_body,
                self.model.shape_transform,
                self.model.shape_type,
                self.model.shape_scale,
                self.model.shape_source_ptr,
                self.model.shape_heightfield_index,
                self.model.heightfield_data,
                self.model.heightfield_elevations,
                self.max_distance,
                self.observation_offset,
            ],
            outputs=[self._ray_hits_w, self.height_scan],
            device=self.device,
        )
