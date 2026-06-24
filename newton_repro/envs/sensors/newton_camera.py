# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone Newton depth camera for repro MDPs.

Mirrors the depth path of :class:`isaaclab_newton.renderers.NewtonWarpRenderer` without any
Isaac Lab imports: it owns a :class:`newton.sensors.SensorTiledCamera`, builds the shape BVH
once, refits it against the live state each frame, and renders a depth image from fixed
per-environment camera poses captured from the live sensor.

Camera poses are supplied as Isaac Lab's ``CameraData.quat_w_world`` (xyzw layout, world camera
convention) and converted to the OpenGL convention the renderer feeds the Newton sensor via the
constant quaternion multiply ``q_world * (0.5, -0.5, -0.5, 0.5)`` (xyzw), matching
:func:`isaaclab.utils.warp.warp_math.convert_camera_frame_orientation_convention_wp`.

The base camera in the Lift-KukaAllegro task is static per environment, so the camera
transforms are computed once; only the BVH refit + ray cast run per step.
"""

from __future__ import annotations

import math

import newton
import numpy as np
import torch
import warp as wp

# Constant quaternion (xyzw) applied as ``q_world * Q`` to convert a world-convention
# (+X forward, +Z up) camera orientation to the OpenGL convention (-Z forward, +Y up).
_WORLD_TO_OPENGL_XYZW = np.asarray([0.5, -0.5, -0.5, 0.5], dtype=np.float32)


def _quat_mul_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two ``(..., 4)`` quaternions in xyzw layout."""
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        axis=-1,
    )


class NewtonCamera:
    """Render a per-environment depth image from a finalized Newton model.

    Args:
        model: Finalized :class:`newton.Model` (shared with the sim).
        positions_w: Per-env camera world positions [m], shape ``(num_envs, 3)``.
        quats_w_world: Per-env camera world orientations (xyzw), shape ``(num_envs, 4)``.
        intrinsics: Per-env ``3x3`` pinhole intrinsic matrices, shape ``(num_envs, 3, 3)``.
        width: Image width [px].
        height: Image height [px].
        device: Warp/torch device.
        max_distance: Far ray distance [m]; matches ``NewtonWarpRendererCfg.max_distance``.
        enable_textures: Enable textured shading (matches the live renderer default).
        enable_shadows: Enable shadow casting.
        create_default_light: Add the renderer's default directional light.
    """

    def __init__(
        self,
        model,
        *,
        positions_w: np.ndarray,
        quats_w_world: np.ndarray,
        intrinsics: np.ndarray,
        width: int,
        height: int,
        device: str,
        max_distance: float = 1000.0,
        enable_textures: bool = True,
        enable_shadows: bool = False,
        create_default_light: bool = True,
    ) -> None:
        self.model = model
        self.device = device
        self.width = int(width)
        self.height = int(height)
        self.num_envs = int(np.asarray(positions_w).shape[0])

        if model.shape_count > 0 and getattr(model, "bvh_shapes", None) is None:
            newton.geometry.build_bvh_shape(model, model.state())

        self.sensor = newton.sensors.SensorTiledCamera(
            model,
            config=newton.sensors.SensorTiledCamera.RenderConfig(
                enable_textures=enable_textures,
                enable_shadows=enable_shadows,
                max_distance=max_distance,
            ),
        )
        if create_default_light:
            self.sensor.utils.create_default_light(enable_shadows=enable_shadows)

        self._camera_transforms = self._build_camera_transforms(positions_w, quats_w_world)
        fov = 2.0 * math.atan(self.height / (2.0 * float(np.asarray(intrinsics)[0, 1, 1])))
        fov_wp = wp.array(np.asarray([fov], dtype=np.float32), dtype=wp.float32, device=device)
        self._camera_rays = self.sensor.utils.compute_pinhole_camera_rays(self.width, self.height, fov_wp)
        self._depth_buf = wp.zeros((self.num_envs, 1, self.height, self.width), dtype=wp.float32, device=device)

    def _build_camera_transforms(self, positions_w: np.ndarray, quats_w_world: np.ndarray) -> wp.array:
        """Pack static OpenGL-convention camera transforms into a ``(1, num_envs)`` array.

        ``quats_w_world`` is Isaac Lab's ``CameraData.quat_w_world``, stored **xyzw** (it is fed to
        the warp renderer as a ``quatf``), so it is used directly without reordering.
        """
        pos = np.asarray(positions_w, dtype=np.float32).reshape(self.num_envs, 3)
        quat_xyzw = np.asarray(quats_w_world, dtype=np.float32).reshape(self.num_envs, 4)
        quat_gl = _quat_mul_xyzw(quat_xyzw, np.broadcast_to(_WORLD_TO_OPENGL_XYZW, quat_xyzw.shape))
        transforms = np.zeros((1, self.num_envs, 7), dtype=np.float32)
        transforms[0, :, 0:3] = pos
        transforms[0, :, 3:7] = quat_gl
        return wp.array(transforms, dtype=wp.transformf, device=self.device)

    def render(self, state) -> wp.array:
        """Refit the BVH against *state* and render the depth buffer ``(num_envs, 1, H, W)`` [m]."""
        if self.model.shape_count > 0:
            newton.geometry.refit_bvh_shape(self.model, state)
        self.sensor.update(
            state,
            self._camera_transforms,
            self._camera_rays,
            depth_image=self._depth_buf,
            clear_data=newton.sensors.SensorTiledCamera.ClearData(clear_color=0xFFEEEEEE),
        )
        return self._depth_buf

    def depth_normalized(self, state) -> torch.Tensor:
        """Return the normalized depth observation ``(num_envs, 1, H, W)`` matching ``mdp.vision_camera``.

        Applies ``nan_to_num`` -> ``tanh(d / 2) * 2`` -> per-image spatial-mean subtraction ->
        ``NCHW`` permute, then clips to ``[-1, 1]`` (the ``base_image`` observation-term clip).
        """
        depth = self.render(state)
        images = wp.to_torch(depth).view(self.num_envs, self.height, self.width, 1)
        torch.nan_to_num_(images, nan=1.0e6)
        images = torch.tanh(images / 2.0) * 2.0
        images = images - torch.mean(images, dim=(1, 2), keepdim=True)
        images = images.permute(0, 3, 1, 2).contiguous()
        return torch.clamp(images, -1.0, 1.0)
