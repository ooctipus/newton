# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
import re
from typing import Any

import numpy as np
import warp as wp

import newton

from ..core.types import nparray, override

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
except ImportError:
    Gf = Sdf = Usd = UsdGeom = UsdShade = Vt = None

from .viewer import ViewerBase


# transforms a cylinder such that it connects the two points pos0, pos1
def _compute_segment_xform(pos0, pos1):
    mid = (pos0 + pos1) * 0.5
    height = (pos1 - pos0).GetLength()

    dir = (pos1 - pos0) / height

    rot = Gf.Rotation()
    rot.SetRotateInto((0.0, 0.0, 1.0), Gf.Vec3d(dir))

    scale = Gf.Vec3f(1.0, 1.0, height)

    return (mid, Gf.Quath(rot.GetQuat()), scale)


def _usd_add_xform(prim):
    prim = UsdGeom.Xform(prim)
    prim.ClearXformOpOrder()

    prim.AddTranslateOp()
    prim.AddOrientOp()
    prim.AddScaleOp()


def _usd_set_xform(
    xform,
    pos: tuple | None = None,
    rot: tuple | None = None,
    scale: tuple | None = None,
    time: float = 0.0,
):
    xform = UsdGeom.Xform(xform)

    xform_ops = xform.GetOrderedXformOps()

    if pos is not None:
        xform_ops[0].Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])), time)
    if rot is not None:
        xform_ops[1].Set(Gf.Quatf(float(rot[3]), float(rot[0]), float(rot[1]), float(rot[2])), time)
    if scale is not None:
        xform_ops[2].Set(Gf.Vec3d(float(scale[0]), float(scale[1]), float(scale[2])), time)


class ViewerUSD(ViewerBase):
    """
    USD viewer backend for Newton physics simulations.

    This backend creates a USD stage and manages mesh prototypes and instanced rendering
    using PointInstancers. It supports time-sampled transforms for efficient playback
    and visualization of simulation data.
    """

    def __init__(
        self,
        output_path: str,
        fps: int = 60,
        up_axis: str = "Z",
        num_frames: int | None = 100,
        scaling: float = 1.0,
    ):
        """
        Initialize the USD viewer backend for Newton physics simulations.

        Args:
            output_path: Path to the output USD file.
            fps: Frames per second for time sampling. Default is 60.
            up_axis: USD up axis, either 'Y' or 'Z'. Default is 'Z'.
            num_frames: Maximum number of frames to record. Default is 100. If None, recording is unlimited.
            scaling: Uniform scaling applied to the scene root. Default is 1.0.

        Raises:
            ImportError: If the usd-core package is not installed.
        """
        if Usd is None:
            raise ImportError("usd-core package is required for ViewerUSD. Install with: pip install usd-core")

        super().__init__()

        self.output_path = os.path.abspath(output_path)
        self.fps = fps
        self.up_axis = up_axis
        self.num_frames = num_frames

        # Create USD stage. If this output path is already registered in the
        # current process, reuse and clear the existing layer instead of
        # calling CreateNew() again (which raises for duplicate identifiers).
        existing_layer = Sdf.Layer.Find(self.output_path)
        if existing_layer is not None:
            existing_layer.Clear()
            self.stage = Usd.Stage.Open(existing_layer)
        else:
            self.stage = Usd.Stage.CreateNew(self.output_path)
        self.stage.SetTimeCodesPerSecond(fps)  # number of timeCodes per second for data storage
        self.stage.SetFramesPerSecond(fps)  # display frame rate (timeline FPS in DCC tools)
        self.stage.SetStartTimeCode(0)

        axis_token = {
            "X": UsdGeom.Tokens.x,
            "Y": UsdGeom.Tokens.y,
            "Z": UsdGeom.Tokens.z,
        }.get(self.up_axis.strip().upper())

        UsdGeom.SetStageUpAxis(self.stage, axis_token)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)

        self.root = UsdGeom.Xform.Define(self.stage, "/root")

        # apply root scaling
        self.root.ClearXformOpOrder()
        s = self.root.AddScaleOp()
        s.Set(Gf.Vec3d(float(scaling), float(scaling), float(scaling)), 0.0)

        self.stage.SetDefaultPrim(self.root.GetPrim())

        # Track meshes and instancers
        self._meshes = {}  # mesh_name -> prototype_path
        self._instancers = {}  # instancer_name -> UsdGeomPointInstancer
        self._points = {}  # point_name -> UsdGeomPoints

        # Track current frame
        self._frame_index = 0
        self._frame_count = 0

        self._instance_paths: dict[str, list[str]] = {}
        self._instance_prim_cache: dict[str, Any] = {}

        self.set_model(None)

    @override
    def set_model(self, model: newton.Model | None, max_worlds: int | None = None):
        super().set_model(model, max_worlds)
        if model is not None:
            self._build_instance_paths()

    def _build_instance_paths(self):
        """Build semantically meaningful USD paths for each shape instance.

        Uses model.shape_label and model.body_label to create a hierarchy like:
            /objects/env_0/shadow_hand/robot0/link_0/visual/mesh_0
        instead of the generic:
            /model/shapes/shape_0/instance_0
        """
        model = self.model
        if model is None:
            return

        shape_labels = model.shape_label
        shape_body_np = model.shape_body.numpy()
        shape_world_np = model.shape_world.numpy()

        num_worlds = int(shape_world_np.max()) + 1 if len(shape_world_np) > 0 else 1
        multi_env = num_worlds > 1

        def _sanitize(s: str) -> str:
            """Sanitize a string to be a valid USD prim path segment."""
            s = s.strip("/")
            s = re.sub(r"[^a-zA-Z0-9_/]", "_", s)
            s = "/".join(seg for seg in s.split("/") if seg)
            return s

        def _strip_env_prefix(label: str) -> str:
            """Strip common Isaac Lab env scaffolding like World/envs/env_N/."""
            return re.sub(r"^World/envs/env_\d+/", "", label)

        def _world_prefix(world_idx: int) -> str:
            if world_idx < 0:
                return "/objects/shared"
            if multi_env:
                return f"/objects/env_{world_idx}"
            return "/objects"

        for batch in self._shape_instances.values():
            paths = []
            for i, shape_idx in enumerate(batch.model_shapes):
                label = shape_labels[shape_idx] if shape_idx < len(shape_labels) else None
                world_idx = int(shape_world_np[shape_idx])
                prefix = _world_prefix(world_idx)

                if label:
                    clean = _sanitize(label)
                    if multi_env and clean:
                        clean = _strip_env_prefix(clean)
                    if clean:
                        instance_path = f"{prefix}/{clean}"
                    else:
                        instance_path = f"{prefix}/shape_{shape_idx}"
                else:
                    body_idx = int(shape_body_np[shape_idx])
                    body_lbl = (
                        model.body_label[body_idx]
                        if 0 <= body_idx < len(model.body_label)
                        else "static"
                    )
                    body_clean = _sanitize(body_lbl) or f"body_{body_idx}"
                    if multi_env:
                        body_clean = _strip_env_prefix(body_clean)
                    instance_path = f"{prefix}/{body_clean}/shape_{i}"

                paths.append(instance_path)
            self._instance_paths[batch.name] = paths

    @override
    def begin_frame(self, time: float):
        """
        Begin a new frame at the given simulation time.

        Args:
            time: The simulation time for the new frame.
        """
        super().begin_frame(time)
        self._frame_index = int(time * self.fps)
        self._frame_count += 1

        # Update stage end time if needed
        if self._frame_index > self.stage.GetEndTimeCode():
            self.stage.SetEndTimeCode(self._frame_index)

    @override
    def end_frame(self):
        """
        End the current frame.

        This method is a placeholder for any end-of-frame logic required by the backend.
        """
        pass

    @override
    def is_running(self):
        """
        Check if the viewer is still running.

        Returns:
            bool: False if the frame limit is exceeded, True otherwise.
        """
        if self.num_frames is not None:
            return self._frame_count < self.num_frames
        return True

    @override
    def close(self):
        """
        Finalize and save the USD stage.

        This should be called when all logging is complete to ensure the USD file is written.
        """
        self.stage.GetRootLayer().Save()
        self.stage = None

        if self.output_path:
            print(f"USD output saved in: {os.path.abspath(self.output_path)}")

    def _get_path(self, name):
        # Handle both absolute and relative paths correctly
        if name.startswith("/"):
            return "/root" + name
        else:
            return "/root/" + name

    @override
    def log_mesh(
        self,
        name: str,
        points: wp.array(dtype=wp.vec3),
        indices: wp.array(dtype=wp.int32) | wp.array(dtype=wp.uint32),
        normals: wp.array(dtype=wp.vec3) | None = None,
        uvs: wp.array(dtype=wp.vec2) | None = None,
        texture: np.ndarray | str | None = None,
        hidden: bool = False,
        backface_culling: bool = True,
        color: tuple[float, float, float] | None = None,
        roughness: float | None = None,
        metallic: float | None = None,
    ):
        """
        Create a USD mesh prototype from vertex and index data.

        Args:
            name: Mesh name or Sdf.Path string.
            points: Vertex positions as a warp array of wp.vec3.
            indices: Triangle indices as a warp array of wp.uint32.
            normals: Vertex normals as a warp array of wp.vec3.
            uvs: UV coordinates as a warp array of wp.vec2.
            texture: Optional texture path/URL or image array.
            hidden: If True, mesh will be hidden.
            backface_culling: If True, enable backface culling.
            color: Optional base color as (r, g, b) in [0, 1].
            roughness: Optional roughness in [0, 1].
            metallic: Optional metallic in [0, 1].
        """

        # Convert warp arrays to numpy
        points_np = points.numpy().astype(np.float32)
        indices_np = indices.numpy().astype(np.uint32)

        first_time = name not in self._meshes
        if first_time:
            self._ensure_scopes_for_path(self.stage, self._get_path(name))

            mesh_prim = UsdGeom.Mesh.Define(self.stage, self._get_path(name))

            # setup topology once (do not set every frame)
            face_vertex_counts = [3] * (len(indices_np) // 3)
            mesh_prim.GetFaceVertexCountsAttr().Set(face_vertex_counts)
            mesh_prim.GetFaceVertexIndicesAttr().Set(indices_np)

            # Store the prototype path
            self._meshes[name] = mesh_prim

        mesh_prim = self._meshes[name]
        mesh_prim.GetPointsAttr().Set(points_np, self._frame_index)

        # Set normals if provided
        if normals is not None:
            normals_np = normals.numpy().astype(np.float32)
            mesh_prim.GetNormalsAttr().Set(normals_np, self._frame_index)
            mesh_prim.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

        if first_time:
            # Write UVs as a "st" primvar (only on first creation)
            if uvs is not None:
                uvs_np = uvs.numpy().astype(np.float32)
                num_uvs = len(uvs_np)
                num_verts = len(points_np)
                num_face_verts = len(indices_np)
                pv_api = UsdGeom.PrimvarsAPI(mesh_prim)
                if num_uvs == num_verts:
                    st_pv = pv_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
                    st_pv.Set(uvs_np)
                elif num_uvs == num_face_verts:
                    st_pv = pv_api.CreatePrimvar(
                        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
                    )
                    st_pv.Set(uvs_np)

            # Create and bind UsdPreviewSurface material
            if texture is not None or color is not None:
                self._create_and_bind_material(
                    mesh_prim, name, texture=texture, color=color, roughness=roughness, metallic=metallic
                )

        mesh_prim.GetVisibilityAttr().Set("inherited" if not hidden else "invisible", self._frame_index)

    def _create_and_bind_material(
        self,
        mesh_prim: "UsdGeom.Mesh",
        mesh_name: str,
        *,
        texture: np.ndarray | str | None = None,
        color: tuple[float, float, float] | None = None,
        roughness: float | None = None,
        metallic: float | None = None,
    ):
        """Create a UsdPreviewSurface material and bind it to a mesh prim.

        Handles both file-path textures and numpy-array textures (saved as
        PNG alongside the USD). Falls back to a solid diffuseColor when no
        texture is provided.
        """
        safe_name = mesh_name.strip("/").replace("/", "_")
        mat_path = f"/root/materials/mat_{safe_name}"
        self._ensure_scopes_for_path(self.stage, mat_path)

        material = UsdShade.Material.Define(self.stage, mat_path)
        shader = UsdShade.Shader.Define(self.stage, mat_path + "/preview_surface")
        shader.CreateIdAttr("UsdPreviewSurface")

        # Roughness / metallic
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
            roughness if roughness is not None else 0.5
        )
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(
            metallic if metallic is not None else 0.0
        )

        # Resolve texture path (save numpy arrays to disk)
        texture_file = None
        if texture is not None:
            if isinstance(texture, str):
                texture_file = texture
            elif isinstance(texture, np.ndarray):
                tex_dir = os.path.dirname(self.output_path)
                tex_filename = f"tex_{safe_name}.png"
                tex_path = os.path.join(tex_dir, tex_filename)
                try:
                    from PIL import Image  # noqa: PLC0415

                    img = texture
                    if img.dtype != np.uint8:
                        img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
                    Image.fromarray(img).save(tex_path)
                    texture_file = f"./{tex_filename}"
                except ImportError:
                    pass

        if texture_file is not None:
            # Create texture reader shader
            tex_reader = UsdShade.Shader.Define(self.stage, mat_path + "/diffuse_texture")
            tex_reader.CreateIdAttr("UsdUVTexture")
            tex_reader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_file)
            tex_reader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            tex_reader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            tex_reader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

            # Create ST reader shader
            st_reader = UsdShade.Shader.Define(self.stage, mat_path + "/st_reader")
            st_reader.CreateIdAttr("UsdPrimvarReader_float2")
            st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

            # Connect texture → st reader
            tex_reader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                st_reader.ConnectableAPI(), "result"
            )

            # Connect surface diffuseColor → texture rgb
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                tex_reader.ConnectableAPI(), "rgb"
            )
        elif color is not None:
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
            )
        else:
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.5, 0.5, 0.5))

        # Wire surface output
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        # Bind material to the mesh prim
        UsdShade.MaterialBindingAPI.Apply(mesh_prim.GetPrim())
        UsdShade.MaterialBindingAPI(mesh_prim).Bind(material)

    # log a set of instances as individual mesh prims, slower but makes it easier
    # to do post-editing of instance materials etc. default for Newton shapes
    @override
    def log_instances(
        self,
        name: str,
        mesh: str,
        xforms: wp.array(dtype=wp.transform) | None,
        scales: wp.array(dtype=wp.vec3) | None,
        colors: wp.array(dtype=wp.vec3) | None,
        materials: wp.array(dtype=wp.vec4) | None,
        hidden: bool = False,
    ):
        """
        Log a batch of mesh instances for rendering.

        Args:
            name: Unique name for the instancer.
            mesh: Name of the base mesh.
            xforms: Array of transforms.
            scales: Array of scales.
            colors: Array of colors.
            materials: Array of materials.
            hidden: Whether the instances are hidden.
        """
        # Get prototype path
        if mesh not in self._meshes:
            msg = f"Mesh prototype '{mesh}' not found for log_instances(). Call log_mesh() first."
            raise RuntimeError(msg)

        self._ensure_scopes_for_path(self.stage, self._get_path(name) + "/scope")

        if xforms is not None:
            xforms = xforms.numpy()
        else:
            xforms = np.empty((0, 7), dtype=np.float32)

        if scales is not None:
            scales = scales.numpy()
        else:
            scales = np.ones((len(xforms), 3), dtype=np.float32)

        if colors is not None:
            colors = colors.numpy()

        semantic_paths = self._instance_paths.get(name)
        mesh_proto_path = self._get_path(mesh)

        for i in range(len(xforms)):
            if semantic_paths is not None and i < len(semantic_paths):
                instance_path = self._get_path(semantic_paths[i])
            else:
                instance_path = self._get_path(name) + f"/instance_{i}"

            instance = self._instance_prim_cache.get(instance_path)
            if instance is None:
                instance = self.stage.GetPrimAtPath(instance_path)
                if not instance:
                    self._ensure_scopes_for_path(self.stage, instance_path)
                    instance = self.stage.DefinePrim(instance_path)
                    instance.GetReferences().AddInternalReference(mesh_proto_path)
                    UsdGeom.Imageable(instance).GetVisibilityAttr().Set(
                        "inherited" if not hidden else "invisible"
                    )
                    _usd_add_xform(instance)
                self._instance_prim_cache[instance_path] = instance

            if xforms is not None:
                pos = xforms[i][:3]
                rot = xforms[i][3:7]
                _usd_set_xform(instance, pos, rot, scales[i], self._frame_index)

            if colors is not None:
                displayColor = UsdGeom.PrimvarsAPI(instance).GetPrimvar("displayColor")
                displayColor.Set(colors[i], self._frame_index)

    # log a set of instances as a point instancer, faster but less flexible
    def log_instances_point_instancer(
        self,
        name: str,
        mesh: str,
        xforms: wp.array(dtype=wp.transform) | None,
        scales: wp.array(dtype=wp.vec3) | nparray | None,
        colors: (
            wp.array(dtype=wp.vec3)
            | wp.array(dtype=wp.float32)
            | tuple[float, float, float]
            | list[float]
            | nparray
            | None
        ),
        materials: wp.array(dtype=wp.vec4) | None,
    ):
        """
        Create or update a PointInstancer for mesh instances.

        Args:
            name: Instancer name or Sdf.Path string.
            mesh: Mesh prototype name (must be previously logged).
            xforms: Instance transforms as a warp array of wp.transform.
            scales: Instance scales as a warp array of wp.vec3.
            colors: Instance colors as a warp array of wp.vec3.
            materials: Instance materials as a warp array of wp.vec4.

        Raises:
            RuntimeError: If the mesh prototype is not found.
        """
        # Get prototype path
        if mesh not in self._meshes:
            msg = f"Mesh prototype '{mesh}' not found for log_instances(). Call log_mesh() first."
            raise RuntimeError(msg)

        num_instances = len(xforms)

        # Create instancer if it doesn't exist
        if name not in self._instancers:
            self._ensure_scopes_for_path(self.stage, self._get_path(name))

            instancer = UsdGeom.PointInstancer.Define(self.stage, self._get_path(name))
            instancer.CreateIdsAttr().Set(list(range(num_instances)))
            instancer.CreateProtoIndicesAttr().Set([0] * num_instances)
            UsdGeom.PrimvarsAPI(instancer).CreatePrimvar(
                "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex, 1
            )

            # Set the prototype relationship
            instancer.GetPrototypesRel().AddTarget(self._get_path(mesh))

            self._instancers[name] = instancer

        instancer = self._instancers[name]

        # Convert transforms to USD format
        if xforms is not None:
            xforms_np = xforms.numpy()

            # Extract positions from warp transforms using vectorized operations
            # Warp transform format: [x, y, z, qx, qy, qz, qw]
            positions = xforms_np[:, :3].astype(np.float32)

            # Convert quaternion format: Warp (x, y, z, w) → USD (w, (x,y,z))
            # USD expects quaternions as Gf.Quath(real, imag_vec3)
            quat_w = xforms_np[:, 6].astype(np.float32)
            quat_xyz = xforms_np[:, 3:6].astype(np.float32)

            # Create orientations list with proper USD quaternion format
            orientations = []
            for i in range(num_instances):
                quat = Gf.Quath(
                    float(quat_w[i]), Gf.Vec3h(float(quat_xyz[i, 0]), float(quat_xyz[i, 1]), float(quat_xyz[i, 2]))
                )
                orientations.append(quat)

            # Handle scales with numpy operations
            if scales is None:
                scales = np.ones((num_instances, 3), dtype=np.float32)
            elif isinstance(scales, wp.array):
                scales = scales.numpy().astype(np.float32)

            # Set attributes at current time
            instancer.GetPositionsAttr().Set(positions, self._frame_index)
            instancer.GetOrientationsAttr().Set(orientations, self._frame_index)

            if scales is not None:
                instancer.GetScalesAttr().Set(scales, self._frame_index)

            if colors is not None:
                # Promote colors to proper numpy array format
                colors_np = self._promote_colors_to_array(colors, num_instances)

                # Set color per-instance
                displayColor = UsdGeom.PrimvarsAPI(instancer).GetPrimvar("displayColor")
                displayColor.Set(colors_np, self._frame_index)

                # Explicit identity indices [0, 1, 2, ...], otherwise OV won't pick them up
                indices = Vt.IntArray(range(num_instances))
                displayColor.SetIndices(indices, self._frame_index)

    # Abstract methods that need basic implementations
    @override
    def log_lines(
        self,
        name: str,
        starts: wp.array(dtype=wp.vec3) | None,
        ends: wp.array(dtype=wp.vec3) | None,
        colors: (
            wp.array(dtype=wp.vec3) | wp.array(dtype=wp.float32) | tuple[float, float, float] | list[float] | None
        ),
        width: float = 0.01,
        hidden: bool = False,
    ):
        """Debug helper to add a line list as a set of capsules

        Args:
            name: Unique name for the line batch.
            starts: The vertices of the lines (wp.array)
            ends: The vertices of the lines (wp.array)
            colors: The colors of the lines (wp.array)
            width: The width of the lines.
            hidden: Whether the lines are hidden.
        """

        if name not in self._instancers:
            self._ensure_scopes_for_path(self.stage, self._get_path(name))

            instancer = UsdGeom.PointInstancer.Define(self.stage, self._get_path(name))

            # define nested capsule prim
            instancer_capsule = UsdGeom.Capsule.Define(self.stage, instancer.GetPath().AppendChild("capsule"))
            instancer_capsule.GetRadiusAttr().Set(width)

            instancer.CreatePrototypesRel().SetTargets([instancer_capsule.GetPath()])
            UsdGeom.PrimvarsAPI(instancer).CreatePrimvar(
                "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex, 1
            )

            self._instancers[name] = instancer

        instancer = self._instancers[name]

        if starts is not None and ends is not None:
            num_lines = int(len(starts))
            if num_lines > 0:
                # bring to host
                starts = starts.numpy()
                ends = ends.numpy()

                line_positions = []
                line_rotations = []
                line_scales = []

                for i in range(num_lines):
                    pos0 = starts[i]
                    pos1 = ends[i]

                    (pos, rot, scale) = _compute_segment_xform(
                        Gf.Vec3f(float(pos0[0]), float(pos0[1]), float(pos0[2])),
                        Gf.Vec3f(float(pos1[0]), float(pos1[1]), float(pos1[2])),
                    )

                    line_positions.append(pos)
                    line_rotations.append(rot)
                    line_scales.append(scale)

                instancer.GetPositionsAttr().Set(line_positions, self._frame_index)
                instancer.GetOrientationsAttr().Set(line_rotations, self._frame_index)
                instancer.GetScalesAttr().Set(line_scales, self._frame_index)
                instancer.GetProtoIndicesAttr().Set([0] * num_lines, self._frame_index)
                instancer.CreateIdsAttr().Set(list(range(num_lines)))

                if colors is not None:
                    # Promote colors to proper numpy array format
                    colors_np = self._promote_colors_to_array(colors, num_lines)

                    # Set color per-instance
                    displayColor = UsdGeom.PrimvarsAPI(instancer).GetPrimvar("displayColor")
                    displayColor.Set(colors_np, self._frame_index)

                    # Explicit identity indices [0, 1, 2, ...], otherwise OV won't pick them up
                    indices = Vt.IntArray(range(num_lines))
                    displayColor.SetIndices(indices, self._frame_index)

        instancer.GetVisibilityAttr().Set("inherited" if not hidden else "invisible", self._frame_index)

    @override
    def log_points(
        self,
        name: str,
        points: wp.array(dtype=wp.vec3) | None,
        radii: wp.array(dtype=wp.float32) | float | None = None,
        colors: (
            wp.array(dtype=wp.vec3) | wp.array(dtype=wp.float32) | tuple[float, float, float] | list[float] | None
        ) = None,
        hidden: bool = False,
    ):
        """Log points as a USD `Points` primitive.

        Args:
            name: Unique name for the point primitive.
            points: Point positions.
            radii: Point radii or a single shared radius.
            colors: Optional per-point colors or a shared RGB triplet.
            hidden: Whether the point primitive is hidden.

        Returns:
            Sdf.Path of the created/updated points primitive.
        """
        if points is None:
            return

        num_points = len(points)

        if radii is None:
            radii = 0.1

        if np.isscalar(radii):
            radius_interp = "constant"
        else:
            radius_interp = "vertex"

        colors, color_interp = self._normalize_point_colors(colors, num_points)

        path = self._get_path(name)
        instancer = UsdGeom.Points.Get(self.stage, path)
        if not instancer:
            self._ensure_scopes_for_path(self.stage, path)
            instancer = UsdGeom.Points.Define(self.stage, path)

            UsdGeom.Primvar(instancer.GetWidthsAttr()).SetInterpolation(radius_interp)
            UsdGeom.Primvar(instancer.GetDisplayColorAttr()).SetInterpolation(color_interp)

        instancer.GetPointsAttr().Set(points.numpy(), self._frame_index)

        # convert radii to widths for USD
        if np.isscalar(radii):
            widths = (radii * 2.0,)
        elif isinstance(radii, wp.array):
            widths = radii.numpy() * 2.0
        else:
            widths = np.array(radii) * 2.0

        instancer.GetWidthsAttr().Set(widths, self._frame_index)

        if colors is not None:
            instancer.GetDisplayColorAttr().Set(colors, self._frame_index)

        instancer.GetVisibilityAttr().Set("inherited" if not hidden else "invisible", self._frame_index)
        return instancer.GetPath()

    @override
    def log_array(self, name: str, array: wp.array(dtype=Any) | nparray):
        """
        Log array data (not implemented for USD backend).

        This method is a placeholder and does not log array data in the USD backend.

        Args:
            name: Unique path/name for the array signal.
            array: Array data to visualize.
        """
        pass

    @override
    def log_scalar(self, name: str, value: int | float | bool | np.number):
        """
        Log scalar value (not implemented for USD backend).

        This method is a placeholder and does not log scalar values in the USD backend.

        Args:
            name: Unique path/name for the scalar signal.
            value: Scalar value to visualize.
        """
        pass

    @override
    def apply_forces(self, state: newton.State):
        """USD backend does not apply interactive forces.

        Args:
            state: Current simulation state.
        """
        pass

    def _promote_colors_to_array(self, colors, num_items):
        """
        Helper method to promote colors to a numpy array format.

        Parameters:
            colors: Input colors in various formats (wp.array, list/tuple, np.ndarray)
            num_items: Number of items that need colors

        Returns:
            np.ndarray: Colors as numpy array with shape (num_items, 3)
        """
        if colors is None:
            return None

        if isinstance(colors, wp.array):
            # Convert warp array to numpy
            return colors.numpy()
        elif isinstance(colors, list | tuple) and len(colors) == 3 and all(np.isscalar(x) for x in colors):
            # Single color (list/tuple of 3 floats) - promote to array with one value per item
            return np.tile(colors, (num_items, 1))
        elif isinstance(colors, np.ndarray):
            # Already numpy array - pass through
            return colors
        else:
            # Fallback for other formats
            return np.array(colors)

    @staticmethod
    def _is_single_rgb_triplet(colors) -> bool:
        """Returns True when colors represent one RGB triplet."""
        if isinstance(colors, np.ndarray):
            return colors.ndim == 1 and colors.shape[0] == 3

        if isinstance(colors, list | tuple):
            return len(colors) == 3 and all(np.isscalar(x) for x in colors)

        return False

    def _normalize_point_colors(self, colors, num_points):
        """Normalize point colors and return (values, interpolation token)."""
        if colors is None:
            return None, "constant"

        if isinstance(colors, wp.array):
            colors = colors.numpy()

        if self._is_single_rgb_triplet(colors):
            colors_arr = np.asarray(colors, dtype=np.float32)
            return colors_arr.reshape(1, 3), "constant"

        if isinstance(colors, np.ndarray):
            return colors, "vertex"

        if isinstance(colors, list | tuple):
            # Keep list/tuple inputs as-is for existing valid per-point color inputs.
            if len(colors) == num_points:
                return colors, "vertex"
            return np.asarray(colors), "vertex"

        return np.asarray(colors), "vertex"

    @staticmethod
    def _ensure_scopes_for_path(stage: Usd.Stage, prim_path_str: str):
        """
        Ensure that all parent prims in the hierarchy exist as 'Scope' prims.

        If a prim does not exist at the given path, this method creates all
        non-existent parent prims in its hierarchy as 'Scope' prims. This is
        useful for ensuring a valid hierarchy before defining a prim.

        Parameters:
            stage: The USD stage to operate on.
            prim_path_str: The Sdf.Path string for the target prim.
        """
        # Convert the string to an Sdf.Path object for robust manipulation
        prim_path = Sdf.Path(prim_path_str)

        # First, check if the target prim already exists.
        if stage.GetPrimAtPath(prim_path):
            return

        # We only want to create the parent hierarchy, not the final prim itself.
        parent_path = prim_path.GetParentPath()

        # GetPrefixes() provides a convenient list of all ancestor paths.
        # For "/A/B/C", it returns ["/", "/A", "/A/B"].
        for path in parent_path.GetPrefixes():
            # The absolute root path ('/') always exists, so we can skip it.
            if path == Sdf.Path.absoluteRootPath:
                continue

            # Check if a prim exists at the current ancestor path.
            if not stage.GetPrimAtPath(path):
                stage.DefinePrim(path, "Scope")
