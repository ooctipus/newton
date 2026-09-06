# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.metadata as importlib_metadata
import math
import os
import re
import sys
import warnings
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from ...core.types import MAXVAL, override, vec5, vec10
from ...geometry import GeoType, Mesh, ShapeFlags
from ...geometry.inertia import verify_and_correct_inertia
from ...sim import (
    BodyFlags,
    Contacts,
    Control,
    JointTargetMode,
    JointType,
    Model,
    ModelBuilder,
    ModelFlags,
    State,
    StateFlags,
)
from ...sim.articulation import eval_articulation_fk, eval_fk
from ...sim.contacts import GENERATION_SENTINEL as _GENERATION_SENTINEL
from ...sim.graph_coloring import color_graph, plot_graph
from ...utils import topological_sort
from ...utils.benchmark import event_scope
from ...utils.import_utils import string_to_warp
from ..coupled.interface import CouplingEndpointKind, CouplingInterface
from ..solver import SolverBase
from . import kernels
from .collision_masks import (
    MUJOCO_COLLISION_MASK_DOMAIN_UNSET,
    MUJOCO_COLLISION_MASK_UNSET,
    NEWTON_COLLISION_MASK_MAX_SHAPE_COUNT,
    compile_newton_collision_graph,
    mujoco_mask_to_signed,
)
from .constants import (
    DEFAULT_LIMIT_GAIN_RTOL,
    DEFAULT_LIMIT_KD,
    DEFAULT_LIMIT_KE,
    DEFAULT_LIMIT_SOLREF,
    HINGE_CONNECT_AXIS_OFFSET,
    KINEMATIC_ARMATURE,
    MJ_MINVAL,
    SOLREF_MODE_FORCE_SPACE,
    SOLREF_MODE_MJCF_DEFAULT,
    SOLREF_MODE_RAW,
)
from .enums import EqType as _EqType
from .enums import _ActuatorBiasType, _ActuatorDynamicsType, _ActuatorGainType
from .equality import MJC_OBJ_BODY, MjcEqualityTargetKind, _register_equality_constraint_attributes
from .kernels import (
    CONTACT_CONVERSION_MAX_THREADS,
    MeshVariantBody,
    MeshVariantShape,
    _finish_wake_injection,
    _snapshot_nacon_count,
    apply_body_sleep_override_kernel,
    apply_mjc_body_f_kernel,
    apply_mjc_control_kernel,
    apply_mjc_free_joint_f_to_body_f_kernel,
    apply_mjc_inputs_kernel,
    apply_mjc_qfrc_kernel,
    build_ref_q_kernel,
    collect_woken_dormant_slabs_kernel,
    compact_reset_world_ids_kernel,
    convert_mj_coords_to_warp_kernel,
    convert_newton_contacts_to_mjwarp_kernel,
    convert_qfrc_actuator_from_mj_kernel,
    convert_rigid_forces_from_mj_kernel,
    convert_solref,
    convert_warp_coords_to_mj_kernel,
    convert_warp_coords_to_mj_worlds_kernel,
    create_convert_mjw_contacts_to_newton_kernel,
    create_inverse_shape_mapping_kernel,
    eval_mujoco_coupling_effective_mass_block_kernel,
    eval_mujoco_coupling_effective_mass_kernel,
    eval_mujoco_coupling_gravity_acceleration_kernel,
    inject_dormant_slab_contacts_kernel,
    prepare_contact_conversion_kernel,
    recompute_jnt_eq_anchor1_kernel,
    refresh_tree_awake_kernel,
    repeat_array_kernel,
    reset_joint_state_kernel,
    reset_sleeping_state_kernel,
    reset_world_buffers_kernel,
    restore_sleeping_state_kernel,
    set_mesh_variant_index_kernel,
    set_selected_body_sleep_override_kernel,
    sync_qpos0_kernel,
    sync_site_xposes_kernel,
    sync_worldbody_geom_xposes_kernel,
    update_axis_properties_kernel,
    update_body_inertia_kernel,
    update_body_mass_ipos_kernel,
    update_body_properties_kernel,
    update_connect_constraint_anchors_kernel,
    update_connect_constraint_rel_body_poses_at_qref_kernel,
    update_ctrl_direct_actuator_properties_kernel,
    update_dof_properties_kernel,
    update_eq_data_and_active_kernel,
    update_eq_properties_kernel,
    update_geom_properties_kernel,
    update_jnt_connect_constraint_anchors_kernel,
    update_jnt_connect_constraint_rel_body_poses_at_qref_kernel,
    update_jnt_properties_kernel,
    update_jnt_solref_from_invweight0_kernel,
    update_joint_transforms_kernel,
    update_mimic_eq_data_and_active_kernel,
    update_mocap_transforms_kernel,
    update_model_properties_kernel,
    update_pair_properties_kernel,
    update_shape_mappings_kernel,
    update_site_properties_kernel,
    update_solver_options_kernel,
    update_tendon_properties_kernel,
    wake_changed_trees_kernel,
    wake_contact_trees_kernel,
    wake_selected_tree_kernel,
)

if TYPE_CHECKING:
    from mujoco import MjData, MjModel
    from mujoco_warp import Data as MjWarpData
    from mujoco_warp import Model as MjWarpModel
else:
    MjModel = object
    MjData = object
    MjWarpModel = object
    MjWarpData = object

AttributeAssignment = Model.AttributeAssignment
AttributeFrequency = Model.AttributeFrequency


def _required_specifier(package: str, requirements: Iterable[str]) -> str | None:
    pattern = re.compile(rf"^{re.escape(package)}(?=[<>=!~])([^;]+)")
    for requirement in requirements:
        match = pattern.match(requirement)
        if match:
            return match.group(1).strip().replace(" ", "")
    return None


def _warn_if_mujoco_versions_mismatch(mujoco: Any, mujoco_warp: Any) -> None:
    try:
        metadata_text = importlib_metadata.distribution("newton").read_text("METADATA")
    except importlib_metadata.PackageNotFoundError:
        return
    if metadata_text is None:
        return

    requirements = [
        line.removeprefix("Requires-Dist:").strip()
        for line in metadata_text.splitlines()
        if line.startswith("Requires-Dist:")
    ]

    mismatches = []
    for package, module in (("mujoco", mujoco), ("mujoco-warp", mujoco_warp)):
        specifier = _required_specifier(package, requirements)
        installed_version = _installed_version(package, module)
        if specifier and installed_version and not _version_satisfies(installed_version, specifier):
            mismatches.append(f"{package}=={installed_version} (requires {specifier})")

    if mismatches:
        warnings.warn(
            "MuJoCo dependency version mismatch with Newton's declared requirements: "
            + "; ".join(mismatches)
            + '. Reinstall Newton dependencies, for example `uv pip install -e ".[examples]"`.',
            RuntimeWarning,
            stacklevel=3,
        )


def _installed_version(package: str, module: Any) -> str | None:
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        module_version = getattr(module, "__version__", None)
        return str(module_version) if module_version is not None else None


def _version_satisfies(installed_version: str, specifier: str) -> bool:
    installed = _release(installed_version)
    if not installed:
        return True

    for required_version in re.findall(r">=\s*([0-9][^,;]*)", specifier):
        if _version_lt(installed, _release(required_version)):
            return False
    for required_version in re.findall(r"~=\s*([0-9][^,;]*)", specifier):
        required = _release(required_version)
        prefix_width = max(len(required) - 1, 1)
        if _version_lt(installed, required) or installed[:prefix_width] != required[:prefix_width]:
            return False
    return True


def _release(version: str) -> tuple[int, ...]:
    match = re.match(r"\d+(?:\.\d+)*", version)
    return tuple(int(component) for component in match.group(0).split(".")) if match else ()


def _version_lt(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    width = max(len(left), len(right))
    return left + (0,) * (width - len(left)) < right + (0,) * (width - len(right))


def _mujoco_warp_deterministic_modules() -> list[Any]:
    """Return loaded MJWarp implementation modules."""
    return [
        module for name, module in sys.modules.items() if name.startswith("mujoco_warp._src.") and module is not None
    ]


_MUJOCO_WARP_DYNAMIC_RECORD_MODULES = frozenset({"mujoco_warp._src.smooth"})


def _mujoco_warp_max_constraint_row_width(mj_model: MjModel) -> int:
    """Return a model-derived upper bound for sparse constraint row width."""
    nv = int(mj_model.nv)
    if nv == 0:
        return 0

    chain_widths = []
    seen_welded_bodies = set()
    for body_id in range(int(mj_model.nbody)):
        welded_body_id = int(mj_model.body_weldid[body_id])
        if welded_body_id in seen_welded_bodies:
            continue
        seen_welded_bodies.add(welded_body_id)

        dof_id = int(mj_model.body_dofadr[welded_body_id]) + int(mj_model.body_dofnum[welded_body_id]) - 1
        width = 0
        while dof_id >= 0:
            width += 1
            dof_id = int(mj_model.dof_parentid[dof_id])
        chain_widths.append(width)

    chain_widths.sort(reverse=True)
    body_pair_width = sum(chain_widths[:2])
    tendon_pair_width = 2 * max((int(width) for width in mj_model.ten_J_rownnz), default=0)
    flex_width = max((int(width) for width in mj_model.flexedge_J_rownnz), default=0)

    # Actuator moment rows are state-dependent for some transmission types, so
    # retain the full-DOF bound whenever the model contains actuators.
    actuator_width = nv if int(mj_model.nu) > 0 else 0
    return min(nv, max(1, body_pair_width, tendon_pair_width, flex_width, actuator_width))


def _mujoco_warp_deterministic_max_records(mj_model: MjModel, mjw_data: MjWarpData) -> int:
    """Compute a safe per-thread deterministic atomic record bound."""
    # Generated dense contact-Jacobian kernels let one thread accumulate a
    # record for every allocated constraint row. Sparse Hessian kernels can
    # visit every element in a constraint row's lower-triangular product.
    constraint_records = int(mjw_data.njmax)
    row_width = _mujoco_warp_max_constraint_row_width(mj_model)
    hessian_records = row_width * (row_width + 1) // 2

    # Spatial tendon kernels walk both endpoint chains for every path segment.
    # Tendon armature kernels can write both halves of a dense matrix, which is
    # the one-segment lower bound used here for fixed tendons.
    tendon_records = 0
    for path_size, row_width in zip(mj_model.tendon_num, mj_model.ten_J_rownnz, strict=True):
        segment_count = max(int(path_size) - 1, 1)
        tendon_records = max(tendon_records, 2 * segment_count * int(row_width))

    return max(1, constraint_records, hessian_records, tendon_records)


def _mesh_scale_key(mesh: Mesh, scale: np.ndarray) -> tuple[int, tuple[float, float, float]]:
    return id(mesh), tuple(float(s) for s in scale)


def _prepare_mujoco_mesh(mesh: Mesh, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """Prepare one Newton mesh asset for MuJoCo compilation."""
    vertices = mesh.vertices * scale
    indices = mesh.indices.flatten()
    maxhullvert = mesh.maxhullvert
    extent_axis = vertices.max(axis=0) - vertices.min(axis=0)
    is_planar = _mujoco_mesh_vertices_are_planar(vertices, extent_axis)
    if is_planar:
        vertices, indices, maxhullvert = _make_nonplanar_mujoco_mesh(vertices, indices, maxhullvert, extent_axis)
    return vertices, indices, maxhullvert, is_planar


def _mujoco_mesh_vertices_are_planar(
    vertices: np.ndarray, extent_axis: np.ndarray | None = None, eps: float = 1.0e-6
) -> bool:
    if len(vertices) < 3:
        return False

    vertices = np.asarray(vertices)
    if extent_axis is None:
        extent_axis = vertices.max(axis=0) - vertices.min(axis=0)
    extent_axis = np.asarray(extent_axis)
    tolerance = eps * max(float(np.linalg.norm(extent_axis)), eps)
    if np.all(extent_axis <= tolerance):
        return True
    if np.any(extent_axis <= tolerance):
        return True

    if len(vertices) > 64:
        sample_indices = np.linspace(0, len(vertices) - 1, 64, dtype=np.int32)
        if not _points_are_planar(vertices[sample_indices], tolerance):
            return False

    return _points_are_planar(vertices, tolerance)


def _points_are_planar(points: np.ndarray, tolerance: float) -> bool:
    p0 = points[0]
    offsets = points - p0
    distances_from_p0 = np.linalg.norm(offsets, axis=1)
    p1 = int(np.argmax(distances_from_p0))
    line_length = distances_from_p0[p1]
    if line_length <= tolerance:
        return True

    line = points[p1] - p0
    cross = np.cross(offsets, line)
    cross_lengths = np.linalg.norm(cross, axis=1)
    p2 = int(np.argmax(cross_lengths))
    normal_length = cross_lengths[p2]
    if normal_length <= tolerance * line_length:
        return True

    normal = cross[p2] / normal_length
    plane_distances = np.abs(offsets @ normal)
    return bool(np.all(plane_distances <= tolerance))


def _make_nonplanar_mujoco_mesh(
    vertices: np.ndarray,
    indices: np.ndarray,
    maxhullvert: int,
    extent_axis: np.ndarray | None = None,
    eps: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray, int]:
    vertices = np.asarray(vertices).reshape(-1, 3)
    indices = np.asarray(indices, dtype=np.int32).flatten()
    if len(vertices) < 3 or indices.size < 3 or indices.size % 3 != 0:
        raise ValueError("Unable to build a temporary non-planar MuJoCo mesh from invalid mesh data")

    if extent_axis is None:
        extent_axis = vertices.max(axis=0) - vertices.min(axis=0)
    extent_axis = np.asarray(extent_axis)
    extent = float(np.linalg.norm(extent_axis))
    edge_tolerance = eps * max(extent, eps)
    axis = int(np.argmin(extent_axis))
    if extent_axis[axis] <= edge_tolerance:
        normal = np.zeros(3, dtype=np.float64)
        normal[axis] = 1.0
    else:
        vertices64 = np.asarray(vertices, dtype=np.float64)
        centered = vertices64 - vertices64.mean(axis=0)
        _, singular_values, vh = np.linalg.svd(centered, full_matrices=True)
        largest = float(singular_values[0]) if singular_values.size else 0.0
        if int(np.count_nonzero(singular_values > eps * max(largest, eps))) >= 3:
            return vertices, indices, maxhullvert

        normal = vh[-1]
        normal_length = np.linalg.norm(normal)
        if normal_length <= eps:
            raise ValueError("Unable to build a temporary non-planar MuJoCo mesh from coincident vertices")
        normal /= normal_length

    apex = vertices.mean(axis=0, dtype=np.float64) + normal * max(1.0e-3, 1.0e-3 * extent)

    edge = None
    for tri in indices.reshape(-1, 3):
        for edge_indices in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            a, b = int(edge_indices[0]), int(edge_indices[1])
            if np.linalg.norm(vertices[a] - vertices[b]) > edge_tolerance:
                edge = (a, b)
                break
        if edge is not None:
            break
    if edge is None:
        raise ValueError("Unable to build a temporary non-planar MuJoCo mesh without a non-degenerate edge")

    apex_index = len(vertices)
    inflated_vertices = np.vstack((vertices, apex))
    inflated_indices = np.concatenate((indices, np.array([edge[0], edge[1], apex_index], dtype=np.int32)))
    return inflated_vertices, inflated_indices, max(maxhullvert, 4)


class SolverMuJoCo(SolverBase, CouplingInterface):
    """
    This solver provides an interface to simulate physics using the `MuJoCo <https://github.com/google-deepmind/mujoco>`_ physics engine,
    optimized with GPU acceleration through `mujoco_warp <https://github.com/google-deepmind/mujoco_warp>`_. It supports both MuJoCo and
    mujoco_warp backends, enabling efficient simulation of articulated systems with
    contacts and constraints.

    .. note::

        - This solver requires `mujoco_warp`_ and its dependencies to be installed.
        - For installation instructions, see the `mujoco_warp`_ repository.
        - ``shape_collision_radius`` from Newton models is not used by MuJoCo. Instead, MuJoCo computes
          bounding sphere radii (``geom_rbound``) internally based on the geometry definition.

    Joint support:
        - Supported joint types: PRISMATIC, REVOLUTE, BALL, FIXED, FREE, D6.
          DISTANCE and ROD joints are not supported.
        - :attr:`~newton.Model.joint_armature`, :attr:`~newton.Model.joint_friction`,
          :attr:`~newton.Model.joint_effort_limit`, :attr:`~newton.Model.joint_limit_ke`/:attr:`~newton.Model.joint_limit_kd`,
          :attr:`~newton.Model.joint_target_ke`/:attr:`~newton.Model.joint_target_kd`,
          :attr:`~newton.Model.joint_target_mode`, and :attr:`~newton.Control.joint_f` are supported.
        - Equality constraints (CONNECT, WELD, JOINT) and mimic constraints (REVOLUTE and PRISMATIC only) are supported.
        - :attr:`~newton.Model.joint_velocity_limit` and :attr:`~newton.Model.joint_enabled`
          are not supported.

        See :ref:`Joint feature support` for the full comparison across solvers.

    Example
    -------

    .. code-block:: python

        solver = newton.solvers.SolverMuJoCo(model)

        # simulation loop
        for i in range(100):
            solver.step(state_in, state_out, control, contacts, dt)
            state_in, state_out = state_out, state_in

    Debugging
    ---------

    To debug the SolverMuJoCo, you can save the MuJoCo model that is created from the :class:`newton.Model` in the constructor of the SolverMuJoCo:

    .. code-block:: python

        solver = newton.solvers.SolverMuJoCo(model, save_to_mjcf="model.xml")

    This will save the MuJoCo model as an MJCF file, which can be opened in the MuJoCo simulator.

    It is also possible to visualize the simulation running in the SolverMuJoCo through MuJoCo's own viewer.
    This may help to debug the simulation and see how the MuJoCo model looks like when it is created from the Newton model.

    .. code-block:: python

        import newton

        solver = newton.solvers.SolverMuJoCo(model)

        for _ in range(num_frames):
            # step the solver
            solver.step(state_in, state_out, control, contacts, dt)
            state_in, state_out = state_out, state_in

            solver.render_mujoco_viewer()
    """

    EqType = _EqType
    """MuJoCo equality constraint type."""

    class CtrlSource(IntEnum):
        """Control source for MuJoCo actuators.

        Determines where an actuator gets its control input from:

        - :attr:`JOINT_TARGET`: Maps from Newton's :attr:`~newton.Control.joint_target_q`/:attr:`~newton.Control.joint_target_qd` arrays.
        - :attr:`CTRL_DIRECT`: Uses ``control.mujoco.ctrl`` directly (for MuJoCo-native control)
        """

        JOINT_TARGET = 0
        CTRL_DIRECT = 1

    class CtrlType(IntEnum):
        """Control type for MuJoCo actuators.

        For :attr:`~newton.solvers.SolverMuJoCo.CtrlSource.JOINT_TARGET` mode, determines which target array to read from:

        - :attr:`POSITION`: Maps from :attr:`~newton.Control.joint_target_q`, syncs gains from
          :attr:`~newton.Model.joint_target_ke`. For :attr:`~newton.JointTargetMode.POSITION`-only actuators,
          also syncs damping from :attr:`~newton.Model.joint_target_kd`. For
          :attr:`~newton.JointTargetMode.POSITION_VELOCITY` mode, kd is handled by the separate velocity actuator.
        - :attr:`VELOCITY`: Maps from :attr:`~newton.Control.joint_target_qd`, syncs gains from :attr:`~newton.Model.joint_target_kd`
        - :attr:`GENERAL`: Used with :attr:`~newton.solvers.SolverMuJoCo.CtrlSource.CTRL_DIRECT` mode for motor/general actuators
        """

        POSITION = 0
        VELOCITY = 1
        GENERAL = 2

    class TrnType(IntEnum):
        """Transmission type values for MuJoCo actuators."""

        UNDEFINED = -1

        JOINT = 0
        JOINT_IN_PARENT = 1
        TENDON = 2
        SITE = 3
        BODY = 4
        SLIDERCRANK = 5

    class SleepPolicy(IntEnum):
        """Sleep policy for a MuJoCo kinematic tree."""

        AUTO = 0
        """Let the MuJoCo compiler choose whether the tree can sleep."""
        NEVER = 3
        """Prevent the tree from sleeping."""
        ALLOWED = 4
        """Allow the tree to sleep."""
        INIT = 5
        """Initialize the tree asleep."""

    class MeshVariantSet:
        """Fixed-topology rigid-object builders selectable at runtime.

        Args:
            name: Set name.
            shape_indices: Target Newton shape indices, shape ``[world_count, shape_count]``.
            variant_builders: One-body source builders, in variant-index order.
            initial_variant_ids: Initial variant per world, shape ``[world_count]``.
            source_shape_indices: Optional compiled resource shapes, shape
                ``[variant_count, shape_count]``. When omitted, every variant must
                be present in an initial world.
            inertia_diagonal_offset: Value [kg·m²] added to finalized body inertia.
        """

        def __init__(
            self,
            *,
            name: str,
            shape_indices: Sequence[Sequence[int]],
            variant_builders: Sequence[ModelBuilder],
            initial_variant_ids: Sequence[int] | None = None,
            source_shape_indices: Sequence[Sequence[int]] | None = None,
            inertia_diagonal_offset: float = 0.0,
        ):
            if not math.isfinite(inertia_diagonal_offset) or inertia_diagonal_offset < 0.0:
                raise ValueError("inertia_diagonal_offset must be finite and non-negative")
            self.name = name
            self.shape_indices = np.asarray(shape_indices, dtype=np.int32)
            self.variant_builders = tuple(variant_builders)
            self.inertia_diagonal_offset = float(inertia_diagonal_offset)
            self.initial_variant_ids = (
                np.zeros(self.shape_indices.shape[0], dtype=np.int32)
                if initial_variant_ids is None
                else np.asarray(initial_variant_ids, dtype=np.int32)
            )
            self.source_shape_indices = (
                None if source_shape_indices is None else np.asarray(source_shape_indices, dtype=np.int32)
            )

    @dataclass
    class _MeshVariantBank:
        """Device data for one mesh variant set."""

        shape_indices: wp.array2d[wp.int32]
        geom_indices: wp.array[wp.int32]
        mj_body_index: int
        mj_dof_index: int
        shapes: wp.array2d[MeshVariantShape]
        bodies: wp.array[MeshVariantBody]
        variant_ids: wp.array[wp.int32]

    # Class variables to cache the imported modules
    _mujoco = None
    _mujoco_warp = None
    _versions_checked = False
    _convert_mjw_contacts_to_newton_kernel = None
    _generated_kernel_deterministic_options: tuple[wp.DeterministicMode, int] | None = None

    @classmethod
    def import_mujoco(cls):
        """Import the MuJoCo Warp dependencies and cache them as class variables."""
        if cls._mujoco is None or cls._mujoco_warp is None:
            try:
                with warnings.catch_warnings():
                    # Set a filter to make all ImportWarnings "always" appear
                    # This is useful to debug import errors on Windows, for example
                    warnings.simplefilter("always", category=ImportWarning)

                    import mujoco
                    import mujoco_warp

                    cls._mujoco = mujoco
                    cls._mujoco_warp = mujoco_warp
            except ImportError as e:
                raise ImportError(
                    "MuJoCo backend not installed. Please refer to https://github.com/google-deepmind/mujoco_warp for installation instructions."
                ) from e
        if not cls._versions_checked:
            try:
                _warn_if_mujoco_versions_mismatch(cls._mujoco, cls._mujoco_warp)
            except Exception:
                pass
            cls._versions_checked = True
        return cls._mujoco, cls._mujoco_warp

    def _validate_mesh_variant_definitions(self, model: Model) -> None:
        """Validate the rigid-object topology needed by the indexed update path."""
        names = set()
        shape_world = model.shape_world.numpy()
        shape_type = model.shape_type.numpy()
        shape_body = model.shape_body.numpy()
        shape_flags = model.shape_flags.numpy()
        body_world = model.body_world.numpy()
        claimed_bodies = set()

        for definition in self._mesh_variant_definitions:
            if not definition.name or definition.name in names:
                raise ValueError(f"Mesh variant set name must be non-empty and unique: {definition.name!r}")
            names.add(definition.name)
            if definition.shape_indices.ndim != 2 or definition.shape_indices.shape[0] != model.world_count:
                raise ValueError(f"Mesh variant {definition.name!r} has invalid shape indices")
            if definition.shape_indices.shape[1] == 0:
                raise ValueError(f"Mesh variant {definition.name!r} has invalid rigid-object topology")
            if len(definition.variant_builders) < 2:
                raise ValueError(f"Mesh variant {definition.name!r} needs at least two source builders")
            if definition.initial_variant_ids.shape != (model.world_count,):
                raise ValueError(f"Mesh variant {definition.name!r} has invalid initial indices")
            if np.any(
                (definition.initial_variant_ids < 0)
                | (definition.initial_variant_ids >= len(definition.variant_builders))
            ):
                raise ValueError(f"Mesh variant {definition.name!r} has an initial index outside its source range")
            if definition.source_shape_indices is None and np.any(
                np.bincount(definition.initial_variant_ids, minlength=len(definition.variant_builders)) == 0
            ):
                raise ValueError(f"Mesh variant {definition.name!r} needs every source in an initial world")

            shape_indices = definition.shape_indices
            if np.any((shape_indices < 0) | (shape_indices >= model.shape_count)):
                raise ValueError(f"Mesh variant {definition.name!r} targets an invalid model index")
            if any(len(set(row)) != shape_indices.shape[1] for row in shape_indices):
                raise ValueError(f"Mesh variant {definition.name!r} repeats a target shape")
            body_indices = shape_body[shape_indices[:, 0]]
            if np.any(body_indices < 0) or np.any(shape_body[shape_indices] != body_indices[:, None]):
                raise ValueError(f"Mesh variant {definition.name!r} shapes must belong to one body per world")
            overlap = claimed_bodies.intersection(int(body) for body in body_indices)
            if overlap:
                raise ValueError(f"Mesh variant {definition.name!r} shares body ownership with another set")
            claimed_bodies.update(int(body) for body in body_indices)
            for world in range(model.world_count):
                if np.any(shape_world[shape_indices[world]] != world) or body_world[body_indices[world]] != world:
                    raise ValueError(f"Mesh variant {definition.name!r} crosses world boundaries")
            if not np.all(np.isin(shape_type[shape_indices], (GeoType.MESH, GeoType.CONVEX_MESH))):
                raise ValueError(f"Mesh variant {definition.name!r} targets a non-mesh shape")
            if not np.all(shape_flags[shape_indices] & int(ShapeFlags.COLLIDE_SHAPES)):
                raise ValueError(f"Mesh variant {definition.name!r} targets a non-collision shape")

            for builder in definition.variant_builders:
                if builder.body_count != 1:
                    raise ValueError(f"Mesh variant {definition.name!r} sources must contain one body")
                shapes = self._mesh_variant_shape_indices(builder)
                if len(shapes) != shape_indices.shape[1]:
                    raise ValueError(f"Mesh variant {definition.name!r} changes collision-shape topology")
            if definition.source_shape_indices is not None:
                expected_shape = (len(definition.variant_builders), shape_indices.shape[1])
                if definition.source_shape_indices.shape != expected_shape:
                    raise ValueError(f"Mesh variant {definition.name!r} has invalid source shape indices")
                if np.any(
                    (definition.source_shape_indices < 0) | (definition.source_shape_indices >= model.shape_count)
                ):
                    raise ValueError(f"Mesh variant {definition.name!r} targets an invalid source shape")
                if not np.all(
                    np.isin(shape_type[definition.source_shape_indices], (GeoType.MESH, GeoType.CONVEX_MESH))
                ):
                    raise ValueError(f"Mesh variant {definition.name!r} has a non-mesh source shape")

    @staticmethod
    def _mesh_variant_shape_indices(builder: ModelBuilder) -> tuple[int, ...]:
        shapes = tuple(
            shape
            for shape in builder.body_shapes[0]
            if int(builder.shape_flags[shape]) & int(ShapeFlags.COLLIDE_SHAPES)
        )
        if not shapes or any(
            builder.shape_type[shape] not in (GeoType.MESH, GeoType.CONVEX_MESH)
            or not isinstance(builder.shape_source[shape], Mesh)
            for shape in shapes
        ):
            raise ValueError("Mesh variants require mesh-only collision geometry")
        return shapes

    @staticmethod
    def _mesh_variant_body(builder: ModelBuilder) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
        """Return finalized mass, inverse mass, COM, inertia, and inverse inertia."""
        mass, inertia, _ = verify_and_correct_inertia(
            builder.body_mass[0],
            builder.body_inertia[0],
            builder.balance_inertia,
            builder.bound_mass,
            builder.bound_inertia,
            builder.body_label[0],
        )
        inertia = np.asarray(inertia, dtype=np.float64).reshape(3, 3)
        inv_mass = 1.0 / mass if mass > 0.0 else 0.0
        inv_inertia = np.linalg.inv(inertia) if np.any(inertia) else inertia
        return mass, inv_mass, np.asarray(builder.body_com[0], dtype=np.float64), inertia, inv_inertia

    def _add_mesh_variant_assets(self, spec: Any, model: Model, shape_mapping: dict[int, str]) -> None:
        """Add every candidate mesh to the MuJoCo asset table."""
        if not self._mesh_variant_definitions:
            return

        mesh_asset_names: dict[tuple[int, tuple[float, float, float]], str] = {}
        shape_scale = model.shape_scale.numpy()
        shape_type = model.shape_type.numpy()
        for shape, name in shape_mapping.items():
            if shape_type[shape] in (GeoType.MESH, GeoType.CONVEX_MESH):
                mesh_asset_names[_mesh_scale_key(model.shape_source[shape], shape_scale[shape])] = name

        for set_index, definition in enumerate(self._mesh_variant_definitions):
            asset_rows = []
            for variant_index, builder in enumerate(definition.variant_builders):
                names: list[str] = []
                for shape_slot, shape in enumerate(self._mesh_variant_shape_indices(builder)):
                    mesh = builder.shape_source[shape]
                    scale = np.asarray(builder.shape_scale[shape], dtype=np.float32)
                    key = _mesh_scale_key(mesh, scale)
                    asset_name = mesh_asset_names.get(key)
                    if asset_name is None:
                        asset_name = f"mesh_variant_{set_index}_{variant_index}_{shape_slot}"
                        vertices, indices, maxhullvert, is_planar = _prepare_mujoco_mesh(mesh, scale)
                        if is_planar and self._use_mujoco_contacts:
                            raise ValueError(f"Mesh variant {definition.name!r} contains a planar collider")
                        spec.add_mesh(
                            name=asset_name,
                            uservert=vertices.flatten(),
                            userface=indices.flatten(),
                            maxhullvert=maxhullvert,
                        )
                        mesh_asset_names[key] = asset_name
                    names.append(asset_name)
                asset_rows.append(tuple(names))
            self._mesh_variant_asset_names[definition.name] = tuple(asset_rows)

    def _create_mesh_variant_banks(self, model: Model) -> None:
        """Pack compiled geometry and inertial rows for device selection."""
        if not self._mesh_variant_definitions:
            return

        mujoco = self._mujoco
        geom_to_shape = self.mjc_geom_to_newton_shape.numpy()
        body_to_newton = self.mjc_body_to_newton.numpy()
        shape_body = model.shape_body.numpy()
        shape_scales = model.shape_scale.numpy()
        shape_transforms = model.shape_transform.numpy()
        shape_collision_radii = model.shape_collision_radius.numpy()
        shape_source_ptrs = model.shape_source_ptr.numpy()
        shape_mesh_properties = model._shape_mesh_properties.numpy()
        shape_sdf_indices = model._shape_sdf_index.numpy()
        shape_edge_ranges = model.shape_edge_range.numpy()
        shape_collision_aabb_lowers = model.shape_collision_aabb_lower.numpy()
        shape_collision_aabb_uppers = model.shape_collision_aabb_upper.numpy()
        shape_voxel_resolutions = model._shape_voxel_resolution.numpy()
        body_masses = model.body_mass.numpy()
        body_inv_masses = model.body_inv_mass.numpy()
        body_coms = model.body_com.numpy()
        body_inertias = model.body_inertia.numpy()
        body_inv_inertias = model.body_inv_inertia.numpy()
        device = model.device
        for definition in self._mesh_variant_definitions:
            geom_indices = np.asarray(
                [np.flatnonzero(geom_to_shape[0] == shape).item() for shape in definition.shape_indices[0]],
                dtype=np.int32,
            )
            for world in range(model.world_count):
                if not np.array_equal(geom_to_shape[world, geom_indices], definition.shape_indices[world]):
                    raise ValueError(f"Mesh variant {definition.name!r} does not preserve shape ordering")

            body_indices = shape_body[definition.shape_indices[:, 0]]
            mj_body = int(self.mj_model.geom_bodyid[geom_indices[0]])
            if np.any(self.mj_model.geom_bodyid[geom_indices] != mj_body) or not np.array_equal(
                body_to_newton[:, mj_body], body_indices
            ):
                raise ValueError(f"Mesh variant {definition.name!r} does not preserve body ordering")
            if self.mj_model.body_parentid[mj_body] != 0 or np.any(self.mj_model.body_parentid[1:] == mj_body):
                raise ValueError(f"Mesh variant {definition.name!r} must target a root leaf body")

            joint_count = int(self.mj_model.body_jntnum[mj_body])
            if joint_count == 0:
                mj_dof = -1
            else:
                joint = int(self.mj_model.body_jntadr[mj_body])
                if joint_count != 1 or self.mj_model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
                    raise ValueError(f"Mesh variant {definition.name!r} must target a free or jointless body")
                mj_dof = int(self.mj_model.jnt_dofadr[joint])

            shape_rows = np.zeros(
                (len(definition.variant_builders), definition.shape_indices.shape[1]),
                dtype=MeshVariantShape.numpy_dtype(),
            )
            body_rows = np.zeros(len(definition.variant_builders), dtype=MeshVariantBody.numpy_dtype())
            for variant, (builder, asset_names) in enumerate(
                zip(definition.variant_builders, self._mesh_variant_asset_names[definition.name], strict=True)
            ):
                if definition.source_shape_indices is None:
                    source_world = int(np.flatnonzero(definition.initial_variant_ids == variant)[0])
                    source_shapes = definition.shape_indices[source_world]
                else:
                    source_world = None
                    source_shapes = definition.source_shape_indices[variant]
                for slot, (builder_shape, source_shape, asset_name) in enumerate(
                    zip(self._mesh_variant_shape_indices(builder), source_shapes, asset_names, strict=True)
                ):
                    if hash(model.shape_source[source_shape]) != hash(
                        builder.shape_source[builder_shape]
                    ) or not np.allclose(shape_scales[source_shape], builder.shape_scale[builder_shape]):
                        raise ValueError(
                            f"Mesh variant {definition.name!r} initial world {source_world} does not contain source {variant}"
                        )
                    dataid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_MESH, asset_name)
                    start = int(self.mj_model.mesh_vertadr[dataid])
                    vertices = self.mj_model.mesh_vert[start : start + int(self.mj_model.mesh_vertnum[dataid])]
                    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
                    size = np.maximum(np.abs(lower), np.abs(upper))
                    shape_rows["source_ptr"][variant, slot] = shape_source_ptrs[source_shape]
                    shape_rows["mesh_properties"][variant, slot] = shape_mesh_properties[source_shape]
                    shape_rows["sdf_index"][variant, slot] = shape_sdf_indices[source_shape]
                    shape_rows["edge_range"][variant, slot] = shape_edge_ranges[source_shape]
                    shape_rows["collision_aabb_lower"][variant, slot] = shape_collision_aabb_lowers[source_shape]
                    shape_rows["collision_aabb_upper"][variant, slot] = shape_collision_aabb_uppers[source_shape]
                    shape_rows["voxel_resolution"][variant, slot] = shape_voxel_resolutions[source_shape]
                    shape_rows["dataid"][variant, slot] = dataid
                    shape_rows["size"][variant, slot] = size
                    shape_rows["rbound"][variant, slot] = np.linalg.norm(size)
                    shape_rows["aabb_center"][variant, slot] = 0.5 * (lower + upper)
                    shape_rows["aabb_size"][variant, slot] = 0.5 * (upper - lower)
                    shape_rows["scale"][variant, slot] = shape_scales[source_shape]
                    shape_rows["xform"][variant, slot] = shape_transforms[source_shape]
                    shape_rows["collision_radius"][variant, slot] = shape_collision_radii[source_shape]
                    xform = np.asarray(shape_transforms[source_shape], dtype=np.float64)
                    xquat = xform[[6, 3, 4, 5]]
                    geom_quat = np.empty(4)
                    geom_pos = np.empty(3)
                    mujoco.mju_mulQuat(geom_quat, xquat, self.mj_model.mesh_quat[dataid])
                    mujoco.mju_rotVecQuat(geom_pos, self.mj_model.mesh_pos[dataid], xquat)
                    shape_rows["geom_pos"][variant, slot] = xform[:3] + geom_pos
                    shape_rows["geom_quat"][variant, slot] = geom_quat

                if source_world is None:
                    mass, inv_mass, com, inertia, inv_inertia = self._mesh_variant_body(builder)
                else:
                    source_body = body_indices[source_world]
                    mass = float(body_masses[source_body])
                    inv_mass = float(body_inv_masses[source_body])
                    com = np.asarray(body_coms[source_body], dtype=np.float64)
                    inertia = np.asarray(body_inertias[source_body], dtype=np.float64).reshape(3, 3)
                    inv_inertia = np.asarray(body_inv_inertias[source_body], dtype=np.float64).reshape(3, 3)
                if definition.inertia_diagonal_offset:
                    inertia = inertia.copy()
                    inertia.flat[::4] += definition.inertia_diagonal_offset
                    inv_inertia = np.linalg.inv(inertia)
                if mj_dof < 0:
                    inv_mass = 0.0
                    inv_inertia = np.zeros_like(inertia)
                mj_inertia, eigvec, mj_iquat = np.empty(3), np.empty(9), np.empty(4)
                mujoco.mju_eig3(mj_inertia, eigvec, mj_iquat, inertia.ravel())
                if mj_dof >= 0:
                    if mass <= 0.0 or np.any(mj_inertia <= 0.0):
                        raise ValueError(f"Mesh variant {definition.name!r} has non-positive mass or inertia")
                    rotational_invweight = np.trace(inv_inertia) / 3.0
                else:
                    rotational_invweight = 0.0
                cross = np.array(((0.0, -com[2], com[1]), (com[2], 0.0, -com[0]), (-com[1], com[0], 0.0)))
                linear_invweight = 1.0 / mass + np.trace(cross @ inv_inertia @ cross.T) / 3.0 if mj_dof >= 0 else 0.0
                body_rows["mass"][variant] = mass
                body_rows["inv_mass"][variant] = inv_mass
                body_rows["com"][variant] = com
                body_rows["inertia"][variant] = inertia
                body_rows["inv_inertia"][variant] = inv_inertia
                body_rows["mj_inertia"][variant] = mj_inertia
                body_rows["mj_iquat"][variant] = mj_iquat
                body_rows["body_invweight0"][variant] = (1.0 / mass, rotational_invweight) if mj_dof >= 0 else 0.0
                body_rows["dof_invweight0"][variant] = (linear_invweight,) * 3 + (rotational_invweight,) * 3

            bank = SolverMuJoCo._MeshVariantBank(
                shape_indices=wp.array(definition.shape_indices, dtype=wp.int32, device=device),
                geom_indices=wp.array(geom_indices, dtype=wp.int32, device=device),
                mj_body_index=mj_body,
                mj_dof_index=mj_dof,
                shapes=wp.array(shape_rows, dtype=MeshVariantShape, device=device),
                bodies=wp.array(body_rows, dtype=MeshVariantBody, device=device),
                variant_ids=wp.zeros(model.world_count, dtype=wp.int32, device=device),
            )
            self._mesh_variant_banks[definition.name] = bank

    def _prepare_generated_kernels(self) -> None:
        """Invalidate MJWarp's generated kernels when determinism changes."""
        options = (self._deterministic, self._deterministic_max_records)
        if SolverMuJoCo._generated_kernel_deterministic_options == options:
            return

        # MJWarp's factory cache key does not include Warp module options.
        # Recreate unique kernels so they inherit this solver's configuration.
        from mujoco_warp._src import warp_util

        warp_util._KERNEL_CACHE.clear()
        SolverMuJoCo._generated_kernel_deterministic_options = options

    def _set_mujoco_warp_module_options(self) -> None:
        """Configure loaded shared modules without overriding code-generated bounds."""
        for module in [*_mujoco_warp_deterministic_modules(), kernels]:
            max_records = (
                self._deterministic_max_records if module.__name__ in _MUJOCO_WARP_DYNAMIC_RECORD_MODULES else 0
            )
            options = {
                "deterministic": self._deterministic,
                "deterministic_max_records": max_records,
            }
            self._set_module_options(options, module=module)

    @staticmethod
    def _parse_integrator(value: str | int, context: dict[str, Any] | None = None) -> int:
        """Parse integrator option: Euler=0, RK4=1, implicit=2, implicitfast=3."""
        return SolverMuJoCo._parse_named_int(value, {"euler": 0, "rk4": 1, "implicit": 2, "implicitfast": 3})

    @staticmethod
    def _parse_solver(value: str | int, context: dict[str, Any] | None = None) -> int:
        """Parse solver option: CG=1, Newton=2. PGS (0) is not supported."""
        return SolverMuJoCo._parse_named_int(value, {"cg": 1, "newton": 2})

    @staticmethod
    def _parse_cone(value: str | int, context: dict[str, Any] | None = None) -> int:
        """Parse cone option: pyramidal=0, elliptic=1."""
        return SolverMuJoCo._parse_named_int(value, {"pyramidal": 0, "elliptic": 1})

    @staticmethod
    def _parse_jacobian(value: str | int, context: dict[str, Any] | None = None) -> int:
        """Parse jacobian option: dense=0, sparse=1, auto=2."""
        return SolverMuJoCo._parse_named_int(value, {"dense": 0, "sparse": 1, "auto": 2})

    @staticmethod
    def _parse_sleep_policy(value: str | int, context: dict[str, Any] | None = None) -> int:
        """Parse body sleep policy: auto=0, never=3, allowed=4, init=5."""
        return SolverMuJoCo._parse_named_int(
            value,
            {
                "auto": SolverMuJoCo.SleepPolicy.AUTO,
                "never": SolverMuJoCo.SleepPolicy.NEVER,
                "allowed": SolverMuJoCo.SleepPolicy.ALLOWED,
                "init": SolverMuJoCo.SleepPolicy.INIT,
            },
        )

    @staticmethod
    def _parse_enable_disable(value: str | bool | int, context: dict[str, Any] | None = None) -> bool:
        """Parse MuJoCo enable/disable values."""
        if isinstance(value, bool | np.bool_):
            return bool(value)
        if isinstance(value, int | np.integer):
            return bool(value)
        parsed = str(value).lower().strip()
        if parsed in ("enable", "true", "1"):
            return True
        if parsed in ("disable", "false", "0"):
            return False
        raise ValueError(f"Expected enable or disable, got {value!r}.")

    @staticmethod
    def _parse_named_int(value: str | int, mapping: dict[str, int], fallback_on_unknown: int | None = None) -> int:
        """Parse string-valued enums to int, otherwise return int(value)."""
        if isinstance(value, int | np.integer):
            return int(value)
        lower_value = str(value).lower().strip()
        if lower_value in mapping:
            return mapping[lower_value]
        # Support MuJoCo enum string reprs like "mjtCone.mjCONE_ELLIPTIC".
        last_component = lower_value.rsplit(".", maxsplit=1)[-1]
        if last_component in mapping:
            return mapping[last_component]
        enum_suffix = last_component.rsplit("_", maxsplit=1)[-1]
        if enum_suffix in mapping:
            return mapping[enum_suffix]
        if fallback_on_unknown is not None:
            return fallback_on_unknown
        return int(lower_value)

    @staticmethod
    def _angle_value_transformer(value: str, context: dict[str, Any] | None) -> float:
        """Transform angle values from MJCF, converting deg to rad for angular joints.

        For attributes like springref and ref that represent angles,
        parses the string value and multiplies by pi/180 when use_degrees=True and joint is angular.
        """
        parsed = string_to_warp(value, wp.float32, 0.0)
        if context is not None:
            joint_type = context.get("joint_type")
            use_degrees = context.get("use_degrees", False)
            is_angular = joint_type in ["hinge", "ball"]
            if is_angular and use_degrees:
                return parsed * (np.pi / 180)
        return parsed

    @staticmethod
    def _is_mjc_actuator_prim(prim: Any, _context: dict[str, Any]) -> bool:
        """Filter for prims of type ``MjcActuator`` for USD parsing.

        This is used as the ``usd_prim_filter`` for the ``mujoco:actuator`` custom frequency.
        Returns True for USD Prim objects whose type name is ``MjcActuator``.

        Args:
            prim: The USD prim to check.
            _context: Context dictionary with parsing results (path maps, units, etc.).
                This matches the return value of :meth:`newton.ModelBuilder.add_usd`.

        Returns:
            True if the prim is an MjcActuator, False otherwise.
        """
        return prim.GetTypeName() == "MjcActuator"

    @staticmethod
    def _is_mjc_tendon_prim(prim: Any, _context: dict[str, Any]) -> bool:
        """Filter for prims of type ``MjcTendon`` for USD parsing.

        This is used as the ``usd_prim_filter`` for the ``mujoco:tendon`` custom frequency.
        Returns True for USD Prim objects whose type name is ``MjcTendon``.

        Args:
            prim: The USD prim to check.
            _context: Context dictionary with parsing results (path maps, units, etc.).
                This matches the return value of :meth:`newton.ModelBuilder.add_usd`.

        Returns:
            True if the prim is an MjcTendon, False otherwise.
        """
        return prim.GetTypeName() == "MjcTendon"

    @staticmethod
    def _parse_mjc_fixed_tendon_joint_entries(prim, builder: ModelBuilder) -> list[tuple[int, float]]:
        """Parse fixed tendon joint/coefficient entries from an MjcTendon prim.

        Returns:
            List of ``(joint_index, coef)`` entries in authored tendon path order.
        """
        tendon_type_attr = prim.GetAttribute("mjc:type")
        tendon_type = tendon_type_attr.Get() if tendon_type_attr else None
        if tendon_type is None or str(tendon_type).lower() != "fixed":
            return []

        path_rel = prim.GetRelationship("mjc:path")
        path_targets = list(path_rel.GetTargets()) if path_rel else []
        if len(path_targets) == 0:
            return []

        indices_attr = prim.GetAttribute("mjc:path:indices")
        authored_indices = indices_attr.Get() if indices_attr else None
        indices = list(authored_indices) if authored_indices is not None and len(authored_indices) > 0 else None
        if indices is None:
            # If indices are omitted, keep authored relationship order.
            indices = list(range(len(path_targets)))

        coef_attr = prim.GetAttribute("mjc:path:coef")
        authored_coef = coef_attr.Get() if coef_attr else None
        coefs = list(authored_coef) if authored_coef is not None else []

        joint_entries: list[tuple[int, float]] = []
        for i, path_idx in enumerate(indices):
            path_idx_int = int(path_idx)
            if path_idx_int < 0 or path_idx_int >= len(path_targets):
                warnings.warn(
                    f"MjcTendon {prim.GetPath()} has out-of-range mjc:path:indices entry {path_idx_int}. Skipping.",
                    stacklevel=2,
                )
                continue

            joint_path = str(path_targets[path_idx_int])
            try:
                joint_idx = builder.joint_label.index(joint_path)
            except ValueError:
                warnings.warn(
                    f"MjcTendon {prim.GetPath()} references unknown joint path {joint_path}. Skipping.",
                    stacklevel=2,
                )
                continue

            coef = float(coefs[i]) if i < len(coefs) else 1.0
            joint_entries.append((joint_idx, coef))

        return joint_entries

    @staticmethod
    def _expand_mjc_tendon_joint_rows(prim, context: dict[str, Any]) -> Iterable[dict[str, Any]]:
        """Expand one MjcTendon prim into 0..N mujoco:tendon_joint rows."""
        builder = context.get("builder")
        if not isinstance(builder, ModelBuilder):
            return []

        joint_entries = SolverMuJoCo._parse_mjc_fixed_tendon_joint_entries(prim, builder)
        return [
            {
                "mujoco:tendon_joint": joint_idx,
                "mujoco:tendon_coef": coef,
            }
            for joint_idx, coef in joint_entries
        ]

    @override
    @classmethod
    def register_custom_attributes(cls, builder: ModelBuilder) -> None:
        """
        Declare custom attributes to be allocated on the Model object within the ``mujoco`` namespace.
        Custom attributes use ``CustomAttribute.usd_attribute_name`` with the ``mjc:`` prefix (e.g. ``"mjc:condim"``)
        to leverage the MuJoCo USD schema where attributes are named ``"mjc:attr"`` rather than ``"newton:mujoco:attr"``.
        """
        # Register custom frequencies before adding attributes that use them
        # This is required as custom frequencies must be registered before use

        # Note: only attributes with usd_attribute_name defined are parsed from USD at the moment.

        _register_equality_constraint_attributes(builder)

        def parse_solreflimit_mode_usd(_value: Any, context: dict[str, Any]) -> int | None:
            prim = context.get("prim")
            if prim is None:
                return None
            solreflimit_attr = prim.GetAttribute("mjc:solreflimit")
            if solreflimit_attr is not None and solreflimit_attr.HasAuthoredValue():
                return SOLREF_MODE_RAW
            return None

        def parse_solref_mode_usd(_value: Any, context: dict[str, Any]) -> int | None:
            prim = context.get("prim")
            if prim is None:
                return None
            solref_attr = prim.GetAttribute("mjc:solref")
            if solref_attr is not None and solref_attr.HasAuthoredValue():
                return SOLREF_MODE_RAW
            return SOLREF_MODE_MJCF_DEFAULT

        # region custom frequencies
        builder.add_custom_frequency(ModelBuilder.CustomFrequency(name="pair", namespace="mujoco"))
        builder.add_custom_frequency(
            ModelBuilder.CustomFrequency(
                name="actuator",
                namespace="mujoco",
                usd_prim_filter=cls._is_mjc_actuator_prim,
            )
        )
        builder.add_custom_frequency(
            ModelBuilder.CustomFrequency(
                name="tendon",
                namespace="mujoco",
                usd_prim_filter=cls._is_mjc_tendon_prim,
            )
        )
        builder.add_custom_frequency(
            ModelBuilder.CustomFrequency(
                name="tendon_joint",
                namespace="mujoco",
                usd_prim_filter=cls._is_mjc_tendon_prim,
                usd_entry_expander=cls._expand_mjc_tendon_joint_rows,
            )
        )
        builder.add_custom_frequency(
            ModelBuilder.CustomFrequency(
                name="tendon_wrap",
                namespace="mujoco",
            )
        )
        # endregion custom frequencies

        # region geom attributes
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="site_size_is_display",
                frequency=AttributeFrequency.SHAPE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.bool,
                default=False,
                namespace="mujoco",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="contype",
                frequency=AttributeFrequency.SHAPE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int64,
                default=MUJOCO_COLLISION_MASK_UNSET,
                namespace="mujoco",
                usd_attribute_name="mjc:contype",
                mjcf_attribute_name="contype",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="conaffinity",
                frequency=AttributeFrequency.SHAPE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int64,
                default=MUJOCO_COLLISION_MASK_UNSET,
                namespace="mujoco",
                usd_attribute_name="mjc:conaffinity",
                mjcf_attribute_name="conaffinity",
            )
        )
        # Record which add_mjcf() call supplied each shape's masks. This is not
        # another mask: it prevents equal bit numbers from separate files from
        # being treated as if they were authored together.
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="collision_mask_domain",
                frequency=AttributeFrequency.SHAPE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int64,
                default=MUJOCO_COLLISION_MASK_DOMAIN_UNSET,
                namespace="mujoco",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="condim",
                frequency=AttributeFrequency.SHAPE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=3,
                namespace="mujoco",
                usd_attribute_name="mjc:condim",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="geom_group",
                frequency=AttributeFrequency.SHAPE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=0,
                namespace="mujoco",
                usd_attribute_name="mjc:group",
                mjcf_attribute_name="group",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="geom_priority",
                frequency=AttributeFrequency.SHAPE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=0,
                namespace="mujoco",
                usd_attribute_name="mjc:priority",
                mjcf_attribute_name="priority",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="geom_solimp",
                frequency=AttributeFrequency.SHAPE,
                assignment=AttributeAssignment.MODEL,
                dtype=vec5,
                default=vec5(0.9, 0.95, 0.001, 0.5, 2.0),
                namespace="mujoco",
                usd_attribute_name="mjc:solimp",
                mjcf_attribute_name="solimp",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="geom_solmix",
                frequency=AttributeFrequency.SHAPE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=1.0,
                namespace="mujoco",
                usd_attribute_name="mjc:solmix",
                mjcf_attribute_name="solmix",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="solref",
                frequency=AttributeFrequency.SHAPE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.vec2,
                # Sentinel for "not authored". MJCF/USD import fills this only
                # when a raw MuJoCo ``solref`` was explicitly authored.
                default=wp.vec2(0.0, 0.0),
                namespace="mujoco",
                usd_attribute_name="mjc:solref",
                mjcf_attribute_name="solref",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="solref_mode",
                frequency=AttributeFrequency.SHAPE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                # See docs/solvers/mujoco.rst > "Shape-material contact
                # stiffness and damping" for the three modes. Default is
                # MJCF_DEFAULT so existing builder-API ke/kd defaults keep
                # working; opt into force-space scaling by setting
                # model.mujoco.solref_mode[shape] = SOLREF_MODE_FORCE_SPACE.
                default=SOLREF_MODE_MJCF_DEFAULT,
                namespace="mujoco",
                usd_attribute_name="*",
                usd_value_transformer=parse_solref_mode_usd,
            )
        )
        # endregion geom attributes

        # region body and joint attributes
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="limit_margin",
                frequency=AttributeFrequency.JOINT_DOF,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.0,
                namespace="mujoco",
                usd_attribute_name="mjc:margin",
                mjcf_attribute_name="margin",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="solimplimit",
                frequency=AttributeFrequency.JOINT_DOF,
                assignment=AttributeAssignment.MODEL,
                dtype=vec5,
                default=vec5(0.9, 0.95, 0.001, 0.5, 2.0),
                namespace="mujoco",
                usd_attribute_name="mjc:solimplimit",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="solreflimit",
                frequency=AttributeFrequency.JOINT_DOF,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.vec2,
                # Sentinel for "not authored". MJCF/USD import fills this only
                # when a raw MuJoCo solreflimit was explicitly authored.
                default=wp.vec2(0.0, 0.0),
                namespace="mujoco",
                usd_attribute_name="mjc:solreflimit",
                mjcf_attribute_name="solreflimit",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="solreflimit_mode",
                frequency=AttributeFrequency.JOINT_DOF,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                # ``mujoco.solreflimit`` needs out-of-band state because a
                # vec2 value alone cannot distinguish all required cases:
                # SOLREF_MODE_FORCE_SPACE = Newton force-space gains from joint_limit_ke/kd,
                # SOLREF_MODE_RAW = raw MuJoCo solreflimit authored/imported exactly,
                # SOLREF_MODE_MJCF_DEFAULT = implicit MJCF default until gains change.
                # The mode also lets MJCF import preserve authored
                # solreflimit="0 0", which collides with the solreflimit
                # attribute's legacy zero sentinel.
                default=SOLREF_MODE_FORCE_SPACE,
                namespace="mujoco",
                usd_attribute_name="*",
                usd_value_transformer=parse_solreflimit_mode_usd,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="solreffriction",
                frequency=AttributeFrequency.JOINT_DOF,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.vec2,
                default=wp.vec2(0.02, 1.0),
                namespace="mujoco",
                usd_attribute_name="mjc:solreffriction",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="solimpfriction",
                frequency=AttributeFrequency.JOINT_DOF,
                assignment=AttributeAssignment.MODEL,
                dtype=vec5,
                default=vec5(0.9, 0.95, 0.001, 0.5, 2.0),
                namespace="mujoco",
                usd_attribute_name="mjc:solimpfriction",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="gravcomp",
                frequency=AttributeFrequency.BODY,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.0,
                namespace="mujoco",
                usd_attribute_name="mjc:gravcomp",
                mjcf_attribute_name="gravcomp",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="sleep_policy",
                frequency=AttributeFrequency.BODY,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=int(cls.SleepPolicy.AUTO),
                namespace="mujoco",
                usd_attribute_name="mjc:sleep",
                mjcf_attribute_name="sleep",
                mjcf_value_transformer=cls._parse_sleep_policy,
                usd_value_transformer=cls._parse_sleep_policy,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="dof_passive_stiffness",
                frequency=AttributeFrequency.JOINT_DOF,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.0,
                namespace="mujoco",
                usd_attribute_name="mjc:stiffness",
                mjcf_attribute_name="stiffness",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="dof_springref",
                frequency=AttributeFrequency.JOINT_DOF,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.0,
                namespace="mujoco",
                usd_attribute_name="mjc:springref",
                mjcf_attribute_name="springref",
                mjcf_value_transformer=cls._angle_value_transformer,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="dof_ref",
                frequency=AttributeFrequency.JOINT_DOF,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.0,
                namespace="mujoco",
                usd_attribute_name="mjc:ref",
                mjcf_attribute_name="ref",
                mjcf_value_transformer=cls._angle_value_transformer,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="jnt_actgravcomp",
                frequency=AttributeFrequency.JOINT_DOF,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.bool,
                default=False,
                namespace="mujoco",
                usd_attribute_name="mjc:actuatorgravcomp",
                mjcf_attribute_name="actuatorgravcomp",
            )
        )
        # endregion body and joint attributes

        # region solver options
        # Solver options (frequency WORLD for per-world values)
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="impratio",
                frequency=AttributeFrequency.WORLD,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=1.0,
                namespace="mujoco",
                usd_attribute_name="mjc:option:impratio",
                mjcf_attribute_name="impratio",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tolerance",
                frequency=AttributeFrequency.WORLD,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=1e-8,
                namespace="mujoco",
                usd_attribute_name="mjc:option:tolerance",
                mjcf_attribute_name="tolerance",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="ls_tolerance",
                frequency=AttributeFrequency.WORLD,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.01,
                namespace="mujoco",
                usd_attribute_name="mjc:option:ls_tolerance",
                mjcf_attribute_name="ls_tolerance",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="ccd_tolerance",
                frequency=AttributeFrequency.WORLD,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=1e-6,
                namespace="mujoco",
                usd_attribute_name="mjc:option:ccd_tolerance",
                mjcf_attribute_name="ccd_tolerance",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="sleep_tolerance",
                frequency=AttributeFrequency.WORLD,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=1e-3,
                namespace="mujoco",
                usd_attribute_name="mjc:option:sleep_tolerance",
                mjcf_attribute_name="sleep_tolerance",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="enable_sleeping",
                frequency=AttributeFrequency.ONCE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.bool,
                default=False,
                namespace="mujoco",
                mjcf_attribute_name="sleep",
                mjcf_value_transformer=cls._parse_enable_disable,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="density",
                frequency=AttributeFrequency.WORLD,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.0,
                namespace="mujoco",
                usd_attribute_name="mjc:option:density",
                mjcf_attribute_name="density",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="viscosity",
                frequency=AttributeFrequency.WORLD,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.0,
                namespace="mujoco",
                usd_attribute_name="mjc:option:viscosity",
                mjcf_attribute_name="viscosity",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="wind",
                frequency=AttributeFrequency.WORLD,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.vec3,
                default=wp.vec3(0.0, 0.0, 0.0),
                namespace="mujoco",
                usd_attribute_name="mjc:option:wind",
                mjcf_attribute_name="wind",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="magnetic",
                frequency=AttributeFrequency.WORLD,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.vec3,
                default=wp.vec3(0.0, -0.5, 0.0),
                namespace="mujoco",
                usd_attribute_name="mjc:option:magnetic",
                mjcf_attribute_name="magnetic",
            )
        )

        # Solver options (frequency ONCE for single value shared across all worlds)
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="iterations",
                frequency=AttributeFrequency.ONCE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=100,
                namespace="mujoco",
                usd_attribute_name="mjc:option:iterations",
                mjcf_attribute_name="iterations",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="ls_iterations",
                frequency=AttributeFrequency.ONCE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=50,
                namespace="mujoco",
                usd_attribute_name="mjc:option:ls_iterations",
                mjcf_attribute_name="ls_iterations",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="ccd_iterations",
                frequency=AttributeFrequency.ONCE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=35,  # MuJoCo default
                namespace="mujoco",
                usd_attribute_name="mjc:option:ccd_iterations",
                mjcf_attribute_name="ccd_iterations",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="sdf_iterations",
                frequency=AttributeFrequency.ONCE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=10,
                namespace="mujoco",
                usd_attribute_name="mjc:option:sdf_iterations",
                mjcf_attribute_name="sdf_iterations",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="sdf_initpoints",
                frequency=AttributeFrequency.ONCE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=40,
                namespace="mujoco",
                usd_attribute_name="mjc:option:sdf_initpoints",
                mjcf_attribute_name="sdf_initpoints",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="integrator",
                frequency=AttributeFrequency.ONCE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=3,  # Newton default: implicitfast (not MuJoCo's 0/Euler)
                namespace="mujoco",
                usd_attribute_name="mjc:option:integrator",
                mjcf_attribute_name="integrator",
                mjcf_value_transformer=cls._parse_integrator,
                usd_value_transformer=cls._parse_integrator,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="solver",
                frequency=AttributeFrequency.ONCE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=2,  # Newton
                namespace="mujoco",
                usd_attribute_name="mjc:option:solver",
                mjcf_attribute_name="solver",
                mjcf_value_transformer=cls._parse_solver,
                usd_value_transformer=cls._parse_solver,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="cone",
                frequency=AttributeFrequency.ONCE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=0,  # pyramidal
                namespace="mujoco",
                usd_attribute_name="mjc:option:cone",
                mjcf_attribute_name="cone",
                mjcf_value_transformer=cls._parse_cone,
                usd_value_transformer=cls._parse_cone,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="jacobian",
                frequency=AttributeFrequency.ONCE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=2,  # auto
                namespace="mujoco",
                usd_attribute_name="mjc:option:jacobian",
                mjcf_attribute_name="jacobian",
                mjcf_value_transformer=cls._parse_jacobian,
                usd_value_transformer=cls._parse_jacobian,
            )
        )
        # endregion solver options

        # region pair attributes
        # --- Pair attributes (from MJCF <pair> tag) ---
        # Explicit contact pairs with custom properties. Pairs from the template world and
        # global pairs (world < 0) are used.
        # These are parsed automatically from MJCF <contact><pair> elements.
        # All pair attributes share the "mujoco:pair" custom frequency.
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="pair_world",
                frequency="mujoco:pair",
                dtype=wp.int32,
                default=0,
                namespace="mujoco",
                references="world",
                # No mjcf_attribute_name - this is set automatically during parsing
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="pair_geom1",
                frequency="mujoco:pair",
                dtype=wp.int32,
                default=-1,
                namespace="mujoco",
                references="shape",
                mjcf_attribute_name="geom1",  # Maps to shape index via geom name lookup
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="pair_geom2",
                frequency="mujoco:pair",
                dtype=wp.int32,
                default=-1,
                namespace="mujoco",
                references="shape",
                mjcf_attribute_name="geom2",  # Maps to shape index via geom name lookup
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="pair_condim",
                frequency="mujoco:pair",
                dtype=wp.int32,
                default=3,
                namespace="mujoco",
                mjcf_attribute_name="condim",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="pair_solref",
                frequency="mujoco:pair",
                dtype=wp.vec2,
                default=wp.vec2(0.02, 1.0),
                namespace="mujoco",
                mjcf_attribute_name="solref",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="pair_solreffriction",
                frequency="mujoco:pair",
                dtype=wp.vec2,
                default=wp.vec2(0.02, 1.0),
                namespace="mujoco",
                mjcf_attribute_name="solreffriction",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="pair_solimp",
                frequency="mujoco:pair",
                dtype=vec5,
                default=vec5(0.9, 0.95, 0.001, 0.5, 2.0),
                namespace="mujoco",
                mjcf_attribute_name="solimp",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="pair_margin",
                frequency="mujoco:pair",
                dtype=wp.float32,
                default=0.0,
                namespace="mujoco",
                mjcf_attribute_name="margin",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="pair_gap",
                frequency="mujoco:pair",
                dtype=wp.float32,
                default=0.0,
                namespace="mujoco",
                mjcf_attribute_name="gap",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="pair_friction",
                frequency="mujoco:pair",
                dtype=vec5,
                default=vec5(1.0, 1.0, 0.005, 0.0001, 0.0001),
                namespace="mujoco",
                mjcf_attribute_name="friction",
            )
        )
        # endregion pair attributes

        # region actuator attributes
        # --- MuJoCo General Actuator attributes (mujoco:actuator frequency) ---
        # These are used for general/motor actuators parsed from MJCF
        # All actuator attributes share the "mujoco:actuator" custom frequency.
        # Note: actuator_trnid[0] stores the target index, actuator_trntype determines its meaning (joint/tendon/site)
        def parse_actuator_enum(value: Any, mapping: dict[str, int]) -> int:
            """Parse actuator enum values, defaulting to 0 for unknown strings."""
            return int(SolverMuJoCo._parse_named_int(value, mapping, fallback_on_unknown=0))

        actuator_transmission_types = {
            "joint": int(SolverMuJoCo.TrnType.JOINT),
            "jointinparent": int(SolverMuJoCo.TrnType.JOINT_IN_PARENT),
            "tendon": int(SolverMuJoCo.TrnType.TENDON),
            "site": int(SolverMuJoCo.TrnType.SITE),
            "body": int(SolverMuJoCo.TrnType.BODY),
            "slidercrank": int(SolverMuJoCo.TrnType.SLIDERCRANK),
        }
        actuator_dynamics_types = {
            "none": _ActuatorDynamicsType.NONE,
            "integrator": _ActuatorDynamicsType.INTEGRATOR,
            "filter": _ActuatorDynamicsType.FILTER,
            "filterexact": _ActuatorDynamicsType.FILTER_EXACT,
            "muscle": _ActuatorDynamicsType.MUSCLE,
            "user": _ActuatorDynamicsType.USER,
        }
        actuator_gain_types = {
            "fixed": _ActuatorGainType.FIXED,
            "affine": _ActuatorGainType.AFFINE,
            "muscle": _ActuatorGainType.MUSCLE,
            "user": _ActuatorGainType.USER,
        }
        actuator_bias_types = {
            "none": _ActuatorBiasType.NONE,
            "affine": _ActuatorBiasType.AFFINE,
            "muscle": _ActuatorBiasType.MUSCLE,
            "user": _ActuatorBiasType.USER,
        }

        def parse_trntype(s: str, _context: dict[str, Any] | None = None) -> int:
            return parse_actuator_enum(s, actuator_transmission_types)

        def parse_dyntype(s: str, _context: dict[str, Any] | None = None) -> int:
            return parse_actuator_enum(s, actuator_dynamics_types)

        def parse_gaintype(s: str, _context: dict[str, Any] | None = None) -> int:
            return parse_actuator_enum(s, actuator_gain_types)

        def parse_biastype(s: str, _context: dict[str, Any] | None = None) -> int:
            return parse_actuator_enum(s, actuator_bias_types)

        def parse_bool(value: Any, context: dict[str, Any] | None = None) -> bool:
            """Parse MJCF/USD boolean values to bool."""
            if isinstance(value, bool):
                return value
            if isinstance(value, int | np.integer):
                return bool(value)
            s = str(value).strip().lower()
            if s == "auto":
                if context is not None:
                    prim = context.get("prim")
                    attr = context.get("attr")
                    if prim is not None and attr is not None:
                        raise NotImplementedError(
                            f"Error while parsing value '{attr.usd_attribute_name}' at prim '{prim.GetPath()}'. Auto boolean values are not supported at the moment."
                        )
                raise NotImplementedError("Auto boolean values are not supported at the moment.")
            return s in ("true", "1")

        def get_usd_range_if_authored(prim, range_attr_name: str) -> tuple[float, float] | None:
            """Return (min, max) for an authored USD range or None if no bounds are authored."""
            min_attr = prim.GetAttribute(f"{range_attr_name}:min")
            max_attr = prim.GetAttribute(f"{range_attr_name}:max")
            min_authored = bool(min_attr and min_attr.HasAuthoredValue())
            max_authored = bool(max_attr and max_attr.HasAuthoredValue())
            if not min_authored and not max_authored:
                return None

            rmin = min_attr.Get() if min_attr else None
            rmax = max_attr.Get() if max_attr else None
            # Some USD assets omit one bound and rely on schema defaults (often 0).
            # Mirror that behavior to avoid falling back to unrelated parser defaults.
            if rmin is None:
                rmin = 0.0
            if rmax is None:
                rmax = 0.0
            return float(rmin), float(rmax)

        def make_usd_range_transformer(range_attr_name: str):
            """Create a transformer that parses a USD min/max range pair."""

            def transform(_: Any, context: dict[str, Any]) -> wp.vec2 | None:
                range_vals = get_usd_range_if_authored(context["prim"], range_attr_name)
                if range_vals is None:
                    return None
                return wp.vec2(range_vals[0], range_vals[1])

            return transform

        def make_usd_has_range_transformer(range_attr_name: str):
            """Create a transformer that returns 1 when a USD range is authored."""

            def transform(_: Any, context: dict[str, Any]) -> int:
                range_vals = get_usd_range_if_authored(context["prim"], range_attr_name)
                return int(range_vals is not None)

            return transform

        def make_usd_limited_transformer(limited_attr_name: str, range_attr_name: str):
            """Create a transformer for MuJoCo tri-state limited tokens.

            The corresponding USD attributes are token-valued with allowed values
            ``"false"``, ``"true"``, and ``"auto"``. We preserve this tristate
            representation as integers ``0/1/2`` and defer any autolimits-based
            resolution to MuJoCo compilation.
            """

            def transform(_: Any, context: dict[str, Any]) -> int:
                prim = context["prim"]

                limited_attr = prim.GetAttribute(limited_attr_name)
                if limited_attr and limited_attr.HasAuthoredValue():
                    return parse_tristate(limited_attr.Get())
                # Keep MuJoCo's default tri-state semantics: omitted means "auto" (2).
                return 2

            return transform

        def _resolve_inheritrange_as_ctrlrange(prim, context: dict[str, Any]) -> tuple[float, float] | None:
            """Resolve mjc:inheritRange to a concrete (lower, upper) control range.

            Reads the target joint's limits from the builder and computes the
            control range Returns None if inheritRange is not authored, zero, or the target joint cannot be found.
            """
            inherit_attr = prim.GetAttribute("mjc:inheritRange")
            if not inherit_attr or not inherit_attr.HasAuthoredValue():
                return None
            inheritrange = float(inherit_attr.Get())
            if inheritrange <= 0:
                return None
            result = context.get("result")
            b = context.get("builder")
            if not result or not b:
                return None
            try:
                target_path = resolve_actuator_target_path(prim)
            except ValueError:
                return None
            path_joint_map = result.get("path_joint_map", {})
            joint_idx = path_joint_map.get(target_path, -1)
            if joint_idx < 0 or joint_idx >= len(b.joint_qd_start):
                return None
            dof_idx = b.joint_qd_start[joint_idx]
            if dof_idx < 0 or dof_idx >= len(b.joint_limit_lower):
                return None
            lower = b.joint_limit_lower[dof_idx]
            upper = b.joint_limit_upper[dof_idx]
            if lower >= upper:
                return None
            mean = (upper + lower) / 2.0
            radius = (upper - lower) / 2.0 * inheritrange
            return (mean - radius, mean + radius)

        def transform_ctrlrange(_: Any, context: dict[str, Any]) -> wp.vec2 | None:
            """Parse mjc:ctrlRange, falling back to inheritrange-derived range."""
            prim = context["prim"]
            range_vals = get_usd_range_if_authored(prim, "mjc:ctrlRange")
            if range_vals is not None:
                return wp.vec2(range_vals[0], range_vals[1])
            resolved = _resolve_inheritrange_as_ctrlrange(prim, context)
            if resolved is not None:
                return wp.vec2(float(resolved[0]), float(resolved[1]))
            return None

        def transform_has_ctrlrange(_: Any, context: dict[str, Any]) -> int:
            """Return 1 when ctrlRange is authored or inheritrange resolves a range."""
            prim = context["prim"]
            if get_usd_range_if_authored(prim, "mjc:ctrlRange") is not None:
                return 1
            if _resolve_inheritrange_as_ctrlrange(prim, context) is not None:
                return 1
            return 0

        def transform_ctrllimited(_: Any, context: dict[str, Any]) -> int:
            """Parse mjc:ctrlLimited, defaulting to true when inheritrange resolves."""
            prim = context["prim"]
            limited_attr = prim.GetAttribute("mjc:ctrlLimited")
            if limited_attr and limited_attr.HasAuthoredValue():
                return parse_tristate(limited_attr.Get())
            if _resolve_inheritrange_as_ctrlrange(prim, context) is not None:
                return 1
            return 2

        def resolve_prim_name(_: str, context: dict[str, Any]) -> str:
            """Return the USD prim path as the attribute value.

            Used as a ``usd_value_transformer`` for custom attributes whose value
            should simply be the scene path of the prim they are defined on (e.g.
            tendon labels).

            Args:
                _: The attribute name (unused).
                context: A dictionary containing at least a ``"prim"`` key with the
                    USD prim being processed.

            Returns:
                The ``Sdf.Path`` of the prim.
            """
            return str(context["prim"].GetPath())

        def resolve_actuator_target_path(prim) -> str:
            """Resolve the single target path referenced by an ``MjcActuator`` prim."""
            rel = prim.GetRelationship("mjc:target")
            target_paths = rel.GetTargets() if rel else []
            if len(target_paths) == 0:
                raise ValueError(f"MjcActuator {prim.GetPath()} is missing a 'mjc:target' relationship")
            if len(target_paths) != 1:
                raise ValueError(f"MjcActuator {prim.GetPath()} has unsupported number of targets: {len(target_paths)}")
            return str(target_paths[0])

        def get_registered_string_values(attribute_name: str) -> list[str]:
            """Return registered string values for a custom attribute."""
            attr = builder.custom_attributes.get(attribute_name)
            if attr is None or attr.values is None:
                return []
            if isinstance(attr.values, dict):
                return [str(attr.values[idx]) for idx in sorted(attr.values.keys())]
            if isinstance(attr.values, list):
                return [str(value) for value in attr.values]
            return []

        def resolve_actuator_target(
            prim,
        ) -> tuple[int, int, str]:
            """Resolve actuator target to (trntype, target_index, target_path).

            Returns (-1, -1, target_path) when the target path cannot be mapped yet.
            This can happen during USD parsing when tendon rows are authored later in
            prim traversal order.
            """
            target_path = resolve_actuator_target_path(prim)
            joint_dof_names = get_registered_string_values("mujoco:joint_dof_label")
            try:
                return int(SolverMuJoCo.TrnType.JOINT), joint_dof_names.index(target_path), target_path
            except ValueError:
                pass

            tendon_names = get_registered_string_values("mujoco:tendon_label")
            try:
                return int(SolverMuJoCo.TrnType.TENDON), tendon_names.index(target_path), target_path
            except ValueError:
                pass

            # Check if the target matches a site shape label.  Sites are stored
            # as shapes with the SITE flag in the builder.  We return a sentinel
            # target_index of 0 here; the actual site name will be resolved by
            # the SITE branch in ``_init_actuators`` via ``site_label_to_name``.
            for i, label in enumerate(builder.shape_label):
                if label == target_path and (builder.shape_flags[i] & ShapeFlags.SITE):
                    return int(SolverMuJoCo.TrnType.SITE), 0, target_path

            return -1, -1, target_path

        def resolve_joint_dof_label(_: str, context: dict[str, Any]):
            """For each DOF, return the prim path(s) of the DOF(s).

            The returned list length must match the joint's DOF count:

            - PhysicsRevoluteJoint / PhysicsPrismaticJoint: 1 DOF → [prim_path]
            - PhysicsFixedJoint: 0 DOFs → [] (empty list, no DOFs to name)
            - PhysicsSphericalJoint: 3 DOFs → [prim_path:rotX, prim_path:rotY, prim_path:rotZ]
            - PhysicsJoint (D6): N DOFs → one entry per free axis, determined from limit attributes

            Args:
                _: The attribute name (unused).
                context: A dictionary containing at least a ``"prim"`` key with the USD prim
                    for the joint being processed.

            Returns:
                A list of DOF name strings whose length matches the joint's DOF count.
            """
            prim = context["prim"]
            prim_type = prim.GetTypeName()
            prim_path = str(prim.GetPath())

            if prim_type in ["PhysicsRevoluteJoint", "PhysicsPrismaticJoint"]:
                return [prim_path]

            if prim_type == "PhysicsFixedJoint":
                return []

            if prim_type == "PhysicsSphericalJoint":
                # Spherical (ball) joints always have 3 rotational DOFs
                return [f"{prim_path}:rotX", f"{prim_path}:rotY", f"{prim_path}:rotZ"]

            if prim_type == "PhysicsJoint":
                # Determine free axes from limit attributes on the prim.
                # An axis is a DOF when its limit low < high.
                # Linear axes are enumerated first, then angular, to match the DOF
                # ordering used by add_joint_d6 (linear_axes before angular_axes).
                dof_names = []
                for axis_name in ["transX", "transY", "transZ", "rotX", "rotY", "rotZ"]:
                    low_attr = prim.GetAttribute(f"limit:{axis_name}:physics:low")
                    high_attr = prim.GetAttribute(f"limit:{axis_name}:physics:high")
                    if low_attr and high_attr:
                        low = low_attr.Get()
                        high = high_attr.Get()
                        if low is not None and high is not None and low < high:
                            dof_names.append(f"{prim_path}:{axis_name}")
                return dof_names

            warnings.warn(f"Unsupported joint type for DOF name resolution: {prim_type}", stacklevel=2)
            return []

        # First we get a list of all joint DOF names from USD
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="joint_dof_label",
                frequency=AttributeFrequency.JOINT_DOF,
                assignment=AttributeAssignment.MODEL,
                dtype=str,
                default="",
                namespace="mujoco",
                usd_attribute_name="*",
                usd_value_transformer=resolve_joint_dof_label,
            )
        )

        # Then we resolve each USD actuator transmission target from its mjc:target path.
        # If target resolution is not possible yet (for example tendon target parsed later),
        # we preserve sentinel values and resolve deterministically in _init_actuators
        # using actuator_target_label.
        def resolve_actuator_transmission_type(_: str, context: dict[str, Any]) -> int:
            """Resolve transmission type for a USD actuator prim from its target path."""
            prim = context["prim"]
            trntype, _target_idx, _target_path = resolve_actuator_target(prim)
            if trntype < 0:
                return int(SolverMuJoCo.TrnType.JOINT)
            return trntype

        def resolve_actuator_target_label(_: str, context: dict[str, Any]) -> str:
            """Resolve target path label for a USD actuator prim."""
            return resolve_actuator_target_path(context["prim"])

        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="actuator_trnid",
                frequency="mujoco:actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=wp.vec2i,
                default=wp.vec2i(-1, -1),
                namespace="mujoco",
            )
        )

        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="actuator_target_label",
                frequency="mujoco:actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=str,
                default="",
                namespace="mujoco",
                usd_attribute_name="*",
                usd_value_transformer=resolve_actuator_target_label,
            )
        )

        def parse_tristate(value: Any, _context: dict[str, Any] | None = None) -> int:
            """Parse MuJoCo tri-state values to int.

            Accepts ``"false"``, ``"true"``, and ``"auto"`` (or their numeric
            equivalents ``0``, ``1``, and ``2``) and returns the corresponding
            integer code expected by MuJoCo custom attributes.
            """
            return SolverMuJoCo._parse_named_int(value, {"false": 0, "true": 1, "auto": 2})

        def parse_presence(_value: str, _context: dict[str, Any] | None = None) -> int:
            """Return 1 to indicate the attribute was explicitly present in the MJCF."""
            return 1

        # Compiler option (frequency ONCE for single value shared across all worlds)
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="autolimits",
                frequency=AttributeFrequency.ONCE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.bool,
                default=True,  # MuJoCo default: true
                namespace="mujoco",
                mjcf_value_transformer=parse_bool,
            )
        )

        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="actuator_trntype",
                frequency="mujoco:actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=int(SolverMuJoCo.TrnType.JOINT),
                namespace="mujoco",
                mjcf_attribute_name="trntype",
                mjcf_value_transformer=parse_trntype,
                usd_attribute_name="*",
                usd_value_transformer=resolve_actuator_transmission_type,
            )
        )

        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="actuator_dyntype",
                frequency="mujoco:actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=int(_ActuatorDynamicsType.NONE),
                namespace="mujoco",
                mjcf_attribute_name="dyntype",
                mjcf_value_transformer=parse_dyntype,
                usd_attribute_name="mjc:dynType",
                usd_value_transformer=parse_dyntype,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="actuator_gaintype",
                frequency="mujoco:actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=int(_ActuatorGainType.FIXED),
                namespace="mujoco",
                mjcf_attribute_name="gaintype",
                mjcf_value_transformer=parse_gaintype,
                usd_attribute_name="mjc:gainType",
                usd_value_transformer=parse_gaintype,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="actuator_biastype",
                frequency="mujoco:actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=int(_ActuatorBiasType.NONE),
                namespace="mujoco",
                mjcf_attribute_name="biastype",
                mjcf_value_transformer=parse_biastype,
                usd_attribute_name="mjc:biasType",
                usd_value_transformer=parse_biastype,
            )
        )

        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="actuator_world",
                frequency="mujoco:actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=-1,
                namespace="mujoco",
                references="world",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="actuator_ctrllimited",
                frequency="mujoco:actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=2,
                namespace="mujoco",
                mjcf_attribute_name="ctrllimited",
                mjcf_value_transformer=parse_tristate,
                usd_attribute_name="*",
                usd_value_transformer=transform_ctrllimited,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="actuator_forcelimited",
                frequency="mujoco:actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=2,
                namespace="mujoco",
                mjcf_attribute_name="forcelimited",
                mjcf_value_transformer=parse_tristate,
                usd_attribute_name="*",
                usd_value_transformer=make_usd_limited_transformer("mjc:forceLimited", "mjc:forceRange"),
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="actuator_ctrlrange",
                frequency="mujoco:actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=wp.vec2,
                default=wp.vec2(0.0, 0.0),
                namespace="mujoco",
                mjcf_attribute_name="ctrlrange",
                usd_attribute_name="*",
                usd_value_transformer=transform_ctrlrange,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="actuator_has_ctrlrange",
                frequency="mujoco:actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=0,
                namespace="mujoco",
                mjcf_attribute_name="ctrlrange",
                mjcf_value_transformer=parse_presence,
                usd_attribute_name="*",
                usd_value_transformer=transform_has_ctrlrange,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="actuator_forcerange",
                frequency="mujoco:actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=wp.vec2,
                default=wp.vec2(0.0, 0.0),
                namespace="mujoco",
                mjcf_attribute_name="forcerange",
                usd_attribute_name="*",
                usd_value_transformer=make_usd_range_transformer("mjc:forceRange"),
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="actuator_has_forcerange",
                frequency="mujoco:actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=0,
                namespace="mujoco",
                mjcf_attribute_name="forcerange",
                mjcf_value_transformer=parse_presence,
                usd_attribute_name="*",
                usd_value_transformer=make_usd_has_range_transformer("mjc:forceRange"),
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="actuator_gear",
                frequency="mujoco:actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=wp.types.vector(length=6, dtype=wp.float32),
                default=wp.types.vector(length=6, dtype=wp.float32)(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                namespace="mujoco",
                mjcf_attribute_name="gear",
                usd_attribute_name="mjc:gear",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="actuator_cranklength",
                frequency="mujoco:actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.0,
                namespace="mujoco",
                mjcf_attribute_name="cranklength",
            )
        )

        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="actuator_dynprm",
                frequency="mujoco:actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=vec10,
                default=vec10(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                namespace="mujoco",
                mjcf_attribute_name="dynprm",
                usd_attribute_name="mjc:dynPrm",
            )
        )

        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="actuator_gainprm",
                frequency="mujoco:actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=vec10,
                default=vec10(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                namespace="mujoco",
                mjcf_attribute_name="gainprm",
                usd_attribute_name="mjc:gainPrm",
            )
        )

        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="actuator_biasprm",
                frequency="mujoco:actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=vec10,
                default=vec10(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                namespace="mujoco",
                mjcf_attribute_name="biasprm",
                usd_attribute_name="mjc:biasPrm",
            )
        )

        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="actuator_actlimited",
                frequency="mujoco:actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=2,
                namespace="mujoco",
                mjcf_attribute_name="actlimited",
                mjcf_value_transformer=parse_tristate,
                usd_attribute_name="*",
                usd_value_transformer=make_usd_limited_transformer("mjc:actLimited", "mjc:actRange"),
            )
        )

        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="actuator_actrange",
                frequency="mujoco:actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=wp.vec2,
                default=wp.vec2(0.0, 0.0),
                namespace="mujoco",
                mjcf_attribute_name="actrange",
                usd_attribute_name="*",
                usd_value_transformer=make_usd_range_transformer("mjc:actRange"),
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="actuator_has_actrange",
                frequency="mujoco:actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=0,
                namespace="mujoco",
                mjcf_attribute_name="actrange",
                mjcf_value_transformer=parse_presence,
                usd_attribute_name="*",
                usd_value_transformer=make_usd_has_range_transformer("mjc:actRange"),
            )
        )

        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="actuator_actdim",
                frequency="mujoco:actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=-1,
                namespace="mujoco",
                mjcf_attribute_name="actdim",
                usd_attribute_name="mjc:actDim",
            )
        )

        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="actuator_actearly",
                frequency="mujoco:actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=wp.bool,
                default=False,
                namespace="mujoco",
                mjcf_attribute_name="actearly",
                mjcf_value_transformer=parse_bool,
                usd_attribute_name="mjc:actEarly",
                usd_value_transformer=parse_bool,
            )
        )

        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="ctrl",
                frequency="mujoco:actuator",
                assignment=AttributeAssignment.CONTROL,
                dtype=wp.float32,
                default=0.0,
                namespace="mujoco",
            )
        )

        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="ctrl_source",
                frequency="mujoco:actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=int(SolverMuJoCo.CtrlSource.CTRL_DIRECT),
                namespace="mujoco",
            )
        )
        # Actuator kind (position/velocity/general), classified once at import.
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="ctrl_type",
                frequency="mujoco:actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=int(SolverMuJoCo.CtrlType.GENERAL),
                namespace="mujoco",
            )
        )
        # endregion actuator attributes

        # region tendon attributes
        # --- Fixed Tendon attributes (variable-length, from MJCF <tendon><fixed> tag) ---
        # Fixed tendons compute length as a linear combination of joint positions.
        # Only tendons from the template world are used; MuJoCo replicates them across worlds.

        # Tendon-level attributes (one per tendon)
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_world",
                frequency="mujoco:tendon",
                dtype=wp.int32,
                default=0,
                namespace="mujoco",
                references="world",
            )
        )

        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_stiffness",
                frequency="mujoco:tendon",
                dtype=wp.float32,
                default=0.0,
                namespace="mujoco",
                mjcf_attribute_name="stiffness",
                usd_attribute_name="mjc:stiffness",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_damping",
                frequency="mujoco:tendon",
                dtype=wp.float32,
                default=0.0,
                namespace="mujoco",
                mjcf_attribute_name="damping",
                usd_attribute_name="mjc:damping",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_frictionloss",
                frequency="mujoco:tendon",
                dtype=wp.float32,
                default=0.0,
                namespace="mujoco",
                mjcf_attribute_name="frictionloss",
                usd_attribute_name="mjc:frictionloss",
            )
        )

        def resolve_context_builder(context: dict[str, Any]) -> ModelBuilder:
            """Resolve builder from transformer context, falling back to current builder."""
            context_builder = context.get("builder")
            if isinstance(context_builder, ModelBuilder):
                return context_builder
            return builder

        def resolve_tendon_joint_adr(_: Any, context: dict[str, Any]) -> int:
            context_builder = resolve_context_builder(context)
            tendon_joint_attr = context_builder.custom_attributes.get("mujoco:tendon_joint")
            if tendon_joint_attr is None or not isinstance(tendon_joint_attr.values, list):
                return 0
            return len(tendon_joint_attr.values)

        def resolve_tendon_joint_num(_: Any, context: dict[str, Any]) -> int:
            context_builder = resolve_context_builder(context)
            joint_entries = cls._parse_mjc_fixed_tendon_joint_entries(context["prim"], context_builder)
            return len(joint_entries)

        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_limited",
                frequency="mujoco:tendon",
                dtype=wp.int32,
                default=2,  # 0=false, 1=true, 2=auto
                namespace="mujoco",
                mjcf_attribute_name="limited",
                mjcf_value_transformer=parse_tristate,
                usd_attribute_name="mjc:limited",
                usd_value_transformer=parse_tristate,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_range",
                frequency="mujoco:tendon",
                dtype=wp.vec2,
                default=wp.vec2(0.0, 0.0),
                namespace="mujoco",
                mjcf_attribute_name="range",
                usd_attribute_name="mjc:range:min",
                usd_value_transformer=make_usd_range_transformer("mjc:range"),
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_margin",
                frequency="mujoco:tendon",
                dtype=wp.float32,
                default=0.0,
                namespace="mujoco",
                mjcf_attribute_name="margin",
                usd_attribute_name="mjc:margin",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_solref_limit",
                frequency="mujoco:tendon",
                dtype=wp.vec2,
                default=wp.vec2(0.02, 1.0),
                namespace="mujoco",
                mjcf_attribute_name="solreflimit",
                usd_attribute_name="mjc:solreflimit",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_solimp_limit",
                frequency="mujoco:tendon",
                dtype=vec5,
                default=vec5(0.9, 0.95, 0.001, 0.5, 2.0),
                namespace="mujoco",
                mjcf_attribute_name="solimplimit",
                usd_attribute_name="mjc:solimplimit",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_solref_friction",
                frequency="mujoco:tendon",
                dtype=wp.vec2,
                default=wp.vec2(0.02, 1.0),
                namespace="mujoco",
                mjcf_attribute_name="solreffriction",
                usd_attribute_name="mjc:solreffriction",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_solimp_friction",
                frequency="mujoco:tendon",
                dtype=vec5,
                default=vec5(0.9, 0.95, 0.001, 0.5, 2.0),
                namespace="mujoco",
                mjcf_attribute_name="solimpfriction",
                usd_attribute_name="mjc:solimpfriction",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_armature",
                frequency="mujoco:tendon",
                dtype=wp.float32,
                default=0.0,
                namespace="mujoco",
                mjcf_attribute_name="armature",
                usd_attribute_name="mjc:armature",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_springlength",
                frequency="mujoco:tendon",
                dtype=wp.vec2,
                default=wp.vec2(-1.0, -1.0),  # -1 means use default (model length)
                namespace="mujoco",
                mjcf_attribute_name="springlength",
                usd_attribute_name="mjc:springlength",
            )
        )
        # Addressing into joint arrays (one per tendon)
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_joint_adr",
                frequency="mujoco:tendon",
                dtype=wp.int32,
                default=0,
                namespace="mujoco",
                references="mujoco:tendon_joint",  # Offset by joint entry count during merge
                usd_attribute_name="*",
                usd_value_transformer=resolve_tendon_joint_adr,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_joint_num",
                frequency="mujoco:tendon",
                dtype=wp.int32,
                default=0,
                namespace="mujoco",
                usd_attribute_name="*",
                usd_value_transformer=resolve_tendon_joint_num,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_actuator_force_range",
                frequency="mujoco:tendon",
                dtype=wp.vec2,
                default=wp.vec2(0.0, 0.0),
                namespace="mujoco",
                mjcf_attribute_name="actuatorfrcrange",
                usd_attribute_name="mjc:actuatorfrcrange:min",
                usd_value_transformer=make_usd_range_transformer("mjc:actuatorfrcrange"),
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_actuator_force_limited",
                frequency="mujoco:tendon",
                dtype=wp.int32,
                default=2,  # 0=false, 1=true, 2=auto
                namespace="mujoco",
                mjcf_attribute_name="actuatorfrclimited",
                mjcf_value_transformer=parse_tristate,
                usd_attribute_name="mjc:actuatorfrclimited",
                usd_value_transformer=parse_tristate,
            )
        )
        # Tendon names (string attribute - stored as list[str], not warp array)
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_label",
                frequency="mujoco:tendon",
                dtype=str,
                default="",
                namespace="mujoco",
                mjcf_attribute_name="name",
                usd_attribute_name="*",
                usd_value_transformer=resolve_prim_name,
            )
        )

        # Joint arrays (one entry per joint in a fixed tendon's linear combination)
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_joint",
                frequency="mujoco:tendon_joint",
                dtype=wp.int32,
                default=-1,
                namespace="mujoco",
                references="joint",  # Offset by joint count during merge
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_coef",
                frequency="mujoco:tendon_joint",
                dtype=wp.float32,
                default=0.0,
                namespace="mujoco",
                mjcf_attribute_name="coef",
            )
        )
        # endregion tendon attributes

        # --- Spatial tendon attributes ---
        # Tendon type distinguishes fixed (0) from spatial (1) tendons.
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_type",
                frequency="mujoco:tendon",
                dtype=wp.int32,
                default=0,
                namespace="mujoco",
            )
        )
        # Addressing into wrap path arrays (one per tendon, used by spatial tendons)
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_wrap_adr",
                frequency="mujoco:tendon",
                dtype=wp.int32,
                default=0,
                namespace="mujoco",
                references="mujoco:tendon_wrap",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_wrap_num",
                frequency="mujoco:tendon",
                dtype=wp.int32,
                default=0,
                namespace="mujoco",
            )
        )

        # Wrap path arrays (one entry per wrap element in a spatial tendon's path)
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_wrap_type",
                frequency="mujoco:tendon_wrap",
                dtype=wp.int32,
                default=0,  # 0=site, 1=geom, 2=pulley
                namespace="mujoco",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_wrap_shape",
                frequency="mujoco:tendon_wrap",
                dtype=wp.int32,
                default=-1,
                namespace="mujoco",
                references="shape",  # Offset by shape count during merge
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_wrap_sidesite",
                frequency="mujoco:tendon_wrap",
                dtype=wp.int32,
                default=-1,
                namespace="mujoco",
                references="shape",  # Offset by shape count during merge
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_wrap_prm",
                frequency="mujoco:tendon_wrap",
                dtype=wp.float32,
                default=0.0,
                namespace="mujoco",
            )
        )

    def _init_pairs(self, model: Model, spec: Any, shape_mapping: dict[int, str], template_world: int) -> None:
        """
        Initialize MuJoCo contact pairs from custom attributes.

        Pairs belonging to the template world and global pairs (world < 0) are
        added to the MuJoCo spec. MuJoCo will replicate these pairs across all
        worlds automatically.

        Args:
            model: The Newton model.
            spec: The MuJoCo spec to add pairs to.
            shape_mapping: Mapping from Newton shape index to MuJoCo geom name.
            template_world: The world index to use as the template (typically first_group).
        """
        pair_count = model.custom_frequency_counts.get("mujoco:pair", 0)
        if pair_count == 0:
            return

        mujoco_attrs = model.mujoco

        def get_numpy(name):
            attr = getattr(mujoco_attrs, name, None)
            return attr.numpy() if attr is not None else None

        pair_world = get_numpy("pair_world")
        pair_geom1 = get_numpy("pair_geom1")
        pair_geom2 = get_numpy("pair_geom2")
        if pair_world is None or pair_geom1 is None or pair_geom2 is None:
            return

        pair_condim = get_numpy("pair_condim")
        pair_solref = get_numpy("pair_solref")
        pair_solreffriction = get_numpy("pair_solreffriction")
        pair_solimp = get_numpy("pair_solimp")
        pair_margin = get_numpy("pair_margin")
        pair_gap = get_numpy("pair_gap")
        pair_friction = get_numpy("pair_friction")

        for i in range(pair_count):
            # Only include pairs from the template world or global pairs (world < 0)
            pw = int(pair_world[i])
            if pw != template_world and pw >= 0:
                continue

            # Map Newton shape indices to MuJoCo geom names
            newton_shape1 = int(pair_geom1[i])
            newton_shape2 = int(pair_geom2[i])

            # Skip invalid pairs
            if newton_shape1 < 0 or newton_shape2 < 0:
                continue

            geom_name1 = shape_mapping.get(newton_shape1)
            geom_name2 = shape_mapping.get(newton_shape2)

            if geom_name1 is None or geom_name2 is None:
                warnings.warn(
                    f"Skipping pair {i}: Newton shapes ({newton_shape1}, {newton_shape2}) "
                    f"not found in MuJoCo shape mapping.",
                    stacklevel=2,
                )
                continue

            # Build pair kwargs
            pair_kwargs: dict[str, Any] = {
                "geomname1": geom_name1,
                "geomname2": geom_name2,
            }

            if pair_condim is not None:
                pair_kwargs["condim"] = int(pair_condim[i])
            if pair_solref is not None:
                pair_kwargs["solref"] = pair_solref[i].tolist()
            if pair_solreffriction is not None:
                pair_kwargs["solreffriction"] = pair_solreffriction[i].tolist()
            if pair_solimp is not None:
                pair_kwargs["solimp"] = pair_solimp[i].tolist()
            authored_margin = float(pair_margin[i]) if pair_margin is not None else 0.0
            authored_gap = float(pair_gap[i]) if pair_gap is not None else 0.0
            if self._zero_margins_for_native_ccd:
                # NATIVECCD/MULTICCD reject non-zero margin at put_model (#2106);
                # gap is unrestricted under MuJoCo 3.9, so forward it.
                pair_kwargs["margin"] = 0.0
                if pair_gap is not None:
                    pair_kwargs["gap"] = authored_gap
                if self._use_mujoco_contacts and authored_margin > 0.0:
                    warnings.warn(
                        f"Pair ({geom_name1}, {geom_name2}): authored margin="
                        f"{authored_margin} zeroed for NATIVECCD/MULTICCD "
                        f"compatibility (#2106). To honor this value, switch "
                        f"to Newton's collision pipeline by constructing the "
                        f"solver with use_mujoco_contacts=False and feeding "
                        f"Newton-generated contacts into step().",
                        stacklevel=2,
                    )
            else:
                if pair_margin is not None:
                    pair_kwargs["margin"] = authored_margin
                if pair_gap is not None:
                    pair_kwargs["gap"] = authored_gap
            if pair_friction is not None:
                pair_kwargs["friction"] = pair_friction[i].tolist()

            spec.add_pair(**pair_kwargs)

    @staticmethod
    def _validate_tendon_attributes(model: Model) -> tuple[int, int, int]:
        """
        Validate that all tendon attributes have consistent lengths.

        Args:
            model: The Newton model to validate.

        Returns:
            tuple[int, int, int]: (tendon_count, joint_entry_count, wrap_entry_count).

        Raises:
            ValueError: If tendon attributes have inconsistent lengths.
        """
        mujoco_attrs = getattr(model, "mujoco", None)
        if mujoco_attrs is None:
            return 0, 0, 0

        # Tendon-level attributes
        tendon_attr_names = [
            "tendon_world",
            "tendon_type",
            "tendon_stiffness",
            "tendon_damping",
            "tendon_frictionloss",
            "tendon_limited",
            "tendon_range",
            "tendon_margin",
            "tendon_actuator_force_limited",
            "tendon_actuator_force_range",
            "tendon_solref_limit",
            "tendon_solimp_limit",
            "tendon_solref_friction",
            "tendon_solimp_friction",
            "tendon_springlength",
            "tendon_armature",
            "tendon_joint_adr",
            "tendon_joint_num",
            "tendon_wrap_adr",
            "tendon_wrap_num",
        ]

        # If the list above has N parameters then each tendon should have exactly N parameters.
        # Count the number of parameters that we have for each tendon.
        # Each entry in the array of counts should be N.
        # We can then extract the number of unique entries in our array of counts.
        # The number of unique entries should be 1 because every entry should be N.
        # If the number of unique entries is not 1 then we are missing an attribute on at least one tendon.
        tendon_lengths: dict[str, int] = {}
        for name in tendon_attr_names:
            attr = getattr(mujoco_attrs, name, None)
            if attr is not None:
                tendon_lengths[name] = len(attr)
        if not tendon_lengths:
            return 0, 0, 0
        # Check all tendon-level lengths are the same
        unique_tendon_lengths = set(tendon_lengths.values())
        if len(unique_tendon_lengths) > 1:
            raise ValueError(
                f"MuJoCo tendon attributes have inconsistent lengths: {tendon_lengths}. "
                "All tendon-level attributes must have the same number of elements."
            )

        # Compute the number of tendons.
        tendon_count = next(iter(unique_tendon_lengths))

        # Attributes per joint in the tendon that allow the tendon length to
        # be calculated as a linear sum of coefficient and joint position.
        # For each joint in a tendon (specified by joint index) there must be a corresponding coefficient.
        joint_attr_names = ["tendon_joint", "tendon_coef"]
        joint_lengths: dict[str, int] = {}
        for name in joint_attr_names:
            attr = getattr(mujoco_attrs, name, None)
            if attr is not None:
                joint_lengths[name] = len(attr)
        if not joint_lengths:
            joint_entry_count = 0
        else:
            unique_joint_lengths = set(joint_lengths.values())
            if len(unique_joint_lengths) > 1:
                raise ValueError(
                    f"MuJoCo tendon joint attributes have inconsistent lengths: {joint_lengths}. "
                    "All joint-level attributes must have the same number of elements."
                )
            joint_entry_count = next(iter(unique_joint_lengths))

        # Wrap path attributes for spatial tendons
        wrap_attr_names = ["tendon_wrap_type", "tendon_wrap_shape", "tendon_wrap_sidesite", "tendon_wrap_prm"]
        wrap_lengths: dict[str, int] = {}
        for name in wrap_attr_names:
            attr = getattr(mujoco_attrs, name, None)
            if attr is not None:
                wrap_lengths[name] = len(attr)
        if not wrap_lengths:
            wrap_entry_count = 0
        else:
            unique_wrap_lengths = set(wrap_lengths.values())
            if len(unique_wrap_lengths) > 1:
                raise ValueError(
                    f"MuJoCo tendon wrap attributes have inconsistent lengths: {wrap_lengths}. "
                    "All wrap-level attributes must have the same number of elements."
                )
            wrap_entry_count = next(iter(unique_wrap_lengths))

        return tendon_count, joint_entry_count, wrap_entry_count

    def _init_tendons(
        self,
        model: Model,
        spec: Any,
        joint_mapping: dict[int, str],
        shape_mapping: dict[int, str],
        site_mapping: dict[int, str],
        template_world: int,
    ) -> tuple[list[int], list[str]]:
        """
        Initialize MuJoCo fixed and spatial tendons from custom attributes.

        Only tendons belonging to the template world are added to the MuJoCo spec.
        MuJoCo will replicate these tendons across all worlds automatically.

        Args:
            model: The Newton model.
            spec: The MuJoCo spec to add tendons to.
            joint_mapping: Mapping from Newton joint index to MuJoCo joint name.
            shape_mapping: Mapping from Newton shape index to MuJoCo geom name.
            site_mapping: Mapping from Newton shape index (sites) to MuJoCo site name.
            template_world: The world index to use as the template (typically first_group).

        Returns:
            tuple[list[int], list[str]]: Tuple of (Newton tendon indices, MuJoCo tendon names).
        """

        tendon_count, joint_entry_count, wrap_entry_count = self._validate_tendon_attributes(model)
        if tendon_count == 0:
            return [], []

        mujoco_attrs = model.mujoco

        # Get tendon-level arrays
        tendon_world = mujoco_attrs.tendon_world.numpy()
        tendon_type_attr = getattr(mujoco_attrs, "tendon_type", None)
        tendon_type_np = tendon_type_attr.numpy() if tendon_type_attr is not None else None
        tendon_stiffness = getattr(mujoco_attrs, "tendon_stiffness", None)
        tendon_stiffness_np = tendon_stiffness.numpy() if tendon_stiffness is not None else None
        tendon_damping = getattr(mujoco_attrs, "tendon_damping", None)
        tendon_damping_np = tendon_damping.numpy() if tendon_damping is not None else None
        tendon_frictionloss = getattr(mujoco_attrs, "tendon_frictionloss", None)
        tendon_frictionloss_np = tendon_frictionloss.numpy() if tendon_frictionloss is not None else None
        tendon_limited = getattr(mujoco_attrs, "tendon_limited", None)
        tendon_limited_np = tendon_limited.numpy() if tendon_limited is not None else None
        tendon_range = getattr(mujoco_attrs, "tendon_range", None)
        tendon_range_np = tendon_range.numpy() if tendon_range is not None else None
        tendon_actuator_force_limited = getattr(mujoco_attrs, "tendon_actuator_force_limited", None)
        tendon_actuator_force_limited_np = (
            tendon_actuator_force_limited.numpy() if tendon_actuator_force_limited is not None else None
        )
        tendon_actuator_force_range = getattr(mujoco_attrs, "tendon_actuator_force_range", None)
        tendon_actuator_force_range_np = (
            tendon_actuator_force_range.numpy() if tendon_actuator_force_range is not None else None
        )
        tendon_margin = getattr(mujoco_attrs, "tendon_margin", None)
        tendon_margin_np = tendon_margin.numpy() if tendon_margin is not None else None
        tendon_solref_limit = getattr(mujoco_attrs, "tendon_solref_limit", None)
        tendon_solref_limit_np = tendon_solref_limit.numpy() if tendon_solref_limit is not None else None
        tendon_solimp_limit = getattr(mujoco_attrs, "tendon_solimp_limit", None)
        tendon_solimp_limit_np = tendon_solimp_limit.numpy() if tendon_solimp_limit is not None else None
        tendon_solref_friction = getattr(mujoco_attrs, "tendon_solref_friction", None)
        tendon_solref_friction_np = tendon_solref_friction.numpy() if tendon_solref_friction is not None else None
        tendon_solimp_friction = getattr(mujoco_attrs, "tendon_solimp_friction", None)
        tendon_solimp_friction_np = tendon_solimp_friction.numpy() if tendon_solimp_friction is not None else None
        tendon_armature = getattr(mujoco_attrs, "tendon_armature", None)
        tendon_armature_np = tendon_armature.numpy() if tendon_armature is not None else None
        tendon_springlength = getattr(mujoco_attrs, "tendon_springlength", None)
        tendon_springlength_np = tendon_springlength.numpy() if tendon_springlength is not None else None
        tendon_label_arr = getattr(mujoco_attrs, "tendon_label", None)

        # Fixed tendon arrays
        tendon_joint_adr = mujoco_attrs.tendon_joint_adr.numpy()
        tendon_joint_num = mujoco_attrs.tendon_joint_num.numpy()
        tendon_joint = mujoco_attrs.tendon_joint.numpy() if joint_entry_count > 0 else None
        tendon_coef = mujoco_attrs.tendon_coef.numpy() if joint_entry_count > 0 else None

        # Spatial tendon wrap path arrays
        tendon_wrap_adr_np = mujoco_attrs.tendon_wrap_adr.numpy() if wrap_entry_count > 0 else None
        tendon_wrap_num_np = mujoco_attrs.tendon_wrap_num.numpy() if wrap_entry_count > 0 else None
        tendon_wrap_type_np = mujoco_attrs.tendon_wrap_type.numpy() if wrap_entry_count > 0 else None
        tendon_wrap_shape_np = mujoco_attrs.tendon_wrap_shape.numpy() if wrap_entry_count > 0 else None
        tendon_wrap_sidesite_np = mujoco_attrs.tendon_wrap_sidesite.numpy() if wrap_entry_count > 0 else None
        tendon_wrap_prm_np = mujoco_attrs.tendon_wrap_prm.numpy() if wrap_entry_count > 0 else None

        model_joint_type_np = model.joint_type.numpy()

        # Track which Newton tendon indices are added to MuJoCo and their names
        selected_tendons: list[int] = []
        tendon_names: list[str] = []
        used_tendon_names: set[str] = set()

        for i in range(tendon_count):
            # Only include tendons from the template world or global tendons (world < 0)
            tw = int(tendon_world[i])
            if tw != template_world and tw >= 0:
                continue

            # Resolve tendon label early so it can be included in warnings.
            tendon_label = ""
            if isinstance(tendon_label_arr, list) and i < len(tendon_label_arr):
                tendon_label = str(tendon_label_arr[i]).strip()
            if tendon_label == "":
                tendon_label = f"tendon_{i}"

            ttype = int(tendon_type_np[i]) if tendon_type_np is not None else 0

            # Pre-validate wrapping path before creating the tendon in the spec.
            if ttype == 0:
                # Fixed tendon: build joint wraps list
                joint_start = int(tendon_joint_adr[i])
                joint_num = int(tendon_joint_num[i])
                if joint_num <= 0:
                    if wp.config.log_level <= wp.LOG_DEBUG:
                        print(f"Warning: Skipping tendon {i} during MuJoCo export because it has no joint wraps.")
                    continue

                if joint_start < 0 or joint_start + joint_num > joint_entry_count:
                    warnings.warn(
                        f"Skipping fixed tendon '{tendon_label}': joint range "
                        f"[{joint_start}, {joint_start + joint_num}) "
                        f"out of bounds for joint entries ({joint_entry_count}).",
                        stacklevel=2,
                    )
                    continue

                fixed_wraps: list[tuple[str, float]] = []
                for j in range(joint_start, joint_start + joint_num):
                    if tendon_joint is None or tendon_coef is None:
                        break
                    newton_joint = int(tendon_joint[j])
                    coef = float(tendon_coef[j])
                    if newton_joint < 0:
                        warnings.warn(
                            f"Skipping joint entry {j} for tendon {i}: invalid joint index {newton_joint}.",
                            stacklevel=2,
                        )
                        continue
                    if model_joint_type_np[newton_joint] == JointType.D6:
                        warnings.warn(
                            f"Skipping joint entry {j} for tendon {i}: invalid D6 joint type {newton_joint}.",
                            stacklevel=2,
                        )
                        continue
                    joint_name = joint_mapping.get(newton_joint)
                    if joint_name is None:
                        warnings.warn(
                            f"Skipping joint entry {j} for tendon {i}: Newton joint {newton_joint} "
                            f"not found in MuJoCo joint mapping.",
                            stacklevel=2,
                        )
                        continue
                    fixed_wraps.append((joint_name, coef))

                if len(fixed_wraps) == 0:
                    if wp.config.log_level <= wp.LOG_DEBUG:
                        print(
                            f"Warning: Skipping tendon {i} during MuJoCo export "
                            "because no valid joint wraps were resolved."
                        )
                    continue

            elif ttype == 1:
                # Spatial tendon: validate wrap path arrays and bounds
                if tendon_wrap_adr_np is None or tendon_wrap_num_np is None:
                    warnings.warn(
                        f"Spatial tendon '{tendon_label}' has no wrap path arrays, skipping.",
                        stacklevel=2,
                    )
                    continue

                wrap_start = int(tendon_wrap_adr_np[i])
                wrap_num = int(tendon_wrap_num_np[i])
                if wrap_start < 0 or wrap_num <= 0 or wrap_start + wrap_num > wrap_entry_count:
                    warnings.warn(
                        f"Skipping spatial tendon '{tendon_label}': wrap range "
                        f"[{wrap_start}, {wrap_start + wrap_num}) "
                        f"out of bounds for wrap entries ({wrap_entry_count}).",
                        stacklevel=2,
                    )
                    continue

                # Pre-resolve all wrap elements; skip entire tendon if any element fails
                spatial_wraps_valid = True
                for w in range(wrap_start, wrap_start + wrap_num):
                    if tendon_wrap_type_np is None or tendon_wrap_shape_np is None:
                        spatial_wraps_valid = False
                        break
                    wtype = int(tendon_wrap_type_np[w])
                    if wtype == 0:  # site
                        if site_mapping.get(int(tendon_wrap_shape_np[w])) is None:
                            warnings.warn(
                                f"Skipping spatial tendon '{tendon_label}': wrap site at index {w} "
                                f"(shape {int(tendon_wrap_shape_np[w])}) not in site mapping.",
                                stacklevel=2,
                            )
                            spatial_wraps_valid = False
                            break
                    elif wtype == 1:  # geom
                        if shape_mapping.get(int(tendon_wrap_shape_np[w])) is None:
                            warnings.warn(
                                f"Skipping spatial tendon '{tendon_label}': wrap geom at index {w} "
                                f"(shape {int(tendon_wrap_shape_np[w])}) not in shape mapping.",
                                stacklevel=2,
                            )
                            spatial_wraps_valid = False
                            break
                    elif wtype == 2:  # pulley
                        divisor = float(tendon_wrap_prm_np[w]) if tendon_wrap_prm_np is not None else 0.0
                        if divisor <= 0.0:
                            warnings.warn(
                                f"Skipping spatial tendon '{tendon_label}': pulley at index {w} "
                                f"has non-positive divisor {divisor}.",
                                stacklevel=2,
                            )
                            spatial_wraps_valid = False
                            break
                    else:
                        warnings.warn(
                            f"Skipping spatial tendon '{tendon_label}': unknown wrap type {wtype} at index {w}.",
                            stacklevel=2,
                        )
                        spatial_wraps_valid = False
                        break
                if not spatial_wraps_valid:
                    continue

            else:
                warnings.warn(f"Skipping tendon '{tendon_label}': unknown tendon type {ttype}.", stacklevel=2)
                continue

            # Track this tendon only after confirming it can be exported.
            selected_tendons.append(i)

            # Use the label resolved earlier; ensure unique names for the spec.
            tendon_name = tendon_label
            suffix = 1
            while tendon_name in used_tendon_names:
                tendon_name = f"{tendon_label}_{suffix}"
                suffix += 1
            used_tendon_names.add(tendon_name)

            tendon_names.append(tendon_name)
            t = spec.add_tendon()
            t.name = tendon_name

            # Set tendon properties (shared between fixed and spatial)
            if tendon_stiffness_np is not None:
                t.stiffness[0] = tendon_stiffness_np[i]
            if tendon_damping_np is not None:
                t.damping[0] = tendon_damping_np[i]
            if tendon_frictionloss_np is not None:
                t.frictionloss = float(tendon_frictionloss_np[i])
            if tendon_limited_np is not None:
                t.limited = int(tendon_limited_np[i])
            if tendon_range_np is not None:
                t.range = tendon_range_np[i].tolist()
            if tendon_actuator_force_limited_np is not None:
                t.actfrclimited = int(tendon_actuator_force_limited_np[i])
            if tendon_actuator_force_range_np is not None:
                t.actfrcrange = tendon_actuator_force_range_np[i].tolist()
            if tendon_margin_np is not None:
                t.margin = float(tendon_margin_np[i])
            if tendon_armature_np is not None:
                t.armature = float(tendon_armature_np[i])
            if tendon_solref_limit_np is not None:
                t.solref_limit = tendon_solref_limit_np[i].tolist()
            if tendon_solimp_limit_np is not None:
                t.solimp_limit = tendon_solimp_limit_np[i].tolist()
            if tendon_solref_friction_np is not None:
                t.solref_friction = tendon_solref_friction_np[i].tolist()
            if tendon_solimp_friction_np is not None:
                t.solimp_friction = tendon_solimp_friction_np[i].tolist()
            if tendon_springlength_np is not None:
                val = tendon_springlength_np[i]
                has_automatic_length_computation = val[0] == -1.0
                has_dead_zone = val[1] >= val[0]
                if has_automatic_length_computation:
                    t.springlength[0] = -1.0
                    t.springlength[1] = -1.0
                elif has_dead_zone:
                    t.springlength[0] = val[0]
                    t.springlength[1] = val[1]
                else:
                    t.springlength[0] = val[0]
                    t.springlength[1] = val[0]

            # Add wrapping path (all elements pre-validated above)
            if ttype == 0:
                for joint_name, coef in fixed_wraps:
                    t.wrap_joint(joint_name, coef)
            elif ttype == 1:
                for w in range(wrap_start, wrap_start + wrap_num):
                    wtype = int(tendon_wrap_type_np[w])
                    if wtype == 0:
                        t.wrap_site(site_mapping[int(tendon_wrap_shape_np[w])])
                    elif wtype == 1:
                        geom_name = shape_mapping[int(tendon_wrap_shape_np[w])]
                        sidesite_name = ""
                        if tendon_wrap_sidesite_np is not None:
                            sidesite_idx = int(tendon_wrap_sidesite_np[w])
                            if sidesite_idx >= 0:
                                sidesite_name = site_mapping.get(sidesite_idx)
                                if sidesite_name is None:
                                    warnings.warn(
                                        f"Wrap geom {w} for tendon {i} references sidesite "
                                        f"{sidesite_idx} not in site mapping; ignoring sidesite.",
                                        stacklevel=2,
                                    )
                                    sidesite_name = ""
                        t.wrap_geom(geom_name, sidesite_name)
                    elif wtype == 2:
                        t.wrap_pulley(float(tendon_wrap_prm_np[w]))
                    # else: unknown wtype — already rejected during pre-validation

        return selected_tendons, tendon_names

    def _init_actuators(
        self,
        model: Model,
        spec: Any,
        template_world: int,
        actuator_args: dict[str, Any],
        mjc_actuator_ctrl_source_list: list[int],
        mjc_actuator_to_newton_idx_list: list[int],
        mjc_actuator_to_target_q_idx_list: list[int],
        mjc_actuator_to_target_q_axis_idx_list: list[int],
        mjc_actuator_to_newton_ball_jnt_list: list[int],
        dof_to_mjc_joint: np.ndarray,
        mjc_joint_names: list[str],
        selected_tendons: list[int],
        mjc_tendon_names: list[str],
        body_name_mapping: dict[int, str],
        site_mapping: dict[int, str] | None = None,
    ) -> int:
        """Initialize MuJoCo general actuators from custom attributes.

        Only processes CTRL_DIRECT actuators (motor, general, etc.) from the
        mujoco:actuator custom attributes. JOINT_TARGET actuators (position/velocity)
        are handled separately in the joint iteration loop.

        For CTRL_DIRECT actuators targeting joints, this method uses the DOF index
        stored in actuator_trnid (see import_mjcf.py) to look up the correct MuJoCo
        joint name. This is necessary because Newton may combine multiple MJCF joints
        into one, but MuJoCo needs the specific joint name (e.g., "joint_ang1" not "joint").

        Args:
            model: The Newton model.
            spec: The MuJoCo spec to add actuators to.
            template_world: The world index to use as the template.
            actuator_args: Default actuator arguments.
            mjc_actuator_ctrl_source_list: List to append control sources to.
            mjc_actuator_to_newton_idx_list: List to append Newton indices to.
            dof_to_mjc_joint: Mapping from Newton DOF index to MuJoCo joint index.
                Used to resolve CTRL_DIRECT joint actuators to their MuJoCo targets.
            mjc_joint_names: List of MuJoCo joint names indexed by MuJoCo joint index.
                Used together with dof_to_mjc_joint to get the correct joint name.
            body_name_mapping: Mapping from Newton body index to de-duplicated MuJoCo body name.
            site_mapping: Mapping from Newton shape index (sites) to MuJoCo site name.
                Used to resolve CTRL_DIRECT actuators targeting sites.
        Returns:
            int: Number of actuators added.
        """
        if site_mapping is None:
            site_mapping = {}
        mujoco = self._mujoco

        mujoco_attrs = getattr(model, "mujoco", None)
        mujoco_actuator_count = model.custom_frequency_counts.get("mujoco:actuator", 0)

        if mujoco_actuator_count == 0 or mujoco_attrs is None or not hasattr(mujoco_attrs, "actuator_trnid"):
            return 0

        actuator_count = 0

        # actuator_trnid[:,0] is the target index, actuator_trntype determines its meaning
        actuator_trnid = mujoco_attrs.actuator_trnid.numpy()
        trntype_arr = mujoco_attrs.actuator_trntype.numpy() if hasattr(mujoco_attrs, "actuator_trntype") else None
        ctrl_source_arr = mujoco_attrs.ctrl_source.numpy() if hasattr(mujoco_attrs, "ctrl_source") else None
        actuator_world_arr = mujoco_attrs.actuator_world.numpy() if hasattr(mujoco_attrs, "actuator_world") else None
        actuator_target_label_arr = getattr(mujoco_attrs, "actuator_target_label", None)
        joint_dof_label_arr = getattr(mujoco_attrs, "joint_dof_label", None)
        tendon_label_arr = getattr(mujoco_attrs, "tendon_label", None)

        # Build reverse lookup from shape label (prim path) to site name
        # so we can resolve actuator target labels that reference sites.
        site_label_to_name: dict[str, str] = {}
        for shape_idx, site_name in site_mapping.items():
            if shape_idx < len(model.shape_label):
                site_label_to_name[model.shape_label[shape_idx]] = site_name

        def resolve_target_from_label(target_label: str) -> tuple[int, int]:
            if isinstance(joint_dof_label_arr, list):
                try:
                    return int(SolverMuJoCo.TrnType.JOINT), joint_dof_label_arr.index(target_label)
                except ValueError:
                    pass
            if isinstance(tendon_label_arr, list):
                try:
                    return int(SolverMuJoCo.TrnType.TENDON), tendon_label_arr.index(target_label)
                except ValueError:
                    pass
            # Check if the target label matches a site shape label.
            # For site targets, return trntype=SITE with target_idx=0
            # (the actual site name is resolved in the SITE branch below).
            if target_label in site_label_to_name:
                return int(SolverMuJoCo.TrnType.SITE), 0
            return -1, -1

        # Pre-fetch range/limited arrays to avoid per-element .numpy() calls
        has_ctrlrange_arr = (
            mujoco_attrs.actuator_has_ctrlrange.numpy() if hasattr(mujoco_attrs, "actuator_has_ctrlrange") else None
        )
        ctrlrange_arr = mujoco_attrs.actuator_ctrlrange.numpy() if hasattr(mujoco_attrs, "actuator_ctrlrange") else None
        ctrllimited_arr = (
            mujoco_attrs.actuator_ctrllimited.numpy() if hasattr(mujoco_attrs, "actuator_ctrllimited") else None
        )
        has_forcerange_arr = (
            mujoco_attrs.actuator_has_forcerange.numpy() if hasattr(mujoco_attrs, "actuator_has_forcerange") else None
        )
        forcerange_arr = (
            mujoco_attrs.actuator_forcerange.numpy() if hasattr(mujoco_attrs, "actuator_forcerange") else None
        )
        forcelimited_arr = (
            mujoco_attrs.actuator_forcelimited.numpy() if hasattr(mujoco_attrs, "actuator_forcelimited") else None
        )
        has_actrange_arr = (
            mujoco_attrs.actuator_has_actrange.numpy() if hasattr(mujoco_attrs, "actuator_has_actrange") else None
        )
        actrange_arr = mujoco_attrs.actuator_actrange.numpy() if hasattr(mujoco_attrs, "actuator_actrange") else None
        actlimited_arr = (
            mujoco_attrs.actuator_actlimited.numpy() if hasattr(mujoco_attrs, "actuator_actlimited") else None
        )
        for mujoco_act_idx in range(mujoco_actuator_count):
            # Skip JOINT_TARGET actuators - they're already added via joint_target_mode path
            if ctrl_source_arr is not None:
                ctrl_source = int(ctrl_source_arr[mujoco_act_idx])
                if ctrl_source == SolverMuJoCo.CtrlSource.JOINT_TARGET:
                    continue  # Already handled in joint iteration

            # Only include actuators from the first world (template) or global actuators
            if actuator_world_arr is not None:
                actuator_world = int(actuator_world_arr[mujoco_act_idx])
                if actuator_world != template_world and actuator_world != -1:
                    continue  # Skip actuators from other worlds

            target_idx = int(actuator_trnid[mujoco_act_idx, 0])
            target_idx_alt = int(actuator_trnid[mujoco_act_idx, 1])
            slider_site_name = None
            refsite_name = None

            # Determine target type from trntype enum (JOINT, TENDON, SITE, BODY, ...).
            trntype = int(trntype_arr[mujoco_act_idx]) if trntype_arr is not None else 0
            target_label = ""
            if isinstance(actuator_target_label_arr, list) and mujoco_act_idx < len(actuator_target_label_arr):
                target_label = actuator_target_label_arr[mujoco_act_idx]

            # Backward compatibility for older USD parsing that wrote tendon index to trnid[1].
            if trntype == int(SolverMuJoCo.TrnType.TENDON):
                if target_idx < 0 and target_idx_alt >= 0:
                    target_idx = target_idx_alt
                elif target_idx == 0 and target_idx_alt > 0:
                    target_idx = target_idx_alt

            # Deferred target resolution: when USD parsing ran before tendon rows were available,
            # keep actuator_target_label and resolve the final (type, index) here.
            if target_idx < 0 and target_label:
                resolved_type, resolved_idx = resolve_target_from_label(target_label)
                if resolved_idx >= 0:
                    trntype = resolved_type
                    target_idx = resolved_idx
            if target_idx < 0:
                warnings.warn(
                    f"MuJoCo actuator {mujoco_act_idx} has unresolved target '{target_label}'. Skipping actuator.",
                    stacklevel=2,
                )
                continue

            if trntype in (
                int(SolverMuJoCo.TrnType.JOINT),
                int(SolverMuJoCo.TrnType.JOINT_IN_PARENT),
            ):
                # For CTRL_DIRECT joint actuators, actuator_trnid stores a DOF index
                # (not a Newton joint index). This allows us to find the specific MuJoCo
                # joint when Newton has combined multiple MJCF joints into one.
                dof_idx = target_idx
                dofs_per_world = len(dof_to_mjc_joint)
                if dof_idx < 0 or dof_idx >= dofs_per_world:
                    if wp.config.log_level <= wp.LOG_DEBUG:
                        print(f"Warning: MuJoCo actuator {mujoco_act_idx} has invalid DOF target {dof_idx}")
                    continue
                mjc_joint_idx = dof_to_mjc_joint[dof_idx]
                if mjc_joint_idx < 0 or mjc_joint_idx >= len(mjc_joint_names):
                    if wp.config.log_level <= wp.LOG_DEBUG:
                        print(f"Warning: MuJoCo actuator {mujoco_act_idx} DOF {dof_idx} not mapped to MuJoCo joint")
                    continue
                target_name = mjc_joint_names[mjc_joint_idx]
            elif trntype == int(SolverMuJoCo.TrnType.TENDON):
                try:
                    mjc_tendon_idx = selected_tendons.index(target_idx)
                    target_name = mjc_tendon_names[mjc_tendon_idx]
                except (ValueError, IndexError):
                    if wp.config.log_level <= wp.LOG_DEBUG:
                        print(f"Warning: MuJoCo actuator {mujoco_act_idx} references tendon {target_idx} not in MuJoCo")
                    continue
            elif trntype == int(SolverMuJoCo.TrnType.BODY):
                if target_idx < 0 or target_idx >= len(model.body_label):
                    if wp.config.log_level <= wp.LOG_DEBUG:
                        print(f"Warning: MuJoCo actuator {mujoco_act_idx} has invalid body target {target_idx}")
                    continue
                target_name = body_name_mapping.get(target_idx)
                if target_name is None:
                    if wp.config.log_level <= wp.LOG_DEBUG:
                        print(
                            f"Warning: MuJoCo actuator {mujoco_act_idx} references body {target_idx} "
                            "not present in the MuJoCo export."
                        )
                    continue
            elif trntype == int(SolverMuJoCo.TrnType.SITE):
                # Resolve site target: prefer label when available (USD path),
                # then fall back to index-based lookup (MJCF/direct trnid path).
                # Label-first avoids sentinel target_idx=0 colliding with a real site.
                site_name = None
                if target_label:
                    site_name = site_label_to_name.get(target_label)
                if site_name is None:
                    site_name = site_mapping.get(target_idx)
                if site_name is None:
                    if wp.config.log_level <= wp.LOG_DEBUG:
                        print(
                            f"Warning: MuJoCo actuator {mujoco_act_idx} site target "
                            f"'{target_label}' not found in site mapping"
                        )
                    continue
                target_name = site_name
                if target_idx_alt >= 0:
                    refsite_name = site_mapping.get(target_idx_alt)
                    if refsite_name is None:
                        if wp.config.log_level <= wp.LOG_DEBUG:
                            print(
                                f"Warning: MuJoCo actuator {mujoco_act_idx} references site {target_idx_alt} "
                                "as refsite, but it is not present in the MuJoCo export."
                            )
                        continue
            elif trntype == int(SolverMuJoCo.TrnType.SLIDERCRANK):
                target_name = site_mapping.get(target_idx)
                slider_site_name = site_mapping.get(target_idx_alt)
                if target_name is None or slider_site_name is None:
                    if wp.config.log_level <= wp.LOG_DEBUG:
                        print(
                            f"Warning: MuJoCo slider-crank actuator {mujoco_act_idx} references "
                            f"unavailable sites {target_idx}, {target_idx_alt}"
                        )
                    continue
            else:
                # TODO: Support remaining MuJoCo transmission types.
                if wp.config.log_level <= wp.LOG_DEBUG:
                    print(f"Warning: MuJoCo actuator {mujoco_act_idx} has unsupported trntype {trntype}")
                continue

            general_args = dict(actuator_args)
            if slider_site_name is not None:
                general_args["slidersite"] = slider_site_name

            # Get custom attributes for this MuJoCo actuator
            if hasattr(mujoco_attrs, "actuator_gainprm"):
                gainprm = mujoco_attrs.actuator_gainprm.numpy()[mujoco_act_idx]
                general_args["gainprm"] = list(gainprm)  # All 10 elements
            if hasattr(mujoco_attrs, "actuator_biasprm"):
                biasprm = mujoco_attrs.actuator_biasprm.numpy()[mujoco_act_idx]
                general_args["biasprm"] = list(biasprm)  # All 10 elements
            if hasattr(mujoco_attrs, "actuator_dynprm"):
                dynprm = mujoco_attrs.actuator_dynprm.numpy()[mujoco_act_idx]
                general_args["dynprm"] = list(dynprm)  # All 10 elements
            if hasattr(mujoco_attrs, "actuator_gear"):
                gear_arr = mujoco_attrs.actuator_gear.numpy()[mujoco_act_idx]
                general_args["gear"] = list(gear_arr)
            if hasattr(mujoco_attrs, "actuator_cranklength"):
                cranklength = float(mujoco_attrs.actuator_cranklength.numpy()[mujoco_act_idx])
                general_args["cranklength"] = cranklength
            # Only pass range to MuJoCo when explicitly set in MJCF (has_*range flags),
            # so MuJoCo can correctly resolve auto-limited flags via spec.compiler.autolimits.
            if has_ctrlrange_arr is not None and has_ctrlrange_arr[mujoco_act_idx]:
                general_args["ctrlrange"] = tuple(ctrlrange_arr[mujoco_act_idx])
            if ctrllimited_arr is not None:
                general_args["ctrllimited"] = int(ctrllimited_arr[mujoco_act_idx])
            if has_forcerange_arr is not None and has_forcerange_arr[mujoco_act_idx]:
                general_args["forcerange"] = tuple(forcerange_arr[mujoco_act_idx])
            if forcelimited_arr is not None:
                general_args["forcelimited"] = int(forcelimited_arr[mujoco_act_idx])
            if has_actrange_arr is not None and has_actrange_arr[mujoco_act_idx]:
                general_args["actrange"] = tuple(actrange_arr[mujoco_act_idx])
            if actlimited_arr is not None:
                general_args["actlimited"] = int(actlimited_arr[mujoco_act_idx])
            if hasattr(mujoco_attrs, "actuator_actearly"):
                actearly = mujoco_attrs.actuator_actearly.numpy()[mujoco_act_idx]
                general_args["actearly"] = bool(actearly)
            if hasattr(mujoco_attrs, "actuator_actdim"):
                actdim = mujoco_attrs.actuator_actdim.numpy()[mujoco_act_idx]
                if actdim >= 0:  # -1 means auto
                    general_args["actdim"] = int(actdim)
            if hasattr(mujoco_attrs, "actuator_dyntype"):
                dyntype = int(mujoco_attrs.actuator_dyntype.numpy()[mujoco_act_idx])
                general_args["dyntype"] = dyntype
            if hasattr(mujoco_attrs, "actuator_gaintype"):
                gaintype = int(mujoco_attrs.actuator_gaintype.numpy()[mujoco_act_idx])
                general_args["gaintype"] = gaintype
            if hasattr(mujoco_attrs, "actuator_biastype"):
                biastype = int(mujoco_attrs.actuator_biastype.numpy()[mujoco_act_idx])
                general_args["biastype"] = biastype
            # Detect position/velocity actuator shortcuts. Use set_to_position/
            # set_to_velocity after add_actuator so MuJoCo's compiler computes kd
            # from dampratio via mj_setConst (kd = dampratio * 2 * sqrt(kp * acc0)).
            shortcut = None  # "position" or "velocity" if detected
            shortcut_args: dict[str, float] = {}
            if general_args.get("biastype") == mujoco.mjtBias.mjBIAS_AFFINE and general_args.get("gainprm", [0])[0] > 0:
                kp = general_args["gainprm"][0]
                bp = general_args.get("biasprm", [0, 0, 0])
                # Position shortcut: biasprm = [0, -kp, -kv]
                # A positive biasprm[2] indicates a dampratio placeholder
                if bp[0] == 0 and abs(bp[1] + kp) < 1e-8:
                    shortcut = "position"
                    shortcut_args["kp"] = kp
                    if bp[2] < 0.0:
                        shortcut_args["kv"] = -bp[2]
                    elif bp[2] > 0.0:
                        shortcut_args["dampratio"] = bp[2]
                    for key in ("biasprm", "biastype", "gainprm", "gaintype"):
                        general_args.pop(key, None)
                # Velocity shortcut: biasprm = [0, 0, -kv] where kv = gainprm[0]
                elif bp[0] == 0 and bp[1] == 0 and bp[2] != 0:
                    kv = general_args["gainprm"][0]
                    if abs(bp[2] + kv) < 1e-8:
                        shortcut = "velocity"
                        shortcut_args["kv"] = kv
                        for key in ("biasprm", "biastype", "gainprm", "gaintype"):
                            general_args.pop(key, None)

            # Map trntype integer to MuJoCo enum and override default in general_args
            trntype_enum = {
                int(SolverMuJoCo.TrnType.JOINT): mujoco.mjtTrn.mjTRN_JOINT,
                int(SolverMuJoCo.TrnType.JOINT_IN_PARENT): mujoco.mjtTrn.mjTRN_JOINTINPARENT,
                int(SolverMuJoCo.TrnType.TENDON): mujoco.mjtTrn.mjTRN_TENDON,
                int(SolverMuJoCo.TrnType.SITE): mujoco.mjtTrn.mjTRN_SITE,
                int(SolverMuJoCo.TrnType.BODY): mujoco.mjtTrn.mjTRN_BODY,
                int(SolverMuJoCo.TrnType.SLIDERCRANK): mujoco.mjtTrn.mjTRN_SLIDERCRANK,
            }.get(trntype, mujoco.mjtTrn.mjTRN_JOINT)
            general_args["trntype"] = trntype_enum
            if refsite_name is not None:
                general_args["refsite"] = refsite_name
            act = spec.add_actuator(target=target_name, **general_args)
            if shortcut == "position":
                act.set_to_position(**shortcut_args)
            elif shortcut == "velocity":
                act.set_to_velocity(**shortcut_args)
            # CTRL_DIRECT actuators - store MJCF-order index into control.mujoco.ctrl
            # mujoco_act_idx is the index in Newton's mujoco:actuator frequency (MJCF order)
            mjc_actuator_ctrl_source_list.append(1)  # CTRL_DIRECT
            mjc_actuator_to_newton_idx_list.append(mujoco_act_idx)
            mjc_actuator_to_target_q_idx_list.append(-1)
            mjc_actuator_to_target_q_axis_idx_list.append(-1)
            mjc_actuator_to_newton_ball_jnt_list.append(-1)
            actuator_count += 1

        return actuator_count

    def __init__(
        self,
        model: Model,
        *,
        separate_worlds: bool | None = None,
        njmax: int | None = None,
        nconmax: int | None = None,
        iterations: int | None = None,
        ls_iterations: int | None = None,
        ccd_iterations: int | None = None,
        sdf_iterations: int | None = None,
        sdf_initpoints: int | None = None,
        solver: int | str | None = None,
        integrator: int | str | None = None,
        cone: int | str | None = None,
        jacobian: int | str | None = None,
        impratio: float | None = None,
        tolerance: float | None = None,
        ls_tolerance: float | None = None,
        ccd_tolerance: float | None = None,
        density: float | None = None,
        viscosity: float | None = None,
        wind: tuple | None = None,
        magnetic: tuple | None = None,
        use_mujoco_cpu: bool = False,
        enable_multiccd: bool = False,
        enable_sleeping: bool | None = None,
        nvmax: int | None = None,
        sleep_tolerance: float | None = None,
        world_solver: bool = False,
        disable_contacts: bool = False,
        disable_sensors: bool = False,
        update_data_interval: int = 1,
        dormant_contact_filter: bool = True,
        save_to_mjcf: str | None = None,
        use_mujoco_contacts: bool = True,
        include_sites: bool = True,
        skip_visual_only_geoms: bool = True,
        mesh_variant_sets: Sequence[SolverMuJoCo.MeshVariantSet] | None = None,
        deterministic: wp.DeterministicMode | None = None,
    ):
        """
        Solver options (e.g., ``impratio``) follow this resolution priority:

        1. **Constructor argument** - If provided, same value is used for all worlds.
        2. **Newton model custom attribute** (``model.mujoco.<option>``) - Supports per-world values.
        3. **MuJoCo default** - Used if neither of the above is set.

        Args:
            model: The model to be simulated.
            separate_worlds: If True, each Newton world is mapped to a separate MuJoCo world. Defaults to `not use_mujoco_cpu`.
            njmax: Maximum number of constraints per world. If None, a default value is estimated from the initial state. Note that the larger of the user-provided value or the default value is used.
            nconmax: Number of contact points per world. If None, a default value is estimated from the initial state. Note that the larger of the user-provided value or the default value is used.
            iterations: Number of solver iterations. If None, uses model custom attribute or MuJoCo's default (100).
            ls_iterations: Number of line search iterations for the solver. If None, uses model custom attribute or MuJoCo's default (50).
            ccd_iterations: Maximum CCD iterations. If None, uses model custom attribute or MuJoCo's default (35).
            sdf_iterations: Maximum SDF iterations. If None, uses model custom attribute or MuJoCo's default (10).
            sdf_initpoints: Number of SDF initialization points. If None, uses model custom attribute or MuJoCo's default (40).
            solver: Solver type. Can be "cg" or "newton", or their corresponding MuJoCo integer constants. If None, uses model custom attribute or Newton's default ("newton").
            integrator: Integrator type. Can be "euler", "rk4", or "implicitfast", or their corresponding MuJoCo integer constants. If None, uses model custom attribute or Newton's default ("implicitfast").
            cone: The type of contact friction cone. Can be "pyramidal", "elliptic", or their corresponding MuJoCo integer constants. If None, uses model custom attribute or Newton's default ("pyramidal").
            jacobian: Jacobian computation method. Can be "dense", "sparse", or "auto", or their corresponding MuJoCo integer constants. If None, uses model custom attribute or MuJoCo's default ("auto").
            impratio: Frictional-to-normal constraint impedance ratio. If None, uses model custom attribute or MuJoCo's default (1.0).
            tolerance: Solver tolerance for early termination. If None, uses model custom attribute or MuJoCo's default (1e-8).
            ls_tolerance: Line search tolerance for early termination. If None, uses model custom attribute or MuJoCo's default (0.01).
            ccd_tolerance: Continuous collision detection tolerance. If None, uses model custom attribute or MuJoCo's default (1e-6).
            density: Medium density for lift and drag forces. If None, uses model custom attribute or MuJoCo's default (0.0).
            viscosity: Medium viscosity for lift and drag forces. If None, uses model custom attribute or MuJoCo's default (0.0).
            wind: Wind velocity vector (x, y, z) for lift and drag forces. If None, uses model custom attribute or MuJoCo's default (0, 0, 0).
            magnetic: Global magnetic flux vector (x, y, z). If None, uses model custom attribute or MuJoCo's default (0, -0.5, 0).
            use_mujoco_cpu: If True, use the MuJoCo-C CPU backend instead of `mujoco_warp`.
            enable_multiccd: If True, enable multi-CCD contact generation (up to 4 contact points per geom pair instead of 1). Note: geom pairs where either geom has ``margin > 0`` always produce a single contact regardless of this flag.
            enable_sleeping: Whether to enable MuJoCo Warp's sleeping optimization. If None, uses the model custom attribute or defaults to False. Sleeping requires the GPU backend, the Newton solver, and a non-RK4 integrator.
            nvmax: Maximum number of active degrees of freedom per world when sleeping is enabled. Must accommodate every initially awake degree of freedom. If None, allocates space for every degree of freedom, which is safe but provides no compact-solver memory savings.
            sleep_tolerance: Sleep velocity tolerance. If None, uses model custom attribute or MuJoCo default (0.001).
            world_solver: Solve each world with MuJoCo Warp's per-world solver on the sleeping path; worlds it does not
                certify fall back to the stock solver inside the same step. Requires the sparse Jacobian and a MuJoCo
                Warp build with ``Option.world_solver``.
            disable_contacts: If True, disable contact computation in MuJoCo.
            disable_sensors: If True, disable sensor computation in MuJoCo.
            update_data_interval: Frequency (in simulation steps) at which to update the MuJoCo Data object from the Newton state. If 0, Data is never updated after initialization.
            dormant_contact_filter: When sleeping is enabled and contacts come from Newton's collision pipeline, skip converting contacts whose bodies are all asleep or immovable and inject them only when a tree wakes. Sleeping DOFs are frozen and excluded from the solve, so these rows never influence the solution; skipping them keeps ``nacon``/``nefc`` proportional to awake work. Independently of this flag, rows held in a pipeline :attr:`~newton.Contacts.dormant_contact_store` are injected in the substep their tree wakes. Injection runs after the adapter's own wake pass (external forces, pose edits, contact with an awake tree); trees that MJWarp wakes inside the step through tendon or equality constraints receive their rows one substep later.
            save_to_mjcf: Optional path to save the generated MJCF model file.
            use_mujoco_contacts: If True, use the MuJoCo contact solver. If False, use the Newton contact solver (newton contacts must be passed in through the step function in that case).
            include_sites: If ``True`` (default), Newton shapes marked with ``ShapeFlags.SITE`` are exported as MuJoCo sites. Sites are non-colliding reference points used for sensor attachment, debugging, or as frames of reference. If ``False``, sites are skipped during export. Defaults to ``True``.
            skip_visual_only_geoms: If ``True`` (default), geometries used only for visualization (i.e. not involved in collision) are excluded from the exported MuJoCo spec. This avoids mismatches with models that use explicit ``<contact>`` definitions for collision geometry.
            mesh_variant_sets: Optional fixed-topology mesh variant banks to compile
                into the GPU model. Runtime selection is available through
                :meth:`set_mesh_variant_index`. This requires separate-world GPU
                simulation.
            deterministic: Deterministic mode for MuJoCo Warp solver kernels. Pass a
                :class:`warp.DeterministicMode`, or ``None`` to inherit
                ``wp.config.deterministic``.
        """
        super().__init__(model)

        self._mesh_variant_definitions = tuple(mesh_variant_sets or ())
        self._mesh_variant_banks: dict[str, SolverMuJoCo._MeshVariantBank] = {}
        self._mesh_variant_asset_names: dict[str, tuple[tuple[str, ...], ...]] = {}

        # Import and cache MuJoCo modules (only happens once per class)
        mujoco, _ = self.import_mujoco()
        mujoco_attrs = getattr(model, "mujoco", None)
        if enable_sleeping is None:
            if mujoco_attrs is not None and hasattr(mujoco_attrs, "enable_sleeping"):
                enable_sleeping = bool(mujoco_attrs.enable_sleeping.numpy()[0])
            else:
                enable_sleeping = False
        elif not isinstance(enable_sleeping, bool | np.bool_):
            raise TypeError(f"enable_sleeping must be a bool or None, got {type(enable_sleeping).__name__}.")
        enable_sleeping = bool(enable_sleeping)
        if enable_sleeping and use_mujoco_cpu:
            raise ValueError("enable_sleeping=True requires the MuJoCo Warp GPU backend (use_mujoco_cpu=False).")
        if nvmax is not None:
            if isinstance(nvmax, bool) or not isinstance(nvmax, int | np.integer):
                raise TypeError(f"nvmax must be an integer or None, got {type(nvmax).__name__}.")
            if nvmax < 0:
                raise ValueError(f"nvmax must be non-negative, got {nvmax}.")
            if not enable_sleeping:
                raise ValueError("nvmax is only supported when sleeping is enabled.")
            if use_mujoco_cpu:
                raise ValueError("nvmax is only supported by the MuJoCo Warp GPU backend.")
            nvmax = int(nvmax)
        if sleep_tolerance is not None and (not math.isfinite(sleep_tolerance) or sleep_tolerance < 0.0):
            raise ValueError(f"sleep_tolerance must be finite and non-negative, got {sleep_tolerance}.")
        self.enable_sleeping = enable_sleeping
        """Whether MuJoCo Warp sleeping is enabled."""
        self.nvmax = nvmax
        """Maximum number of active degrees of freedom allocated per world."""
        self._sleep_awake_value = -(1 + int(mujoco.mjMINAWAKE))
        self._sleep_qpos_treeid: wp.array[wp.int32] | None = None
        self._sleep_tree_changed: wp.array2d[wp.int32] | None = None
        # Reset-world selection: ids compacted from the reset mask with their device count, and the
        # identity list used when every world resets (see _reset_world_selection).
        self._reset_world_ids: wp.array[wp.int32] | None = None
        self._reset_world_count: wp.array[wp.int32] | None = None
        self._all_world_ids: wp.array[wp.int32] | None = None
        self._initial_tree_asleep: wp.array[wp.int32] | None = None
        self._initial_tree_awake: wp.array[wp.int32] | None = None
        self._initial_body_awake: wp.array[wp.int32] | None = None
        self._initial_body_awake_ind: wp.array[wp.int32] | None = None
        self._initial_dof_awake_ind: wp.array[wp.int32] | None = None
        self._initial_ntree_awake = 0
        self._initial_nbody_awake = 0
        self._initial_nv_awake = 0
        self._initial_model_sync = True
        self._deterministic = deterministic if deterministic is not None else wp.config.deterministic
        self._deterministic_max_records = 0
        if not use_mujoco_cpu:
            # MJWarp's step pipeline spans several modules (forward dynamics,
            # smooth dynamics, constraints, solver, and optional collision).
            # Newton-to-MJWarp contact conversion also belongs to this solver
            # path and compacts contacts with atomics. Keep the resolved mode
            # as the single source of truth instead of relying on global Warp
            # determinism during module import.
            self._set_mujoco_warp_module_options()

        # Deferred from module scope: wp.static() in this kernel imports mujoco_warp.
        if not use_mujoco_cpu and use_mujoco_contacts and SolverMuJoCo._convert_mjw_contacts_to_newton_kernel is None:
            SolverMuJoCo._convert_mjw_contacts_to_newton_kernel = create_convert_mjw_contacts_to_newton_kernel()

        # --- New unified mappings: MuJoCo[world, entity] -> Newton[entity] ---
        self.mjc_body_to_newton: wp.array2d[wp.int32] | None = None
        """Mapping from MuJoCo [world, body] to Newton body index. Shape [nworld, nbody], dtype int32."""
        self.mjc_geom_to_newton_shape: wp.array2d[wp.int32] | None = None
        """Mapping from MuJoCo [world, geom] to Newton shape index. Shape [nworld, ngeom], dtype int32."""
        self._collision_shape_sleep_index: wp.array[wp.vec2i] | None = None
        self._body_sleep_index: wp.array[wp.vec2i] | None = None
        self._body_sleep_override: wp.array2d[wp.int32] | None = None
        self._has_body_sleep_overrides = False
        # Template-relative for per-world sites and absolute for global sites.
        self._mjc_site_shape_index: wp.array[wp.int32] | None = None
        self._mjc_site_is_global: wp.array[bool] | None = None
        self.mjc_jnt_to_newton_jnt: wp.array2d[wp.int32] | None = None
        """Mapping from MuJoCo [world, joint] to Newton joint index. Shape [nworld, njnt], dtype int32."""
        self.mjc_jnt_to_newton_dof: wp.array2d[wp.int32] | None = None
        """Mapping from MuJoCo [world, joint] to Newton DOF index. Shape [nworld, njnt], dtype int32."""
        self.mjc_dof_to_newton_dof: wp.array2d[wp.int32] | None = None
        """Mapping from MuJoCo [world, dof] to Newton DOF index. Shape [nworld, nv], dtype int32."""
        self.newton_dof_to_body: wp.array[wp.int32] | None = None
        """Mapping from Newton DOF index to child body index. Shape [joint_dof_count], dtype int32."""
        self.mjc_mocap_to_newton_jnt: wp.array2d[wp.int32] | None = None
        """Mapping from MuJoCo [world, mocap] to Newton joint index. Shape [nworld, nmocap], dtype int32."""
        self.mjc_actuator_ctrl_source: wp.array[wp.int32] | None = None
        """Control source for each MuJoCo actuator.

        Values: 0=JOINT_TARGET (uses joint_target_q/joint_target_qd), 1=CTRL_DIRECT (uses mujoco.ctrl)
        Shape [nu], dtype int32."""
        self.mjc_actuator_to_newton_idx: wp.array[wp.int32] | None = None
        """Mapping from MuJoCo actuator to Newton index.

        For JOINT_TARGET: sign-encoded DOF index (>=0: position, -1: unmapped, <=-2: velocity with -(idx+2))
        For CTRL_DIRECT: MJCF-order index into control.mujoco.ctrl array

        Shape [nu], dtype int32."""
        self.mjc_actuator_to_newton_target_q_idx: wp.array[wp.int32] | None = None
        """Per-actuator coordinate lookup for target conversion.

        For position actuators, entries index :attr:`Control.joint_target_q` using
        :attr:`Model.joint_target_q_start`. For BALL-joint velocity actuators, entries index the
        current quaternion in :attr:`State.joint_q` using :attr:`Model.joint_q_start`, so the
        target velocity can be rotated into MuJoCo's current ball-joint frame. ``-1`` means unused.
        Note that the BALL velocity case reuses this existing target-q lookup slot.
        Shape ``[nu]``, dtype ``int32``."""
        self.mjc_actuator_to_target_q_axis_idx: wp.array[wp.int32] | None = None
        """Angular-axis selector (``0``/``1``/``2``) for BALL-joint position and
        velocity actuators, ``-1`` otherwise. Shape ``[nu]``."""
        self.mjc_actuator_to_newton_ball_jnt: wp.array[wp.int32] | None = None
        """Per-actuator template-world Newton joint index for BALL-joint actuators, ``-1`` otherwise.

        The control kernel uses this to read the current world's child-anchor rotation from
        ``joint_X_c[world * joints_per_world + jnt % joints_per_world]``.
        Shape ``[nu]``, dtype ``int32``."""
        self.mjc_eq_to_newton_eq: wp.array2d[wp.int32] | None = None
        """Mapping from MuJoCo [world, eq] to Newton equality constraint index.

        Corresponds to the equality constraints that are created in MuJoCo from Newton's equality constraints.
        A value of -1 indicates the entry is unmapped -- the MuJoCo constraint was synthesized from a loop-closure
        joint (CONNECT or WELD) rather than from an explicit Newton equality constraint. For CONNECT-only
        entries see :attr:`mjc_eq_to_newton_jnt`.

        Shape [nworld, neq], dtype int32."""
        self.mjc_eq_to_newton_jnt: wp.array2d[wp.int32] | None = None
        """Mapping from MuJoCo [world, eq] to Newton joint index for CONNECT constraints only.

        Corresponds to the CONNECT equality constraints synthesized from
        loop-closure joints (revolute or ball) that have no associated
        articulation (``joint_articulation == -1``).  WELD constraints
        (from FIXED loop joints) are excluded — their ``eq_data`` is
        managed entirely by MuJoCo and must not be overwritten by the
        CONNECT anchor kernels.  A value of -1 indicates an unmapped
        entry (either an explicit Newton equality constraint or a WELD).

        Shape [nworld, neq], dtype int32."""
        self.mjc_eq_to_newton_mimic: wp.array2d[wp.int32] | None = None
        """Mapping from MuJoCo [world, eq] to Newton mimic constraint index.

        Corresponds to the equality constraints that are created in MuJoCo from Newton's mimic constraints.
        A value of -1 indicates that the MuJoCo equality constraint is not associated with a Newton mimic constraint.

        Shape [nworld, neq], dtype int32."""
        self.mjc_tendon_to_newton_tendon: wp.array2d[wp.int32] | None = None
        """Mapping from MuJoCo [world, tendon] to Newton tendon index.

        Shape [nworld, ntendon], dtype int32."""
        self.body_free_qd_start: wp.array[wp.int32] | None = None
        """Per-body mapping to the free-joint qd_start index (or -1 if not free)."""

        # --- Conditional/lazy mappings ---
        self.newton_shape_to_mjc_geom: wp.array[wp.int32] | None = None
        """Inverse mapping from Newton shape index to MuJoCo geom index. Only created when use_mujoco_contacts=False. Shape [nshape], dtype int32."""

        # --- Helper arrays for actuator types ---

        # --- Internal state for mapping creation ---
        self._shapes_per_world: int = 0
        """Number of shapes per world (for computing Newton shape indices from template)."""
        self._first_env_shape_base: int = 0
        """Base shape index for the first environment."""

        # --- Internal state for connect constraint anchor computation ---
        self.has_connect_constraints: bool = False
        """Whether the model contains any CONNECT equality constraints."""
        self.has_jnt_connect_constraints: bool = False
        """Whether the model contains CONNECT constraints synthesized from loop-closure joints (revolute or ball)."""
        self.connect_constraint_q_rel: wp.array | None = None
        """Relative rotation ``inv(q2) * q1`` at the reference pose per equality constraint, ``wp.array[wp.quat]``, shape ``[equality_constraint_count]``."""
        self.connect_constraint_t_rel: wp.array | None = None
        """Relative translation [m] at the reference pose per equality constraint, ``wp.array[wp.vec3]``, shape ``[equality_constraint_count]``."""

        # --- Internal state for anchor computation of joint-synthesized CONNECT constraints ---
        self.jnt_eq_anchor1: wp.array2d[wp.vec3] | None = None
        """Body1-local anchor [m] per ``[world, eq]`` for joint-synthesized CONNECT constraints, ``wp.array2d[wp.vec3]``, shape ``[world_count, neq]``."""
        self.jnt_eq_anchor1_has_axis_offset: wp.array2d[wp.int32] | None = None
        """Whether each ``[world, eq]`` entry is the second hinge CONNECT offset along the joint axis, ``wp.array2d[wp.int32]``, shape ``[world_count, neq]``."""
        self.jnt_connect_constraint_q_rel: wp.array2d[wp.quat] | None = None
        """Relative rotation per ``[world, eq]`` for joint-synthesized CONNECT constraints, ``wp.array2d[wp.quat]``, shape ``[world_count, neq]``."""
        self.jnt_connect_constraint_t_rel: wp.array2d[wp.vec3] | None = None
        """Relative translation [m] per ``[world, eq]`` for joint-synthesized CONNECT constraints, ``wp.array2d[wp.vec3]``, shape ``[world_count, neq]``."""

        self._viewer = None
        """Instance of the MuJoCo viewer for debugging."""

        self._use_mujoco_contacts = use_mujoco_contacts
        """Whether MuJoCo handles collision detection.

        Controls margin zeroing: when True, geom/pair margins on the MuJoCo
        model are kept at zero for NATIVECCD compatibility (#2106)."""

        # mujoco_warp.put_model() rejects non-zero margins on box-box pairs
        # (default NATIVECCD path) or any box/mesh pair when MULTICCD is enabled.
        # Skip the workaround entirely when no such geom types exist in the model.
        shape_types_arr = model.shape_type.numpy()
        has_box = bool(np.any(shape_types_arr == GeoType.BOX))
        has_mesh = bool(np.any((shape_types_arr == GeoType.MESH) | (shape_types_arr == GeoType.CONVEX_MESH)))
        self._zero_margins_for_native_ccd = has_box or (enable_multiccd and has_mesh)
        """True when the NATIVECCD/MULTICCD margin workaround applies (#2106)."""

        enableflags = 0
        disableflags = 0
        if enable_sleeping:
            enableflags |= mujoco.mjtEnableBit.mjENBL_SLEEP
        if not enable_multiccd:
            disableflags |= mujoco.mjtDisableBit.mjDSBL_MULTICCD
        if disable_contacts:
            disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        if disable_sensors:
            disableflags |= mujoco.mjtDisableBit.mjDSBL_SENSOR
        self.use_mujoco_cpu = use_mujoco_cpu
        if use_mujoco_contacts or use_mujoco_cpu:
            mujoco_attrs_for_warn = getattr(model, "mujoco", None)
            solref_mode_attr = (
                getattr(mujoco_attrs_for_warn, "solref_mode", None) if mujoco_attrs_for_warn is not None else None
            )
            if solref_mode_attr is not None:
                force_space_shapes = int(np.count_nonzero(solref_mode_attr.numpy() == SOLREF_MODE_FORCE_SPACE))
                if force_space_shapes > 0:
                    backends = [
                        name
                        for name, flag in (
                            ("use_mujoco_contacts=True", use_mujoco_contacts),
                            ("use_mujoco_cpu=True", use_mujoco_cpu),
                        )
                        if flag
                    ]
                    warnings.warn(
                        f"{force_space_shapes} shape(s) have mujoco.solref_mode == SOLREF_MODE_FORCE_SPACE "
                        f"but SolverMuJoCo is running with {', '.join(backends)}. The per-contact "
                        "body_invweight0 override only fires on the Newton-contacts GPU path "
                        "(use_mujoco_contacts=False); on this backend these shapes silently fall back to "
                        "the legacy convert_solref(ke, kd, 1, 1) approximation and shape_material_ke/kd "
                        "will not behave as force-space gains. See "
                        "docs/solvers/mujoco.rst > 'Shape-material contact stiffness and damping'.",
                        stacklevel=2,
                    )
        if separate_worlds is None:
            separate_worlds = not use_mujoco_cpu and model.world_count > 1
        if self._mesh_variant_definitions:
            if use_mujoco_cpu:
                raise ValueError("mesh_variant_sets requires the MuJoCo Warp GPU backend.")
            if not separate_worlds:
                raise ValueError("mesh_variant_sets requires separate_worlds=True.")
            self._validate_mesh_variant_definitions(model)
        # Buffers for the fast-path contact conversion optimisation.
        # See _prepare_contact_conversion / convert_newton_contacts_to_mjwarp_kernel.
        # Initialised before _convert_to_mjc because notify_model_changed (called
        # during conversion) may call _invalidate_contact_fast_path.
        #
        # Allocate the generation and count buffers before conversion because
        # notify_model_changed() may invalidate them during conversion. The
        # contact map is allocated below once the converted capacity is known.
        self._contact_tid_to_cid: wp.array[wp.int32] | None = None
        self._last_contact_generation = wp.full(1, _GENERATION_SENTINEL, dtype=wp.int32, device=self.device)
        self._last_nacon_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        # Track the Contacts instance and its capacity, plus the MJWarp
        # naconmax used during the last full pass.  Any change to these
        # invariants invalidates the cached tid_to_cid mapping because the
        # cached cid values would index into a different output buffer.
        # Note: we key on id(contacts.contact_generation) (a stable per-Contacts
        # device array) rather than id(contacts).  Empirically, keying on the
        # outer Contacts wrapper produces broken binaries in the dexsuite
        # workload (root cause unclear; the inner array's id is what works).
        self._last_contacts_id: int | None = None
        self._last_rigid_contact_max: int | None = None
        self._last_naconmax: int | None = None

        # One-shot dedup for ``_update_solref_from_invweight0``'s authored
        # ``mujoco.solreflimit`` domain validator. Re-armed by
        # ``notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)``.
        self._raw_solreflimit_validated: bool = False

        with wp.ScopedTimer("convert_model_to_mujoco", active=False):
            self._convert_to_mjc(
                model,
                enableflags=enableflags,
                disableflags=disableflags,
                disable_contacts=disable_contacts,
                separate_worlds=separate_worlds,
                njmax=njmax,
                nconmax=nconmax,
                nvmax=nvmax,
                iterations=iterations,
                ls_iterations=ls_iterations,
                ccd_iterations=ccd_iterations,
                sdf_iterations=sdf_iterations,
                sdf_initpoints=sdf_initpoints,
                cone=cone,
                jacobian=jacobian,
                impratio=impratio,
                tolerance=tolerance,
                ls_tolerance=ls_tolerance,
                ccd_tolerance=ccd_tolerance,
                sleep_tolerance=sleep_tolerance,
                density=density,
                viscosity=viscosity,
                wind=wind,
                magnetic=magnetic,
                solver=solver,
                integrator=integrator,
                target_filename=save_to_mjcf,
                include_sites=include_sites,
                skip_visual_only_geoms=skip_visual_only_geoms,
            )
        if world_solver:
            if use_mujoco_cpu or not hasattr(self.mjw_model.opt, "world_solver"):
                raise ValueError("world_solver=True requires MuJoCo Warp with Option.world_solver.")
            self.mjw_model.opt.world_solver = True
        self._mesh_variant_definitions = ()
        self._mesh_variant_asset_names.clear()
        if not use_mujoco_cpu and not use_mujoco_contacts:
            self._contact_tid_to_cid = wp.full(self.mjw_data.naconmax, -1, dtype=wp.int32, device=self.device)
        self._initial_model_sync = False
        self.update_data_interval = update_data_interval
        self.dormant_contact_filter = bool(dormant_contact_filter) and self.enable_sleeping and not use_mujoco_contacts
        """Whether contacts between asleep/immovable bodies are parked until a tree wakes."""
        self._dormant_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._wake_event = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._tree_asleep_prev = (
            wp.zeros(self.mjw_data.tree_asleep.shape, dtype=wp.int32, device=self.device)
            if self.enable_sleeping and self.mjw_data is not None
            else None
        )
        self._dormant_flag: wp.array[wp.int32] | None = None
        self._dormant_tids: wp.array[wp.int32] | None = None
        if self._contact_tid_to_cid is not None:
            self._dormant_flag = wp.zeros(self._contact_tid_to_cid.shape[0], dtype=wp.int32, device=self.device)
            self._dormant_tids = wp.zeros(self._contact_tid_to_cid.shape[0], dtype=wp.int32, device=self.device)
        self._step = 0
        # opt.timestep is written at the first substep of a tick (and whenever dt changes); the
        # intermediate substeps of the same tick reuse it.
        self._timestep_fill_pending = True
        self._last_timestep: float | None = None
        self.publish_intermediate_body_state: bool = True
        """Whether intermediate substeps (:meth:`_step_intermediate`) publish ``body_q``/``body_qd``.

        With the in-step contact conversion the converted contacts read their body poses from
        MuJoCo Warp's kinematics, so Newton's forward kinematics is only needed when the state is
        consumed outside the solver. Set ``False`` when nothing reads Newton body state between the
        substeps of a tick (no mid-tick collision, no per-substep force callbacks); the finalizing
        :meth:`step` always publishes. Ignored when the in-step conversion is unavailable.
        """
        self.publish_intermediate_joint_state: bool = True
        """Whether intermediate substeps publish ``joint_q``/``joint_qd`` and re-read them.

        Set ``False`` when nothing reads or writes Newton joint state between the substeps of a
        tick: intermediate substeps then skip the joint-coordinate publish, the ``update_data_interval``
        re-sync of ``qpos``/``qvel`` from the state runs on the first substep of a tick only, and the
        finalizing :meth:`step` publishes as usual. Implies the body-state publish is skipped as well
        unless :attr:`publish_intermediate_body_state` requests it.
        """
        self.mjwarp_publish_derived: bool = False
        """Whether MuJoCo Warp's fused forward publishes the derived data Newton never reads.

        ``Option.fused_world_publish_derived`` (MuJoCo Warp versions that have it) is set from this
        flag before every step: ``False`` lets the fused per-world forward skip the derived fields
        no stage of the step and no Newton conversion reads (``xmat``, ``ximat``, ``xanchor``,
        ``xaxis``, ``geom_xpos``, ``geom_xmat``, ``site_xpos``, ``site_xmat``, ``crb``,
        ``actuator_length``, ``actuator_velocity``, ``actuator_moment``, ``cdof_dot``,
        ``qfrc_spring``, ``qfrc_damper``, ``qfrc_adhesion``, ``cacc``, ``cfrc_int``). Requests
        for ``body_qdd``/``body_parent_f`` (which read ``cacc``/``cfrc_int``) and sensors publish
        them regardless. Set ``True`` when reading those fields from :attr:`mjw_data` directly.
        """
        self.conditional_wake_injection: bool = False
        """Whether the dormant-contact injection runs under a conditional graph node.

        ``False`` (default) launches the injection kernels every substep; without a wake event they
        exit early. ``True`` wraps them in ``wp.capture_if`` on the wake event instead. Measured in a
        CUDA graph, the conditional node costs about 9 us of scheduling per substep whether or not
        the branch is taken, more than the near-empty launches it skips, and the wake event is
        shared by all worlds so large batches take the branch on most substeps anyway.
        """
        # Newton-side contacts are converted inside the MJWarp step, after its wake pass and
        # kinematics, through the post_position callback (MJWarp versions without the hook keep the
        # pre-step conversion from body_q).
        self._contact_pose_hook = (
            not use_mujoco_cpu
            and not use_mujoco_contacts
            and self.mjw_model is not None
            and hasattr(self.mjw_model.callback, "post_position")
        )
        self._hook_contacts: Contacts | None = None
        self._hook_state_in: State | None = None
        # Per-MJWarp-contact records for the in-step refresh: Newton row, MuJoCo bodies and the
        # pose-independent offset scales / surface radii (see _convert_one_contact).
        self._hook_tid: wp.array[wp.int32] | None = None
        self._hook_body: wp.array[wp.vec2i] | None = None
        self._hook_geom: wp.array[wp.vec4] | None = None
        if self._contact_pose_hook:
            self._allocate_hook_records(self.mjw_data.naconmax)

        if self.mjw_model is not None:
            self.mjw_model.opt.run_collision_detection = use_mujoco_contacts

    @contextmanager
    def _scoped_deterministic_config(self):
        """Apply solver-local determinism for lazily-created MJWarp kernels."""
        original_mode = wp.config.deterministic
        original_max_records = wp.config.deterministic_max_records
        try:
            # MJWarp creates several ``module="unique"`` kernels lazily while
            # stepping. Those generated modules do not inherit options from the
            # source Python modules above, so keep the solver options active
            # while the step path compiles/captures its kernels.
            wp.config.deterministic = self._deterministic
            wp.config.deterministic_max_records = self._deterministic_max_records
            yield
        finally:
            wp.config.deterministic = original_mode
            wp.config.deterministic_max_records = original_max_records

    @contextmanager
    def _scoped_mujoco_warp_execution(self):
        """Prepare and apply all solver-local MJWarp compilation options."""
        self._apply_module_options()
        self._prepare_generated_kernels()
        with self._scoped_deterministic_config():
            yield

    @event_scope
    def _mujoco_warp_step(self, finalize: bool = True, contacts: Contacts | None = None, state_in: State | None = None):
        """Advance MJWarp by one substep.

        Args:
            finalize: Whether this substep materializes MJWarp's deferred derived data.
            contacts: Newton contacts converted inside the step (:meth:`_convert_contacts_in_step`);
                ``None`` leaves the ``post_position`` callback unbound.
            state_in: Newton input state of the substep (contact conversion inputs).
        """
        callback = self.mjw_model.callback
        if contacts is not None:
            self._hook_contacts = contacts
            self._hook_state_in = state_in
            callback.post_position = self._convert_contacts_in_step
        try:
            if finalize:
                self._mujoco_warp.step(self.mjw_model, self.mjw_data)
            else:
                self._mujoco_warp._step_intermediate(self.mjw_model, self.mjw_data)
        finally:
            if contacts is not None:
                # scoped to the step so other MJWarp passes on this model (reset) leave the rows alone
                callback.post_position = None
                self._hook_contacts = None
                self._hook_state_in = None

    def _convert_contacts_in_step(self, m: MjWarpModel, d: MjWarpData) -> None:
        """Convert the Newton contacts inside the MJWarp step, after its wake pass and kinematics.

        Bound to ``m.callback.post_position`` for the duration of :meth:`_mujoco_warp_step`, so it
        runs after MJWarp's own ``sleep.wake`` (forces, velocities, user requests) and kinematics
        and before ``make_constraint`` reads the contact rows. Mirrors the collision prelude of
        MJWarp's internal step for externally supplied contacts: conversion of the live rows (or
        the by-contact refresh of ``dist``/``pos`` from ``xpos``/``xquat`` on substeps that keep
        the contact set), the collision wake, and the injection of dormant contacts of trees that
        woke since the previous substep. Trees woken here take part in this substep's solve, as
        with MJWarp's ``wake_collision``; the awake counters and body/DOF compaction are left to
        MJWarp's forward pass, which recomputes them before they are read.
        """
        contacts = self._hook_contacts
        state_in = self._hook_state_in
        model = self.model
        inject, store = self._wake_injection_store(contacts) if self.enable_sleeping else (False, None)
        self._launch_contact_conversion(
            model,
            state_in,
            contacts,
            xpos=d.xpos,
            xquat=d.xquat,
            tree_asleep_prev=self._tree_asleep_prev if inject else None,
        )
        if self.enable_sleeping:
            self._wake_contact_trees(contacts, tree_asleep_prev=self._tree_asleep_prev if inject else None)
            if inject:
                self._inject_woken_dormant_contacts(model, state_in, contacts, store, xpos=d.xpos, xquat=d.xquat)
        else:
            self._snapshot_contact_generation(contacts)

    def _allocate_hook_records(self, naconmax: int) -> None:
        """Allocate the per-MJWarp-contact records of the in-step refresh for ``naconmax`` contacts."""
        self._hook_tid = wp.zeros(naconmax, dtype=wp.int32, device=self.device)
        self._hook_body = wp.zeros(naconmax, dtype=wp.vec2i, device=self.device)
        self._hook_geom = wp.zeros(naconmax, dtype=wp.vec4, device=self.device)

    def _apply_mjwarp_publish_option(self, state_out: State) -> None:
        """Set ``Option.fused_world_publish_derived`` for this step (see :attr:`mjwarp_publish_derived`)."""
        m = self.mjw_model
        if not hasattr(m.opt, "fused_world_publish_derived"):
            return
        m.opt.fused_world_publish_derived = bool(
            self.mjwarp_publish_derived
            or m.nsensor > 0
            or m.sensor_rne_postconstraint
            or state_out.body_qdd is not None
            or state_out.body_parent_f is not None
        )

    @property
    def mesh_variant_names(self) -> tuple[str, ...]:
        """Names of the compiled mesh variant sets."""
        return tuple(self._mesh_variant_banks)

    def mesh_variant_ids(self, name: str) -> wp.array[wp.int32]:
        """Return the selected variant id in each world."""
        try:
            return self._mesh_variant_banks[name].variant_ids
        except KeyError as error:
            raise KeyError(f"Unknown mesh variant set {name!r}; available sets: {self.mesh_variant_names}.") from error

    def set_mesh_variant_index(
        self,
        name: str,
        *,
        variant_ids: wp.array[wp.int32],
        world_ids: wp.array[wp.int32],
    ) -> None:
        """Select precompiled geometry and inertia rows at reset.

        Args:
            name: Mesh variant set name.
            variant_ids: Variant id per selected world, shape ``[selection_count]``.
                Every value must be in the set's variant range.
            world_ids: Valid, unique world indices, shape ``[selection_count]``.
        """
        try:
            bank = self._mesh_variant_banks[name]
        except KeyError as error:
            raise KeyError(f"Unknown mesh variant set {name!r}; available sets: {self.mesh_variant_names}.") from error
        for argument_name, array in (("variant_ids", variant_ids), ("world_ids", world_ids)):
            if not isinstance(array, wp.array) or array.dtype != wp.int32:
                raise TypeError(f"{argument_name} must be a one-dimensional wp.array with dtype wp.int32.")
            if array.ndim != 1:
                raise ValueError(f"{argument_name} must be one-dimensional; got shape {array.shape}.")
            if array.device != self.model.device:
                raise ValueError(f"{argument_name} must be on {self.model.device}, got {array.device}.")
        if variant_ids.shape != world_ids.shape:
            raise ValueError(
                f"variant_ids and world_ids must have equal shapes; got {variant_ids.shape} and {world_ids.shape}."
            )
        if not variant_ids.shape[0]:
            return

        with wp.ScopedDevice(self.model.device), self._scoped_mujoco_warp_execution():
            wp.launch(
                set_mesh_variant_index_kernel,
                dim=(variant_ids.shape[0], bank.shape_indices.shape[1]),
                inputs=[
                    variant_ids,
                    world_ids,
                    bank.shape_indices,
                    bank.geom_indices,
                    self.model.shape_body,
                    bank.mj_body_index,
                    bank.mj_dof_index,
                    bank.shapes,
                    bank.bodies,
                    self.mjw_model.nv,
                    _GENERATION_SENTINEL,
                ],
                outputs=[
                    self.model.shape_scale,
                    self.model.shape_transform,
                    self.model.shape_collision_radius,
                    self.model.shape_source_ptr,
                    self.model._shape_mesh_properties,
                    self.model._shape_sdf_index,
                    self.model.shape_edge_range,
                    self.model.shape_collision_aabb_lower,
                    self.model.shape_collision_aabb_upper,
                    self.model._shape_voxel_resolution,
                    self.model.body_mass,
                    self.model.body_inv_mass,
                    self.model.body_com,
                    self.model.body_inertia,
                    self.model.body_inv_inertia,
                    self.mjw_model.geom_dataid,
                    self.mjw_model.geom_size,
                    self.mjw_model.geom_rbound,
                    self.mjw_model.geom_aabb,
                    self.mjw_model.geom_pos,
                    self.mjw_model.geom_quat,
                    self.mjw_model.body_mass,
                    self.mjw_model.body_subtreemass,
                    self.mjw_model.body_ipos,
                    self.mjw_model.body_inertia,
                    self.mjw_model.body_iquat,
                    self.mjw_model.body_invweight0,
                    self.mjw_model.dof_invweight0,
                    self.mjw_model.stat.meaninertia,
                    self.mjw_data.qacc_warmstart,
                    self.mjw_data.nacon,
                    self._last_contact_generation,
                    self._last_nacon_count,
                    bank.variant_ids,
                ],
                device=self.model.device,
            )
            if self.enable_sleeping and bank.mj_dof_index >= 0:
                wp.launch(
                    wake_selected_tree_kernel,
                    dim=world_ids.shape[0],
                    inputs=[
                        world_ids,
                        int(self.mj_model.body_treeid[bank.mj_body_index]),
                        self.mjw_model.ntree,
                        self._sleep_awake_value,
                    ],
                    outputs=[self.mjw_data.tree_asleep],
                    device=self.model.device,
                )

    def _step_impl(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts,
        dt: float,
        *,
        finalize: bool,
    ) -> None:
        if self.use_mujoco_cpu:
            self._apply_mjc_control(self.model, state_in, control, self.mj_data)
            if self.update_data_interval > 0 and self._step % self.update_data_interval == 0:
                # XXX updating the mujoco state at every step may introduce numerical instability
                self._update_mjc_data(self.mj_data, self.model, state_in)
            self.mj_model.opt.timestep = dt
            self._mujoco.mj_step(self.mj_model, self.mj_data)
            self._update_newton_state(self.model, state_out, self.mj_data, state_prev=state_in)
        else:
            with wp.ScopedDevice(self.model.device), self._scoped_mujoco_warp_execution():
                self._enable_rne_postconstraint(state_out)
                self._apply_mjwarp_publish_option(state_out)
                convert_contacts = not self.mjw_model.opt.run_collision_detection
                prepared = self._apply_mjc_inputs(
                    self.model, state_in, control, contacts, prepare_contacts=convert_contacts
                )
                first_of_tick = self._timestep_fill_pending
                if self.update_data_interval > 0 and (
                    first_of_tick
                    if not self.publish_intermediate_joint_state
                    else self._step % self.update_data_interval == 0
                ):
                    self._update_mjc_data(self.mjw_data, self.model, state_in)
                if first_of_tick or dt != self._last_timestep:
                    self.mjw_model.opt.timestep.fill_(dt)
                    self._last_timestep = dt
                self._timestep_fill_pending = finalize
                hook = convert_contacts and self._contact_pose_hook
                if convert_contacts:
                    self._prepare_contact_conversion(contacts, prepared=prepared)
                    if not hook:
                        # pre-step conversion from body_q (MJWarp without the post_position hook)
                        self._launch_contact_conversion(self.model, state_in, contacts)
                        if self.enable_sleeping:
                            self._wake_before_step(self.model, state_in, contacts)
                        else:
                            self._snapshot_contact_generation(contacts)
                self._mujoco_warp_step(finalize, contacts if hook else None, state_in)
                publish_body_state = finalize or not hook or self.publish_intermediate_body_state
                self._update_newton_state(
                    self.model,
                    state_out,
                    self.mjw_data,
                    state_prev=state_in,
                    publish_body_state=publish_body_state,
                    publish_joint_state=publish_body_state or self.publish_intermediate_joint_state,
                )
        self._step += 1

    @event_scope
    @override
    def step(self, state_in: State, state_out: State, control: Control, contacts: Contacts, dt: float) -> None:
        self._step_impl(state_in, state_out, control, contacts, dt, finalize=True)

    @event_scope
    def _step_intermediate(
        self, state_in: State, state_out: State, control: Control, contacts: Contacts, dt: float
    ) -> None:
        """Advance an intermediate GPU substep through MJWarp's safe deferred-data path."""
        self._step_impl(state_in, state_out, control, contacts, dt, finalize=False)

    @override
    def reset(
        self,
        state: State,
        world_mask: wp.array[wp.bool] | None = None,
        flags: StateFlags | int | None = None,
    ) -> None:
        """Reset joint state to model defaults and clear MuJoCo's internal buffers.

        MuJoCo carries solver state (acceleration warm-start, actuator
        activations) and applied-force inputs across :meth:`step` calls. After a
        divergence (e.g. NaNs), these buffers can poison the next step even once
        the joint state has been reset, because :meth:`step` warm-starts from
        them. This method therefore always zeros, per world, ``qacc_warmstart``,
        ``qfrc_applied``, ``xfrc_applied``, ``act`` and ``ctrl``. (``qacc`` is
        left alone: the solver overwrites it from ``qacc_warmstart`` at the start
        of every step.)

        In addition, the requested entries of the Newton :class:`~newton.State`
        are reset to the model defaults (``model.joint_q`` / ``model.joint_qd``)
        for the selected worlds, controlled by *flags*:

        * :attr:`~newton.StateFlags.JOINT_Q` resets ``state.joint_q``.
        * :attr:`~newton.StateFlags.JOINT_QD` resets ``state.joint_qd``.

        Because MuJoCo is a reduced-coordinate solver, ``state.body_q`` /
        ``state.body_qd`` are derived from the joint coordinates by forward
        kinematics on the next :meth:`step`; the corresponding
        :attr:`~newton.StateFlags.BODY_Q` / :attr:`~newton.StateFlags.BODY_QD`
        (and particle) flags are not actionable here and are ignored.

        ``qpos`` / ``qvel`` are normally synced from the reset ``state`` at the
        start of the next :meth:`step`. When ``update_data_interval != 1`` that
        per-step sync is disabled or sparse, so the reset joint coordinates are
        pushed into ``qpos`` / ``qvel`` immediately for the selected worlds.
        When sleeping is enabled, reset synchronizes immediately, rebuilds
        MuJoCo Warp's cached position- and velocity-dependent data, and restores
        the initial sleep state in the selected worlds.

        With MuJoCo Warp the work is proportional to the number of selected
        worlds: the mask is compacted on the device into a world id list (no
        host synchronization), every reset kernel runs over that list, and the
        derived data is rebuilt with ``mujoco_warp.forward_worlds``, which
        leaves unselected worlds untouched. Models outside MuJoCo Warp's fused
        per-world path fall back to rebuilding every world.

        Args:
            state: The simulation state to reset (modified in place).
            world_mask: Optional boolean mask of shape ``(world_count + 1,)``
                selecting which worlds to reset. The final entry represents
                global world ``-1`` and is a no-op because MuJoCo does not
                support global dynamic objects. If ``None``, all worlds are
                reset.

                .. deprecated:: 1.5
                    Passing a mask with shape ``(world_count,)`` is deprecated.
                    Use shape ``(world_count + 1,)`` with a final ``False`` entry
                    to select local worlds only.
            flags: Optional :class:`~newton.StateFlags` bitmask controlling which
                joint-state quantities are reset. If ``None``, all are reset.
                The internal MuJoCo buffers are always cleared regardless.
        """
        if state is None:
            raise ValueError("'state' argument is required.")

        world_mask = self._normalize_reset_world_mask(world_mask)
        world_count = self.model.world_count
        native_template_world_selected = not self.use_mujoco_cpu or world_mask is None or bool(world_mask.numpy()[0])

        # Reset joint coordinates/velocities to model defaults for the selected
        # worlds. body_q/body_qd are FK outputs and intentionally not touched.
        flags_value = int(StateFlags.ALL if flags is None else flags)
        reset_q = bool(flags_value & StateFlags.JOINT_Q) and state.joint_q is not None
        reset_qd = bool(flags_value & StateFlags.JOINT_QD) and state.joint_qd is not None
        if reset_q or reset_qd:
            coords_per_world = self.model.joint_coord_count // world_count
            dofs_per_world = self.model.joint_dof_count // world_count
            joint_dim = max(coords_per_world if reset_q else 0, dofs_per_world if reset_qd else 0)
            if joint_dim > 0:
                wp.launch(
                    reset_joint_state_kernel,
                    dim=(world_count, joint_dim),
                    inputs=[
                        world_mask,
                        coords_per_world,
                        dofs_per_world,
                        self.model.joint_q,
                        self.model.joint_qd,
                        state.joint_q if reset_q else None,
                        state.joint_qd if reset_qd else None,
                    ],
                    device=self.model.device,
                )
                # Without sleeping, the default update interval propagates the
                # reset on the next step. Otherwise push it now so it is not
                # lost before the next sync. The sleeping path synchronizes
                # below before rebuilding its cached data.
                if self.update_data_interval != 1 and not self.enable_sleeping:
                    data = self.mj_data if self.use_mujoco_cpu else self.mjw_data
                    if data is not None and native_template_world_selected:
                        self._update_mjc_data(data, self.model, state, world_mask=world_mask)

        # Clear the internal buffers that persist between steps.
        if self.use_mujoco_cpu:
            d = self.mj_data
            if d is None:
                return
            # Native MuJoCo owns one template-world MjData instance even when
            # separate_worlds=True was requested for a multi-world Newton
            # model. Its persistent buffers therefore belong to local world 0.
            if not native_template_world_selected:
                return
            # Single MjData instance: clear the whole buffers (no per-world mask).
            d.qacc_warmstart[:] = 0.0
            d.qfrc_applied[:] = 0.0
            d.ctrl[:] = 0.0
            d.act[:] = 0.0
            d.xfrc_applied[:] = 0.0
            return

        d = self.mjw_data
        if d is None:
            return

        world_ids, world_count, launch_worlds = self._reset_world_selection(world_mask)
        buffers = (d.qacc_warmstart, d.qfrc_applied, d.ctrl, d.act, d.xfrc_applied)
        buffer_dim = max(buffer.shape[1] for buffer in buffers)
        wp.launch(
            reset_world_buffers_kernel,
            dim=(launch_worlds, buffer_dim),
            inputs=[world_ids, world_count, *buffers],
            device=self.model.device,
        )
        if self.enable_sleeping:
            m = self.mjw_model
            self._update_mjc_data_worlds(d, self.model, state, world_ids, world_count, launch_worlds)
            self._launch_reset_sleeping_state(world_ids, world_count, launch_worlds, clear_overflow=True)
            # Sleeping trees retain derived state, so rebuild it at the reset
            # coordinates before restoring the initial sleep bookkeeping.
            with wp.ScopedDevice(self.model.device), self._scoped_mujoco_warp_execution():
                self._forward_worlds(d, world_ids, launch_worlds if world_count is None else world_count)
            self._launch_restore_sleeping_state(world_ids, world_count, launch_worlds, clear_overflow=True)
            if self._has_body_sleep_overrides:
                # The overrides edit tree_asleep, so the kernel also republishes the
                # sleep bookkeeping of these worlds (MuJoCo Warp's update_sleep).
                wp.launch(
                    apply_body_sleep_override_kernel,
                    dim=launch_worlds,
                    inputs=[
                        world_ids,
                        world_count,
                        self._body_sleep_override,
                        m.nv,
                        m.nbody,
                        m.ntree,
                        self._sleep_awake_value,
                        int(self._mujoco.mjtSleepState.mjS_STATIC),
                        int(self._mujoco.mjtSleepState.mjS_AWAKE),
                        int(self._mujoco.mjtSleepState.mjS_ASLEEP),
                        m.body_rootid,
                        m.body_mocapid,
                        m.body_treeid,
                        m.dof_bodyid,
                    ],
                    outputs=[
                        d.tree_asleep,
                        d.tree_awake,
                        d.body_awake,
                        d.body_awake_ind,
                        d.dof_awake_ind,
                        d.ntree_awake,
                        d.nbody_awake,
                        d.nv_awake,
                    ],
                    device=self.model.device,
                )

    def _reset_world_selection(
        self, world_mask: wp.array[wp.bool] | None
    ) -> tuple[wp.array[wp.int32], wp.array[wp.int32] | None, int]:
        """Turn a reset mask into ``(world_ids, device count, launch worlds)`` for the reset kernels.

        ``None`` selects every world through a cached identity list and a host
        count (no device count). Otherwise one block-wide scan compacts the mask
        into ascending world ids and writes their number to a one-element device
        array; the reset launches then span ``nworld`` slots and their kernels
        exit the unused ones, so no host synchronization is needed and the
        sequence stays CUDA-graph capturable.
        """
        nworld = self.mjw_data.nworld
        device = self.model.device
        if world_mask is None:
            if self._all_world_ids is None:
                self._all_world_ids = wp.array(np.arange(nworld, dtype=np.int32), dtype=wp.int32, device=device)
            return self._all_world_ids, None, nworld
        if self._reset_world_ids is None:
            self._reset_world_ids = wp.empty(nworld, dtype=wp.int32, device=device)
            self._reset_world_count = wp.zeros(1, dtype=wp.int32, device=device)
        wp.launch_tiled(
            compact_reset_world_ids_kernel,
            dim=[1],
            inputs=[world_mask, nworld, self._reset_world_ids, self._reset_world_count],
            block_dim=256,
            device=device,
        )
        return self._reset_world_ids, self._reset_world_count, nworld

    def _update_mjc_data_worlds(
        self,
        mj_data: MjWarpData,
        model: Model,
        state: State,
        world_ids: wp.array[wp.int32],
        world_count: wp.array[wp.int32] | None,
        launch_worlds: int,
    ) -> None:
        """Push the Newton joint state of the listed worlds into MuJoCo Warp ``qpos`` / ``qvel``.

        Reset-only sibling of :meth:`_update_mjc_data`. The caller wakes every
        tree of these worlds right after, so the per-tree edit detection of the
        step path is not needed here.
        """
        joints_per_world = model.joint_count // mj_data.nworld
        mujoco_attrs = getattr(model, "mujoco", None)
        dof_ref = getattr(mujoco_attrs, "dof_ref", None) if mujoco_attrs is not None else None
        wp.launch(
            convert_warp_coords_to_mj_worlds_kernel,
            dim=(launch_worlds, joints_per_world),
            inputs=[
                state.joint_q,
                state.joint_qd,
                world_ids,
                world_count,
                joints_per_world,
                model.joint_type,
                model.joint_q_start,
                model.joint_qd_start,
                model.joint_dof_dim,
                model.joint_child,
                model.joint_X_p,
                model.joint_X_c,
                model.body_com,
                dof_ref,
                self.mj_q_start,
                self.mj_qd_start,
                None,
                1.0e-6,
            ],
            outputs=[mj_data.qpos, mj_data.qvel, None],
            device=model.device,
        )

    def _forward_worlds(self, mj_data: MjWarpData, world_ids: wp.array[wp.int32], count: int | wp.array) -> None:
        """Rebuild MuJoCo Warp's position- and velocity-dependent data for the listed worlds."""
        forward_worlds = getattr(self._mujoco_warp, "forward_worlds", None)
        if forward_worlds is None:
            # MuJoCo Warp without world subsets: rebuild every world.
            self._mujoco_warp.fwd_position(self.mjw_model, mj_data, factorize=False)
            self._mujoco_warp.fwd_velocity(self.mjw_model, mj_data)
            return
        forward_worlds(self.mjw_model, mj_data, world_ids, count)

    def _capture_initial_sleeping_state(self) -> None:
        """Capture the template world's initial sleep bookkeeping."""
        if not self.enable_sleeping or self.mjw_data is None:
            return

        d = self.mjw_data
        device = self.model.device
        self._initial_tree_asleep = wp.array(
            d.tree_asleep.numpy()[0].copy(),
            dtype=wp.int32,
            device=device,
        )
        self._initial_tree_awake = wp.array(
            d.tree_awake.numpy()[0].copy(),
            dtype=wp.int32,
            device=device,
        )
        self._initial_body_awake = wp.array(
            d.body_awake.numpy()[0].copy(),
            dtype=wp.int32,
            device=device,
        )
        self._initial_body_awake_ind = wp.array(
            d.body_awake_ind.numpy()[0].copy(),
            dtype=wp.int32,
            device=device,
        )
        self._initial_dof_awake_ind = wp.array(
            d.dof_awake_ind.numpy()[0].copy(),
            dtype=wp.int32,
            device=device,
        )
        self._initial_ntree_awake = int(d.ntree_awake.numpy()[0])
        self._initial_nbody_awake = int(d.nbody_awake.numpy()[0])
        self._initial_nv_awake = int(d.nv_awake.numpy()[0])

    def _restore_initial_sleeping_state(
        self,
        world_mask: wp.array[wp.bool] | None = None,
        *,
        clear_overflow: bool = False,
    ) -> None:
        """Restore the initial sleep state in selected MuJoCo Warp worlds."""
        if not self.enable_sleeping or self.mjw_data is None:
            return
        self._launch_restore_sleeping_state(*self._reset_world_selection(world_mask), clear_overflow=clear_overflow)

    def _launch_restore_sleeping_state(
        self,
        world_ids: wp.array[wp.int32],
        world_count: wp.array[wp.int32] | None,
        launch_worlds: int,
        *,
        clear_overflow: bool,
    ) -> None:
        if self._initial_tree_asleep is None:
            raise RuntimeError("Initial MuJoCo Warp sleep state was not captured.")

        m = self.mjw_model
        d = self.mjw_data
        sleep_dim = max(m.nv, m.nbody, m.ntree)
        if sleep_dim == 0:
            return
        wp.launch(
            restore_sleeping_state_kernel,
            dim=(launch_worlds, sleep_dim),
            inputs=[
                world_ids,
                world_count,
                int(clear_overflow),
                m.nv,
                m.nbody,
                m.ntree,
                self._initial_ntree_awake,
                self._initial_nbody_awake,
                self._initial_nv_awake,
                self._initial_tree_asleep,
                self._initial_tree_awake,
                self._initial_body_awake,
                self._initial_body_awake_ind,
                self._initial_dof_awake_ind,
                d.tree_asleep,
                d.tree_awake,
                d.body_awake,
                d.body_awake_ind,
                d.dof_awake_ind,
                d.ntree_awake,
                d.nbody_awake,
                d.nv_awake,
                d.overflow,
            ],
            device=self.model.device,
        )

    def _wake_sleeping_worlds(
        self,
        world_mask: wp.array[wp.bool] | None = None,
        *,
        clear_overflow: bool = False,
    ) -> None:
        """Wake all dynamic trees in the selected MuJoCo Warp worlds."""
        if not self.enable_sleeping or self.mjw_data is None:
            return
        self._launch_reset_sleeping_state(*self._reset_world_selection(world_mask), clear_overflow=clear_overflow)

    def _launch_reset_sleeping_state(
        self,
        world_ids: wp.array[wp.int32],
        world_count: wp.array[wp.int32] | None,
        launch_worlds: int,
        *,
        clear_overflow: bool,
    ) -> None:
        m = self.mjw_model
        d = self.mjw_data
        sleep_dim = max(m.nv, m.nbody, m.ntree)
        if sleep_dim == 0:
            return
        wp.launch(
            reset_sleeping_state_kernel,
            dim=(launch_worlds, sleep_dim),
            inputs=[
                world_ids,
                world_count,
                int(clear_overflow),
                m.nv,
                m.nbody,
                m.ntree,
                self._sleep_awake_value,
                int(self._mujoco.mjtSleepState.mjS_STATIC),
                int(self._mujoco.mjtSleepState.mjS_AWAKE),
                m.body_rootid,
                m.body_mocapid,
                m.body_treeid,
                d.tree_asleep,
                d.tree_awake,
                d.body_awake,
                d.body_awake_ind,
                d.dof_awake_ind,
                d.ntree_awake,
                d.nbody_awake,
                d.nv_awake,
                d.overflow,
            ],
            device=self.model.device,
        )

    def _enable_rne_postconstraint(self, state_out: State):
        """Request computation of RNE forces if required for state fields."""
        rne_postconstraint_fields = {"body_qdd", "body_parent_f"}
        # TODO: handle use_mujoco_cpu
        m = self.mjw_model
        if m.sensor_rne_postconstraint:
            return
        if any(getattr(state_out, field) is not None for field in rne_postconstraint_fields):
            # required for cfrc_ext, cfrc_int, cacc
            if wp.config.log_level <= wp.LOG_DEBUG:
                print("Setting model.sensor_rne_postconstraint True")
            m.sensor_rne_postconstraint = True

    def _invalidate_contact_fast_path(self):
        """Force the next contact conversion to take the full path.

        Called when cached MJWarp contact fields (friction, solref, solimp,
        etc.) may be stale — e.g. after :meth:`notify_model_changed` updates
        geom or body properties, or when the bound Contacts instance / MJWarp
        ``naconmax`` changes (which would make cached ``cid`` values index
        into a different output buffer).
        """
        self._last_contact_generation.fill_(_GENERATION_SENTINEL)
        self._last_nacon_count.zero_()

    @override
    def coupling_eval_gravity_acceleration(
        self,
        out_body_acceleration: wp.array[wp.vec3] | None,
        out_particle_acceleration: wp.array[wp.vec3] | None,
    ) -> None:
        """Evaluate MuJoCo's internally applied gravity acceleration for coupling."""
        if out_particle_acceleration is not None:
            super().coupling_eval_gravity_acceleration(None, out_particle_acceleration)

        if out_body_acceleration is None or out_body_acceleration.shape[0] == 0:
            return

        body_gravcomp = getattr(self.mjw_model, "body_gravcomp", None) if self.mjw_model is not None else None
        if body_gravcomp is None or self.mjc_body_to_newton is None or self.model.body_world is None:
            super().coupling_eval_gravity_acceleration(out_body_acceleration, None)
            return

        wp.launch(
            eval_mujoco_coupling_gravity_acceleration_kernel,
            dim=out_body_acceleration.shape[0],
            inputs=[
                self.model.gravity,
                self.model.body_world,
                self.mjc_body_to_newton,
                body_gravcomp,
            ],
            outputs=[out_body_acceleration],
            device=self.model.device,
        )

    @override
    def coupling_eval_effective_mass(
        self,
        endpoint_kind: wp.array[int],
        endpoint_index: wp.array[int],
        endpoint_local_pos: wp.array[wp.vec3],
        out: wp.array[float],
    ) -> None:
        """Evaluate MuJoCo articulated effective masses for coupling endpoints."""
        if (
            self.mjw_model is None
            or self.mjc_body_to_newton is None
            or self.model.body_world is None
            or self.model.body_mass is None
            or self.model.particle_mass is None
        ):
            super().coupling_eval_effective_mass(endpoint_kind, endpoint_index, endpoint_local_pos, out)
            return

        wp.launch(
            eval_mujoco_coupling_effective_mass_kernel,
            dim=out.shape[0],
            inputs=[
                endpoint_kind,
                endpoint_index,
                endpoint_local_pos,
                int(CouplingEndpointKind.BODY),
                int(CouplingEndpointKind.PARTICLE),
                self.model.body_mass,
                self.model.particle_mass,
                self.model.body_world,
                self.mjc_body_to_newton,
                self.mjw_model.body_invweight0,
            ],
            outputs=[out],
            device=self.model.device,
        )

    @override
    def coupling_eval_effective_mass_block(
        self,
        endpoint_kind: wp.array[int],
        endpoint_index: wp.array[int],
        endpoint_local_pos: wp.array[wp.vec3],
        out_mass: wp.array[float],
        out_inertia: wp.array[wp.mat33] | None = None,
    ) -> None:
        """Evaluate MuJoCo articulated effective mass and inertia blocks."""
        if out_inertia is None:
            self.coupling_eval_effective_mass(endpoint_kind, endpoint_index, endpoint_local_pos, out_mass)
            return

        if (
            self.mjw_model is None
            or self.mjc_body_to_newton is None
            or self.model.body_world is None
            or self.model.body_mass is None
            or self.model.body_inertia is None
            or self.model.particle_mass is None
        ):
            super().coupling_eval_effective_mass_block(
                endpoint_kind,
                endpoint_index,
                endpoint_local_pos,
                out_mass,
                out_inertia,
            )
            return

        wp.launch(
            eval_mujoco_coupling_effective_mass_block_kernel,
            dim=out_mass.shape[0],
            inputs=[
                endpoint_kind,
                endpoint_index,
                endpoint_local_pos,
                int(CouplingEndpointKind.BODY),
                int(CouplingEndpointKind.PARTICLE),
                self.model.body_mass,
                self.model.body_inertia,
                self.model.particle_mass,
                self.model.body_world,
                self.mjc_body_to_newton,
                self.mjw_model.body_invweight0,
            ],
            outputs=[out_mass, out_inertia],
            device=self.model.device,
        )

    def _dormant_dummy_tree_asleep(self) -> wp.array:
        dummy = getattr(self, "_dormant_dummy_tree_asleep_array", None)
        if dummy is None:
            dummy = wp.zeros((1, 1), dtype=wp.int32, device=self.model.device)
            self._dormant_dummy_tree_asleep_array = dummy
        return dummy

    def _contact_conversion_inputs(
        self,
        model: Model,
        state_in: State,
        contacts: Contacts,
        *,
        inject_mode: int,
        total_num_threads: int,
        xpos: wp.array | None = None,
        xquat: wp.array | None = None,
    ) -> list:
        """Argument list shared by the conversion and slab-injection kernels.

        ``xpos``/``xquat`` select MJWarp's body kinematics of the current substep as the pose
        source (in-step conversion); ``None`` uses ``state_in.body_q``.
        """
        bodies_per_world = self.model.body_count // self.model.world_count
        mujoco_attrs = getattr(model, "mujoco", None)
        shape_mjc_solref_mode = getattr(mujoco_attrs, "solref_mode", None) if mujoco_attrs is not None else None
        return [
            state_in.body_q,
            model.shape_body,
            model.body_flags,
            self.mjw_model.geom_bodyid,
            self.mjw_model.body_weldid,
            self.mjw_model.body_treeid,
            self.mjw_data.tree_asleep if self.enable_sleeping else self._dormant_dummy_tree_asleep(),
            int(self.dormant_contact_filter),
            inject_mode,
            self._dormant_flag,
            self._dormant_tids,
            self._dormant_count,
            self._wake_event,
            self.mjw_model.body_invweight0,
            self.mjw_model.geom_condim,
            self.mjw_model.geom_priority,
            self.mjw_model.geom_solmix,
            self.mjw_model.geom_solref,
            self.mjw_model.geom_solimp,
            self.mjw_model.geom_friction,
            self.mjw_model.geom_margin,
            self.mjw_model.geom_gap,
            # Newton shape-material force-space inputs (issue #2009)
            model.shape_material_ke,
            model.shape_material_kd,
            shape_mjc_solref_mode,
            # Newton contacts
            contacts.rigid_contact_count,
            contacts.rigid_contact_shape0,
            contacts.rigid_contact_shape1,
            contacts.rigid_contact_point0,
            contacts.rigid_contact_point1,
            contacts.rigid_contact_normal,
            contacts.rigid_contact_offset0,
            contacts.rigid_contact_offset1,
            contacts.rigid_contact_margin0,
            contacts.rigid_contact_margin1,
            contacts.rigid_contact_stiffness,
            contacts.rigid_contact_damping,
            contacts.rigid_contact_friction,
            model.shape_margin,
            model.shape_material_kf,
            self.mjw_model.opt.impratio_invsqrt,
            self.mjw_model.opt.cone == self._mujoco.mjtCone.mjCONE_ELLIPTIC,
            bodies_per_world,
            self.newton_shape_to_mjc_geom,
            # Mujoco warp contacts
            self.mjw_data.naconmax,
            self.mjw_data.nacon,
            self.mjw_data.contact.dist,
            self.mjw_data.contact.pos,
            self.mjw_data.contact.frame,
            self.mjw_data.contact.includemargin,
            self.mjw_data.contact.friction,
            self.mjw_data.contact.solref,
            self.mjw_data.contact.solreffriction,
            self.mjw_data.contact.solimp,
            self.mjw_data.contact.dim,
            self.mjw_data.contact.geom,
            self.mjw_data.contact.efc_address,
            self.mjw_data.contact.worldid,
            # Data to clear
            self.mjw_data.nworld,
            self.mjw_data.ncollision,
            # Fast-path generation tracking
            contacts.contact_generation,
            self._last_contact_generation,
            self._contact_tid_to_cid,
            self._last_nacon_count,
            total_num_threads,
            # In-step conversion: MJWarp kinematics and the per-contact refresh records
            xpos,
            xquat,
            self._hook_tid,
            self._hook_body,
            self._hook_geom,
        ]

    def _wake_injection_store(self, contacts: Contacts) -> tuple[bool, Any]:
        """Return whether wake injection runs this substep and the dormant store it reads."""
        if self._contact_tid_to_cid is None:
            return False, None
        store = getattr(contacts, "dormant_contact_store", None)
        if store is not None and store.slab_count_total == 0:
            store = None
        if not self.dormant_contact_filter and store is None:
            return False, None
        if not self.enable_sleeping or self._tree_asleep_prev is None:
            # A store configured without sleeping has nothing to wake; the tree buffers do not exist.
            return False, None
        return True, store

    def _wake_before_step(self, model: Model, state_in: State, contacts: Contacts) -> None:
        """Run the adapter's pre-step wake pass and inject dormant contacts of woken trees.

        Mirrors the prelude of MJWarp's own collision step for externally supplied
        contacts: ``sleep.wake`` (applied forces, velocities, user wake requests),
        the ``tree_awake`` refresh and the collision wake, so a tree touched by an
        awake tree wakes in this substep. The refresh kernel also snapshots the
        converted contact count for the fast conversion path and, together with the
        collision wake, records asleep->awake transitions for the injection gate.
        The awake counters and body/DOF compaction are left to MJWarp's forward pass,
        which recomputes them before they are read.
        """
        from mujoco_warp._src import sleep

        m = self.mjw_model
        d = self.mjw_data
        inject, store = self._wake_injection_store(contacts)
        tree_asleep_prev = self._tree_asleep_prev if inject else None
        wake_event = self._wake_event if inject else None
        sleep.wake(m, d)
        wp.launch(
            refresh_tree_awake_kernel,
            dim=d.tree_asleep.shape,
            inputs=[d.tree_asleep, d.nacon, contacts.contact_generation],
            outputs=[
                d.tree_awake,
                tree_asleep_prev,
                wake_event,
                self._last_nacon_count,
                self._last_contact_generation,
            ],
            device=model.device,
        )
        self._wake_contact_trees(contacts, tree_asleep_prev=tree_asleep_prev)
        if inject:
            self._inject_woken_dormant_contacts(model, state_in, contacts, store)

    def _wake_contact_trees(self, contacts: Contacts, *, tree_asleep_prev: wp.array | None) -> None:
        """Wake sleeping trees touched by awake ones and snapshot the converted count and generation."""
        m = self.mjw_model
        d = self.mjw_data
        threads = min(d.naconmax, CONTACT_CONVERSION_MAX_THREADS)
        wp.launch(
            wake_contact_trees_kernel,
            dim=threads,
            inputs=[
                m.ntree,
                m.body_treeid,
                m.geom_bodyid,
                d.tree_awake,
                d.contact.geom,
                d.contact.worldid,
                d.nacon,
                contacts.contact_generation,
                threads,
            ],
            outputs=[
                d.tree_asleep,
                tree_asleep_prev,
                self._wake_event if tree_asleep_prev is not None else None,
                self._last_nacon_count,
                self._last_contact_generation,
            ],
            device=self.model.device,
        )

    def _inject_woken_dormant_contacts(
        self,
        model: Model,
        state_in: State,
        contacts: Contacts,
        store: Any,
        *,
        xpos: wp.array | None = None,
        xquat: wp.array | None = None,
    ) -> None:
        """Give trees that woke this substep their dormant contacts.

        Runs after the collision wake (inside the step through :meth:`_convert_contacts_in_step`,
        or after the adapter's pre-step wake pass :meth:`_wake_before_step`), so a
        tree woken by an external force, a pose edit or contact with an awake tree
        receives its support contacts in the same substep, exactly as when every
        cached contact was converted unconditionally. It is not the last wake point,
        however: with ``run_collision_detection=False`` MJWarp still runs
        ``sleep.wake_tendon`` in ``fwd_kinematics`` (models with tendons) and
        ``sleep.wake_equality`` after ``make_constraint`` (models with equality
        constraints) inside the step, and trees woken there receive their dormant
        rows one substep late, from the next call. Two sources are handled: rows the
        full conversion pass parked (``dormant_contact_filter``) and slabs of the
        pipeline's :attr:`Contacts.dormant_contact_store`, which are appended to the
        Newton buffer as well so the fast path tracks them on later substeps.
        Body poses come from ``xpos``/``xquat`` when given and from ``state_in.body_q`` otherwise.
        """
        d = self.mjw_data
        live_capacity = min(contacts.rigid_contact_max, self._contact_tid_to_cid.shape[0])
        parked_threads = min(self._contact_tid_to_cid.shape[0], CONTACT_CONVERSION_MAX_THREADS)

        def _inject():
            if self.dormant_contact_filter:
                wp.launch(
                    convert_newton_contacts_to_mjwarp_kernel,
                    dim=parked_threads,
                    inputs=[
                        *self._contact_conversion_inputs(
                            model,
                            state_in,
                            contacts,
                            inject_mode=1,
                            total_num_threads=parked_threads,
                            xpos=xpos,
                            xquat=xquat,
                        ),
                        None,  # tree_asleep_prev: the injection pass detects no wake events
                    ],
                    device=model.device,
                )
            if store is not None:
                # Two-level launch: list the woken slabs, then convert their rows with a
                # fixed grid so a substep waking many trees stays one short launch.
                wp.launch(
                    collect_woken_dormant_slabs_kernel,
                    dim=store.slab_count_total,
                    inputs=[
                        d.tree_asleep,
                        store.slab_world_tree,
                        store.slabs.slab_count,
                        store.slabs.slab_live_gen,
                        contacts.contact_generation,
                        self._wake_event,
                        store.inject_slabs,
                        store.inject_count,
                    ],
                    device=model.device,
                )
                inject_threads = min(store.slab_count_total * store.rows_per_shape, CONTACT_CONVERSION_MAX_THREADS)
                wp.launch(
                    inject_dormant_slab_contacts_kernel,
                    dim=inject_threads,
                    inputs=[
                        store.inject_slabs,
                        store.inject_count,
                        store.slabs,
                        live_capacity,
                        contacts.rigid_contact_tids,
                        *self._contact_conversion_inputs(
                            model,
                            state_in,
                            contacts,
                            inject_mode=0,
                            total_num_threads=inject_threads,
                            xpos=xpos,
                            xquat=xquat,
                        ),
                    ],
                    device=model.device,
                )
            wp.launch(
                _finish_wake_injection,
                dim=1,
                inputs=[
                    d.nacon,
                    self._last_nacon_count,
                    store.inject_count if store is not None else None,
                    self._wake_event,
                ],
                device=model.device,
            )

        if self.conditional_wake_injection:
            # Quiet substeps (no asleep->awake transition) skip the injection through a
            # conditional graph node.
            wp.capture_if(self._wake_event, on_true=_inject)
        else:
            # Every injection kernel exits early without a wake event (see conditional_wake_injection).
            _inject()

    def _prepare_contact_conversion(self, contacts: Contacts, *, prepared: bool = False) -> None:
        """Host-side bookkeeping and device prelude of this substep's contact conversion.

        Args:
            contacts: Newton contact buffer.
            prepared: Whether :func:`apply_mjc_inputs_kernel` already zeroed ``nacon`` and
                restarted the parked-contact list for this substep.
        """
        # Ensure the inverse shape mapping exists (lazy creation)
        if self.newton_shape_to_mjc_geom is None:
            self._create_inverse_shape_mapping()

        # The kernel only produces valid output for tid < naconmax (the full
        # path clamps count and rejects cid >= naconmax), so the tid->cid map is
        # sized by min(rigid_contact_max, naconmax). The launch itself grid-strides
        # over the live count with a fixed thread budget.
        naconmax = self.mjw_data.naconmax
        launch_dim = min(contacts.rigid_contact_max, naconmax)

        # Grow the tid_to_cid buffer if the MJWarp data capacity changed after
        # construction.
        # Invalidate the cached tid_to_cid mapping whenever any of the
        # invariants it depends on change:
        #
        #  - Contacts identity: keyed on id(contacts.contact_generation), the
        #    inner per-Contacts device array.  Empirically, keying on the outer
        #    id(contacts) wrapper produces broken binaries in dexsuite training
        #    (root cause unclear; the inner array's id is what works).
        #  - rigid_contact_max: changes the meaning of tid indices.
        #  - mjw_data.naconmax: changes the meaning of cid indices; if the
        #    underlying contact buffers were reallocated (e.g. set_const_fixed
        #    after notify_model_changed), cached cid values could index into
        #    freed memory or out-of-bounds.
        contacts_id = id(contacts.contact_generation)
        needs_realloc = self._contact_tid_to_cid is None or self._contact_tid_to_cid.shape[0] < launch_dim
        contacts_changed = (
            self._last_contacts_id != contacts_id
            or self._last_rigid_contact_max != contacts.rigid_contact_max
            or self._last_naconmax != naconmax
        )

        if needs_realloc or contacts_changed:
            if needs_realloc:
                self._contact_tid_to_cid = wp.full(launch_dim, -1, dtype=wp.int32, device=self.model.device)
                self._dormant_flag = wp.zeros(launch_dim, dtype=wp.int32, device=self.model.device)
                self._dormant_tids = wp.zeros(launch_dim, dtype=wp.int32, device=self.model.device)
            if self._contact_pose_hook and (self._hook_tid is None or self._hook_tid.shape[0] < naconmax):
                self._allocate_hook_records(naconmax)
            # Reset existing device buffers (always pre-allocated in __init__).
            self._invalidate_contact_fast_path()
            self._last_contacts_id = contacts_id
            self._last_rigid_contact_max = contacts.rigid_contact_max
            self._last_naconmax = naconmax

        if not prepared:
            # Zero nacon before the kernel (the full path counts with atomic_add; the fast path
            # restores the count from last_nacon_count) and restart the parked (dormant) contact
            # list on a new collision generation.
            wp.launch(
                prepare_contact_conversion_kernel,
                dim=1,
                inputs=[contacts.contact_generation, self._last_contact_generation],
                outputs=[self.mjw_data.nacon, self._dormant_count],
                device=self.model.device,
            )

    def _launch_contact_conversion(
        self,
        model: Model,
        state_in: State,
        contacts: Contacts,
        *,
        xpos: wp.array | None = None,
        xquat: wp.array | None = None,
        tree_asleep_prev: wp.array | None = None,
    ) -> None:
        """Convert the Newton contact buffer into MJWarp contacts for this substep.

        Args:
            model: Newton model.
            state_in: Newton state providing the body poses when ``xpos`` is ``None``.
            contacts: Newton contact buffer.
            xpos: MJWarp body positions of the current substep (in-step conversion).
            xquat: MJWarp body orientations of the current substep (in-step conversion).
            tree_asleep_prev: When given, the kernel also flags trees that woke since this
                snapshot (``wake_event``) and refreshes the snapshot.
        """
        thread_count = min(contacts.rigid_contact_max, self.mjw_data.naconmax, CONTACT_CONVERSION_MAX_THREADS)
        wp.launch(
            convert_newton_contacts_to_mjwarp_kernel,
            dim=thread_count,
            inputs=[
                *self._contact_conversion_inputs(
                    model,
                    state_in,
                    contacts,
                    inject_mode=0,
                    total_num_threads=thread_count,
                    xpos=xpos,
                    xquat=xquat,
                ),
                tree_asleep_prev,
            ],
            device=model.device,
        )

    def _snapshot_contact_generation(self, contacts: Contacts) -> None:
        """Snapshot the converted count and generation for the fast conversion path.

        Runs as a separate dim=1 kernel after the conversion kernel completed so that ``nacon``
        has its final value and ``last_contact_generation`` is only updated once every thread
        of the conversion kernel has read it. With sleeping enabled the collision-wake kernel
        takes the snapshot instead.
        """
        wp.launch(
            _snapshot_nacon_count,
            dim=1,
            inputs=[
                self.mjw_data.nacon,
                self._last_nacon_count,
                contacts.contact_generation,
                self._last_contact_generation,
            ],
            device=self.model.device,
        )

    def _sync_mjw_inertias_to_mjc_cpu(self) -> None:
        """Synchronize the complete MJWarp inertial representation to MuJoCo CPU."""
        mjw_body_inertia = self.mjw_model.body_inertia.numpy()[0]
        mjw_body_iquat = self.mjw_model.body_iquat.numpy()[0]

        inertia_changed = ~np.isclose(
            self.mj_model.body_inertia,
            mjw_body_inertia,
            rtol=1.0e-6,
            atol=1.0e-8,
        ).all(axis=1)
        iquat_changed = ~np.isclose(
            self.mj_model.body_iquat,
            mjw_body_iquat,
            rtol=1.0e-6,
            atol=1.0e-8,
        ).all(axis=1)
        changed_bodies = inertia_changed | iquat_changed

        if not np.any(changed_bodies):
            return

        self.mj_model.body_inertia[:] = mjw_body_inertia
        self.mj_model.body_iquat[:] = mjw_body_iquat

        # ``body_inertia`` and ``body_iquat`` are coupled. Once the inertial
        # frame changes, MuJoCo CPU's compiled simple-path metadata may still
        # describe the old frame, so invalidate it before ``mj_setConst()``.
        self.mj_model.body_sameframe[changed_bodies] = int(self._mujoco.mjtSameFrame.mjSAMEFRAME_NONE)
        self.mj_model.body_simple[changed_bodies] = 0
        self.mj_model.dof_simplenum[:] = 0

    @override
    def notify_model_changed(self, flags: ModelFlags | int, world_mask: wp.array[wp.bool] | None = None) -> None:
        """Refresh solver buffers after model edits.

        Args:
            flags: Bit-mask of :class:`~newton.ModelFlags` describing what changed.
            world_mask: Optional per-world bool mask of shape ``(world_count + 1,)`` selecting
                the worlds whose properties changed. When sleeping is enabled only those worlds
                are woken; ``None`` wakes every world (previous behavior).
        """
        if self.use_mujoco_cpu:
            self._notify_model_changed(flags, world_mask)
        else:
            with self._scoped_mujoco_warp_execution():
                self._notify_model_changed(flags, world_mask)

    def _notify_model_changed(self, flags: ModelFlags | int, world_mask: wp.array[wp.bool] | None = None) -> None:
        need_const_fixed = False
        need_const_0 = False
        need_length_range = False

        if flags & ModelFlags.BODY_INERTIAL_PROPERTIES:
            self._update_model_inertial_properties()
            # set_const_fixed / set_const_0 (called below) recompute MuJoCo
            # constants that feed into contact solver parameters (invweight0,
            # subtreemass, etc.).  Cached MJWarp contact fields written by
            # the fast path are derived from those constants, so invalidate
            # to force a full re-pack on the next contact conversion.
            self._invalidate_contact_fast_path()
            need_const_fixed = True
            need_const_0 = True
        if flags & ModelFlags.JOINT_PROPERTIES:
            self._update_joint_properties()
        if flags & ModelFlags.BODY_PROPERTIES:
            self._update_body_properties()
            self._invalidate_contact_fast_path()
            need_const_0 = True
        if flags & ModelFlags.JOINT_DOF_PROPERTIES:
            self._update_joint_dof_properties()
            self._invalidate_contact_fast_path()
            # Allow ``_update_solref_from_invweight0`` to re-validate authored
            # ``mujoco.solreflimit`` values after the user reassigns them.
            self._raw_solreflimit_validated = False
            need_const_0 = True
            need_length_range = True
        if flags & ModelFlags.SHAPE_PROPERTIES:
            self._update_geom_properties()
            self._update_site_properties()
            self._update_pair_properties()
            self._invalidate_contact_fast_path()
        if flags & ModelFlags.MODEL_PROPERTIES:
            self._update_model_properties()
            self._invalidate_contact_fast_path()
        if flags & ModelFlags.CONSTRAINT_PROPERTIES:
            self._update_eq_properties()
            self._update_mimic_eq_properties()
        if flags & ModelFlags.TENDON_PROPERTIES:
            self._update_tendon_properties()
            need_const_0 = True
            need_length_range = True
        if flags & ModelFlags.ACTUATOR_PROPERTIES:
            self._update_actuator_properties()
            need_const_0 = True
            need_length_range = True

        has_any_connect = self.has_connect_constraints or self.has_jnt_connect_constraints
        update_connect_constraint_anchor_rel_xform_at_ref_pose = has_any_connect and bool(
            flags & (ModelFlags.JOINT_PROPERTIES | ModelFlags.JOINT_DOF_PROPERTIES)
        )
        update_connect_constraint_anchors = self.has_connect_constraints and bool(
            flags & ModelFlags.CONSTRAINT_PROPERTIES
        )

        # ``need_const_0`` already covers every update that changes the derived
        # ``dof_invweight0`` factors or the source joint-limit data, so it also
        # captures every case that needs ``jnt_solref`` to be re-scaled.
        need_solref_update = need_const_0

        if self.use_mujoco_cpu:
            if flags & ModelFlags.BODY_INERTIAL_PROPERTIES:
                self.mj_model.body_ipos[:] = self.mjw_model.body_ipos.numpy()[0]
                self.mj_model.body_mass[:] = self.mjw_model.body_mass.numpy()[0]
                self.mj_model.body_gravcomp[:] = self.mjw_model.body_gravcomp.numpy()[0]
                self._sync_mjw_inertias_to_mjc_cpu()
            if flags & (ModelFlags.BODY_PROPERTIES | ModelFlags.JOINT_DOF_PROPERTIES):
                self.mj_model.dof_armature[:] = self.mjw_model.dof_armature.numpy()[0]
                self.mj_model.dof_frictionloss[:] = self.mjw_model.dof_frictionloss.numpy()[0]
                self.mj_model.dof_damping[:] = self.mjw_model.dof_damping.numpy()[0]
                self.mj_model.dof_solimp[:] = self.mjw_model.dof_solimp.numpy()[0]
                self.mj_model.dof_solref[:] = self.mjw_model.dof_solref.numpy()[0]
                self.mj_model.qpos0[:] = self.mjw_model.qpos0.numpy()[0]
                self.mj_model.qpos_spring[:] = self.mjw_model.qpos_spring.numpy()[0]
            if flags & ModelFlags.JOINT_DOF_PROPERTIES:
                self.mj_model.jnt_solimp[:] = self.mjw_model.jnt_solimp.numpy()[0]
                self.mj_model.jnt_stiffness[:] = self.mjw_model.jnt_stiffness.numpy()[0]
                self.mj_model.jnt_margin[:] = self.mjw_model.jnt_margin.numpy()[0]
                self.mj_model.jnt_range[:] = self.mjw_model.jnt_range.numpy()[0]
                self.mj_model.jnt_actfrcrange[:] = self.mjw_model.jnt_actfrcrange.numpy()[0]
            if need_length_range or need_const_fixed or need_const_0:
                self._set_const_0_with_physical_meaninertia()
            if need_solref_update:
                # ``mj_setConst`` refreshes the derived ``dof_invweight0``
                # factors; ``jnt_solimp`` was already written by
                # ``_update_joint_dof_properties`` above.
                self._update_solref_from_invweight0()
            # Must be called last — mj_setConst/set_const_0 computes CONNECT anchor2
            # without accounting for Newton's dof_ref, so we overwrite with the
            # correctly computed values.
            self._notify_connect_constraints_changed(
                update_connect_constraint_anchor_rel_xform_at_ref_pose,
                update_connect_constraint_anchors,
            )
            if flags & ModelFlags.CONSTRAINT_PROPERTIES:
                self._sync_equality_properties_to_mujoco_cpu()

        else:
            if (
                need_length_range
                or need_const_fixed
                or need_const_0
                or need_solref_update
                or update_connect_constraint_anchor_rel_xform_at_ref_pose
                or update_connect_constraint_anchors
            ):
                with wp.ScopedDevice(self.model.device):
                    if need_length_range:
                        self._mujoco_warp.set_length_range(self.mjw_model, self.mjw_data)
                    if need_const_fixed:
                        self._mujoco_warp.set_const_fixed(self.mjw_model, self.mjw_data)
                    if need_const_0:
                        self._set_const_0_with_physical_meaninertia()
                    if need_solref_update:
                        # ``set_const_0`` refreshes ``dof_invweight0`` and
                        # ``jnt_solimp`` was already written by
                        # ``_update_joint_dof_properties`` above.
                        self._update_solref_from_invweight0()
                    # Must be called last — mj_setConst/set_const_0 computes CONNECT anchor2
                    # without accounting for Newton's dof_ref, so we overwrite with the
                    # correctly computed values.
                    self._notify_connect_constraints_changed(
                        update_connect_constraint_anchor_rel_xform_at_ref_pose,
                        update_connect_constraint_anchors,
                    )

            if flags & ModelFlags.SHAPE_PROPERTIES:
                self._sync_worldbody_geom_xposes()
                self._sync_site_xposes()

            if flags and getattr(self, "enable_sleeping", False) and not getattr(self, "_initial_model_sync", False):
                # MuJoCo Warp cannot infer that Newton-side property updates
                # invalidate sleeping islands, so explicitly wake them. Only the
                # worlds whose properties changed need to wake; a global wake
                # defeats sleeping in every untouched world for MJ_MINAWAKE steps.
                self._wake_sleeping_worlds(world_mask)

    def _sync_equality_properties_to_mujoco_cpu(self) -> None:
        """Mirror equality properties from MJWarp buffers to MuJoCo-C CPU buffers."""
        if self.mj_model.neq == 0:
            return

        self.mj_model.eq_data[:] = self.mjw_model.eq_data.numpy()[0]
        self.mj_model.eq_solref[:] = self.mjw_model.eq_solref.numpy()[0]
        self.mj_model.eq_solimp[:] = self.mjw_model.eq_solimp.numpy()[0]
        self.mj_data.eq_active[:] = self.mjw_data.eq_active.numpy()[0]

    def _sync_worldbody_geom_xposes(self) -> None:
        """Refresh derived poses that MJWarp leaves fixed after data creation.

        MJWarp initializes direct worldbody geoms from the single CPU template
        and skips them during forward kinematics. Newton's batched model can
        carry distinct per-world geometry transforms, so copy them to the
        corresponding derived data after shape properties change.
        """
        if self.mj_model.ngeom == 0:
            return
        wp.launch(
            sync_worldbody_geom_xposes_kernel,
            dim=(self.mjw_data.nworld, self.mj_model.ngeom),
            inputs=[
                self.mjw_model.geom_bodyid,
                self.mjw_model.geom_pos,
                self.mjw_model.geom_quat,
            ],
            outputs=[
                self.mjw_data.geom_xpos,
                self.mjw_data.geom_xmat,
            ],
            device=self.model.device,
        )

    def _sync_site_xposes(self) -> None:
        """Refresh derived site poses after per-world model updates."""
        if self.mj_model.nsite == 0:
            return
        wp.launch(
            sync_site_xposes_kernel,
            dim=(self.mjw_data.nworld, self.mj_model.nsite),
            inputs=[
                self.mjw_model.site_bodyid,
                self.mjw_model.site_pos,
                self.mjw_model.site_quat,
                self.mjw_data.xpos,
                self.mjw_data.xquat,
            ],
            outputs=[
                self.mjw_data.site_xpos,
                self.mjw_data.site_xmat,
            ],
            device=self.model.device,
        )

    def _create_inverse_shape_mapping(self):
        """
        Create the inverse shape mapping (Newton shape -> MuJoCo [world, geom]).
        This is lazily created only when use_mujoco_contacts=False.
        """
        nworld = self.mjc_geom_to_newton_shape.shape[0]
        ngeom = self.mjc_geom_to_newton_shape.shape[1]

        # Create the inverse mapping array
        self.newton_shape_to_mjc_geom = wp.full(self.model.shape_count, -1, dtype=wp.int32, device=self.model.device)

        # Launch kernel to populate the inverse mapping
        wp.launch(
            create_inverse_shape_mapping_kernel,
            dim=(nworld, ngeom),
            inputs=[
                self.mjc_geom_to_newton_shape,
            ],
            outputs=[
                self.newton_shape_to_mjc_geom,
            ],
            device=self.model.device,
        )

    @staticmethod
    def _data_is_mjwarp(data):
        # Check if the data is a mujoco_warp Data object
        return hasattr(data, "nworld")

    def _apply_mjc_inputs(
        self,
        model: Model,
        state: State,
        control: Control | None,
        contacts: Contacts | None,
        *,
        prepare_contacts: bool,
    ) -> bool:
        """Apply controls, joint forces and body forces to MJWarp in one launch.

        Fuses the four per-substep input kernels of :meth:`_apply_mjc_control` for the
        MJWarp path and, when ``prepare_contacts`` is set, the contact-conversion prelude
        (``nacon`` reset and parked-list restart).

        Returns:
            Whether the contact-conversion prelude ran.
        """
        d = self.mjw_data
        apply_ctrl = (
            control is not None
            and self.mjc_actuator_ctrl_source is not None
            and self.mjc_actuator_to_newton_idx is not None
        )
        apply_qfrc = control is not None and control.joint_f is not None
        apply_body_f = state.body_f is not None
        apply_free_joint_f = control is not None and control.joint_f is not None
        if not (apply_ctrl or apply_qfrc or apply_body_f or apply_free_joint_f or prepare_contacts):
            return False
        nworld = d.nworld
        joints_per_world = model.joint_count // nworld
        nbody = self.mjc_body_to_newton.shape[1]
        nu = self.mjc_actuator_ctrl_source.shape[0] if apply_ctrl else 0
        dofs_per_world = model.joint_dof_count // nworld if nworld > 0 else model.joint_dof_count
        coords_per_world = model.joint_coord_count // nworld if nworld > 0 else model.joint_coord_count
        joint_target_q = control.joint_target_q if apply_ctrl else None
        joint_target_qd = control.joint_target_qd if apply_ctrl else None
        target_q_total = joint_target_q.shape[0] if joint_target_q is not None else 0
        target_q_per_world = target_q_total // nworld if nworld > 0 else target_q_total
        mujoco_ctrl = None
        if apply_ctrl:
            # Get mujoco.ctrl (None if not available - won't be accessed if no CTRL_DIRECT actuators)
            mujoco_ctrl_ns = getattr(control, "mujoco", None)
            mujoco_ctrl = getattr(mujoco_ctrl_ns, "ctrl", None) if mujoco_ctrl_ns is not None else None
        ctrls_per_world = mujoco_ctrl.shape[0] // nworld if mujoco_ctrl is not None and nworld > 0 else 0
        width = max(
            nu if apply_ctrl else 0,
            joints_per_world if apply_qfrc else 0,
            nbody if apply_body_f or apply_free_joint_f else 0,
            1,
        )
        wp.launch(
            apply_mjc_inputs_kernel,
            dim=(nworld, width),
            inputs=[
                int(apply_ctrl),
                int(apply_qfrc),
                int(apply_body_f),
                int(apply_free_joint_f),
                int(prepare_contacts),
                self.mjc_actuator_ctrl_source,
                self.mjc_actuator_to_newton_idx,
                self.mjc_actuator_to_newton_target_q_idx,
                self.mjc_actuator_to_target_q_axis_idx,
                self.mjc_actuator_to_newton_ball_jnt,
                model.joint_X_c,
                joint_target_q,
                joint_target_qd,
                state.joint_q,
                mujoco_ctrl,
                target_q_per_world,
                coords_per_world,
                dofs_per_world,
                ctrls_per_world,
                joints_per_world,
                model.use_coord_layout_targets,
                control.joint_f if control is not None else None,
                model.joint_type,
                model.joint_child,
                model.body_flags,
                model.joint_q_start,
                model.joint_qd_start,
                model.joint_dof_dim,
                self.mj_qd_start,
                self.mjc_body_to_newton,
                state.body_f,
                model.body_mass,
                model.body_world,
                model.gravity,
                self.mjw_model.body_gravcomp,
                self.body_free_qd_start,
                contacts.contact_generation if prepare_contacts else None,
                self._last_contact_generation,
            ],
            outputs=[d.ctrl, d.qfrc_applied, d.xfrc_applied, d.nacon, self._dormant_count],
            device=model.device,
        )
        return prepare_contacts

    def _apply_mjc_control(self, model: Model, state: State, control: Control | None, mj_data: MjWarpData | MjData):
        if control is None or control.joint_f is None:
            if state.body_f is None:
                return
        is_mjwarp = SolverMuJoCo._data_is_mjwarp(mj_data)
        single_world_template = False
        if is_mjwarp:
            ctrl = mj_data.ctrl
            qfrc = mj_data.qfrc_applied
            xfrc = mj_data.xfrc_applied
            nworld = mj_data.nworld
        else:
            effective_dof_count = model.joint_dof_count - self._total_loop_joint_dofs
            single_world_template = len(mj_data.qfrc_applied) < effective_dof_count
            ctrl = wp.zeros((1, len(mj_data.ctrl)), dtype=wp.float32, device=model.device)
            qfrc = wp.zeros((1, len(mj_data.qfrc_applied)), dtype=wp.float32, device=model.device)
            xfrc = wp.zeros((1, len(mj_data.xfrc_applied)), dtype=wp.spatial_vector, device=model.device)
            nworld = 1
        joints_per_world = (
            model.joint_count // model.world_count if single_world_template else model.joint_count // nworld
        )
        if control is not None:
            # Use instance arrays (built during MuJoCo model construction)
            if self.mjc_actuator_ctrl_source is not None and self.mjc_actuator_to_newton_idx is not None:
                nu = self.mjc_actuator_ctrl_source.shape[0]
                dofs_per_world = model.joint_dof_count // nworld if nworld > 0 else model.joint_dof_count
                target_q_total = control.joint_target_q.shape[0] if control.joint_target_q is not None else 0
                target_q_per_world = target_q_total // nworld if nworld > 0 else target_q_total
                coords_per_world = model.joint_coord_count // nworld if nworld > 0 else model.joint_coord_count

                # Get mujoco.ctrl (None if not available - won't be accessed if no CTRL_DIRECT actuators)
                mujoco_ctrl_ns = getattr(control, "mujoco", None)
                mujoco_ctrl = getattr(mujoco_ctrl_ns, "ctrl", None) if mujoco_ctrl_ns is not None else None
                ctrls_per_world = mujoco_ctrl.shape[0] // nworld if mujoco_ctrl is not None and nworld > 0 else 0

                wp.launch(
                    apply_mjc_control_kernel,
                    dim=(nworld, nu),
                    inputs=[
                        self.mjc_actuator_ctrl_source,
                        self.mjc_actuator_to_newton_idx,
                        self.mjc_actuator_to_newton_target_q_idx,
                        self.mjc_actuator_to_target_q_axis_idx,
                        self.mjc_actuator_to_newton_ball_jnt,
                        model.joint_X_c,
                        control.joint_target_q,
                        control.joint_target_qd,
                        state.joint_q,
                        mujoco_ctrl,
                        target_q_per_world,
                        coords_per_world,
                        dofs_per_world,
                        ctrls_per_world,
                        joints_per_world,
                        model.use_coord_layout_targets,
                    ],
                    outputs=[
                        ctrl,
                    ],
                    device=model.device,
                )
            wp.launch(
                apply_mjc_qfrc_kernel,
                dim=(nworld, joints_per_world),
                inputs=[
                    control.joint_f,
                    state.joint_q,
                    model.joint_type,
                    model.joint_child,
                    model.body_flags,
                    model.joint_q_start,
                    model.joint_qd_start,
                    model.joint_dof_dim,
                    model.joint_X_c,
                    joints_per_world,
                    self.mj_qd_start,
                ],
                outputs=[
                    qfrc,
                ],
                device=model.device,
            )

        if state.body_f is not None:
            # Launch over MuJoCo bodies
            nbody = self.mjc_body_to_newton.shape[1]
            wp.launch(
                apply_mjc_body_f_kernel,
                dim=(nworld, nbody),
                inputs=[
                    self.mjc_body_to_newton,
                    model.body_flags,
                    state.body_f,
                    model.body_mass,
                    model.body_world,
                    model.gravity,
                    self.mjw_model.body_gravcomp,
                ],
                outputs=[
                    xfrc,
                ],
                device=model.device,
            )
        if control is not None and control.joint_f is not None:
            # Free/DISTANCE joint forces are applied via xfrc_applied to preserve COM-wrench semantics.
            nbody = self.mjc_body_to_newton.shape[1]
            wp.launch(
                apply_mjc_free_joint_f_to_body_f_kernel,
                dim=(nworld, nbody),
                inputs=[
                    self.mjc_body_to_newton,
                    model.body_flags,
                    self.body_free_qd_start,
                    control.joint_f,
                ],
                outputs=[
                    xfrc,
                ],
                device=model.device,
            )
        if not is_mjwarp:
            mj_data.xfrc_applied = xfrc.numpy()
            mj_data.ctrl[:] = ctrl.numpy().flatten()
            mj_data.qfrc_applied[:] = qfrc.numpy()

    def _update_mjc_data(
        self,
        mj_data: MjWarpData | MjData,
        model: Model,
        state: State | None = None,
        world_mask: wp.array[wp.bool] | None = None,
    ):
        is_mjwarp = SolverMuJoCo._data_is_mjwarp(mj_data)
        single_world_template = False
        if is_mjwarp:
            # we have an MjWarp Data object
            qpos = mj_data.qpos
            qvel = mj_data.qvel
            nworld = mj_data.nworld
        else:
            # we have an MjData object from Mujoco
            effective_coord_count = model.joint_coord_count - self._total_loop_joint_coords
            single_world_template = len(mj_data.qpos) < effective_coord_count
            expected_qpos = (
                effective_coord_count // model.world_count if single_world_template else effective_coord_count
            )
            assert len(mj_data.qpos) >= expected_qpos, (
                f"MuJoCo qpos size ({len(mj_data.qpos)}) < expected joint coords ({expected_qpos})"
            )
            qpos = wp.empty((1, len(mj_data.qpos)), dtype=wp.float32, device=model.device)
            qvel = wp.empty((1, len(mj_data.qvel)), dtype=wp.float32, device=model.device)
            nworld = 1
        if state is None:
            joint_q = model.joint_q
            joint_qd = model.joint_qd
        else:
            joint_q = state.joint_q
            joint_qd = state.joint_qd
        joints_per_world = (
            model.joint_count // model.world_count if single_world_template else model.joint_count // nworld
        )
        mujoco_attrs = getattr(model, "mujoco", None)
        dof_ref = getattr(mujoco_attrs, "dof_ref", None) if mujoco_attrs is not None else None
        # With sleeping, coordinates that moved by more than the tolerance flag their tree
        # (external pose edits) and the walker below wakes the flagged trees.
        detect_tree_changes = is_mjwarp and self.enable_sleeping
        wp.launch(
            convert_warp_coords_to_mj_kernel,
            dim=(nworld, joints_per_world),
            inputs=[
                joint_q,
                joint_qd,
                world_mask,
                joints_per_world,
                model.joint_type,
                model.joint_q_start,
                model.joint_qd_start,
                model.joint_dof_dim,
                model.joint_child,
                model.joint_X_p,
                model.joint_X_c,
                model.body_com,
                dof_ref,
                self.mj_q_start,
                self.mj_qd_start,
                self._sleep_qpos_treeid if detect_tree_changes else None,
                1.0e-6,
            ],
            outputs=[qpos, qvel, self._sleep_tree_changed if detect_tree_changes else None],
            device=model.device,
        )

        if detect_tree_changes:
            wp.launch(
                wake_changed_trees_kernel,
                dim=nworld,
                inputs=[
                    self._sleep_tree_changed,
                    self.mjw_model.ntree,
                    self._sleep_awake_value,
                ],
                outputs=[self.mjw_data.tree_asleep],
                device=model.device,
            )

        if not is_mjwarp:
            mj_data.qpos[:] = qpos.numpy().flatten()[: len(mj_data.qpos)]
            mj_data.qvel[:] = qvel.numpy().flatten()[: len(mj_data.qvel)]

    def _update_newton_state(
        self,
        model: Model,
        state: State,
        mj_data: MjWarpData | MjData,
        state_prev: State,
        *,
        publish_body_state: bool = True,
        publish_joint_state: bool = True,
    ):
        """Update a Newton state from MuJoCo coordinates and kinematics.

        Args:
            model: Newton model that defines the joint and body layout.
            state: Output Newton state to populate from MuJoCo data.
            mj_data: MuJoCo runtime data source, either CPU `MjData` or Warp data.
            state_prev: Previous Newton state. Kinematic joint coordinates and
                velocities are copied from this state because MuJoCo does not
                independently integrate those DOFs.
            publish_body_state: Whether to run forward kinematics for ``body_q``/``body_qd``.
                Intermediate substeps skip it when the converted contacts read their poses from
                MJWarp and nothing else reads Newton body state before the finalizing substep.
            publish_joint_state: Whether to convert ``qpos``/``qvel`` into ``joint_q``/``joint_qd``.
                Intermediate substeps skip it when nothing reads Newton joint state before the
                finalizing substep (see :attr:`publish_intermediate_joint_state`); the body-state
                publish requires it.
        """
        is_mjwarp = SolverMuJoCo._data_is_mjwarp(mj_data)
        single_world_template = False
        if is_mjwarp:
            # we have an MjWarp Data object
            qpos = mj_data.qpos
            qvel = mj_data.qvel
            nworld = mj_data.nworld
        else:
            # we have an MjData object from Mujoco
            effective_coord_count = model.joint_coord_count - self._total_loop_joint_coords
            single_world_template = len(mj_data.qpos) < effective_coord_count
            qpos = wp.array([mj_data.qpos], dtype=wp.float32, device=model.device)
            qvel = wp.array([mj_data.qvel], dtype=wp.float32, device=model.device)
            nworld = 1
        joints_per_world = (
            model.joint_count // model.world_count if single_world_template else model.joint_count // nworld
        )
        mujoco_attrs = getattr(model, "mujoco", None)
        dof_ref = getattr(mujoco_attrs, "dof_ref", None) if mujoco_attrs is not None else None
        if publish_joint_state or publish_body_state:
            wp.launch(
                convert_mj_coords_to_warp_kernel,
                dim=(nworld, joints_per_world),
                inputs=[
                    qpos,
                    qvel,
                    joints_per_world,
                    model.joint_type,
                    model.joint_q_start,
                    model.joint_qd_start,
                    model.joint_dof_dim,
                    model.joint_child,
                    model.joint_X_p,
                    model.joint_X_c,
                    model.body_com,
                    dof_ref,
                    model.body_flags,
                    state_prev.joint_q,
                    state_prev.joint_qd,
                    self.mj_q_start,
                    self.mj_qd_start,
                ],
                outputs=[state.joint_q, state.joint_qd],
                device=model.device,
            )

        if publish_body_state:
            eval_fk(model, state.joint_q, state.joint_qd, state)

        # Update rigid force fields on state.
        if state.body_qdd is not None or state.body_parent_f is not None:
            # Launch over MuJoCo bodies
            nbody = self.mjc_body_to_newton.shape[1]
            wp.launch(
                convert_rigid_forces_from_mj_kernel,
                (nworld, nbody),
                inputs=[
                    self.mjc_body_to_newton,
                    self.mjw_model.body_rootid,
                    self.mjw_model.opt.gravity,
                    self.mjw_data.xipos,
                    self.mjw_data.subtree_com,
                    self.mjw_data.cacc,
                    self.mjw_data.cvel,
                    self.mjw_data.cfrc_int,
                ],
                outputs=[state.body_qdd, state.body_parent_f],
                device=model.device,
            )

        # Update actuator forces in joint DOF space.
        qfrc_actuator = getattr(getattr(state, "mujoco", None), "qfrc_actuator", None)
        if qfrc_actuator is not None:
            if is_mjwarp:
                mjw_qfrc = mj_data.qfrc_actuator
                mjw_qpos = mj_data.qpos
            else:
                mjw_qfrc = wp.array([mj_data.qfrc_actuator], dtype=wp.float32, device=model.device)
                mjw_qpos = wp.array([mj_data.qpos], dtype=wp.float32, device=model.device)
            wp.launch(
                convert_qfrc_actuator_from_mj_kernel,
                dim=(nworld, joints_per_world),
                inputs=[
                    mjw_qfrc,
                    mjw_qpos,
                    joints_per_world,
                    model.joint_type,
                    model.joint_q_start,
                    model.joint_qd_start,
                    model.joint_dof_dim,
                    model.joint_child,
                    model.joint_X_c,
                    model.body_com,
                    self.mj_q_start,
                    self.mj_qd_start,
                ],
                outputs=[qfrc_actuator],
                device=model.device,
            )

    @staticmethod
    def _find_body_collision_filter_pairs(
        model: Model,
        selected_bodies: np.ndarray,
        colliding_shapes: np.ndarray,
    ):
        """For shape collision filter pairs, find body collision filter pairs that are contained within."""

        shape_set = set(colliding_shapes)

        body_shapes = {}
        for body in selected_bodies:
            shapes = model.body_shapes[body]
            shapes = [s for s in shapes if s in shape_set]
            body_shapes[body] = shapes

        # Batch all candidate shape pairs into one bulk filter query; per-pair
        # membership calls dominate this loop for large body selections.
        bodies_a, bodies_b = np.triu_indices(len(selected_bodies), k=1)
        candidate_pairs = []
        pair_counts = np.empty(len(bodies_a), dtype=np.int64)
        for k, (body_a, body_b) in enumerate(zip(bodies_a, bodies_b, strict=True)):
            shapes_1 = body_shapes[selected_bodies[body_a]]
            shapes_2 = body_shapes[selected_bodies[body_b]]
            pair_counts[k] = len(shapes_1) * len(shapes_2)
            candidate_pairs.extend((shape_1, shape_2) for shape_1 in shapes_1 for shape_2 in shapes_2)

        filtered = model.shape_collision_filter_mask(np.asarray(candidate_pairs, dtype=np.int64).reshape((-1, 2)))
        # A body pair is excluded when every one of its shape pairs is
        # filtered (vacuously true when either body has no colliding shapes).
        unfiltered_cumulative = np.concatenate(([0], np.cumsum(~filtered)))
        segment_ends = np.cumsum(pair_counts)
        excluded = unfiltered_cumulative[segment_ends] == unfiltered_cumulative[segment_ends - pair_counts]
        return [(selected_bodies[bodies_a[k]], selected_bodies[bodies_b[k]]) for k in np.flatnonzero(excluded)]

    @staticmethod
    def _compile_newton_collision_masks(model: Model, selected_shapes: np.ndarray):
        """Compile selected Newton groups and filters into MuJoCo masks."""
        groups = model.shape_collision_group.numpy()[selected_shapes]
        # The exact compiler materializes a dense pair graph. Enforce its size
        # limit before even reading sparse filters; the fallback needs only the
        # selected collision groups.
        if selected_shapes.shape[0] > NEWTON_COLLISION_MASK_MAX_SHAPE_COUNT:
            return compile_newton_collision_graph(groups)

        # Query only the bounded selected-shape graph. Materializing the global
        # sparse pair array scales this template conversion with replicated worlds.
        shape_a, shape_b = np.triu_indices(selected_shapes.shape[0], k=1)
        candidate_pairs = np.column_stack((selected_shapes[shape_a], selected_shapes[shape_b]))
        filtered = model.shape_collision_filter_mask(candidate_pairs)
        local_filter_pairs = np.column_stack((shape_a[filtered], shape_b[filtered]))
        return compile_newton_collision_graph(groups, excluded_pairs=local_filter_pairs)

    @staticmethod
    def _preserved_masks_cover_collision_filters(
        model: Model,
        selected_shapes: np.ndarray,
        collision_type: np.ndarray,
        collision_affinity: np.ndarray,
        body_filters: Iterable[tuple[int, int]],
    ) -> bool:
        """Return whether preserved masks and body excludes cover Newton pair filters."""
        filter_pairs = model.shape_collision_filter_pairs_array()
        if filter_pairs.shape[0] == 0:
            return True

        selected = np.zeros(model.shape_count, dtype=bool)
        selected[selected_shapes] = True
        filter_pairs = filter_pairs[selected[filter_pairs[:, 0]] & selected[filter_pairs[:, 1]]]
        if filter_pairs.shape[0] == 0:
            return True

        collision_type = (collision_type.astype(np.int64, copy=False) & np.int64(0xFFFFFFFF)).astype(np.uint32)
        collision_affinity = (collision_affinity.astype(np.int64, copy=False) & np.int64(0xFFFFFFFF)).astype(np.uint32)
        shape_a = filter_pairs[:, 0]
        shape_b = filter_pairs[:, 1]
        raw_masks_allow = ((collision_type[shape_a] & collision_affinity[shape_b]) != 0) | (
            (collision_type[shape_b] & collision_affinity[shape_a]) != 0
        )
        remaining_pairs = filter_pairs[raw_masks_allow]
        if remaining_pairs.shape[0] == 0:
            return True

        body_filter_set = {tuple(sorted((int(body_a), int(body_b)))) for body_a, body_b in body_filters}
        shape_body = model.shape_body.numpy()
        for shape_a, shape_b in remaining_pairs:
            body_a = int(shape_body[shape_a])
            body_b = int(shape_body[shape_b])
            # MuJoCo never generates automatic contacts between geoms on the
            # same body; whole-body Newton filters are emitted as excludes.
            if body_a == body_b or tuple(sorted((body_a, body_b))) in body_filter_set:
                continue
            return False
        return True

    @staticmethod
    def _color_collision_shapes(
        model: Model, selected_shapes: np.ndarray, visualize_graph: bool = False, shape_labels: list[str] | None = None
    ) -> np.ndarray:
        """
        Find a graph coloring of the collision filter pairs in the model.
        Shapes within the same color cannot collide with each other.
        Shapes can only collide with shapes of different colors.

        Args:
            model: The model to color the collision shapes of.
            selected_shapes: The indices of the collision shapes to color.
            visualize_graph: Whether to visualize the graph coloring.
            shape_labels: The labels of the shapes, only used for visualization.

        Returns:
            np.ndarray: An integer array of shape (num_shapes,), where each element is the color of the corresponding shape.
        """
        # we first create a mapping from selected shape to local color shape index
        # to reduce the number of nodes in the graph to only the number of selected shapes
        # without any gaps between the indices (otherwise we have to allocate max(selected_shapes) + 1 nodes)
        to_color_shape_index = {}
        for i, shape in enumerate(selected_shapes):
            to_color_shape_index[shape] = i
        # find graph coloring of collision filter pairs
        num_shapes = len(selected_shapes)
        shape_a, shape_b = np.triu_indices(num_shapes, k=1)
        shape_collision_group_np = model.shape_collision_group.numpy()
        cgroup = shape_collision_group_np[selected_shapes]
        # edges representing colliding shape pairs
        candidate_pairs = np.stack((selected_shapes[shape_a], selected_shapes[shape_b]), axis=1)
        filtered = model.shape_collision_filter_mask(candidate_pairs)
        group_a, group_b = cgroup[shape_a], cgroup[shape_b]
        group_allowed = (group_a != 0) & (group_b != 0)
        group_allowed &= ((group_a > 0) & ((group_a == group_b) | (group_b < 0))) | (
            (group_a < 0) & (group_a != group_b)
        )
        edge_mask = ~filtered & group_allowed
        graph_edges = np.stack((shape_a[edge_mask], shape_b[edge_mask]), axis=1).astype(np.int32)
        shape_color = np.zeros(model.shape_count, dtype=np.int32)
        if len(graph_edges) > 0:
            color_groups = color_graph(
                num_nodes=num_shapes,
                graph_edge_indices=wp.array(graph_edges, dtype=wp.int32),
                balance_colors=False,
            )
            for color, group in enumerate(color_groups):
                shape_color[selected_shapes[group]] = color
            if visualize_graph:
                plot_graph(
                    vertices=np.arange(num_shapes),
                    edges=graph_edges,
                    node_labels=[shape_labels[i] for i in selected_shapes] if shape_labels is not None else None,
                    node_colors=[shape_color[i] for i in selected_shapes],
                )

        return shape_color

    @staticmethod
    def _collision_color_masks(color: int) -> tuple[int, int]:
        """Encode a collision color as signed MuJoCo ``contype`` and ``conaffinity`` masks."""
        if color >= 32:
            return 1, 1

        uint32_mask = (1 << 32) - 1
        contype = 1 << color
        conaffinity = uint32_mask & ~contype

        def as_signed_int32(mask: int) -> int:
            return mask if mask < (1 << 31) else mask - (1 << 32)

        return as_signed_int32(contype), as_signed_int32(conaffinity)

    def get_max_contact_count(self) -> int:
        """Return the maximum number of rigid contacts that can be generated by MuJoCo."""
        if self.use_mujoco_cpu:
            raise NotImplementedError()
        return self.mjw_data.naconmax

    @property
    def collision_sleep_filter(
        self,
    ) -> tuple[wp.array[wp.vec2i], wp.array2d[wp.int32]] | None:
        """Shape-to-tree mapping and mutable sleep state for external collision detection."""
        if not self.enable_sleeping:
            return None
        if self._collision_shape_sleep_index is None:
            raise RuntimeError("Collision sleep mapping was not initialized")
        return self._collision_shape_sleep_index, self.mjw_data.tree_asleep

    def set_body_sleep_state(
        self,
        body_ids: wp.array2d[wp.int32],
        asleep: wp.array2d[wp.bool],
        world_ids: wp.array[wp.int32],
    ) -> None:
        """Set the sleep state restored for selected free-body trees.

        ``body_ids`` and ``asleep`` contain one row per world. ``world_ids``
        selects the rows to update and must contain valid, unique indices. The
        overrides are applied after the next reset reconciles joint coordinates.
        """
        if not self.enable_sleeping:
            raise RuntimeError("Body sleep state requires enable_sleeping=True.")
        if self._body_sleep_index is None or self._body_sleep_override is None:
            raise RuntimeError("Body sleep mapping was not initialized.")
        for name, array, dtype, ndim in (
            ("body_ids", body_ids, wp.int32, 2),
            ("asleep", asleep, wp.bool, 2),
            ("world_ids", world_ids, wp.int32, 1),
        ):
            if not isinstance(array, wp.array) or array.dtype != dtype:
                raise TypeError(f"{name} must be a {ndim}-dimensional wp.array with dtype {dtype}.")
            if array.ndim != ndim:
                raise ValueError(f"{name} must be {ndim}-dimensional; got shape {array.shape}.")
            if array.device != self.model.device:
                raise ValueError(f"{name} must be on {self.model.device}, got {array.device}.")
        if body_ids.shape != asleep.shape:
            raise ValueError(f"body_ids and asleep must have equal shapes; got {body_ids.shape} and {asleep.shape}.")
        if body_ids.shape[0] != self.mjw_data.nworld:
            raise ValueError(f"body_ids must have {self.mjw_data.nworld} rows; got {body_ids.shape[0]}.")
        if not world_ids.shape[0]:
            return

        with wp.ScopedDevice(self.model.device), self._scoped_mujoco_warp_execution():
            wp.launch(
                set_selected_body_sleep_override_kernel,
                dim=world_ids.shape[0],
                inputs=[
                    world_ids,
                    body_ids,
                    asleep,
                    self._body_sleep_index,
                    self._body_sleep_override,
                ],
                device=self.model.device,
            )
        self._has_body_sleep_overrides = True

    @override
    def update_contacts(self, contacts: Contacts, state: State | None = None) -> None:
        """Update `contacts` from MuJoCo contacts when running with ``use_mujoco_contacts``."""
        self._apply_module_options()
        if not self._use_mujoco_contacts:
            raise RuntimeError("update_contacts() requires use_mujoco_contacts=True.")
        if self.use_mujoco_cpu:
            raise NotImplementedError()

        # TODO: ensure that class invariants are preserved
        # TODO: fill actual contact arrays instead of creating new ones
        mj_data = self.mjw_data
        mj_contact = mj_data.contact

        if mj_data.naconmax > contacts.rigid_contact_max:
            raise ValueError(
                f"MuJoCo naconmax ({mj_data.naconmax}) exceeds contacts.rigid_contact_max "
                f"({contacts.rigid_contact_max}). Create Contacts with at least "
                f"rigid_contact_max={mj_data.naconmax}."
            )

        wp.launch(
            self._convert_mjw_contacts_to_newton_kernel,
            dim=mj_data.naconmax,
            inputs=[
                self.mjc_geom_to_newton_shape,
                self.mjw_model.opt.cone,
                mj_data.nacon,
                mj_contact.pos,
                mj_contact.frame,
                mj_contact.friction,
                mj_contact.dist,
                mj_contact.dim,
                mj_contact.geom,
                mj_contact.efc_address,
                mj_contact.worldid,
                mj_contact.adhesion,
                mj_data.efc.force,
                self.mjw_model.geom_bodyid,
                mj_data.xpos,
                mj_data.xquat,
                mj_data.njmax,
            ],
            outputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.force,
            ],
            device=self.model.device,
        )
        contacts.n_contacts = mj_data.nacon

    def _convert_to_mjc(
        self,
        model: Model,
        state: State | None = None,
        *,
        separate_worlds: bool | None = None,
        iterations: int | None = None,
        ls_iterations: int | None = None,
        ccd_iterations: int | None = None,
        sdf_iterations: int | None = None,
        sdf_initpoints: int | None = None,
        njmax: int | None = None,  # number of constraints per world
        nconmax: int | None = None,
        nvmax: int | None = None,
        solver: int | str | None = None,
        integrator: int | str | None = None,
        enableflags: int = 0,
        disableflags: int = 0,
        disable_contacts: bool = False,
        impratio: float | None = None,
        tolerance: float | None = None,
        ls_tolerance: float | None = None,
        ccd_tolerance: float | None = None,
        sleep_tolerance: float | None = None,
        density: float | None = None,
        viscosity: float | None = None,
        wind: tuple | None = None,
        magnetic: tuple | None = None,
        cone: int | str | None = None,
        jacobian: int | str | None = None,
        target_filename: str | None = None,
        skip_visual_only_geoms: bool = True,
        include_sites: bool = True,
    ) -> tuple[MjWarpModel, MjWarpData, MjModel, MjData]:
        """
        Convert a Newton model and state to MuJoCo (Warp) model and data.

        See ``docs/solvers/mujoco.rst`` for user-facing documentation of
        all conversions performed here.  Keep that file in sync when changing
        this method.

        Solver options (e.g., ``impratio``) follow this resolution priority:

        1. **Constructor argument** - If provided, same value is used for all worlds.
        2. **Newton model custom attribute** (``model.mujoco.<option>``) - Supports per-world values.
        3. **MuJoCo default** - Used if neither of the above is set.

        Args:
            model: The Newton model to convert.
            state: The Newton state to convert (optional).
            separate_worlds: If True, each world is a separate MuJoCo simulation. If None, defaults to True for GPU mode (not use_mujoco_cpu).
            iterations: Maximum solver iterations. If None, uses model custom attribute or MuJoCo's default (100).
            ls_iterations: Maximum line search iterations. If None, uses model custom attribute or MuJoCo's default (50).
            njmax: Maximum number of constraints per world.
            nconmax: Maximum number of contacts.
            nvmax: Maximum number of active degrees of freedom per world.
            solver: Constraint solver type ("cg" or "newton"). If None, uses model custom attribute or Newton's default ("newton").
            integrator: Integration method ("euler", "rk4", "implicit", "implicitfast"). If None, uses model custom attribute or Newton's default ("implicitfast").
            enableflags: MuJoCo enable flags bitmask.
            disableflags: MuJoCo disable flags bitmask.
            disable_contacts: If True, disable contact computation.
            impratio: Impedance ratio for contacts. If None, uses model custom attribute or MuJoCo default (1.0).
            tolerance: Solver tolerance. If None, uses model custom attribute or MuJoCo default (1e-8).
            ls_tolerance: Line search tolerance. If None, uses model custom attribute or MuJoCo default (0.01).
            ccd_tolerance: CCD tolerance. If None, uses model custom attribute or MuJoCo default (1e-6).
            sleep_tolerance: Sleep velocity tolerance. If None, uses model custom attribute or MuJoCo default (0.001).
            density: Medium density. If None, uses model custom attribute or MuJoCo default (0.0).
            viscosity: Medium viscosity. If None, uses model custom attribute or MuJoCo default (0.0).
            wind: Wind velocity vector (x, y, z). If None, uses model custom attribute or MuJoCo default (0, 0, 0).
            magnetic: Magnetic flux vector (x, y, z). If None, uses model custom attribute or MuJoCo default (0, -0.5, 0).
            cone: Friction cone type ("pyramidal" or "elliptic"). If None, uses model custom attribute or Newton's default ("pyramidal").
            jacobian: Jacobian computation method ("dense", "sparse", or "auto"). If None, uses model custom attribute or MuJoCo default ("auto").
            target_filename: Optional path to save generated MJCF file.
            skip_visual_only_geoms: If True, skip geoms that are visual-only.
            include_sites: If True, include sites in the model.

        Returns:
            tuple[MjWarpModel, MjWarpData, MjModel, MjData]: Model and data objects for
                ``mujoco_warp`` and MuJoCo.
        """
        if not model.joint_count:
            raise ValueError("The model must have at least one joint to be able to convert it to MuJoCo.")

        # Set default for separate_worlds if None
        if separate_worlds is None:
            separate_worlds = True

        # Validate that separate_worlds=False is only used with single world
        if not separate_worlds and model.world_count > 1:
            raise ValueError(
                f"separate_worlds=False is only supported for single-world models. "
                f"Got world_count={model.world_count}. Use separate_worlds=True for multi-world models."
            )

        # Validate model compatibility with separate_worlds mode
        if separate_worlds:
            self._validate_model_for_separate_worlds(model)
            if self.enable_sleeping and model.world_count > 1:
                dofs_per_world = model.joint_dof_count // model.world_count
                joint_qd = model.joint_qd.numpy().reshape(model.world_count, dofs_per_world)
                differing_worlds = np.flatnonzero(np.any(joint_qd != joint_qd[0], axis=1))
                if differing_worlds.size:
                    world = int(differing_worlds[0])
                    raise ValueError(
                        "enable_sleeping=True requires identical default joint velocities in every world "
                        f"because MuJoCo Warp uses one shared initial sleep state; world {world} differs from world 0."
                    )

        mujoco, mujoco_warp = self.import_mujoco()

        actuator_args = {
            # "ctrllimited": True,
            # "ctrlrange": (-1.0, 1.0),
            "gear": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "trntype": mujoco.mjtTrn.mjTRN_JOINT,
            # motor actuation properties (already the default settings in Mujoco)
            "gainprm": [1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "biasprm": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "dyntype": mujoco.mjtDyn.mjDYN_NONE,
            "gaintype": mujoco.mjtGain.mjGAIN_FIXED,
            "biastype": mujoco.mjtBias.mjBIAS_AFFINE,
        }

        # Convert string enum values to integers using the static parser methods
        # (these methods handle both string and int inputs)
        # Only convert if not None - will check custom attributes later if None
        if solver is not None:
            solver = self._parse_solver(solver)
        if integrator is not None:
            integrator = self._parse_integrator(integrator)
        if cone is not None:
            cone = self._parse_cone(cone)
        if jacobian is not None:
            jacobian = self._parse_jacobian(jacobian)

        def quat_to_mjc(q):
            # convert from xyzw to wxyz
            # For Warp kernel equivalent, see quat_xyzw_to_wxyz() in kernels.py
            return [q[3], q[0], q[1], q[2]]

        def quat_from_mjc(q):
            # convert from wxyz to xyzw
            # For Warp kernel equivalent, see quat_wxyz_to_xyzw() in kernels.py
            return [q[1], q[2], q[3], q[0]]

        def fill_arr_from_dict(arr: np.ndarray, d: dict[int, Any]):
            # fast way to fill an array from a dictionary
            # keys and values can also be tuples of integers
            keys = np.array(list(d.keys()), dtype=int)
            vals = np.array(list(d.values()), dtype=int)
            if keys.ndim == 1:
                arr[keys] = vals
            else:
                arr[tuple(keys.T)] = vals

        # Solver option resolution priority (highest to lowest):
        #   1. Constructor argument (e.g., impratio=5.0) - same value for all worlds
        #   2. Newton model custom attribute (model.mujoco.<option>) - supports per-world values
        #   3. MuJoCo default

        # Track which WORLD frequency options were overridden by constructor
        overridden_options = set()

        # Get mujoco custom attributes once
        mujoco_attrs = getattr(model, "mujoco", None)
        if sleep_tolerance is None and mujoco_attrs and hasattr(mujoco_attrs, "sleep_tolerance"):
            sleep_tolerance_values = mujoco_attrs.sleep_tolerance.numpy()
            invalid_sleep_tolerance = ~np.isfinite(sleep_tolerance_values) | (sleep_tolerance_values < 0.0)
            if np.any(invalid_sleep_tolerance):
                first_invalid = int(np.flatnonzero(invalid_sleep_tolerance)[0])
                value = sleep_tolerance_values[first_invalid]
                raise ValueError(
                    "model.mujoco.sleep_tolerance values must be finite and non-negative; "
                    f"got {value} at world {first_invalid}."
                )

        # Helper to resolve scalar option value
        def resolve_option(name: str, constructor_value):
            """Resolve scalar option from constructor > model attribute > None (use MuJoCo default)."""
            if constructor_value is not None:
                overridden_options.add(name)
                return constructor_value
            if mujoco_attrs and hasattr(mujoco_attrs, name):
                # Read from index 0 (template world) for initialization
                return float(getattr(mujoco_attrs, name).numpy()[0])
            return None

        # Helper to resolve vector option value
        def resolve_vector_option(name: str, constructor_value):
            """Resolve vector option from constructor > model attribute > None (use MuJoCo default)."""
            if constructor_value is not None:
                overridden_options.add(name)
                return constructor_value
            if mujoco_attrs and hasattr(mujoco_attrs, name):
                # Read from index 0 (template world) for initialization
                vec = getattr(mujoco_attrs, name).numpy()[0]
                return tuple(vec)
            return None

        # Resolve all WORLD frequency scalar options
        impratio = resolve_option("impratio", impratio)
        tolerance = resolve_option("tolerance", tolerance)
        ls_tolerance = resolve_option("ls_tolerance", ls_tolerance)
        ccd_tolerance = resolve_option("ccd_tolerance", ccd_tolerance)
        sleep_tolerance = resolve_option("sleep_tolerance", sleep_tolerance)
        density = resolve_option("density", density)
        viscosity = resolve_option("viscosity", viscosity)
        if sleep_tolerance is not None and (not math.isfinite(sleep_tolerance) or sleep_tolerance < 0.0):
            raise ValueError(f"sleep_tolerance must be finite and non-negative, got {sleep_tolerance}.")

        # Resolve WORLD frequency vector options
        wind = resolve_vector_option("wind", wind)
        magnetic = resolve_vector_option("magnetic", magnetic)

        # Resolve ONCE frequency numeric options from custom attributes if not provided
        if iterations is None and mujoco_attrs and hasattr(mujoco_attrs, "iterations"):
            iterations = int(mujoco_attrs.iterations.numpy()[0])
        if ls_iterations is None and mujoco_attrs and hasattr(mujoco_attrs, "ls_iterations"):
            ls_iterations = int(mujoco_attrs.ls_iterations.numpy()[0])
        if ccd_iterations is None and mujoco_attrs and hasattr(mujoco_attrs, "ccd_iterations"):
            ccd_iterations = int(mujoco_attrs.ccd_iterations.numpy()[0])
        if sdf_iterations is None and mujoco_attrs and hasattr(mujoco_attrs, "sdf_iterations"):
            sdf_iterations = int(mujoco_attrs.sdf_iterations.numpy()[0])
        if sdf_initpoints is None and mujoco_attrs and hasattr(mujoco_attrs, "sdf_initpoints"):
            sdf_initpoints = int(mujoco_attrs.sdf_initpoints.numpy()[0])

        # Set defaults for numeric options if still None (use MuJoCo defaults)
        if iterations is None:
            iterations = 100
        if ls_iterations is None:
            ls_iterations = 50

        # Resolve ONCE frequency enum options from custom attributes if not provided
        if solver is None and mujoco_attrs and hasattr(mujoco_attrs, "solver"):
            solver = int(mujoco_attrs.solver.numpy()[0])
        if integrator is None and mujoco_attrs and hasattr(mujoco_attrs, "integrator"):
            integrator = int(mujoco_attrs.integrator.numpy()[0])
        if cone is None and mujoco_attrs and hasattr(mujoco_attrs, "cone"):
            cone = int(mujoco_attrs.cone.numpy()[0])
        if jacobian is None and mujoco_attrs and hasattr(mujoco_attrs, "jacobian"):
            jacobian = int(mujoco_attrs.jacobian.numpy()[0])

        # Set defaults for enum options if still None (use Newton defaults, not MuJoCo defaults)
        if solver is None:
            solver = mujoco.mjtSolver.mjSOL_NEWTON  # Newton default (not CG)
        if integrator is None:
            integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST  # Newton default (not Euler)
        if cone is None:
            cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        if jacobian is None:
            jacobian = mujoco.mjtJacobian.mjJAC_AUTO
        if enableflags & mujoco.mjtEnableBit.mjENBL_SLEEP and solver != mujoco.mjtSolver.mjSOL_NEWTON:
            raise ValueError("enable_sleeping=True requires solver='newton'.")
        if enableflags & mujoco.mjtEnableBit.mjENBL_SLEEP and integrator == mujoco.mjtIntegrator.mjINT_RK4:
            raise ValueError("enable_sleeping=True does not support integrator='rk4'.")

        spec = mujoco.MjSpec()
        spec.option.enableflags = enableflags
        spec.option.disableflags = disableflags
        spec.option.gravity = np.array([*model.gravity.numpy()[0]])
        spec.option.solver = solver
        spec.option.integrator = integrator
        spec.option.iterations = iterations
        spec.option.ls_iterations = ls_iterations
        spec.option.cone = cone
        spec.option.jacobian = jacobian

        # Set ONCE frequency numeric options (use MuJoCo defaults if None)
        if ccd_iterations is not None:
            spec.option.ccd_iterations = ccd_iterations
        if sdf_iterations is not None:
            spec.option.sdf_iterations = sdf_iterations
        if sdf_initpoints is not None:
            spec.option.sdf_initpoints = sdf_initpoints

        # Set WORLD frequency options (use MuJoCo defaults if None)
        if impratio is not None:
            spec.option.impratio = impratio
        if tolerance is not None:
            spec.option.tolerance = tolerance
        if ls_tolerance is not None:
            spec.option.ls_tolerance = ls_tolerance
        if ccd_tolerance is not None:
            spec.option.ccd_tolerance = ccd_tolerance
        if sleep_tolerance is not None:
            spec.option.sleep_tolerance = sleep_tolerance
        if density is not None:
            spec.option.density = density
        if viscosity is not None:
            spec.option.viscosity = viscosity
        if wind is not None:
            spec.option.wind = np.array(wind)
        if magnetic is not None:
            spec.option.magnetic = np.array(magnetic)

        spec.compiler.inertiafromgeom = mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_AUTO
        # alignfree would erase the offset used below to force general qM storage.
        spec.compiler.alignfree = False
        if mujoco_attrs and hasattr(mujoco_attrs, "autolimits"):
            spec.compiler.autolimits = bool(mujoco_attrs.autolimits.numpy()[0])

        joint_parent = model.joint_parent.numpy()
        joint_child = model.joint_child.numpy()
        joint_articulation = model.joint_articulation.numpy()
        joint_parent_xform = model.joint_X_p.numpy()
        joint_child_xform = model.joint_X_c.numpy()
        joint_limit_lower = model.joint_limit_lower.numpy()
        joint_limit_upper = model.joint_limit_upper.numpy()
        joint_type = model.joint_type.numpy()
        joint_axis = model.joint_axis.numpy()
        joint_dof_dim = model.joint_dof_dim.numpy()
        joint_qd_start = model.joint_qd_start.numpy()
        joint_target_q_start = model.joint_target_q_start.numpy()
        joint_q_start = model.joint_q_start.numpy()
        joint_armature = model.joint_armature.numpy()
        joint_effort_limit = model.joint_effort_limit.numpy()
        # Per-DOF actuator arrays
        joint_target_mode = model.joint_target_mode.numpy()
        joint_target_ke = model.joint_target_ke.numpy()
        joint_target_kd = model.joint_target_kd.numpy()
        # MoJoCo doesn't have velocity limit
        # joint_velocity_limit = model.joint_velocity_limit.numpy()
        joint_friction = model.joint_friction.numpy()
        joint_world = model.joint_world.numpy()
        body_flags = model.body_flags.numpy()
        body_q = model.body_q.numpy()
        body_mass = model.body_mass.numpy()
        body_inertia = model.body_inertia.numpy()
        body_com = model.body_com.numpy()
        body_world = model.body_world.numpy()
        shape_transform = model.shape_transform.numpy()
        shape_type = model.shape_type.numpy()
        # MuJoCo requires every size component to be positive, so conversion below
        # fills unused zero components. Keep those edits isolated from the model's
        # CPU-backed Warp array, for which ``numpy()`` may return a writable view.
        shape_size = model.shape_scale.numpy().copy()
        shape_flags = model.shape_flags.numpy()
        shape_collision_group = model.shape_collision_group.numpy()
        shape_world = model.shape_world.numpy()
        shape_mu = model.shape_material_mu.numpy()
        shape_ke = model.shape_material_ke.numpy()
        shape_kd = model.shape_material_kd.numpy()
        shape_mu_torsional = model.shape_material_mu_torsional.numpy()
        shape_mu_rolling = model.shape_material_mu_rolling.numpy()
        shape_margin = model.shape_margin.numpy()
        shape_gap = model.shape_gap.numpy()

        # retrieve MuJoCo-specific attributes
        mujoco_attrs = getattr(model, "mujoco", None)

        def get_custom_attribute(name: str) -> np.ndarray | None:
            if mujoco_attrs is None:
                return None
            attr = getattr(mujoco_attrs, name, None)
            if attr is None:
                return None
            return attr.numpy()

        shape_mjc_contype = get_custom_attribute("contype")
        shape_mjc_conaffinity = get_custom_attribute("conaffinity")
        shape_mjc_collision_mask_domain = get_custom_attribute("collision_mask_domain")
        shape_condim = get_custom_attribute("condim")
        shape_geom_group = get_custom_attribute("geom_group")
        shape_priority = get_custom_attribute("geom_priority")
        shape_geom_solimp = get_custom_attribute("geom_solimp")
        shape_geom_solmix = get_custom_attribute("geom_solmix")
        shape_mjc_solref = get_custom_attribute("solref")
        shape_mjc_solref_mode = get_custom_attribute("solref_mode")
        joint_dof_limit_margin = get_custom_attribute("limit_margin")
        joint_solimp_limit = get_custom_attribute("solimplimit")
        joint_solref_limit = get_custom_attribute("solreflimit")
        joint_solref_limit_mode = get_custom_attribute("solreflimit_mode")
        joint_dof_solref = get_custom_attribute("solreffriction")
        joint_dof_solimp = get_custom_attribute("solimpfriction")
        joint_stiffness = get_custom_attribute("dof_passive_stiffness")
        joint_damping = model.joint_damping.numpy() if model.joint_damping is not None else None
        joint_actgravcomp = get_custom_attribute("jnt_actgravcomp")
        body_gravcomp = get_custom_attribute("gravcomp")
        body_sleep_policy = get_custom_attribute("sleep_policy")
        joint_springref = get_custom_attribute("dof_springref")
        joint_ref = get_custom_attribute("dof_ref")

        def joint_has_raw_limit_solref(dof_idx: int) -> bool:
            if joint_solref_limit is None:
                return False
            if joint_solref_limit_mode is not None:
                return int(joint_solref_limit_mode[dof_idx]) == SOLREF_MODE_RAW
            return bool(np.any(joint_solref_limit[dof_idx] != 0.0))

        # Read the per-row equality arrays through the None-safe helper. finalize() materializes
        # these as shape-stable empty arrays even with no constraints, but the None-safe path keeps
        # this robust for models assembled without the standard custom-attribute pipeline.
        eq_constraint_type = get_custom_attribute("equality_constraint_type")
        eq_constraint_body1 = get_custom_attribute("equality_constraint_body1")
        eq_constraint_body2 = get_custom_attribute("equality_constraint_body2")
        eq_constraint_anchor = get_custom_attribute("equality_constraint_anchor")
        eq_constraint_torquescale = get_custom_attribute("equality_constraint_torquescale")
        eq_constraint_relpose = get_custom_attribute("equality_constraint_relpose")
        eq_constraint_joint1 = get_custom_attribute("equality_constraint_joint1")
        eq_constraint_joint2 = get_custom_attribute("equality_constraint_joint2")
        eq_constraint_polycoef = get_custom_attribute("equality_constraint_polycoef")
        eq_constraint_enabled = get_custom_attribute("equality_constraint_enabled")
        eq_constraint_world = get_custom_attribute("equality_constraint_world")
        eq_constraint_solref = get_custom_attribute("eq_solref")
        eq_constraint_solimp = get_custom_attribute("eq_solimp")
        eq_constraint_target_kind = get_custom_attribute("equality_constraint_target_kind")
        eq_constraint_target = get_custom_attribute("equality_constraint_target")
        eq_constraint_objtype = get_custom_attribute("equality_constraint_objtype")

        # Read mimic constraint arrays
        mimic_joint0 = model.constraint_mimic_joint0.numpy()
        mimic_joint1 = model.constraint_mimic_joint1.numpy()
        mimic_coef0 = model.constraint_mimic_coef0.numpy()
        mimic_coef1 = model.constraint_mimic_coef1.numpy()
        mimic_enabled = model.constraint_mimic_enabled.numpy()
        mimic_world = model.constraint_mimic_world.numpy()

        # mapping from joint axis to actuator index
        # axis_to_actuator[i, 0] = position actuator index
        # axis_to_actuator[i, 1] = velocity actuator index
        axis_to_actuator = np.zeros((model.joint_dof_count, 2), dtype=np.int32) - 1
        actuator_count = 0

        # Track actuator mapping as they're created (indexed by MuJoCo actuator order)
        # ctrl_source: 0=JOINT_TARGET, 1=CTRL_DIRECT
        # to_newton_idx: for JOINT_TARGET: >=0 position DOF, -1 unmapped, <=-2 velocity (DOF = -(val+2))
        #                for CTRL_DIRECT: MJCF-order index into control.mujoco.ctrl
        mjc_actuator_ctrl_source_list: list[int] = []
        mjc_actuator_to_newton_idx_list: list[int] = []
        mjc_actuator_to_target_q_idx_list: list[int] = []
        # For BALL-joint actuators (both layouts), this selects which axis-angle component (0/1/2)
        # of the target to feed MuJoCo; -1 means "scalar passthrough" (all other actuators).
        mjc_actuator_to_target_q_axis_idx_list: list[int] = []
        mjc_actuator_to_newton_ball_jnt_list: list[int] = []

        # supported non-fixed joint types in MuJoCo (fixed joints are handled by nesting bodies)
        supported_joint_types = {
            JointType.FREE,
            JointType.BALL,
            JointType.PRISMATIC,
            JointType.REVOLUTE,
            JointType.D6,
        }

        geom_type_mapping = {
            GeoType.SPHERE: mujoco.mjtGeom.mjGEOM_SPHERE,
            GeoType.PLANE: mujoco.mjtGeom.mjGEOM_PLANE,
            GeoType.HFIELD: mujoco.mjtGeom.mjGEOM_HFIELD,
            GeoType.CAPSULE: mujoco.mjtGeom.mjGEOM_CAPSULE,
            GeoType.CYLINDER: mujoco.mjtGeom.mjGEOM_CYLINDER,
            GeoType.BOX: mujoco.mjtGeom.mjGEOM_BOX,
            GeoType.ELLIPSOID: mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            GeoType.MESH: mujoco.mjtGeom.mjGEOM_MESH,
            GeoType.CONVEX_MESH: mujoco.mjtGeom.mjGEOM_MESH,
        }

        mj_bodies = [spec.worldbody]
        full_inertia_bodies = []
        # mapping from Newton body id to MuJoCo body id
        body_mapping = {-1: 0}
        # mapping from Newton shape id to MuJoCo geom name
        shape_mapping = {}
        # mapping from Newton shape id (sites) to MuJoCo site name
        site_mapping = {}
        # Store mapping from Newton joint index to MuJoCo joint name
        joint_mapping = {}
        # Store mapping from Newton body index to MuJoCo body name
        body_name_mapping = {}

        # ensure unique names
        body_name_counts = {}
        joint_names = {}

        if separate_worlds:
            # determine which shapes, bodies and joints belong to the first world
            # based on the body world indices: we pick objects from the first world and global shapes
            non_negatives = body_world[body_world >= 0]
            if len(non_negatives) > 0:
                first_world = np.min(non_negatives)
            else:
                first_world = -1
            selected_shapes = np.where((shape_world == first_world) | (shape_world < 0))[0].astype(np.int32)
            selected_bodies = np.where((body_world == first_world) | (body_world < 0))[0].astype(np.int32)
            selected_joints = np.where((joint_world == first_world) | (joint_world < 0))[0].astype(np.int32)
            if eq_constraint_world is None:
                selected_constraints = np.empty(0, dtype=np.int32)
            else:
                selected_constraints = np.where((eq_constraint_world == first_world) | (eq_constraint_world < 0))[
                    0
                ].astype(np.int32)
            selected_mimic_constraints = np.where((mimic_world == first_world) | (mimic_world < 0))[0].astype(np.int32)
        else:
            # if we are not separating environments to worlds, we use all shapes, bodies, joints
            first_world = 0

            # if we are not separating worlds, we use all shapes, bodies, joints, constraints
            selected_shapes = np.arange(model.shape_count, dtype=np.int32)
            selected_bodies = np.arange(model.body_count, dtype=np.int32)
            selected_joints = np.arange(model.joint_count, dtype=np.int32)
            selected_constraints = np.arange(model.mujoco.equality_constraint_count, dtype=np.int32)
            selected_mimic_constraints = np.arange(model.constraint_mimic_count, dtype=np.int32)

        # get the shapes for the first environment
        first_env_shapes = np.where(shape_world == first_world)[0]

        # Classify joints outside articulations as standalone roots or loop closures.
        joints_unassigned = selected_joints[joint_articulation[selected_joints] == -1]
        joints_articulated = selected_joints[joint_articulation[selected_joints] >= 0]

        # Bodies already owned by an articulation must not be created again as standalone bodies.
        articulated_bodies = {int(body) for body in joint_child[joints_articulated]}
        articulated_bodies.update(int(body) for body in joint_parent[joints_articulated] if body >= 0)

        # Imported MJCF equalities also appear as unassigned joints. Keep them as constraints.
        equality_loop_joints = set()
        if eq_constraint_target_kind is not None and eq_constraint_target is not None:
            for i in selected_constraints:
                if (
                    int(eq_constraint_target_kind[i]) == int(MjcEqualityTargetKind.JOINT)
                    and eq_constraint_target[i] >= 0
                ):
                    equality_loop_joints.add(int(eq_constraint_target[i]))
        joints_static_roots = []
        joints_dynamic_roots = []
        standalone_root_bodies = set()
        for joint in joints_unassigned:
            child = int(joint_child[joint])
            # The first eligible world joint creates the standalone body.
            if (
                joint_parent[joint] == -1
                and child not in articulated_bodies
                and child not in standalone_root_bodies
                and int(joint) not in equality_loop_joints
            ):
                if joint_type[joint] == JointType.FIXED:
                    joints_static_roots.append(int(joint))
                else:
                    joints_dynamic_roots.append(int(joint))
                standalone_root_bodies.add(child)

        # Keep fixed and dynamic roots separate because they use different MuJoCo body creation paths.
        joints_static_roots = np.asarray(joints_static_roots, dtype=np.int32)
        joints_dynamic_roots = np.asarray(joints_dynamic_roots, dtype=np.int32)
        standalone_root_set = set(joints_static_roots) | set(joints_dynamic_roots)

        # Once each standalone body has a world root, its other unassigned joints are loop closures.
        joints_loop = np.asarray(
            [joint for joint in joints_unassigned if joint not in standalone_root_set], dtype=np.int32
        )

        # Every selected body needs exactly one creation path before its shapes and state can be converted.
        instantiated_bodies = articulated_bodies | standalone_root_bodies
        missing_bodies = [int(body) for body in selected_bodies if int(body) not in instantiated_bodies]
        if missing_bodies:
            missing_body_set = set(missing_bodies)
            related_joints = [
                model.joint_label[int(joint)]
                for joint in joints_unassigned
                if int(joint_parent[joint]) in missing_body_set or int(joint_child[joint]) in missing_body_set
            ]
            missing_labels = [model.body_label[body] for body in missing_bodies]
            # Keep conversion errors readable for models with many disconnected bodies.
            if len(missing_labels) > 5:
                missing_labels = [*missing_labels[:5], "..."]
            if len(related_joints) > 5:
                related_joints = [*related_joints[:5], "..."]
            raise ValueError(
                "SolverMuJoCo cannot convert bodies that are outside articulations and have no standalone "
                f"joint to world. Bodies: {missing_labels}. Related joints: {related_joints}."
            )

        if standalone_root_set:
            root_labels = [model.joint_label[int(joint)] for joint in sorted(standalone_root_set)]
            displayed_root_labels = root_labels if len(root_labels) <= 5 else [*root_labels[:5], "..."]
            warnings.warn(
                f"SolverMuJoCo is converting {len(root_labels)} joint(s) outside articulations as standalone "
                f"world roots: {displayed_root_labels}. This fallback is specific to SolverMuJoCo.",
                stacklevel=2,
            )

        # sort joints topologically depth-first since this is the order that will also be used
        # for placing bodies in the MuJoCo model
        joints_simple = [(joint_parent[i], joint_child[i]) for i in joints_articulated]
        if len(joints_articulated) > 0:
            joint_order = topological_sort(joints_simple, use_dfs=True, custom_indices=joints_articulated)
        else:
            joint_order = np.empty(0, dtype=np.int32)
        if any(joint_order[i] != joints_articulated[i] for i in range(len(joints_simple))):
            warnings.warn(
                "Joint order is not in depth-first topological order while converting Newton model to MuJoCo, this may lead to diverging kinematics between MuJoCo and Newton.",
                stacklevel=2,
            )

        # Count the total joint coordinates and DOFs that belong to loop joints
        # across all worlds (not added to MuJoCo as joints). When
        # separate_worlds=True, joints_loop is per-template so we multiply by
        # world_count; otherwise it already spans all worlds.
        joint_q_start_np = model.joint_q_start.numpy()
        joint_qd_start_np = model.joint_qd_start.numpy()
        loop_coord_count = 0
        loop_dof_count = 0
        for j in joints_loop:
            loop_coord_count += int(joint_q_start_np[j + 1]) - int(joint_q_start_np[j])
            loop_dof_count += int(joint_qd_start_np[j + 1]) - int(joint_qd_start_np[j])
        if separate_worlds:
            self._total_loop_joint_coords = loop_coord_count * model.world_count
            self._total_loop_joint_dofs = loop_dof_count * model.world_count
        else:
            self._total_loop_joint_coords = loop_coord_count
            self._total_loop_joint_dofs = loop_dof_count

        # find graph coloring of collision filter pairs
        # filter out shapes that are not colliding with anything
        colliding_shapes = selected_shapes[shape_flags[selected_shapes] & ShapeFlags.COLLIDE_SHAPES != 0]

        # number of shapes we are instantiating in MuJoCo (which will be replicated for the number of envs)
        colliding_shapes_per_world = len(colliding_shapes)

        # filter out non-colliding bodies using excludes
        body_filters = self._find_body_collision_filter_pairs(
            model,
            selected_bodies,
            colliding_shapes,
        )

        # Reuse the original masks only when all shapes came from the same
        # add_mjcf() call and the masks already enforce every Newton filter.
        # Otherwise generate new masks from Newton's final allowed shape pairs.
        use_preserved_collision_masks = (
            shape_mjc_contype is not None
            and shape_mjc_conaffinity is not None
            and shape_mjc_collision_mask_domain is not None
            and np.all(shape_mjc_contype[colliding_shapes] != MUJOCO_COLLISION_MASK_UNSET)
            and np.all(shape_mjc_conaffinity[colliding_shapes] != MUJOCO_COLLISION_MASK_UNSET)
            and np.all(shape_mjc_collision_mask_domain[colliding_shapes] != MUJOCO_COLLISION_MASK_DOMAIN_UNSET)
            and np.unique(shape_mjc_collision_mask_domain[colliding_shapes]).shape[0] <= 1
            and self._preserved_masks_cover_collision_filters(
                model,
                colliding_shapes,
                shape_mjc_contype,
                shape_mjc_conaffinity,
                body_filters,
            )
        )
        compiled_collision_type = None
        compiled_collision_affinity = None
        shape_color = None
        if not use_preserved_collision_masks:
            compiled_masks = self._compile_newton_collision_masks(model, colliding_shapes)
            if compiled_masks.exact:
                compiled_collision_type = np.zeros(model.shape_count, dtype=np.uint32)
                compiled_collision_affinity = np.zeros(model.shape_count, dtype=np.uint32)
                compiled_collision_type[colliding_shapes] = compiled_masks.collision_type
                compiled_collision_affinity[colliding_shapes] = compiled_masks.collision_affinity
            else:
                shape_color = self._color_collision_shapes(
                    model,
                    colliding_shapes,
                    visualize_graph=False,
                    shape_labels=model.shape_label,
                )

        selected_shapes_set = set(selected_shapes)
        mujoco_attrs = getattr(model, "mujoco", None)

        mujoco_pair_contact_shapes: set[int] = set()
        pair_count = model.custom_frequency_counts.get("mujoco:pair", 0)
        if mujoco_attrs is not None and pair_count > 0:
            pair_world_attr = getattr(mujoco_attrs, "pair_world", None)
            pair_geom1_attr = getattr(mujoco_attrs, "pair_geom1", None)
            pair_geom2_attr = getattr(mujoco_attrs, "pair_geom2", None)
            if pair_world_attr is not None and pair_geom1_attr is not None and pair_geom2_attr is not None:
                pair_world_np = pair_world_attr.numpy()
                pair_geom1_np = pair_geom1_attr.numpy()
                pair_geom2_np = pair_geom2_attr.numpy()
                pair_count = min(pair_count, len(pair_world_np), len(pair_geom1_np), len(pair_geom2_np))
                for pair_index in range(pair_count):
                    pair_world = int(pair_world_np[pair_index])
                    if pair_world != first_world and pair_world >= 0:
                        continue
                    for pair_shape in (int(pair_geom1_np[pair_index]), int(pair_geom2_np[pair_index])):
                        if pair_shape >= 0 and pair_shape in selected_shapes_set:
                            mujoco_pair_contact_shapes.add(pair_shape)

        # Compute shapes required by spatial tendons (sites, wrapping geoms, sidesites)
        # so they are not skipped when skip_visual_only_geoms=True or include_sites=False.
        # Only collect from template-world tendons to avoid inflating the count with
        # shape indices from other worlds.
        tendon_required_shapes: set[int] = set()
        if mujoco_attrs is not None:
            _wrap_shape = getattr(mujoco_attrs, "tendon_wrap_shape", None)
            _wrap_sidesite = getattr(mujoco_attrs, "tendon_wrap_sidesite", None)
            _wrap_adr = getattr(mujoco_attrs, "tendon_wrap_adr", None)
            _wrap_num = getattr(mujoco_attrs, "tendon_wrap_num", None)
            _tendon_world = getattr(mujoco_attrs, "tendon_world", None)
            if _wrap_shape is not None and _wrap_adr is not None and _wrap_num is not None:
                wrap_shape_np = _wrap_shape.numpy()
                wrap_sidesite_np = _wrap_sidesite.numpy() if _wrap_sidesite is not None else None
                wrap_adr_np = _wrap_adr.numpy()
                wrap_num_np = _wrap_num.numpy()
                tendon_world_np = _tendon_world.numpy() if _tendon_world is not None else None
                for ti in range(len(wrap_adr_np)):
                    tw = int(tendon_world_np[ti]) if tendon_world_np is not None else 0
                    if tw != first_world and tw >= 0:
                        continue
                    start = int(wrap_adr_np[ti])
                    num = int(wrap_num_np[ti])
                    for w in range(start, start + num):
                        if w < len(wrap_shape_np):
                            idx = int(wrap_shape_np[w])
                            if idx >= 0:
                                tendon_required_shapes.add(idx)
                            if wrap_sidesite_np is not None and w < len(wrap_sidesite_np):
                                ss = int(wrap_sidesite_np[w])
                                if ss >= 0:
                                    tendon_required_shapes.add(ss)

        # Collect shapes required by actuators targeting sites so they are not
        # skipped when include_sites=False.  USD actuators may still carry
        # sentinel trnid/trntype at this point (resolved later from
        # actuator_target_label), so check labels too.
        # Restrict everything to the template world so cost is O(per-world
        # size) instead of O(world_count * per-world size), which dominated
        # solver init for large world counts.
        actuator_required_shapes: set[int] = set()
        if mujoco_attrs is not None:
            _act_trntype = getattr(mujoco_attrs, "actuator_trntype", None)
            _act_trnid = getattr(mujoco_attrs, "actuator_trnid", None)
            _act_target_label = getattr(mujoco_attrs, "actuator_target_label", None)
            _act_world = getattr(mujoco_attrs, "actuator_world", None)
            template_site_mask = (shape_flags[selected_shapes] & int(ShapeFlags.SITE)) != 0
            template_site_indices = selected_shapes[template_site_mask]
            site_shape_by_label = {model.shape_label[int(idx)]: int(idx) for idx in template_site_indices}
            if _act_trntype is not None and _act_trnid is not None:
                act_trntype_np = _act_trntype.numpy()
                act_trnid_np = _act_trnid.numpy()
                if _act_world is not None:
                    act_world_np = _act_world.numpy()
                    template_mask = (act_world_np == first_world) | (act_world_np < 0)
                else:
                    template_mask = np.ones(len(act_trntype_np), dtype=bool)
                # Vectorized: every template-world actuator with trntype==SITE
                # contributes its trnid as a required shape.
                site_trntype_mask = template_mask & (act_trntype_np == int(SolverMuJoCo.TrnType.SITE))
                trnid_targets = act_trnid_np[site_trntype_mask, 0]
                actuator_required_shapes.update(trnid_targets[trnid_targets >= 0].tolist())
                refsite_targets = act_trnid_np[site_trntype_mask, 1]
                actuator_required_shapes.update(refsite_targets[refsite_targets >= 0].tolist())
                slidercrank_mask = template_mask & (act_trntype_np == int(SolverMuJoCo.TrnType.SLIDERCRANK))
                trnid_targets = act_trnid_np[slidercrank_mask]
                actuator_required_shapes.update(trnid_targets[trnid_targets >= 0].tolist())
                # Vectorized: USD-deferred actuators reference sites by label.
                # Intersect template-world target labels with the site label
                # dict instead of iterating over every actuator.
                if isinstance(_act_target_label, list) and site_shape_by_label:
                    template_indices = np.flatnonzero(template_mask).tolist()
                    label_count = len(_act_target_label)
                    template_target_labels = {_act_target_label[ai] for ai in template_indices if ai < label_count}
                    for label in template_target_labels & site_shape_by_label.keys():
                        actuator_required_shapes.add(site_shape_by_label[label])

        required_shapes = tendon_required_shapes | actuator_required_shapes | mujoco_pair_contact_shapes
        mesh_export_cache: dict[tuple[int, tuple[float, float, float]], tuple[np.ndarray, np.ndarray, int, bool]] = {}

        def add_geoms(newton_body_id: int):
            body = mj_bodies[body_mapping[newton_body_id]]
            shapes = model.body_shapes.get(newton_body_id)
            if not shapes:
                return
            for shape in shapes:
                if shape not in selected_shapes_set:
                    # skip shapes that are not selected for this world
                    continue
                # Skip visual-only geoms, but don't skip sites or shapes needed by
                # spatial tendons or actuators.
                is_site = shape_flags[shape] & ShapeFlags.SITE
                if skip_visual_only_geoms and not is_site and not (shape_flags[shape] & ShapeFlags.COLLIDE_SHAPES):
                    if shape not in required_shapes:
                        continue
                stype = shape_type[shape]
                name = f"{model.shape_label[shape]}_{shape}"

                if is_site:
                    if not include_sites and shape not in required_shapes:
                        continue

                    # Map unsupported site types to SPHERE
                    # MuJoCo sites only support: SPHERE, CAPSULE, CYLINDER, BOX
                    supported_site_types = {GeoType.SPHERE, GeoType.CAPSULE, GeoType.CYLINDER, GeoType.BOX}
                    site_geom_type = stype if stype in supported_site_types else GeoType.SPHERE

                    tf = wp.transform(*shape_transform[shape])
                    site_params = {
                        "type": geom_type_mapping[site_geom_type],
                        "name": name,
                        "pos": tf.p,
                        "quat": quat_to_mjc(tf.q),
                    }

                    size = shape_size[shape]
                    # Ensure size is valid for the site type
                    if np.any(size > 0.0):
                        nonzero = size[size > 0.0][0]
                        size[size == 0.0] = nonzero
                        site_params["size"] = size
                    else:
                        site_params["size"] = [0.01, 0.01, 0.01]

                    if shape_flags[shape] & ShapeFlags.VISIBLE:
                        site_params["rgba"] = [0.0, 1.0, 0.0, 0.5]
                    else:
                        site_params["rgba"] = [0.0, 1.0, 0.0, 0.0]

                    body.add_site(**site_params)
                    site_mapping[shape] = name
                    continue

                if stype == GeoType.PLANE and newton_body_id != -1:
                    raise ValueError("Planes can only be attached to static bodies")
                geom_params = {
                    "type": geom_type_mapping[stype],
                    "name": name,
                }
                tf = wp.transform(*shape_transform[shape])
                if stype == GeoType.HFIELD:
                    # Retrieve heightfield source
                    hfield_src = model.shape_source[shape]
                    if hfield_src is None:
                        if wp.config.log_level <= wp.LOG_DEBUG:
                            print(f"Warning: Heightfield shape {shape} has no source data, skipping")
                        continue

                    # Convert Newton heightfield to MuJoCo format
                    # MuJoCo size: (size_x, size_y, size_z, size_base) — all must be positive
                    # Our data is normalized [0,1], height range = max_z - min_z
                    # We set size_base to eps (MuJoCo requires positive) and shift the
                    # geom origin by min_z so the lowest point is at the right world Z.
                    eps = 1e-4
                    mj_size_z = max(hfield_src.max_z - hfield_src.min_z, eps)
                    mj_size = (hfield_src.hx, hfield_src.hy, mj_size_z, eps)
                    elevation_data = hfield_src.data.flatten()

                    hfield_name = f"{model.shape_label[shape].replace('/', '_')}_{shape}"
                    spec.add_hfield(
                        name=hfield_name,
                        nrow=hfield_src.nrow,
                        ncol=hfield_src.ncol,
                        size=mj_size,
                        userdata=elevation_data,
                    )

                    geom_params["hfieldname"] = hfield_name

                    # Shift geom origin so data=0 maps to min_z in world space
                    tf = wp.transform(
                        wp.vec3(tf.p[0], tf.p[1], tf.p[2] + hfield_src.min_z),
                        tf.q,
                    )
                elif stype == GeoType.MESH or stype == GeoType.CONVEX_MESH:
                    mesh_src = model.shape_source[shape]
                    size = shape_size[shape]
                    key = _mesh_scale_key(mesh_src, size)
                    mesh_export = mesh_export_cache.get(key)
                    if mesh_export is None:
                        mesh_export = _prepare_mujoco_mesh(mesh_src, size)
                        mesh_export_cache[key] = mesh_export

                    vertices, indices, maxhullvert, is_planar = mesh_export
                    uses_mujoco_contacts = (
                        bool(shape_flags[shape] & ShapeFlags.COLLIDE_SHAPES)
                        and (
                            (
                                use_preserved_collision_masks
                                and bool(int(shape_mjc_contype[shape]) | int(shape_mjc_conaffinity[shape]))
                            )
                            or (not use_preserved_collision_masks and int(shape_collision_group[shape]) != 0)
                        )
                    ) or shape in mujoco_pair_contact_shapes
                    if is_planar and self._use_mujoco_contacts and not disable_contacts and uses_mujoco_contacts:
                        raise ValueError(
                            f"MuJoCo contact generation does not support planar mesh collider "
                            f"{model.shape_label[shape]!r} (shape {shape}). Use use_mujoco_contacts=False so "
                            "Newton's collision pipeline handles this mesh, or replace it with a plane/box/thick mesh."
                        )
                    spec.add_mesh(
                        name=name,
                        uservert=vertices.flatten(),
                        userface=indices.flatten(),
                        maxhullvert=maxhullvert,
                    )
                    geom_params["meshname"] = name
                geom_params["pos"] = tf.p
                geom_params["quat"] = quat_to_mjc(tf.q)
                size = shape_size[shape]
                if np.any(size > 0.0):
                    # duplicate nonzero entries at places where size is 0
                    nonzero = size[size > 0.0][0]
                    size[size == 0.0] = nonzero
                    geom_params["size"] = size
                else:
                    assert stype == GeoType.PLANE, "Only plane shapes are allowed to have a size of zero"
                    # planes are always infinite for collision purposes in mujoco
                    geom_params["size"] = [5.0, 5.0, 5.0]
                    # make ground plane blue in the MuJoCo viewer (only used for debugging)
                    geom_params["rgba"] = [0.0, 0.3, 0.6, 1.0]

                # encode collision filtering information
                if not (shape_flags[shape] & ShapeFlags.COLLIDE_SHAPES):
                    # Visual-only Newton shapes never participate in automatic
                    # MuJoCo contacts, even if source metadata carried masks.
                    geom_params["contype"] = 0
                    geom_params["conaffinity"] = 0
                elif use_preserved_collision_masks:
                    geom_params["contype"] = mujoco_mask_to_signed(int(shape_mjc_contype[shape]))
                    geom_params["conaffinity"] = mujoco_mask_to_signed(int(shape_mjc_conaffinity[shape]))
                elif shape_collision_group[shape] == 0:
                    geom_params["contype"] = 0
                    geom_params["conaffinity"] = 0
                elif compiled_collision_type is not None and compiled_collision_affinity is not None:
                    geom_params["contype"] = mujoco_mask_to_signed(int(compiled_collision_type[shape]))
                    geom_params["conaffinity"] = mujoco_mask_to_signed(int(compiled_collision_affinity[shape]))
                else:
                    color = shape_color[shape]
                    contype, conaffinity = self._collision_color_masks(int(color))
                    geom_params["contype"] = contype
                    geom_params["conaffinity"] = conaffinity

                # set friction from Newton shape materials
                mu = shape_mu[shape]
                torsional = shape_mu_torsional[shape]
                rolling = shape_mu_rolling[shape]
                geom_params["friction"] = [
                    mu,
                    torsional,
                    rolling,
                ]

                # solref per mujoco.solref_mode. See docs/solvers/mujoco.rst
                # > "Shape-material contact stiffness and damping".
                solref_mode_for_shape = (
                    int(shape_mjc_solref_mode[shape]) if shape_mjc_solref_mode is not None else SOLREF_MODE_MJCF_DEFAULT
                )
                if solref_mode_for_shape == SOLREF_MODE_RAW and shape_mjc_solref is not None:
                    raw_solref = shape_mjc_solref[shape]
                    geom_params["solref"] = (float(raw_solref[0]), float(raw_solref[1]))
                else:
                    geom_params["solref"] = convert_solref(float(shape_ke[shape]), float(shape_kd[shape]), 1.0, 1.0)

                if shape_condim is not None:
                    geom_params["condim"] = shape_condim[shape]
                if shape_geom_group is not None:
                    geom_params["group"] = shape_geom_group[shape]
                if shape_priority is not None:
                    geom_params["priority"] = shape_priority[shape]
                if shape_geom_solimp is not None:
                    geom_params["solimp"] = shape_geom_solimp[shape]
                if shape_geom_solmix is not None:
                    geom_params["solmix"] = shape_geom_solmix[shape]
                geom_params["gap"] = float(shape_gap[shape])
                authored_margin = float(shape_margin[shape])
                if self._zero_margins_for_native_ccd:
                    # Zeroed in the spec for NATIVECCD/MULTICCD compatibility (#2106).
                    # Restored at runtime when use_mujoco_contacts=False via the
                    # update_geom_properties_kernel.
                    geom_params["margin"] = 0.0
                    if self._use_mujoco_contacts and authored_margin > 0.0:
                        warnings.warn(
                            f"Geom {name}: authored margin={authored_margin} zeroed for "
                            f"NATIVECCD/MULTICCD compatibility (#2106). "
                            f"To honor this value, switch to Newton's collision pipeline by "
                            f"constructing the solver with use_mujoco_contacts=False and feeding "
                            f"Newton-generated contacts into step().",
                            stacklevel=2,
                        )
                else:
                    geom_params["margin"] = authored_margin

                body.add_geom(**geom_params)
                # store the geom name instead of assuming index
                shape_mapping[shape] = name

        # add static geoms attached to the worldbody
        add_geoms(-1)

        # Maps from Newton joint index (per-world/template) to MuJoCo DOF start index (per-world/template)
        # Only populated for template joints; in kernels, use joint_in_world to index
        joint_mjc_dof_start = np.full(len(selected_joints), -1, dtype=np.int32)
        joint_mjc_qpos_start = np.full(len(selected_joints), -1, dtype=np.int32)

        # Maps from Newton DOF index to MuJoCo joint index (first world only)
        # Needed because jnt_solimp/jnt_solref are per-joint (not per-DOF) in MuJoCo
        dof_to_mjc_joint = np.full(model.joint_dof_count // model.world_count, -1, dtype=np.int32)

        # This is needed for CTRL_DIRECT actuators targeting joints within combined Newton joints.
        mjc_joint_names: list[str] = []

        # Saved ctrl/force ranges. The rebuild drops them, so re-attach after.
        # Key = (dof, is_position): position and velocity sub-actuators can have
        # different ranges.
        joint_target_ranges: dict[tuple[int, bool], dict[str, Any]] = {}
        if mujoco_attrs is not None and hasattr(mujoco_attrs, "actuator_trnid"):
            jt_count = model.custom_frequency_counts.get("mujoco:actuator", 0)
            jt_trnid = get_custom_attribute("actuator_trnid")
            jt_ctrl_source = get_custom_attribute("ctrl_source")
            jt_trntype = get_custom_attribute("actuator_trntype")
            jt_world = get_custom_attribute("actuator_world")
            jt_ctrl_type = get_custom_attribute("ctrl_type")
            jt_has_ctrlrange = get_custom_attribute("actuator_has_ctrlrange")
            jt_ctrlrange = get_custom_attribute("actuator_ctrlrange")
            jt_ctrllimited = get_custom_attribute("actuator_ctrllimited")
            jt_has_forcerange = get_custom_attribute("actuator_has_forcerange")
            jt_forcerange = get_custom_attribute("actuator_forcerange")
            jt_forcelimited = get_custom_attribute("actuator_forcelimited")

            # Which sub-actuator a row feeds (as is_position): position->position only,
            # velocity->velocity only, unknown->both.
            BOTH_KINDS = (True, False)
            kinds_by_ctrl_type = {
                int(SolverMuJoCo.CtrlType.POSITION): (True,),
                int(SolverMuJoCo.CtrlType.VELOCITY): (False,),
            }

            def classify_joint_target_kinds(row: int) -> tuple[bool, ...]:
                if jt_ctrl_type is None:
                    return BOTH_KINDS
                return kinds_by_ctrl_type.get(int(jt_ctrl_type[row]), BOTH_KINDS)

            for row in range(jt_count):
                if jt_ctrl_source is None or int(jt_ctrl_source[row]) != int(SolverMuJoCo.CtrlSource.JOINT_TARGET):
                    continue
                if jt_trntype is not None and int(jt_trntype[row]) != int(SolverMuJoCo.TrnType.JOINT):
                    continue
                if jt_world is not None:
                    w = int(jt_world[row])
                    if w != first_world and w != -1:
                        continue
                if jt_trnid is None:
                    continue
                dof = int(jt_trnid[row, 0])
                if dof < 0:
                    continue
                info = {
                    "has_ctrlrange": bool(jt_has_ctrlrange[row]) if jt_has_ctrlrange is not None else False,
                    "ctrlrange": tuple(jt_ctrlrange[row]) if jt_ctrlrange is not None else None,
                    "ctrllimited": int(jt_ctrllimited[row]) if jt_ctrllimited is not None else None,
                    "has_forcerange": bool(jt_has_forcerange[row]) if jt_has_forcerange is not None else False,
                    "forcerange": tuple(jt_forcerange[row]) if jt_forcerange is not None else None,
                    "forcelimited": int(jt_forcelimited[row]) if jt_forcelimited is not None else None,
                }
                for is_position in classify_joint_target_kinds(row):
                    joint_target_ranges[(dof, is_position)] = info

        def joint_target_actuator_kwargs(base: dict[str, Any], dof: int, is_position: bool) -> dict[str, Any]:
            """Merge the matching row's authored ctrl/force ranges onto a sub-actuator's kwargs."""
            kwargs = dict(base)
            info = joint_target_ranges.get((dof, is_position))
            if info is None:
                return kwargs
            if info["ctrllimited"] is not None:
                kwargs["ctrllimited"] = info["ctrllimited"]
            if info["has_ctrlrange"] and info["ctrlrange"] is not None:
                kwargs["ctrlrange"] = info["ctrlrange"]
            if info["forcelimited"] is not None:
                kwargs["forcelimited"] = info["forcelimited"]
            if info["has_forcerange"] and info["forcerange"] is not None:
                kwargs["forcerange"] = info["forcerange"]
            return kwargs

        # need to keep track of current dof and joint counts to make the indexing above correct
        num_dofs = 0
        num_qpos = 0
        num_mjc_joints = 0

        def add_body_from_joint(j: int, *, mocap: bool | None):
            parent, child = int(joint_parent[j]), int(joint_child[j])
            child_is_kinematic = (int(body_flags[child]) & int(BodyFlags.KINEMATIC)) != 0
            if mocap is None:
                mocap = child_is_kinematic
            if child in body_mapping:
                raise ValueError(f"Body {child} already exists in the mapping")

            body_mapping[child] = len(mj_bodies)

            j_type = int(joint_type[j])
            # Compute body transform for the MjSpec body pos/quat.
            # A free joint body's initial world pose is stored directly in body_q.
            child_xform = wp.transform(*joint_child_xform[j])
            if j_type == JointType.FREE:
                bq = body_q[child]
                tf = wp.transform(bq[:3], bq[3:])
            else:
                tf = wp.transform(*joint_parent_xform[j])
                tf = tf * wp.transform_inverse(child_xform)

            # ensure unique body name
            name = model.body_label[child].replace("/", "_")
            if name not in body_name_counts:
                body_name_counts[name] = 1
            else:
                while name in body_name_counts:
                    body_name_counts[name] += 1
                    name = f"{name}_{body_name_counts[name]}"
            body_name_mapping[child] = name  # store the final de-duplicated name

            inertia = body_inertia[child]
            mass = body_mass[child]
            # MuJoCo requires positive-definite inertia. For zero-mass bodies
            # (sensor frames, reference links), omit mass and inertia entirely
            # and let MuJoCo handle them natively.
            body_kwargs = {"name": name, "pos": tf.p, "quat": quat_to_mjc(tf.q), "mocap": mocap}
            if body_gravcomp is not None and body_gravcomp[child] != 0.0:
                body_kwargs["gravcomp"] = float(body_gravcomp[child])
            if body_sleep_policy is not None and body_sleep_policy[child] != self.SleepPolicy.AUTO:
                body_kwargs["sleep"] = int(body_sleep_policy[child])
            if mass > 0.0:
                body_kwargs["mass"] = mass
                body_ipos = body_com[child, :].copy()
                compile_ipos = body_ipos.copy()
                compile_ipos[0] += 1.0e-3 if compile_ipos[0] >= 0.0 else -1.0e-3
                # A temporary COM offset forces qM storage that remains valid after inertia edits.
                body_kwargs["ipos"] = compile_ipos
                if inertia[0, 1] == 0.0 and inertia[0, 2] == 0.0 and inertia[1, 2] == 0.0:
                    body_kwargs["inertia"] = [inertia[0, 0], inertia[1, 1], inertia[2, 2]]
                else:
                    body_kwargs["fullinertia"] = [
                        inertia[0, 0],
                        inertia[1, 1],
                        inertia[2, 2],
                        inertia[0, 1],
                        inertia[0, 2],
                        inertia[1, 2],
                    ]
                body_kwargs["explicitinertial"] = True
            body = mj_bodies[body_mapping[parent]].add_body(**body_kwargs)
            mj_bodies.append(body)
            if mass > 0.0:
                full_inertia_bodies.append((body_mapping[child], body, body_ipos))
            return body, parent, child, child_is_kinematic, j_type, child_xform

        # Standalone world-fixed bodies are static (or mocap when kinematic)
        # MuJoCo bodies, not loop constraints or synthetic Newton articulations.
        for j in joints_static_roots:
            _, _, child, _, _, _ = add_body_from_joint(int(j), mocap=None)
            add_geoms(child)

        # Add articulation joints and standalone dynamic roots.
        joints_with_bodies = np.concatenate((joint_order, joints_dynamic_roots))
        for j in joints_with_bodies:
            parent = int(joint_parent[j])
            j_type = int(joint_type[j])
            # Articulated fixed roots remain mocap bodies because Newton can
            # update their root transform at runtime. Fully-locked D6 roots
            # (e.g. imported from a generic USD PhysicsJoint) are equivalent;
            # without mocap they would be baked at the template world's pose
            # for every world.
            is_fixed_root = parent == -1 and (
                j_type == JointType.FIXED or (j_type == JointType.D6 and joint_dof_dim[j][0] + joint_dof_dim[j][1] == 0)
            )
            body, parent, child, child_is_kinematic, j_type, child_xform = add_body_from_joint(
                int(j), mocap=is_fixed_root
            )
            joint_pos = child_xform.p
            joint_rot = child_xform.q

            # add joint
            qd_start = joint_qd_start[j]
            name = model.joint_label[j].replace("/", "_")
            if name not in joint_names:
                joint_names[name] = 1
            else:
                while name in joint_names:
                    joint_names[name] += 1
                    name = f"{name}_{joint_names[name]}"

            # Store mapping from Newton joint index to MuJoCo joint name
            joint_mapping[j] = name

            joint_mjc_dof_start[j] = num_dofs
            joint_mjc_qpos_start[j] = num_qpos

            if j_type == JointType.FREE:
                if parent != -1:
                    warnings.warn(
                        f"Free joint '{model.joint_label[j]}' has parent body {parent} instead of the world (-1). "
                        "SolverMuJoCo requires free joints to attach directly to the world; "
                        "MuJoCo will reject this model at compile time.",
                        UserWarning,
                        stacklevel=2,
                    )
                body.add_joint(
                    name=name,
                    type=mujoco.mjtJoint.mjJNT_FREE,
                    damping=0.0,
                    limited=False,
                )
                mjc_joint_names.append(name)
                for i in range(6):
                    dof_to_mjc_joint[qd_start + i] = num_mjc_joints
                num_dofs += 6
                num_qpos += 7
                num_mjc_joints += 1
            elif j_type == JointType.BALL:
                ball_params = {
                    "name": name,
                    "type": mujoco.mjtJoint.mjJNT_BALL,
                    "axis": wp.quat_rotate(joint_rot, wp.vec3(1.0, 0.0, 0.0)),
                    "pos": joint_pos,
                    "limited": False,
                    "armature": KINEMATIC_ARMATURE if child_is_kinematic else joint_armature[qd_start],
                    "frictionloss": joint_friction[qd_start],
                }
                if joint_stiffness is not None:
                    ball_params["stiffness"] = float(joint_stiffness[qd_start])
                if joint_damping is not None:
                    ball_params["damping"] = float(joint_damping[qd_start])
                body.add_joint(**ball_params)
                mjc_joint_names.append(name)
                # For ball joints, all 3 DOFs map to the same MuJoCo joint
                for i in range(3):
                    dof_to_mjc_joint[qd_start + i] = num_mjc_joints
                num_dofs += 3
                num_qpos += 4
                num_mjc_joints += 1
                # Add actuators for the ball joint using per-DOF arrays
                tq_start = int(joint_target_q_start[j])
                for i in range(3):
                    ai = qd_start + i
                    mode = joint_target_mode[ai]

                    if mode != int(JointTargetMode.NONE):
                        kp = joint_target_ke[ai]
                        kd = joint_target_kd[ai]
                        effort_limit = joint_effort_limit[ai]
                        args = {}
                        args.update(actuator_args)
                        args["gear"] = [0.0] * 6
                        args["gear"][i] = 1.0
                        args["forcerange"] = [-effort_limit, effort_limit]

                        template_dof = ai
                        # Ball targets share one base slot per joint. The kernel reads the coord-layout
                        # quaternion or DOF-layout extrinsic-ZYX triple, then emits component i.
                        template_target_q = tq_start
                        template_target_q_axis = i
                        if mode == JointTargetMode.POSITION:
                            args["gainprm"] = [kp, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                            args["biasprm"] = [0, -kp, -kd, 0, 0, 0, 0, 0, 0, 0]
                            # A ball joint's authored range lives on a single source row
                            # (keyed by the joint base DOF); apply it to every axis actuator.
                            spec.add_actuator(target=name, **joint_target_actuator_kwargs(args, qd_start, True))
                            axis_to_actuator[ai, 0] = actuator_count
                            mjc_actuator_ctrl_source_list.append(0)  # JOINT_TARGET
                            mjc_actuator_to_newton_idx_list.append(template_dof)  # positive = position
                            mjc_actuator_to_target_q_idx_list.append(template_target_q)
                            mjc_actuator_to_target_q_axis_idx_list.append(template_target_q_axis)
                            mjc_actuator_to_newton_ball_jnt_list.append(int(j))
                            actuator_count += 1
                        elif mode == JointTargetMode.POSITION_VELOCITY:
                            args["gainprm"] = [kp, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                            args["biasprm"] = [0, -kp, 0, 0, 0, 0, 0, 0, 0, 0]
                            spec.add_actuator(target=name, **joint_target_actuator_kwargs(args, qd_start, True))
                            axis_to_actuator[ai, 0] = actuator_count
                            mjc_actuator_ctrl_source_list.append(0)  # JOINT_TARGET
                            mjc_actuator_to_newton_idx_list.append(template_dof)  # positive = position
                            mjc_actuator_to_target_q_idx_list.append(template_target_q)
                            mjc_actuator_to_target_q_axis_idx_list.append(template_target_q_axis)
                            mjc_actuator_to_newton_ball_jnt_list.append(int(j))
                            actuator_count += 1

                        if mode in (JointTargetMode.VELOCITY, JointTargetMode.POSITION_VELOCITY):
                            args["gainprm"] = [kd, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                            args["biasprm"] = [0, 0, -kd, 0, 0, 0, 0, 0, 0, 0]
                            spec.add_actuator(target=name, **joint_target_actuator_kwargs(args, qd_start, False))
                            axis_to_actuator[ai, 1] = actuator_count
                            mjc_actuator_ctrl_source_list.append(0)  # JOINT_TARGET
                            mjc_actuator_to_newton_idx_list.append(-(template_dof + 2))  # negative = velocity
                            # target_q_idx for ball velocity points at the coord-indexed q_start
                            # of the ball's quaternion in joint_q (for reading the current r).
                            mjc_actuator_to_target_q_idx_list.append(int(joint_q_start[j]))
                            mjc_actuator_to_target_q_axis_idx_list.append(i)
                            mjc_actuator_to_newton_ball_jnt_list.append(int(j))
                            actuator_count += 1
            elif j_type in supported_joint_types:
                lin_axis_count, ang_axis_count = joint_dof_dim[j]
                multi_axis_joint = lin_axis_count + ang_axis_count > 1
                num_dofs += lin_axis_count + ang_axis_count
                num_qpos += lin_axis_count + ang_axis_count

                tq_start = int(joint_target_q_start[j])

                # linear dofs
                for i in range(lin_axis_count):
                    ai = qd_start + i

                    axis = wp.quat_rotate(joint_rot, wp.vec3(*joint_axis[ai]))

                    joint_params = {
                        "armature": KINEMATIC_ARMATURE if child_is_kinematic else joint_armature[qd_start + i],
                        "pos": joint_pos,
                    }
                    # Set friction
                    joint_params["frictionloss"] = joint_friction[ai]
                    # Set margin if available
                    if joint_dof_limit_margin is not None:
                        joint_params["margin"] = joint_dof_limit_margin[ai]
                    if joint_stiffness is not None:
                        joint_params["stiffness"] = float(joint_stiffness[ai])
                    if joint_damping is not None:
                        joint_params["damping"] = float(joint_damping[ai])
                    if joint_actgravcomp is not None:
                        joint_params["actgravcomp"] = bool(joint_actgravcomp[ai])
                    lower, upper = joint_limit_lower[ai], joint_limit_upper[ai]
                    if lower <= -MAXVAL and upper >= MAXVAL:
                        joint_params["limited"] = False
                    else:
                        joint_params["limited"] = True

                    # Keep the range available for runtime limit enablement.
                    joint_params["range"] = (lower, upper)
                    if joint_params["limited"] and joint_has_raw_limit_solref(ai):
                        # RAW solref_limit values are authored MuJoCo data and
                        # must survive the spec → ``MjModel`` → save_to_mjcf
                        # round-trip. ``SOLREF_MODE_FORCE_SPACE`` and
                        # ``SOLREF_MODE_MJCF_DEFAULT`` joints intentionally omit
                        # the spec-side seed so MuJoCo's compile-time default
                        # ``(0.02, 1.0)`` is used until
                        # ``_update_solref_from_invweight0`` rescales
                        # ``jnt_solref`` post-compilation (issue #2009 review
                        # item I6); skipping the seed also keeps ``save_to_mjcf``
                        # idempotent — re-imported FORCE_SPACE joints recover
                        # the same MJCF_DEFAULT state instead of being frozen
                        # into RAW by a derived solreflimit serialisation.
                        joint_params["solref_limit"] = joint_solref_limit[ai]
                    if joint_solimp_limit is not None:
                        joint_params["solimp_limit"] = joint_solimp_limit[ai]
                    if joint_dof_solref is not None:
                        joint_params["solref_friction"] = joint_dof_solref[ai]
                    if joint_dof_solimp is not None:
                        joint_params["solimp_friction"] = joint_dof_solimp[ai]
                    # Use actfrcrange to clamp total actuator force (P+D sum) on this joint
                    effort_limit = joint_effort_limit[ai]
                    joint_params["actfrclimited"] = True
                    joint_params["actfrcrange"] = (-effort_limit, effort_limit)

                    if joint_springref is not None:
                        joint_params["springref"] = joint_springref[ai]
                    if joint_ref is not None:
                        joint_params["ref"] = joint_ref[ai]

                    axname = name
                    if multi_axis_joint:
                        axname += "_lin"
                    if lin_axis_count > 1:
                        axname += str(i)
                    body.add_joint(
                        name=axname,
                        type=mujoco.mjtJoint.mjJNT_SLIDE,
                        axis=axis,
                        **joint_params,
                    )
                    mjc_joint_names.append(axname)
                    # Map this DOF to the current MuJoCo joint index
                    dof_to_mjc_joint[ai] = num_mjc_joints
                    num_mjc_joints += 1

                    mode = joint_target_mode[ai]
                    if mode != int(JointTargetMode.NONE):
                        kp = joint_target_ke[ai]
                        kd = joint_target_kd[ai]

                        template_dof = ai
                        template_target_q = tq_start + i
                        # Scalar slot (PRISMATIC/REVOLUTE/D6 axis) — no quaternion conversion.
                        if mode == JointTargetMode.POSITION:
                            actuator_args["gainprm"] = [kp, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                            actuator_args["biasprm"] = [0, -kp, -kd, 0, 0, 0, 0, 0, 0, 0]
                            spec.add_actuator(target=axname, **joint_target_actuator_kwargs(actuator_args, ai, True))
                            axis_to_actuator[ai, 0] = actuator_count
                            mjc_actuator_ctrl_source_list.append(0)  # JOINT_TARGET
                            mjc_actuator_to_newton_idx_list.append(template_dof)  # positive = position
                            mjc_actuator_to_target_q_idx_list.append(template_target_q)
                            mjc_actuator_to_target_q_axis_idx_list.append(-1)
                            mjc_actuator_to_newton_ball_jnt_list.append(-1)
                            actuator_count += 1
                        elif mode == JointTargetMode.POSITION_VELOCITY:
                            actuator_args["gainprm"] = [kp, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                            actuator_args["biasprm"] = [0, -kp, 0, 0, 0, 0, 0, 0, 0, 0]
                            spec.add_actuator(target=axname, **joint_target_actuator_kwargs(actuator_args, ai, True))
                            axis_to_actuator[ai, 0] = actuator_count
                            mjc_actuator_ctrl_source_list.append(0)  # JOINT_TARGET
                            mjc_actuator_to_newton_idx_list.append(template_dof)  # positive = position
                            mjc_actuator_to_target_q_idx_list.append(template_target_q)
                            mjc_actuator_to_target_q_axis_idx_list.append(-1)
                            mjc_actuator_to_newton_ball_jnt_list.append(-1)
                            actuator_count += 1

                        if mode in (JointTargetMode.VELOCITY, JointTargetMode.POSITION_VELOCITY):
                            actuator_args["gainprm"] = [kd, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                            actuator_args["biasprm"] = [0, 0, -kd, 0, 0, 0, 0, 0, 0, 0]
                            spec.add_actuator(target=axname, **joint_target_actuator_kwargs(actuator_args, ai, False))
                            axis_to_actuator[ai, 1] = actuator_count
                            mjc_actuator_ctrl_source_list.append(0)  # JOINT_TARGET
                            mjc_actuator_to_newton_idx_list.append(-(template_dof + 2))  # negative = velocity
                            mjc_actuator_to_target_q_idx_list.append(-1)
                            mjc_actuator_to_target_q_axis_idx_list.append(-1)
                            mjc_actuator_to_newton_ball_jnt_list.append(-1)
                            actuator_count += 1

                # angular dofs
                for i in range(lin_axis_count, lin_axis_count + ang_axis_count):
                    ai = qd_start + i

                    axis = wp.quat_rotate(joint_rot, wp.vec3(*joint_axis[ai]))

                    joint_params = {
                        "armature": KINEMATIC_ARMATURE if child_is_kinematic else joint_armature[qd_start + i],
                        "pos": joint_pos,
                    }
                    # Set friction
                    joint_params["frictionloss"] = joint_friction[ai]
                    # Set margin if available
                    if joint_dof_limit_margin is not None:
                        joint_params["margin"] = joint_dof_limit_margin[ai]
                    if joint_stiffness is not None:
                        joint_params["stiffness"] = float(joint_stiffness[ai])
                    if joint_damping is not None:
                        joint_params["damping"] = float(joint_damping[ai])
                    if joint_actgravcomp is not None:
                        joint_params["actgravcomp"] = bool(joint_actgravcomp[ai])
                    lower, upper = joint_limit_lower[ai], joint_limit_upper[ai]
                    if lower <= -MAXVAL and upper >= MAXVAL:
                        joint_params["limited"] = False
                    else:
                        joint_params["limited"] = True

                    # Keep the range available for runtime limit enablement.
                    joint_params["range"] = (np.rad2deg(lower), np.rad2deg(upper))
                    if joint_params["limited"] and joint_has_raw_limit_solref(ai):
                        # See the matching block above for the linear-DOF
                        # joint type: only ``SOLREF_MODE_RAW`` joints seed the
                        # spec so ``save_to_mjcf`` does not freeze
                        # ``SOLREF_MODE_FORCE_SPACE`` joints into a re-import
                        # that loses the Newton force-space semantics
                        # (issue #2009 review item I6).
                        joint_params["solref_limit"] = joint_solref_limit[ai]
                    if joint_solimp_limit is not None:
                        joint_params["solimp_limit"] = joint_solimp_limit[ai]
                    if joint_dof_solref is not None:
                        joint_params["solref_friction"] = joint_dof_solref[ai]
                    if joint_dof_solimp is not None:
                        joint_params["solimp_friction"] = joint_dof_solimp[ai]
                    # Use actfrcrange to clamp total actuator force (P+D sum) on this joint
                    effort_limit = joint_effort_limit[ai]
                    joint_params["actfrclimited"] = True
                    joint_params["actfrcrange"] = (-effort_limit, effort_limit)

                    if joint_springref is not None:
                        joint_params["springref"] = np.rad2deg(joint_springref[ai])
                    if joint_ref is not None:
                        joint_params["ref"] = np.rad2deg(joint_ref[ai])

                    axname = name
                    if multi_axis_joint:
                        axname += "_ang"
                    if ang_axis_count > 1:
                        axname += str(i - lin_axis_count)
                    body.add_joint(
                        name=axname,
                        type=mujoco.mjtJoint.mjJNT_HINGE,
                        axis=axis,
                        **joint_params,
                    )
                    mjc_joint_names.append(axname)
                    # Map this DOF to the current MuJoCo joint index
                    dof_to_mjc_joint[ai] = num_mjc_joints
                    num_mjc_joints += 1

                    mode = joint_target_mode[ai]
                    if mode != int(JointTargetMode.NONE):
                        kp = joint_target_ke[ai]
                        kd = joint_target_kd[ai]

                        template_dof = ai
                        template_target_q = tq_start + i
                        # Scalar slot (REVOLUTE/D6 angular axis) — no quaternion conversion.
                        if mode == JointTargetMode.POSITION:
                            actuator_args["gainprm"] = [kp, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                            actuator_args["biasprm"] = [0, -kp, -kd, 0, 0, 0, 0, 0, 0, 0]
                            spec.add_actuator(target=axname, **joint_target_actuator_kwargs(actuator_args, ai, True))
                            axis_to_actuator[ai, 0] = actuator_count
                            mjc_actuator_ctrl_source_list.append(0)  # JOINT_TARGET
                            mjc_actuator_to_newton_idx_list.append(template_dof)  # positive = position
                            mjc_actuator_to_target_q_idx_list.append(template_target_q)
                            mjc_actuator_to_target_q_axis_idx_list.append(-1)
                            mjc_actuator_to_newton_ball_jnt_list.append(-1)
                            actuator_count += 1
                        elif mode == JointTargetMode.POSITION_VELOCITY:
                            actuator_args["gainprm"] = [kp, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                            actuator_args["biasprm"] = [0, -kp, 0, 0, 0, 0, 0, 0, 0, 0]
                            spec.add_actuator(target=axname, **joint_target_actuator_kwargs(actuator_args, ai, True))
                            axis_to_actuator[ai, 0] = actuator_count
                            mjc_actuator_ctrl_source_list.append(0)  # JOINT_TARGET
                            mjc_actuator_to_newton_idx_list.append(template_dof)  # positive = position
                            mjc_actuator_to_target_q_idx_list.append(template_target_q)
                            mjc_actuator_to_target_q_axis_idx_list.append(-1)
                            mjc_actuator_to_newton_ball_jnt_list.append(-1)
                            actuator_count += 1

                        if mode in (JointTargetMode.VELOCITY, JointTargetMode.POSITION_VELOCITY):
                            actuator_args["gainprm"] = [kd, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                            actuator_args["biasprm"] = [0, 0, -kd, 0, 0, 0, 0, 0, 0, 0]
                            spec.add_actuator(target=axname, **joint_target_actuator_kwargs(actuator_args, ai, False))
                            axis_to_actuator[ai, 1] = actuator_count
                            mjc_actuator_ctrl_source_list.append(0)  # JOINT_TARGET
                            mjc_actuator_to_newton_idx_list.append(-(template_dof + 2))  # negative = velocity
                            mjc_actuator_to_target_q_idx_list.append(-1)
                            mjc_actuator_to_target_q_axis_idx_list.append(-1)
                            mjc_actuator_to_newton_ball_jnt_list.append(-1)
                            actuator_count += 1

                        # Note: MuJoCo general actuators are handled separately via custom attributes

            elif j_type != JointType.FIXED:
                raise NotImplementedError(f"Joint type {j_type} is not supported yet")

            add_geoms(child)

        def get_body_name(body_idx: int) -> str:
            """Get body name, handling world body (-1) correctly."""
            if body_idx == -1:
                return "world"
            target_name = body_name_mapping.get(body_idx)
            if target_name is None:
                target_name = model.body_label[body_idx].replace("/", "_")
                if wp.config.log_level <= wp.LOG_DEBUG:
                    print(
                        f"Warning: MuJoCo equality constraint references body {body_idx} "
                        "not present in the MuJoCo export."
                    )
            return target_name

        def get_eq_target_kind(i: int) -> int:
            if eq_constraint_target_kind is None:
                return int(MjcEqualityTargetKind.NONE)
            return int(eq_constraint_target_kind[i])

        def get_eq_target(i: int) -> int:
            if eq_constraint_target is None:
                return -1
            return int(eq_constraint_target[i])

        def get_eq_objtype(i: int, fallback: int) -> int:
            if eq_constraint_objtype is None:
                return fallback
            objtype = int(eq_constraint_objtype[i])
            return fallback if objtype < 0 else objtype

        def add_body_equality(i: int):
            objtype = get_eq_objtype(i, MJC_OBJ_BODY)
            if objtype == MJC_OBJ_BODY:
                return spec.add_equality(objtype=mujoco.mjtObj.mjOBJ_BODY)
            return spec.add_equality()

        mjc_eq_to_newton_eq_dict = {}
        converted_loop_joint_targets = set()
        converted_mimic_targets = set()
        for i in selected_constraints:
            constraint_type = eq_constraint_type[i]
            target_kind = get_eq_target_kind(i)
            target = get_eq_target(i)
            if target_kind == int(MjcEqualityTargetKind.JOINT) and target >= 0:
                converted_loop_joint_targets.add(target)
            elif target_kind == int(MjcEqualityTargetKind.MIMIC) and target >= 0:
                converted_mimic_targets.add(target)

            if constraint_type == _EqType.CONNECT:
                self.has_connect_constraints = True
                eq = add_body_equality(i)
                eq.type = mujoco.mjtEq.mjEQ_CONNECT
                eq.active = eq_constraint_enabled[i]
                eq.name1 = get_body_name(eq_constraint_body1[i])
                eq.name2 = get_body_name(eq_constraint_body2[i])
                eq.data[0:3] = eq_constraint_anchor[i]
                if eq_constraint_solref is not None:
                    eq.solref = eq_constraint_solref[i]
                if eq_constraint_solimp is not None:
                    eq.solimp = eq_constraint_solimp[i]

            elif constraint_type == _EqType.JOINT:
                eq = spec.add_equality()
                eq.type = mujoco.mjtEq.mjEQ_JOINT
                eq.active = eq_constraint_enabled[i]
                j1_idx = int(eq_constraint_joint1[i])
                j2_idx = int(eq_constraint_joint2[i])
                eq.name1 = joint_mapping.get(j1_idx, model.joint_label[j1_idx].replace("/", "_"))
                eq.name2 = joint_mapping.get(j2_idx, model.joint_label[j2_idx].replace("/", "_"))
                eq.data[0:5] = eq_constraint_polycoef[i]
                if eq_constraint_solref is not None:
                    eq.solref = eq_constraint_solref[i]
                if eq_constraint_solimp is not None:
                    eq.solimp = eq_constraint_solimp[i]

            elif constraint_type == _EqType.WELD:
                eq = add_body_equality(i)
                eq.type = mujoco.mjtEq.mjEQ_WELD
                eq.active = eq_constraint_enabled[i]
                eq.name1 = get_body_name(eq_constraint_body1[i])
                eq.name2 = get_body_name(eq_constraint_body2[i])
                cns_relpose = wp.transform(*eq_constraint_relpose[i])
                eq.data[0:3] = eq_constraint_anchor[i]
                eq.data[3:6] = wp.transform_get_translation(cns_relpose)
                eq.data[6:10] = quat_to_mjc(wp.transform_get_rotation(cns_relpose))
                eq.data[10] = eq_constraint_torquescale[i]
                if eq_constraint_solref is not None:
                    eq.solref = eq_constraint_solref[i]
                if eq_constraint_solimp is not None:
                    eq.solimp = eq_constraint_solimp[i]
            else:
                continue

            mjc_eq_to_newton_eq_dict[eq.id] = i

        # add equality constraints for joints that are excluded from the articulation
        # (the UsdPhysics way of defining loop closures)
        mjc_eq_to_newton_jnt = {}
        jnt_eq_anchor1_dict = {}  # mjc_eq_id -> anchor1 as [x, y, z] for CONNECT constraints from joints
        jnt_eq_anchor1_has_axis_offset = {}  # mjc_eq_id -> bool, True for the second hinge CONNECT
        for j in joints_loop:
            if int(j) in converted_loop_joint_targets:
                continue

            j_type = int(joint_type[j])
            parent_name = get_body_name(joint_parent[j])
            child_name = get_body_name(joint_child[j])
            lin_count, ang_count = joint_dof_dim[j]

            if j_type == JointType.FIXED:
                # Fixed loop joint → weld constraint (constrains all 6 DOFs).
                # Set the anchor on body1; leave data[3:10] (relpose) at zero
                # so that spec.compile() auto-computes it from the body positions
                # at compile time.  Manual relpose computation is fragile because
                # the joint xforms define anchor offsets in body-local frames
                # while the WELD relpose is measured in body2's local frame —
                # these differ whenever the anchor is not at the body origin.
                # Note: WELD entries are intentionally NOT added to
                # mjc_eq_to_newton_jnt so that CONNECT-specific kernels skip
                # them and do not overwrite the WELD relpose data.
                eq = spec.add_equality(objtype=mujoco.mjtObj.mjOBJ_BODY)
                eq.type = mujoco.mjtEq.mjEQ_WELD
                eq.active = True
                eq.name1 = parent_name
                eq.name2 = child_name
                eq.data[0:3] = joint_parent_xform[j][:3]
            elif lin_count == 0 and ang_count == 1:
                # Single-hinge loop joint (revolute): 2x CONNECT at non-coincident
                # points along the hinge axis constrains 5 DOFs (3 trans + 2 rot),
                # leaving exactly 1 rotational DOF around the axis.
                parent_anchor = joint_parent_xform[j][:3]
                parent_xform_tf = wp.transform(*joint_parent_xform[j])
                qd_start = joint_qd_start[j]
                hinge_axis_local = wp.vec3(*joint_axis[qd_start])
                # Rotate axis into the parent body frame (anchor data[0:3] is
                # in the parent body frame, so the offset must be too).
                hinge_axis = wp.quat_rotate(parent_xform_tf.q, hinge_axis_local)
                offset = HINGE_CONNECT_AXIS_OFFSET

                self.has_jnt_connect_constraints = True

                # First CONNECT at the joint anchor
                eq1 = spec.add_equality(objtype=mujoco.mjtObj.mjOBJ_BODY)
                eq1.type = mujoco.mjtEq.mjEQ_CONNECT
                eq1.active = True
                eq1.name1 = parent_name
                eq1.name2 = child_name
                eq1.data[0:3] = parent_anchor
                mjc_eq_to_newton_jnt[eq1.id] = j
                jnt_eq_anchor1_dict[eq1.id] = list(parent_anchor)
                jnt_eq_anchor1_has_axis_offset[eq1.id] = False

                # Second CONNECT offset along the hinge axis
                parent_anchor_offset = np.array(parent_anchor) + offset * np.array(hinge_axis)
                eq2 = spec.add_equality(objtype=mujoco.mjtObj.mjOBJ_BODY)
                eq2.type = mujoco.mjtEq.mjEQ_CONNECT
                eq2.active = True
                eq2.name1 = parent_name
                eq2.name2 = child_name
                eq2.data[0:3] = parent_anchor_offset
                mjc_eq_to_newton_jnt[eq2.id] = j
                jnt_eq_anchor1_dict[eq2.id] = list(parent_anchor_offset)
                jnt_eq_anchor1_has_axis_offset[eq2.id] = True

            elif lin_count == 0 and ang_count == 3:
                # Ball loop joint: 1x CONNECT constrains 3 translational
                # DOFs, leaving all 3 rotational DOFs free.
                eq = spec.add_equality(objtype=mujoco.mjtObj.mjOBJ_BODY)
                eq.type = mujoco.mjtEq.mjEQ_CONNECT
                eq.active = True
                eq.name1 = parent_name
                eq.name2 = child_name
                eq.data[0:3] = joint_parent_xform[j][:3]
                mjc_eq_to_newton_jnt[eq.id] = j
                jnt_eq_anchor1_dict[eq.id] = list(joint_parent_xform[j][:3])
                jnt_eq_anchor1_has_axis_offset[eq.id] = False
                self.has_jnt_connect_constraints = True
            else:
                warnings.warn(
                    f"Loop joint {j} (type {JointType(j_type).name}, "
                    f"{lin_count} linear + {ang_count} angular DOFs) "
                    f"has no supported MuJoCo equality constraint mapping. "
                    f"Skipping loop closure for this joint.",
                    stacklevel=2,
                )
                continue

        # add mimic constraints as mjEQ_JOINT equality constraints
        mjc_eq_to_newton_mimic_dict = {}
        for i in selected_mimic_constraints:
            if int(i) in converted_mimic_targets:
                continue

            j0 = mimic_joint0[i]  # follower
            j1 = mimic_joint1[i]  # leader

            # check that both joints exist in the MuJoCo joint mapping
            j0_name = joint_mapping.get(j0)
            j1_name = joint_mapping.get(j1)
            if j0_name is None or j1_name is None:
                warnings.warn(
                    f"Skipping mimic constraint {i}: follower joint {j0} or leader joint {j1} "
                    f"not found in MuJoCo joint mapping.",
                    stacklevel=2,
                )
                continue

            # mjEQ_JOINT only works with scalar joints (hinge/slide)
            j0_type = joint_type[j0]
            j1_type = joint_type[j1]
            if j0_type not in (JointType.REVOLUTE, JointType.PRISMATIC):
                warnings.warn(
                    f"Skipping mimic constraint {i}: follower joint {j0} has unsupported type "
                    f"{JointType(j0_type).name} for mjEQ_JOINT (only REVOLUTE and PRISMATIC supported).",
                    stacklevel=2,
                )
                continue
            if j1_type not in (JointType.REVOLUTE, JointType.PRISMATIC):
                warnings.warn(
                    f"Skipping mimic constraint {i}: leader joint {j1} has unsupported type "
                    f"{JointType(j1_type).name} for mjEQ_JOINT (only REVOLUTE and PRISMATIC supported).",
                    stacklevel=2,
                )
                continue

            eq = spec.add_equality()
            eq.type = mujoco.mjtEq.mjEQ_JOINT
            eq.active = bool(mimic_enabled[i])
            eq.name1 = j0_name  # follower (constrained joint)
            eq.name2 = j1_name  # leader (driving joint)
            mjc_eq_to_newton_mimic_dict[eq.id] = i
            # polycoef: data[0] + data[1]*q2 + data[2]*q2^2 + ... - q1 = 0
            # mimic: q1 = coef0 + coef1*q2
            eq.data[0] = float(mimic_coef0[i])
            eq.data[1] = float(mimic_coef1[i])
            eq.data[2] = 0.0
            eq.data[3] = 0.0
            eq.data[4] = 0.0

        # Count non-colliding geoms that were kept because they are required by spatial tendons
        tendon_extra_geoms = sum(
            1
            for s in tendon_required_shapes
            if s in selected_shapes_set
            and not (shape_flags[s] & ShapeFlags.SITE)
            and not (shape_flags[s] & ShapeFlags.COLLIDE_SHAPES)
        )
        if skip_visual_only_geoms and len(spec.geoms) != colliding_shapes_per_world + tendon_extra_geoms:
            raise ValueError(
                "The number of geoms in the MuJoCo model does not match the number of colliding shapes in the Newton model."
            )

        if len(spec.bodies) != len(selected_bodies) + 1:  # +1 for the world body
            raise ValueError(
                "The number of bodies in the MuJoCo model does not match the number of selected bodies in the Newton model. "
                "Make sure each body belongs to an articulation or has a standalone joint to world."
            )

        # add contact exclusions between bodies to ensure parent <> child collisions are ignored
        # even when one of the bodies is static
        for b1, b2 in body_filters:
            mb1, mb2 = body_mapping[b1], body_mapping[b2]
            spec.add_exclude(bodyname1=spec.bodies[mb1].name, bodyname2=spec.bodies[mb2].name)

        # add explicit contact pairs from custom attributes
        self._init_pairs(model, spec, shape_mapping, first_world)

        selected_tendons, mjc_tendon_names = self._init_tendons(
            model, spec, joint_mapping, shape_mapping, site_mapping, first_world
        )

        # Process MuJoCo general actuators (motor, general, etc.) from custom attributes
        actuator_count += self._init_actuators(
            model,
            spec,
            first_world,
            actuator_args,
            mjc_actuator_ctrl_source_list,
            mjc_actuator_to_newton_idx_list,
            mjc_actuator_to_target_q_idx_list,
            mjc_actuator_to_target_q_axis_idx_list,
            mjc_actuator_to_newton_ball_jnt_list,
            dof_to_mjc_joint,
            mjc_joint_names,
            selected_tendons,
            mjc_tendon_names,
            body_name_mapping,
            site_mapping,
        )

        # Convert actuator mapping lists to warp arrays
        if mjc_actuator_ctrl_source_list:
            self.mjc_actuator_ctrl_source = wp.array(
                np.array(mjc_actuator_ctrl_source_list, dtype=np.int32),
                dtype=wp.int32,
                device=model.device,
            )
            self.mjc_actuator_to_newton_idx = wp.array(
                np.array(mjc_actuator_to_newton_idx_list, dtype=np.int32),
                dtype=wp.int32,
                device=model.device,
            )
            self.mjc_actuator_to_newton_target_q_idx = wp.array(
                np.array(mjc_actuator_to_target_q_idx_list, dtype=np.int32),
                dtype=wp.int32,
                device=model.device,
            )
            self.mjc_actuator_to_target_q_axis_idx = wp.array(
                np.array(mjc_actuator_to_target_q_axis_idx_list, dtype=np.int32),
                dtype=wp.int32,
                device=model.device,
            )
            self.mjc_actuator_to_newton_ball_jnt = wp.array(
                np.array(mjc_actuator_to_newton_ball_jnt_list, dtype=np.int32),
                dtype=wp.int32,
                device=model.device,
            )
        else:
            self.mjc_actuator_ctrl_source = None
            self.mjc_actuator_to_newton_idx = None
            self.mjc_actuator_to_newton_target_q_idx = None
            self.mjc_actuator_to_target_q_axis_idx = None
            self.mjc_actuator_to_newton_ball_jnt = None

        dampratio_actuators = [
            (actuator.id, actuator.biasprm[2])
            for actuator in spec.actuators
            if actuator.biastype == mujoco.mjtBias.mjBIAS_AFFINE
            and actuator.gaintype == mujoco.mjtGain.mjGAIN_FIXED
            and actuator.gainprm[0] > 0.0
            and actuator.biasprm[0] == 0.0
            and abs(actuator.biasprm[1] + actuator.gainprm[0]) < 1e-8
            and actuator.biasprm[2] > 0.0
        ]

        self._add_mesh_variant_assets(spec, model, shape_mapping)
        self.mj_model = spec.compile()
        # Keep the compiled qM layout, but restore the physical COM and derived constants.
        for body_id, body, body_ipos in full_inertia_bodies:
            body.ipos = body_ipos
            self.mj_model.body_ipos[body_id] = body_ipos
            self.mj_model.body_sameframe[body_id] = mujoco.mjtSameFrame.mjSAMEFRAME_NONE
        # mj_setConst only recomputes dampratio actuators from positive placeholders.
        for actuator_id, dampratio in dampratio_actuators:
            self.mj_model.actuator_biasprm[actuator_id, 2] = dampratio
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_setConst(self.mj_model, self.mj_data)

        # Build MuJoCo qpos/qvel start index arrays for coordinate conversion kernels.
        # These map Newton template joint index → MuJoCo qpos/qvel start.
        # Loop joints get -1 (they have no MuJoCo qpos/qvel slots).
        # Must be created before _update_mjc_data which uses them.
        n_template_joints = len(selected_joints)
        mj_q_start_np = np.full(n_template_joints, -1, dtype=np.int32)
        mj_qd_start_np = np.full(n_template_joints, -1, dtype=np.int32)
        for j_template in range(n_template_joints):
            j_idx = selected_joints[j_template]
            mj_q_start_np[j_template] = joint_mjc_qpos_start[j_idx]
            mj_qd_start_np[j_template] = joint_mjc_dof_start[j_idx]
        # Validate that all non-loop joints got valid MuJoCo start indices
        for j_template in range(n_template_joints):
            j_idx = selected_joints[j_template]
            if joint_articulation[j_idx] >= 0:
                assert mj_q_start_np[j_template] >= 0, (
                    f"Non-loop joint {j_idx} (template {j_template}) has no MuJoCo qpos mapping"
                )
                assert mj_qd_start_np[j_template] >= 0, (
                    f"Non-loop joint {j_idx} (template {j_template}) has no MuJoCo DOF mapping"
                )
        self.mj_q_start = wp.array(mj_q_start_np, dtype=wp.int32, device=model.device)
        self.mj_qd_start = wp.array(mj_qd_start_np, dtype=wp.int32, device=model.device)
        if self.enable_sleeping:
            qpos_treeid_np = np.full(self.mj_model.nq, -1, dtype=np.int32)
            for jointid in range(self.mj_model.njnt):
                qpos_start = int(self.mj_model.jnt_qposadr[jointid])
                qpos_end = (
                    int(self.mj_model.jnt_qposadr[jointid + 1])
                    if jointid + 1 < self.mj_model.njnt
                    else self.mj_model.nq
                )
                bodyid = int(self.mj_model.jnt_bodyid[jointid])
                qpos_treeid_np[qpos_start:qpos_end] = int(self.mj_model.body_treeid[bodyid])
            self._sleep_qpos_treeid = wp.array(qpos_treeid_np, dtype=wp.int32, device=model.device)

        self._update_mjc_data(self.mj_data, model, state)

        # fill some MjWarp model fields that are outdated after _update_mjc_data.
        # just setting qpos0 to d.qpos leads to weird behavior here, needs
        # to be investigated.

        mujoco.mj_forward(self.mj_model, self.mj_data)

        # now that the model is compiled, get the actual geom indices and compute
        # shape transform corrections
        shape_to_geom_idx = {}
        geom_to_shape_idx = {}
        for shape, geom_name in shape_mapping.items():
            geom_idx = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if geom_idx >= 0:
                shape_to_geom_idx[shape] = geom_idx
                geom_to_shape_idx[geom_idx] = shape

        site_to_shape_idx = {}
        for shape, site_name in site_mapping.items():
            site_idx = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if site_idx >= 0:
                site_to_shape_idx[site_idx] = shape

        with wp.ScopedDevice(model.device):
            # create the MuJoCo Warp model
            sleep_policies = self.mj_model.tree_sleep_policy.copy()
            policy_never = self.mj_model.tree_sleep_policy == mujoco.mjtSleepPolicy.mjSLEEP_NEVER
            policy_allowed = self.mj_model.tree_sleep_policy == mujoco.mjtSleepPolicy.mjSLEEP_ALLOWED
            policy_init = self.mj_model.tree_sleep_policy == mujoco.mjtSleepPolicy.mjSLEEP_INIT
            self.mj_model.tree_sleep_policy[policy_never] = mujoco.mjtSleepPolicy.mjSLEEP_AUTO_NEVER
            self.mj_model.tree_sleep_policy[policy_allowed | policy_init] = mujoco.mjtSleepPolicy.mjSLEEP_AUTO_ALLOWED
            try:
                self.mjw_model = mujoco_warp.put_model(self.mj_model)
            finally:
                # MuJoCo Warp consumes only the compiled runtime policy. Keep
                # the authoring policy on the CPU model for inspection and MJCF export.
                self.mj_model.tree_sleep_policy[:] = sleep_policies

            # patch mjw_model with mesh_pos if it doesn't have it
            if not hasattr(self.mjw_model, "mesh_pos"):
                self.mjw_model.mesh_pos = wp.array(self.mj_model.mesh_pos, dtype=wp.vec3)

            # Determine nworld for mapping dimensions
            nworld = model.world_count if separate_worlds else 1

            # --- Create unified mappings: MuJoCo[world, entity] -> Newton[entity] ---

            # Build geom -> shape mapping
            # geom_to_shape_idx maps from MuJoCo geom index to absolute Newton shape index.
            # Convert non-static shapes to template-relative indices for the kernel.
            geom_to_shape_idx_np = np.full((self.mj_model.ngeom,), -1, dtype=np.int32)

            # Find the minimum shape index for the first non-static group to use as the base
            first_env_shape_base = int(np.min(first_env_shapes)) if len(first_env_shapes) > 0 else 0

            # Store for lazy inverse creation
            self._shapes_per_world = len(first_env_shapes)
            self._first_env_shape_base = first_env_shape_base

            # Per-geom static mask (True if static, False otherwise)
            geom_is_static_np = np.zeros((self.mj_model.ngeom,), dtype=bool)

            for geom_idx, abs_shape_idx in geom_to_shape_idx.items():
                if shape_world[abs_shape_idx] < 0:
                    # Static shape - use absolute index and mark mask
                    geom_to_shape_idx_np[geom_idx] = abs_shape_idx
                    geom_is_static_np[geom_idx] = True
                else:
                    # Non-static shape - convert to template-relative offset from first env base
                    geom_to_shape_idx_np[geom_idx] = abs_shape_idx - first_env_shape_base

            geom_to_shape_idx_wp = wp.array(geom_to_shape_idx_np, dtype=wp.int32)
            geom_is_static_wp = wp.array(geom_is_static_np, dtype=bool)

            # Create mjc_geom_to_newton_shape: MuJoCo[world, geom] -> Newton shape
            self.mjc_geom_to_newton_shape = wp.full((nworld, self.mj_model.ngeom), -1, dtype=wp.int32)

            if self.mjw_model.geom_pos.size:
                wp.launch(
                    update_shape_mappings_kernel,
                    dim=(nworld, self.mj_model.ngeom),
                    inputs=[
                        geom_to_shape_idx_wp,
                        geom_is_static_wp,
                        self._shapes_per_world,
                        first_env_shape_base,
                    ],
                    outputs=[
                        self.mjc_geom_to_newton_shape,
                    ],
                    device=model.device,
                )

            site_to_shape_idx_np = np.full((self.mj_model.nsite,), -1, dtype=np.int32)
            site_is_global_np = np.zeros((self.mj_model.nsite,), dtype=bool)
            for site_idx, abs_shape_idx in site_to_shape_idx.items():
                if shape_world[abs_shape_idx] < 0:
                    site_to_shape_idx_np[site_idx] = abs_shape_idx
                    site_is_global_np[site_idx] = True
                else:
                    site_to_shape_idx_np[site_idx] = abs_shape_idx - first_env_shape_base

            self._mjc_site_shape_index = wp.array(site_to_shape_idx_np, dtype=wp.int32)
            self._mjc_site_is_global = wp.array(site_is_global_np, dtype=bool)

            # Create mjc_body_to_newton: MuJoCo[world, body] -> Newton body
            # body_mapping is {newton_body_id: mjc_body_id}, we need to invert it
            # and expand to 2D for all worlds
            nbody = self.mj_model.nbody
            bodies_per_world = model.body_count // model.world_count
            mjc_body_to_newton_np = np.full((nworld, nbody), -1, dtype=np.int32)
            for newton_body, mjc_body in body_mapping.items():
                if newton_body >= 0:  # Skip world body (-1 -> 0)
                    newton_body_in_world = newton_body % bodies_per_world
                    for w in range(nworld):
                        mjc_body_to_newton_np[w, mjc_body] = w * bodies_per_world + newton_body_in_world
            self.mjc_body_to_newton = wp.array(mjc_body_to_newton_np, dtype=wp.int32)

            body_flags_np = model.body_flags.numpy()
            body_sleep_index_np = np.full((model.body_count, 2), (-1, -1), dtype=np.int32)
            kinematic = (body_flags_np & int(BodyFlags.KINEMATIC)) != 0
            body_sleep_index_np[kinematic, 1] = -2
            for worldid in range(nworld):
                for bodyid in range(nbody):
                    newton_body = mjc_body_to_newton_np[worldid, bodyid]
                    treeid = self.mj_model.body_treeid[bodyid]
                    if newton_body >= 0 and treeid >= 0 and not kinematic[newton_body]:
                        body_sleep_index_np[newton_body] = (worldid, treeid)
            shape_body_np = model.shape_body.numpy()
            shape_sleep_index_np = np.full((model.shape_count, 2), (-1, -1), dtype=np.int32)
            body_shapes = shape_body_np >= 0
            shape_sleep_index_np[body_shapes] = body_sleep_index_np[shape_body_np[body_shapes]]
            self._collision_shape_sleep_index = wp.array(shape_sleep_index_np, dtype=wp.vec2i, device=model.device)
            self._body_sleep_index = wp.array(body_sleep_index_np, dtype=wp.vec2i, device=model.device)
            self._body_sleep_override = wp.full((nworld, self.mj_model.ntree), -1, dtype=wp.int32, device=model.device)

            # Common variables for mapping creation
            njnt = self.mj_model.njnt
            joints_per_world = model.joint_count // model.world_count
            dofs_per_world = model.joint_dof_count // model.world_count

            # Map each Newton body to the qd_start of its free/DISTANCE joint (or -1).
            # Use selected_joints as the template and tile offsets across worlds.
            joint_type_np = model.joint_type.numpy()
            joint_child_np = model.joint_child.numpy()
            joint_qd_start_np = model.joint_qd_start.numpy()
            joint_dof_dim_np = model.joint_dof_dim.numpy()

            # Map each Newton DOF to the child body of its parent joint.
            # This is used to apply kinematic body flags to MuJoCo dof_armature.
            newton_dof_to_body_np = np.full(model.joint_dof_count, -1, dtype=np.int32)
            for joint_idx in range(model.joint_count):
                dof_start = int(joint_qd_start_np[joint_idx])
                dof_count = int(joint_dof_dim_np[joint_idx, 0] + joint_dof_dim_np[joint_idx, 1])
                if dof_count > 0:
                    newton_dof_to_body_np[dof_start : dof_start + dof_count] = int(joint_child_np[joint_idx])
            self.newton_dof_to_body = wp.array(newton_dof_to_body_np, dtype=wp.int32)

            template_joint_types = joint_type_np[selected_joints]
            free_mask = np.isin(template_joint_types, (JointType.FREE, JointType.DISTANCE))
            body_free_qd_start_np = np.full(model.body_count, -1, dtype=np.int32)
            if np.any(free_mask):
                template_children = joint_child_np[selected_joints] % bodies_per_world
                template_qd_start = joint_qd_start_np[selected_joints] % dofs_per_world
                child_free = template_children[free_mask]
                qd_start_free = template_qd_start[free_mask]
                world_body_offsets = (np.arange(model.world_count, dtype=np.int32) * bodies_per_world)[:, None]
                world_qd_offsets = (np.arange(model.world_count, dtype=np.int32) * dofs_per_world)[:, None]
                body_indices = (child_free[None, :] + world_body_offsets).ravel()
                qd_starts = (qd_start_free[None, :] + world_qd_offsets).ravel()
                body_free_qd_start_np[body_indices] = qd_starts

            self.body_free_qd_start = wp.array(body_free_qd_start_np, dtype=wp.int32)

            # Create mjc_mocap_to_newton_jnt: MuJoCo[world, mocap] -> Newton joint index.
            # These mocap bodies are Newton roots attached to world by a
            # FIXED or fully-locked D6 joint. Static world shapes are not
            # represented here.
            nmocap = self.mj_model.nmocap
            if nmocap > 0:
                mjc_mocap_to_newton_jnt_np = np.full((nworld, nmocap), -1, dtype=np.int32)
                body_mocapid = self.mj_model.body_mocapid
                for mjc_body in range(nbody):
                    mocap_idx = body_mocapid[mjc_body]
                    if mocap_idx < 0:
                        continue
                    newton_body = mjc_body_to_newton_np[0, mjc_body]
                    if newton_body < 0:
                        continue
                    newton_body_template = newton_body % bodies_per_world
                    for j in range(joints_per_world):
                        if joint_child_np[j] == newton_body_template:
                            for w in range(nworld):
                                mjc_mocap_to_newton_jnt_np[w, mocap_idx] = w * joints_per_world + j
                            break
                self.mjc_mocap_to_newton_jnt = wp.array(mjc_mocap_to_newton_jnt_np, dtype=wp.int32)
            else:
                self.mjc_mocap_to_newton_jnt = None

            # Create mjc_jnt_to_newton_jnt: MuJoCo[world, joint] -> Newton joint index
            # selected_joints[idx] is the Newton template joint index
            mjc_jnt_to_newton_jnt_np = np.full((nworld, njnt), -1, dtype=np.int32)
            # Invert dof_to_mjc_joint to get mjc_jnt -> template_dof, then find the joint
            for template_dof, mjc_jnt in enumerate(dof_to_mjc_joint):
                if mjc_jnt >= 0:
                    # Find which Newton template joint contains this DOF
                    # This is the first DOF of the joint, so we can search for it
                    for _ji, j in enumerate(selected_joints):
                        j_dof_start = joint_qd_start[j] % dofs_per_world
                        j_lin_count, j_ang_count = joint_dof_dim[j]
                        j_dof_end = j_dof_start + j_lin_count + j_ang_count
                        if j_dof_start <= template_dof < j_dof_end:
                            for w in range(nworld):
                                mjc_jnt_to_newton_jnt_np[w, mjc_jnt] = w * joints_per_world + j
                            break
            self.mjc_jnt_to_newton_jnt = wp.array(mjc_jnt_to_newton_jnt_np, dtype=wp.int32)

            # Create mjc_jnt_to_newton_dof: MuJoCo[world, joint] -> Newton DOF start
            # joint_mjc_dof_start[template_joint] -> mjc_dof_start
            # dof_to_mjc_joint[template_dof] -> mjc_joint
            mjc_jnt_to_newton_dof_np = np.full((nworld, njnt), -1, dtype=np.int32)
            for template_dof, mjc_jnt in enumerate(dof_to_mjc_joint):
                if mjc_jnt >= 0:
                    for w in range(nworld):
                        mjc_jnt_to_newton_dof_np[w, mjc_jnt] = w * dofs_per_world + template_dof
            self.mjc_jnt_to_newton_dof = wp.array(mjc_jnt_to_newton_dof_np, dtype=wp.int32)

            # Create mjc_dof_to_newton_dof: MuJoCo[world, dof] -> Newton DOF
            nv = self.mj_model.nv  # Number of DOFs in MuJoCo
            mjc_dof_to_newton_dof_np = np.full((nworld, nv), -1, dtype=np.int32)
            # joint_mjc_dof_start tells us where each Newton template joint's DOFs start in MuJoCo
            for j, mjc_dof_start in enumerate(joint_mjc_dof_start):
                if mjc_dof_start >= 0:
                    newton_dof_start = joint_qd_start[j]
                    lin_count, ang_count = joint_dof_dim[j]
                    total_dofs = lin_count + ang_count
                    for d in range(total_dofs):
                        mjc_dof = mjc_dof_start + d
                        template_newton_dof = (newton_dof_start % dofs_per_world) + d
                        for w in range(nworld):
                            mjc_dof_to_newton_dof_np[w, mjc_dof] = w * dofs_per_world + template_newton_dof
            self.mjc_dof_to_newton_dof = wp.array(mjc_dof_to_newton_dof_np, dtype=wp.int32)

            # Create mjc_eq_to_newton_eq: MuJoCo[world, eq] -> Newton equality constraint
            # selected_constraints[idx] is the Newton template constraint index
            neq = self.mj_model.neq
            eq_constraints_per_world = model.mujoco.equality_constraint_count // model.world_count
            mjc_eq_to_newton_eq_np = np.full((nworld, neq), -1, dtype=np.int32)
            mjc_eq_to_newton_jnt_np = np.full((nworld, neq), -1, dtype=np.int32)
            for mjc_eq, newton_eq in mjc_eq_to_newton_eq_dict.items():
                template_eq = newton_eq % eq_constraints_per_world if eq_constraints_per_world > 0 else newton_eq
                for w in range(nworld):
                    mjc_eq_to_newton_eq_np[w, mjc_eq] = w * eq_constraints_per_world + template_eq
            for mjc_eq, newton_jnt in mjc_eq_to_newton_jnt.items():
                template_jnt = newton_jnt % joints_per_world if joints_per_world > 0 else newton_jnt
                for w in range(nworld):
                    mjc_eq_to_newton_jnt_np[w, mjc_eq] = w * joints_per_world + template_jnt
            self.mjc_eq_to_newton_eq = wp.array(mjc_eq_to_newton_eq_np, dtype=wp.int32)
            self.mjc_eq_to_newton_jnt = wp.array(mjc_eq_to_newton_jnt_np, dtype=wp.int32)

            # Build jnt_eq_anchor1 and jnt_eq_anchor1_has_axis_offset per [world, eq]
            # for joint-synthesized CONNECT constraints.
            jnt_eq_anchor1_np = np.zeros((nworld, neq, 3), dtype=np.float32)
            jnt_eq_anchor1_has_axis_offset_np = np.zeros((nworld, neq), dtype=np.int32)
            for mjc_eq_id, anchor in jnt_eq_anchor1_dict.items():
                has_offset = jnt_eq_anchor1_has_axis_offset.get(mjc_eq_id, False)
                for w in range(nworld):
                    jnt_eq_anchor1_np[w, mjc_eq_id, 0] = anchor[0]
                    jnt_eq_anchor1_np[w, mjc_eq_id, 1] = anchor[1]
                    jnt_eq_anchor1_np[w, mjc_eq_id, 2] = anchor[2]
                    jnt_eq_anchor1_has_axis_offset_np[w, mjc_eq_id] = int(has_offset)
            self.jnt_eq_anchor1 = wp.array(jnt_eq_anchor1_np, dtype=wp.vec3)
            self.jnt_eq_anchor1_has_axis_offset = wp.array(jnt_eq_anchor1_has_axis_offset_np, dtype=wp.int32)

            # Ensure no eq is claimed by both the regular and joint-connect paths.
            assert not np.any((mjc_eq_to_newton_eq_np >= 0) & (mjc_eq_to_newton_jnt_np >= 0)), (
                "mjc_eq_to_newton_eq and mjc_eq_to_newton_jnt overlap -- both kernels would write to the same eq_data slot"
            )

            # Create mjc_eq_to_newton_mimic: MuJoCo[world, eq] -> Newton mimic constraint
            mimic_per_world = (
                model.constraint_mimic_count // model.world_count
                if model.world_count > 0
                else model.constraint_mimic_count
            )
            mjc_eq_to_newton_mimic_np = np.full((nworld, neq), -1, dtype=np.int32)
            for mjc_eq, newton_mimic in mjc_eq_to_newton_mimic_dict.items():
                template_mimic = newton_mimic % mimic_per_world if mimic_per_world > 0 else newton_mimic
                for w in range(nworld):
                    mjc_eq_to_newton_mimic_np[w, mjc_eq] = w * mimic_per_world + template_mimic
            self.mjc_eq_to_newton_mimic = wp.array(mjc_eq_to_newton_mimic_np, dtype=wp.int32)

            # Create mjc_tendon_to_newton_tendon: MuJoCo[world, tendon] -> Newton tendon
            # selected_tendons[idx] is the Newton template tendon index
            ntendon = self.mj_model.ntendon
            if ntendon > 0:
                # Get tendon count per world from custom attributes
                mujoco_attrs = getattr(model, "mujoco", None)
                tendon_world = getattr(mujoco_attrs, "tendon_world", None) if mujoco_attrs else None
                if tendon_world is not None:
                    total_tendons = len(tendon_world)
                    tendons_per_world = total_tendons // model.world_count if model.world_count > 0 else total_tendons
                else:
                    tendons_per_world = ntendon
                mjc_tendon_to_newton_tendon_np = np.full((nworld, ntendon), -1, dtype=np.int32)
                for mjc_tendon, newton_tendon in enumerate(selected_tendons):
                    template_tendon = newton_tendon % tendons_per_world if tendons_per_world > 0 else newton_tendon
                    for w in range(nworld):
                        mjc_tendon_to_newton_tendon_np[w, mjc_tendon] = w * tendons_per_world + template_tendon
                self.mjc_tendon_to_newton_tendon = wp.array(mjc_tendon_to_newton_tendon_np, dtype=wp.int32)

            if separate_worlds:
                nworld = model.world_count
            else:
                nworld = 1

            # TODO find better heuristics to determine nconmax and njmax
            if disable_contacts:
                nconmax = 0
            elif nconmax is not None and nconmax < self.mj_data.ncon:
                warnings.warn(
                    f"[WARNING] Value for nconmax is changed from {nconmax} to {self.mj_data.ncon} following an MjWarp requirement.",
                    stacklevel=2,
                )
                nconmax = self.mj_data.ncon

            if njmax is not None and njmax < self.mj_data.nefc:
                warnings.warn(
                    f"[WARNING] Value for njmax is changed from {njmax} to {self.mj_data.nefc} following an MjWarp requirement.",
                    stacklevel=2,
                )
                njmax = self.mj_data.nefc

            if nvmax is not None:
                if nvmax > self.mj_model.nv:
                    raise ValueError(f"nvmax must not exceed the model's {self.mj_model.nv} degrees of freedom.")
                sleeping_enabled = bool(enableflags & mujoco.mjtEnableBit.mjENBL_SLEEP)
                if not sleeping_enabled:
                    raise ValueError("nvmax is only supported when sleeping is enabled.")
                initially_awake = self.mj_data.tree_asleep < 0
                minimum_nvmax = int(np.sum(self.mj_model.tree_dofnum[initially_awake]))
                if nvmax < minimum_nvmax:
                    raise ValueError(
                        f"nvmax={nvmax} is too small: the initial state has {minimum_nvmax} awake degrees of freedom."
                    )

            self.mjw_data = mujoco_warp.put_data(
                self.mj_model,
                self.mj_data,
                nworld=nworld,
                nconmax=nconmax,
                njmax=njmax,
                nvmax=nvmax,
            )
            self.nvmax = self.mjw_data.nvmax
            if self.enable_sleeping:
                self._capture_initial_sleeping_state()
                # Edited-tree flags set by _update_mjc_data and consumed by wake_changed_trees_kernel.
                self._sleep_tree_changed = wp.zeros(
                    (nworld, self.mjw_model.ntree),
                    dtype=wp.int32,
                    device=self.model.device,
                )

            if not self.use_mujoco_cpu:
                if self._deterministic != wp.DeterministicMode.NOT_GUARANTEED:
                    self._deterministic_max_records = _mujoco_warp_deterministic_max_records(
                        self.mj_model, self.mjw_data
                    )
                self._set_mujoco_warp_module_options()
                self._prepare_generated_kernels()

            # expand model fields that can be expanded:
            self._expand_model_fields(self.mjw_model, nworld)
            self._create_mesh_variant_banks(model)

            # update solver options from Newton model (only if not overridden by constructor)
            self._update_solver_options(overridden_options=overridden_options)

            # so far we have only defined the first world,
            # now complete the data from the Newton model
            self.notify_model_changed(ModelFlags.ALL)
            if self._mesh_variant_definitions:
                world_ids = wp.array(np.arange(model.world_count, dtype=np.int32), dtype=wp.int32, device=model.device)
                for definition in self._mesh_variant_definitions:
                    variant_ids = wp.array(definition.initial_variant_ids, dtype=wp.int32, device=model.device)
                    self.set_mesh_variant_index(definition.name, variant_ids=variant_ids, world_ids=world_ids)

            if target_filename:
                # Only persist ``solreflimit`` for ``SOLREF_MODE_RAW`` joints
                # (authored MuJoCo values). For ``SOLREF_MODE_FORCE_SPACE``
                # the runtime ``jnt_solref`` is the post-``factor`` value;
                # writing it back would, on re-import, look like an
                # authored ``solreflimit`` (RAW mode) and silently freeze
                # the Newton force-space ``joint_limit_ke``/``kd`` semantics
                # — subsequent edits to those Newton gains would no longer
                # change the constraint behaviour. For ``MJCF_DEFAULT``
                # joints we similarly skip so MuJoCo's implicit default
                # ``(0.02, 1.0)`` round-trips as the "no attribute"
                # serialisation that the importer reads back as
                # ``MJCF_DEFAULT``.
                #
                # Note: ``save_to_mjcf`` is *not* a fully semantic
                # round-trip for ``FORCE_SPACE`` joints — MJCF cannot
                # express "use Newton force-space scaling with these
                # gains", only the resulting solref. Users who need to
                # reload force-space joint dynamics must reapply
                # ``joint_limit_ke``/``kd`` (and the FORCE_SPACE mode) on
                # the rebuilt model.
                jnt_to_newton_dof = self.mjc_jnt_to_newton_dof.numpy()[0]
                # ``joint_solref_limit_mode`` is the per-DOF
                # ``mujoco.solreflimit_mode`` array fetched at the start of
                # ``_convert_to_mjc`` (see the ``get_custom_attribute`` block).
                # It is ``None`` only when the custom attribute is not
                # registered, in which case every joint defaults to the
                # FORCE_SPACE branch below.
                for mjc_jnt, solref in enumerate(self.mj_model.jnt_solref):
                    if not self.mj_model.jnt_limited[mjc_jnt]:
                        continue
                    newton_dof = int(jnt_to_newton_dof[mjc_jnt])
                    if newton_dof < 0:
                        continue
                    mode = (
                        int(joint_solref_limit_mode[newton_dof])
                        if joint_solref_limit_mode is not None
                        else SOLREF_MODE_FORCE_SPACE
                    )
                    if mode == SOLREF_MODE_RAW:
                        spec.joints[mjc_jnt].solref_limit = solref
                with open(target_filename, "w") as f:
                    f.write(spec.to_xml())
                    print(f"Saved mujoco model to {os.path.abspath(target_filename)}")

    def _expand_model_fields(self, mj_model: MjWarpModel, nworld: int):
        if nworld == 1:
            return

        model_fields_to_expand = {
            "qpos0",
            "qpos_spring",
            "body_pos",
            "body_quat",
            "body_ipos",
            "body_iquat",
            "body_mass",
            "body_subtreemass",  # Derived from body_mass, computed by set_const_fixed
            "body_inertia",
            "body_invweight0",  # Derived from inertia, computed by set_const_0
            "body_gravcomp",
            "jnt_solref",
            "jnt_solimp",
            "jnt_pos",
            "jnt_axis",
            "jnt_stiffness",
            "jnt_range",
            "jnt_actfrcrange",  # joint-level actuator force range (effort limit)
            "jnt_margin",  # corresponds to newton custom attribute "limit_margin"
            "dof_armature",
            "dof_damping",
            "dof_invweight0",  # Derived from inertia, computed by set_const_0
            "dof_frictionloss",
            "dof_solimp",
            "dof_solref",
            # "geom_matid",
            "geom_solmix",
            "geom_solref",
            "geom_solimp",
            "geom_size",
            "geom_rbound",
            "geom_pos",
            "geom_quat",
            "geom_friction",
            "geom_margin",
            "geom_gap",
            # "geom_rgba",
            "site_pos",
            "site_quat",
            # "cam_pos",
            # "cam_quat",
            # "cam_poscom0",
            # "cam_pos0",
            # "cam_mat0",
            # "light_pos",
            # "light_dir",
            # "light_poscom0",
            # "light_pos0",
            "eq_solref",
            "eq_solimp",
            "eq_data",
            # "actuator_dynprm",
            "actuator_gainprm",
            "actuator_biasprm",
            "actuator_dynprm",
            "actuator_ctrlrange",
            "actuator_forcerange",
            "actuator_actrange",
            "actuator_gear",
            "actuator_cranklength",
            "actuator_acc0",
            "actuator_lengthrange",
            "pair_solref",
            "pair_solreffriction",
            "pair_solimp",
            "pair_margin",
            "pair_gap",
            "pair_friction",
            "tendon_world",
            "tendon_solref_lim",
            "tendon_solimp_lim",
            "tendon_solref_fri",
            "tendon_solimp_fri",
            "tendon_range",
            "tendon_actfrcrange",
            "tendon_margin",
            "tendon_stiffness",
            "tendon_damping",
            "tendon_armature",
            "tendon_frictionloss",
            "tendon_lengthspring",
            "tendon_length0",  # Derived from tendon config, computed by set_const_0
            "tendon_invweight0",  # Derived from inertia, computed by set_const_0
            # "mat_rgba",
        }
        if self._mesh_variant_definitions:
            model_fields_to_expand.update(("geom_dataid", "geom_aabb"))

        # Solver option fields to expand (nested in mj_model.opt)
        opt_fields_to_expand = {
            # "timestep",  # Excluded: conflicts with step() function parameter
            "impratio_invsqrt",
            "tolerance",
            "ls_tolerance",
            "ccd_tolerance",
            "sleep_tolerance",
            "density",
            "viscosity",
            "gravity",
            "wind",
            "magnetic",
        }

        def tile(x: wp.array):
            # Create new array with same shape but first dim multiplied by nworld
            new_shape = list(x.shape)
            new_shape[0] = nworld
            wp_array = {1: wp.array, 2: wp.array2d, 3: wp.array3d, 4: wp.array4d}[len(new_shape)]
            dst = wp_array(shape=new_shape, dtype=x.dtype, device=x.device)

            # Flatten arrays for kernel
            src_flat = x.flatten()
            dst_flat = dst.flatten()

            # Launch kernel to repeat data - one thread per destination element
            n_elems_per_world = dst_flat.shape[0] // nworld
            wp.launch(
                repeat_array_kernel,
                dim=dst_flat.shape[0],
                inputs=[src_flat, n_elems_per_world],
                outputs=[dst_flat],
                device=x.device,
            )
            return dst

        for field in mj_model.__dataclass_fields__:
            if field in model_fields_to_expand:
                array = getattr(mj_model, field)
                setattr(mj_model, field, tile(array))

        mj_model.stat.meaninertia = tile(mj_model.stat.meaninertia)

        for field in mj_model.opt.__dataclass_fields__:
            if field in opt_fields_to_expand:
                array = getattr(mj_model.opt, field)
                setattr(mj_model.opt, field, tile(array))

    def _update_solver_options(self, overridden_options: set[str] | None = None):
        """Update WORLD frequency solver options from Newton model to MuJoCo Warp.

        Called after tile() to update per-world option arrays in mjw_model.opt.
        Only updates WORLD frequency options; ONCE frequency options are already
        set on MjSpec before put_model() and shared across all worlds.

        Args:
            overridden_options: Set of option names that were overridden by constructor.
                These options should not be updated from model custom attributes.
        """
        if overridden_options is None:
            overridden_options = set()

        mujoco_attrs = getattr(self.model, "mujoco", None)
        nworld = self.model.world_count

        # Helper to get WORLD frequency option array or None
        def get_option(name: str):
            if name in overridden_options or not mujoco_attrs or not hasattr(mujoco_attrs, name):
                return None
            return getattr(mujoco_attrs, name)

        # Get all WORLD frequency scalar arrays
        newton_impratio = get_option("impratio")
        newton_tolerance = get_option("tolerance")
        newton_ls_tolerance = get_option("ls_tolerance")
        newton_ccd_tolerance = get_option("ccd_tolerance")
        newton_sleep_tolerance = get_option("sleep_tolerance")
        newton_density = get_option("density")
        newton_viscosity = get_option("viscosity")

        # Get WORLD frequency vector arrays
        newton_wind = get_option("wind")
        newton_magnetic = get_option("magnetic")

        # Skip kernel if all options are None
        if all(
            x is None
            for x in [
                newton_impratio,
                newton_tolerance,
                newton_ls_tolerance,
                newton_ccd_tolerance,
                newton_sleep_tolerance,
                newton_density,
                newton_viscosity,
                newton_wind,
                newton_magnetic,
            ]
        ):
            return

        wp.launch(
            update_solver_options_kernel,
            dim=nworld,
            inputs=[
                newton_impratio,
                newton_tolerance,
                newton_ls_tolerance,
                newton_ccd_tolerance,
                newton_sleep_tolerance,
                newton_density,
                newton_viscosity,
                newton_wind,
                newton_magnetic,
            ],
            outputs=[
                self.mjw_model.opt.impratio_invsqrt,
                self.mjw_model.opt.tolerance,
                self.mjw_model.opt.ls_tolerance,
                self.mjw_model.opt.ccd_tolerance,
                self.mjw_model.opt.sleep_tolerance,
                self.mjw_model.opt.density,
                self.mjw_model.opt.viscosity,
                self.mjw_model.opt.wind,
                self.mjw_model.opt.magnetic,
            ],
            device=self.model.device,
        )

    def _update_model_inertial_properties(self):
        if self.model.body_count == 0:
            return

        # Get gravcomp if available
        mujoco_attrs = getattr(self.model, "mujoco", None)
        gravcomp = getattr(mujoco_attrs, "gravcomp", None) if mujoco_attrs is not None else None

        # Launch over MuJoCo bodies [nworld, nbody]
        nworld = self.mjc_body_to_newton.shape[0]
        nbody = self.mjc_body_to_newton.shape[1]

        wp.launch(
            update_body_mass_ipos_kernel,
            dim=(nworld, nbody),
            inputs=[
                self.mjc_body_to_newton,
                self.model.body_com,
                self.model.body_mass,
                gravcomp,
            ],
            outputs=[
                self.mjw_model.body_ipos,
                self.mjw_model.body_mass,
                self.mjw_model.body_gravcomp,
            ],
            device=self.model.device,
        )

        wp.launch(
            update_body_inertia_kernel,
            dim=(nworld, nbody),
            inputs=[
                self.mjc_body_to_newton,
                self.model.body_inertia,
            ],
            outputs=[self.mjw_model.body_inertia, self.mjw_model.body_iquat],
            device=self.model.device,
        )
        # Gravity compensation may have moved between bodies: rebuild the compact launch list.
        refresh = getattr(getattr(self._mujoco_warp, "_src", None), "io", None)
        refresh = getattr(refresh, "refresh_gravcomp_bodies", None) if refresh is not None else None
        if refresh is not None and self.mjw_model is not None:
            refresh(self.mjw_model)

    def _set_const_0_with_physical_meaninertia(self) -> None:
        """Recompute constants without counting kinematic locking armature in solver statistics."""
        has_kinematic_bodies = bool(np.any((self.model.body_flags.numpy() & int(BodyFlags.KINEMATIC)) != 0))
        if not has_kinematic_bodies:
            if self.use_mujoco_cpu:
                self._mujoco.mj_setConst(self.mj_model, self.mj_data)
            else:
                self._mujoco_warp.set_const_0(self.mjw_model, self.mjw_data)
            return

        # Subtracting the locking armature in float32 would lose the physical inertia.
        self._update_body_properties(apply_kinematic_armature=False)
        if self.use_mujoco_cpu:
            actuator_biasprm = self.mj_model.actuator_biasprm.copy()
            self.mj_model.dof_armature[:] = self.mjw_model.dof_armature.numpy()[0]
            self._mujoco.mj_setConst(self.mj_model, self.mj_data)
            physical_meaninertia = float(self.mj_model.stat.meaninertia)

            self._update_body_properties()
            self.mj_model.actuator_biasprm[:] = actuator_biasprm
            self.mj_model.dof_armature[:] = self.mjw_model.dof_armature.numpy()[0]
            self._mujoco.mj_setConst(self.mj_model, self.mj_data)
            self.mj_model.stat.meaninertia = physical_meaninertia
        else:
            actuator_biasprm = wp.clone(self.mjw_model.actuator_biasprm)
            self._mujoco_warp.set_const_0(self.mjw_model, self.mjw_data)
            physical_meaninertia = wp.clone(self.mjw_model.stat.meaninertia)

            self._update_body_properties()
            wp.copy(self.mjw_model.actuator_biasprm, actuator_biasprm)
            self._mujoco_warp.set_const_0(self.mjw_model, self.mjw_data)
            wp.copy(self.mjw_model.stat.meaninertia, physical_meaninertia)

    def _update_body_properties(self, apply_kinematic_armature: bool = True):
        """Update body-property dependent MuJoCo DOF parameters.

        This currently applies kinematic body flags by rewriting MuJoCo
        ``dof_armature`` from Newton ``body_flags`` and ``joint_armature``.
        """
        if self.model.joint_dof_count == 0:
            return
        if self.mjc_dof_to_newton_dof is None or self.newton_dof_to_body is None:
            return

        nworld = self.mjc_dof_to_newton_dof.shape[0]
        nv = self.mjc_dof_to_newton_dof.shape[1]

        wp.launch(
            update_body_properties_kernel,
            dim=(nworld, nv),
            inputs=[
                self.mjc_dof_to_newton_dof,
                self.newton_dof_to_body,
                self.model.body_flags,
                self.model.joint_armature,
                KINEMATIC_ARMATURE,
                apply_kinematic_armature,
            ],
            outputs=[self.mjw_model.dof_armature],
            device=self.model.device,
        )

    def _update_joint_dof_properties(self):
        """Update joint DOF properties in the MuJoCo model.

        Updates effort limits, friction, damping, solimp/solref, passive
        stiffness, and limit ranges. Armature is updated for dynamic DOFs only;
        DOFs attached to kinematic bodies are preserved.
        """
        if self.model.joint_dof_count == 0:
            return
        if self.newton_dof_to_body is None:
            return

        # Update actuator gains for JOINT_TARGET mode actuators
        if self.mjc_actuator_ctrl_source is not None and self.mjc_actuator_to_newton_idx is not None:
            nu = self.mjc_actuator_ctrl_source.shape[0]
            nworld = self.mjw_model.actuator_biasprm.shape[0]
            dofs_per_world = self.model.joint_dof_count // nworld if nworld > 0 else self.model.joint_dof_count

            wp.launch(
                update_axis_properties_kernel,
                dim=(nworld, nu),
                inputs=[
                    self.mjc_actuator_ctrl_source,
                    self.mjc_actuator_to_newton_idx,
                    self.model.joint_target_ke,
                    self.model.joint_target_kd,
                    self.model.joint_target_mode,
                    dofs_per_world,
                ],
                outputs=[
                    self.mjw_model.actuator_biasprm,
                    self.mjw_model.actuator_gainprm,
                ],
                device=self.model.device,
            )

        # Update DOF properties (armature, friction, damping, solimp, solref) - iterate over MuJoCo DOFs
        mujoco_attrs = getattr(self.model, "mujoco", None)
        dof_solimp = getattr(mujoco_attrs, "solimpfriction", None) if mujoco_attrs is not None else None
        dof_solref = getattr(mujoco_attrs, "solreffriction", None) if mujoco_attrs is not None else None

        nworld = self.mjc_dof_to_newton_dof.shape[0]
        nv = self.mjc_dof_to_newton_dof.shape[1]
        wp.launch(
            update_dof_properties_kernel,
            dim=(nworld, nv),
            inputs=[
                self.mjc_dof_to_newton_dof,
                self.newton_dof_to_body,
                self.model.body_flags,
                self.model.joint_armature,
                self.model.joint_friction,
                self.model.joint_damping,
                dof_solimp,
                dof_solref,
            ],
            outputs=[
                self.mjw_model.dof_armature,
                self.mjw_model.dof_frictionloss,
                self.mjw_model.dof_damping,
                self.mjw_model.dof_solimp,
                self.mjw_model.dof_solref,
            ],
            device=self.model.device,
        )

        # Update joint properties (limits, stiffness, solimp) per MuJoCo joint.
        solimplimit = getattr(mujoco_attrs, "solimplimit", None) if mujoco_attrs is not None else None
        joint_dof_limit_margin = getattr(mujoco_attrs, "limit_margin", None) if mujoco_attrs is not None else None
        joint_stiffness = getattr(mujoco_attrs, "dof_passive_stiffness", None) if mujoco_attrs is not None else None

        njnt = self.mjc_jnt_to_newton_dof.shape[1]
        wp.launch(
            update_jnt_properties_kernel,
            dim=(nworld, njnt),
            inputs=[
                self.mjc_jnt_to_newton_dof,
                self.model.joint_limit_lower,
                self.model.joint_limit_upper,
                self.model.joint_effort_limit,
                solimplimit,
                joint_stiffness,
                joint_dof_limit_margin,
            ],
            outputs=[
                self.mjw_model.jnt_solimp,
                self.mjw_model.jnt_stiffness,
                self.mjw_model.jnt_margin,
                self.mjw_model.jnt_range,
                self.mjw_model.jnt_actfrcrange,
            ],
            device=self.model.device,
        )
        # Joint-limit solref is updated later, after ``set_const_0`` /
        # ``mj_setConst`` refresh ``dof_invweight0``. ``jnt_solimp`` already
        # comes from the launch above.

        # Sync qpos0 and qpos_spring from Newton model data before set_const.
        # set_const copies qpos0 → d.qpos and runs FK to compute derived fields,
        # so qpos0 must be correct before calling it.
        dof_ref = getattr(mujoco_attrs, "dof_ref", None) if mujoco_attrs is not None else None
        dof_springref = getattr(mujoco_attrs, "dof_springref", None) if mujoco_attrs is not None else None
        joints_per_world = self.model.joint_count // nworld
        bodies_per_world = self.model.body_count // nworld
        wp.launch(
            sync_qpos0_kernel,
            dim=(nworld, joints_per_world),
            inputs=[
                joints_per_world,
                bodies_per_world,
                self.model.joint_type,
                self.model.joint_q_start,
                self.model.joint_qd_start,
                self.model.joint_dof_dim,
                self.model.joint_child,
                self.model.body_q,
                dof_ref,
                dof_springref,
                self.mj_q_start,
            ],
            outputs=[
                self.mjw_model.qpos0,
                self.mjw_model.qpos_spring,
            ],
            device=self.model.device,
        )

    def _update_joint_properties(self):
        """Update joint properties including joint positions, joint axes, and relative body transforms in the MuJoCo model."""
        if self.model.joint_count == 0:
            return

        # Update mocap body transforms first (fixed-root bodies have no MuJoCo joints).
        if self.mjc_mocap_to_newton_jnt is not None and self.mjc_mocap_to_newton_jnt.shape[1] > 0:
            nworld = self.mjc_mocap_to_newton_jnt.shape[0]
            nmocap = self.mjc_mocap_to_newton_jnt.shape[1]
            wp.launch(
                update_mocap_transforms_kernel,
                dim=(nworld, nmocap),
                inputs=[
                    self.mjc_mocap_to_newton_jnt,
                    self.model.joint_X_p,
                    self.model.joint_X_c,
                ],
                outputs=[
                    self.mjw_data.mocap_pos,
                    self.mjw_data.mocap_quat,
                ],
                device=self.model.device,
            )

        # Update joint positions, joint axes, and relative body transforms
        # Iterates over MuJoCo joints [world, jnt]
        if self.mjc_jnt_to_newton_jnt is not None and self.mjc_jnt_to_newton_jnt.shape[1] > 0:
            nworld = self.mjc_jnt_to_newton_jnt.shape[0]
            njnt = self.mjc_jnt_to_newton_jnt.shape[1]

            wp.launch(
                update_joint_transforms_kernel,
                dim=(nworld, njnt),
                inputs=[
                    self.mjc_jnt_to_newton_jnt,
                    self.mjc_jnt_to_newton_dof,
                    self.mjw_model.jnt_bodyid,
                    self.mjw_model.jnt_type,
                    # Newton model data (joint-indexed)
                    self.model.joint_X_p,
                    self.model.joint_X_c,
                    # Newton model data (DOF-indexed)
                    self.model.joint_axis,
                ],
                outputs=[
                    self.mjw_model.jnt_pos,
                    self.mjw_model.jnt_axis,
                    self.mjw_model.body_pos,
                    self.mjw_model.body_quat,
                ],
                device=self.model.device,
            )

    @staticmethod
    def _copy_dof_ref_to_qref(model: Model) -> wp.array:
        """Build reference joint coordinates from model data and ``dof_ref``.

        Launches ``build_ref_q_kernel`` to produce joint coordinates in
        Newton convention (xyzw quaternions). FREE/DISTANCE joints copy
        position and orientation from ``joint_q``, BALL
        joints use identity, and hinge/slide/D6 joints use ``dof_ref``.

        Args:
            model: The Newton :class:`Model`.

        Returns:
            Reference joint coordinates [m or rad],
            ``wp.array[wp.float32]``, shape ``[joint_coord_count]``.
        """
        mujoco_attrs = getattr(model, "mujoco", None)
        dof_ref = getattr(mujoco_attrs, "dof_ref", None) if mujoco_attrs is not None else None

        ref_q = wp.zeros(model.joint_coord_count, dtype=wp.float32, device=model.device)
        wp.launch(
            kernel=build_ref_q_kernel,
            dim=model.joint_count,
            inputs=[
                model.joint_type,
                model.joint_q,
                model.joint_q_start,
                model.joint_qd_start,
                model.joint_dof_dim,
                dof_ref,
            ],
            outputs=[
                ref_q,
            ],
            device=model.device,
        )
        return ref_q

    @staticmethod
    def _compute_body_poses_at_qref(model: Model, ref_q: wp.array) -> wp.array:
        """Compute body transforms at the reference joint configuration.

        Runs :func:`newton.eval_articulation_fk` with the given ``ref_q``
        and zero velocities to obtain world-space body transforms at the
        reference pose.

        Args:
            model: The Newton :class:`Model`.
            ref_q: Reference joint coordinates [m or rad],
                ``wp.array[wp.float32]``, shape ``[joint_coord_count]``.

        Returns:
            Body transforms at the reference pose [m],
            ``wp.array[wp.transform]``, shape ``[body_count]``.
        """
        ref_qd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=model.device)
        ref_body_q = wp.zeros(model.body_count, dtype=wp.transform, device=model.device)
        ref_body_qd = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=model.device)

        wp.launch(
            kernel=eval_articulation_fk,
            dim=model.articulation_count,
            inputs=[
                model.articulation_start,
                model.articulation_end,
                model.articulation_count,
                None,
                None,
                model.joint_articulation,
                ref_q,
                ref_qd,
                model.joint_q_start,
                model.joint_qd_start,
                model.joint_type,
                model.joint_parent,
                model.joint_child,
                model.joint_X_p,
                model.joint_X_c,
                model.joint_axis,
                model.joint_dof_dim,
                model.body_com,
                model.body_flags,
                int(BodyFlags.ALL),
            ],
            outputs=[ref_body_q, ref_body_qd],
            device=model.device,
        )
        return ref_body_q

    @staticmethod
    def _compute_connect_constraint_rel_xform_at_qref(model: Model, ref_body_q: wp.array) -> tuple[wp.array, wp.array]:
        """Compute relative body transforms for CONNECT constraints at the reference pose.

        Launches ``update_connect_constraint_rel_body_poses_at_qref_kernel``
        to compute per-constraint ``(q_rel, t_rel)`` pairs such that::

            anchor2 = quat_rotate(q_rel, anchor1) + t_rel

        Args:
            model: The Newton :class:`Model`.
            ref_body_q: Body transforms at the reference pose [m],
                ``wp.array[wp.transform]``, shape ``[body_count]``.

        Returns:
            Tuple of ``(q_rel, t_rel)`` where ``q_rel`` is
            ``wp.array[wp.quat]`` and ``t_rel`` is
            ``wp.array[wp.vec3]`` [m], each of
            shape ``[equality_constraint_count]``.
        """
        neq = model.mujoco.equality_constraint_count

        q_rel = wp.zeros(neq, dtype=wp.quat, device=model.device)
        t_rel = wp.zeros(neq, dtype=wp.vec3, device=model.device)
        # Nothing to launch with no equality constraints; the per-row arrays are present but
        # empty (finalize keeps them shape-stable), so skip the zero-width launch.
        if neq == 0:
            return q_rel, t_rel

        wp.launch(
            update_connect_constraint_rel_body_poses_at_qref_kernel,
            dim=neq,
            inputs=[
                model.mujoco.equality_constraint_type,
                model.mujoco.equality_constraint_body1,
                model.mujoco.equality_constraint_body2,
                ref_body_q,
            ],
            outputs=[
                q_rel,
                t_rel,
            ],
            device=model.device,
        )

        return q_rel, t_rel

    @staticmethod
    def _update_connect_constraint_anchors(
        model: Model,
        mjw_model: MjWarpModel,
        mjc_eq_to_newton_eq: wp.array,
        connect_anchor2_q: wp.array,
        connect_anchor2_t: wp.array,
    ):
        """Write CONNECT constraint anchors into the MuJoCo Warp model.

        Launches ``update_connect_constraint_anchors_kernel`` to copy
        ``anchor1`` [m] and compute
        ``anchor2 = quat_rotate(q_rel, anchor1) + t_rel`` [m] into
        ``mjw_model.eq_data``. Skips immediately when ``mjw_model.neq == 0``.

        Args:
            model: The Newton :class:`Model` (source for constraint types
                and anchor positions).
            mjw_model: The MuJoCo Warp model (target for ``eq_data`` output).
            mjc_eq_to_newton_eq: Mapping from MuJoCo ``[world, eq]`` to
                Newton equality constraint index,
                ``wp.array2d[wp.int32]``.
            connect_anchor2_q: Precomputed relative rotation per constraint,
                ``wp.array[wp.quat]``,
                shape ``[equality_constraint_count]``.
            connect_anchor2_t: Precomputed relative translation [m] per
                constraint, ``wp.array[wp.vec3]``,
                shape ``[equality_constraint_count]``.
        """
        if mjw_model.neq == 0:
            return

        world_count = mjc_eq_to_newton_eq.shape[0]

        wp.launch(
            update_connect_constraint_anchors_kernel,
            dim=(world_count, mjw_model.neq),
            inputs=[
                mjc_eq_to_newton_eq,
                model.mujoco.equality_constraint_type,
                model.mujoco.equality_constraint_anchor,
                connect_anchor2_q,
                connect_anchor2_t,
            ],
            outputs=[
                mjw_model.eq_data,
            ],
            device=model.device,
        )

    def _recompute_jnt_eq_anchor1(self):
        """Recompute ``jnt_eq_anchor1`` from the current ``joint_X_p`` and ``joint_axis``.

        Launches :func:`recompute_jnt_eq_anchor1_kernel` to update the
        body1-local anchor positions for joint-synthesized CONNECT
        constraints.  Must be called when joint frames change so that
        ``jnt_eq_anchor1`` stays in sync with ``joint_X_p``.
        """
        if self.mjw_model.neq == 0:
            return

        world_count = self.mjc_eq_to_newton_jnt.shape[0]

        wp.launch(
            recompute_jnt_eq_anchor1_kernel,
            dim=(world_count, self.mjw_model.neq),
            inputs=[
                self.mjc_eq_to_newton_jnt,
                self.jnt_eq_anchor1_has_axis_offset,
                HINGE_CONNECT_AXIS_OFFSET,
                self.model.joint_X_p,
                self.model.joint_axis,
                self.model.joint_qd_start,
            ],
            outputs=[
                self.jnt_eq_anchor1,
            ],
            device=self.model.device,
        )

    def _notify_connect_constraints_changed(
        self,
        update_anchor_rel_xform_at_ref_pose: bool,
        update_anchors: bool,
    ):
        """Update CONNECT constraint anchors in response to model changes.

        Optionally recomputes the relative body transforms at the reference
        pose, then writes the resulting anchors into ``mjw_model.eq_data``
        (and ``mj_model.eq_data`` on the CPU path).  Handles both
        equality-constraint-based and joint-synthesized CONNECT constraints.

        Must be called **after** other model updates (``mj_setConst``,
        ``set_const_0``, etc.) because it overwrites ``eq_data``.

        Args:
            update_anchor_rel_xform_at_ref_pose: Recompute ``(q_rel, t_rel)``
                from ``dof_ref`` / joint properties.
            update_anchors: Recompute anchors from
                ``model.mujoco.equality_constraint_anchor``.
        """
        if update_anchor_rel_xform_at_ref_pose:
            ref_q = SolverMuJoCo._copy_dof_ref_to_qref(self.model)
            ref_body_q = SolverMuJoCo._compute_body_poses_at_qref(self.model, ref_q)
            self.connect_constraint_q_rel, self.connect_constraint_t_rel = (
                SolverMuJoCo._compute_connect_constraint_rel_xform_at_qref(self.model, ref_body_q)
            )
            if self.has_jnt_connect_constraints:
                self.jnt_connect_constraint_q_rel, self.jnt_connect_constraint_t_rel = (
                    SolverMuJoCo._compute_jnt_connect_constraint_rel_xform_at_qref(
                        self.model,
                        self.mjc_eq_to_newton_jnt,
                        self.mjw_model.neq,
                        ref_body_q,
                    )
                )
        # connect_constraint_q_rel is guaranteed non-None when update_anchors
        # is True because _convert_to_mjc calls notify_model_changed(ALL),
        # which includes JOINT_DOF_PROPERTIES and therefore always computes
        # q_rel before any CONSTRAINT_PROPERTIES-only notification can occur.
        # The None check is a defensive guard for the case where the model
        # has no explicit connect constraints (has_connect_constraints is
        # False and both flags are False, so this branch is unreachable).
        wrote_eq_data = False
        if (
            self.has_connect_constraints
            and (update_anchor_rel_xform_at_ref_pose or update_anchors)
            and self.connect_constraint_q_rel is not None
        ):
            SolverMuJoCo._update_connect_constraint_anchors(
                self.model,
                self.mjw_model,
                self.mjc_eq_to_newton_eq,
                self.connect_constraint_q_rel,
                self.connect_constraint_t_rel,
            )
            wrote_eq_data = True
        if update_anchor_rel_xform_at_ref_pose and self.has_jnt_connect_constraints:
            self._recompute_jnt_eq_anchor1()
            SolverMuJoCo._update_jnt_connect_constraint_anchors(
                self.model,
                self.mjw_model,
                self.mjc_eq_to_newton_jnt,
                self.jnt_eq_anchor1,
                self.jnt_connect_constraint_q_rel,
                self.jnt_connect_constraint_t_rel,
            )
            wrote_eq_data = True
        if wrote_eq_data and self.use_mujoco_cpu:
            self.mj_model.eq_data[:] = self.mjw_model.eq_data.numpy()[0]

    @staticmethod
    def _compute_jnt_connect_constraint_rel_xform_at_qref(
        model: Model,
        mjc_eq_to_newton_jnt: wp.array2d[wp.int32],
        mjw_neq: int,
        ref_body_q: wp.array[wp.transform],
    ) -> tuple[wp.array2d[wp.quat], wp.array2d[wp.vec3]]:
        """Compute relative body transforms for joint-synthesized CONNECT constraints at the reference pose.

        Launches
        :func:`update_jnt_connect_constraint_rel_body_poses_at_qref_kernel`
        to produce per-``[world, eq]`` ``(q_rel, t_rel)`` arrays from the
        precomputed ``ref_body_q``.

        Args:
            model: The Newton :class:`Model`.
            mjc_eq_to_newton_jnt: Mapping from MuJoCo ``[world, eq]`` to
                Newton joint index, ``wp.array2d[wp.int32]``,
                shape ``[world_count, neq]``.
            mjw_neq: Number of MuJoCo equality constraints (``neq``).
            ref_body_q: Body transforms at the reference pose [m],
                ``wp.array[wp.transform]``, shape ``[body_count]``.

        Returns:
            Tuple of ``(jnt_connect_constraint_q_rel, jnt_connect_constraint_t_rel)`` where
            ``jnt_connect_constraint_q_rel`` is ``wp.array2d[wp.quat]`` and
            ``jnt_connect_constraint_t_rel`` is ``wp.array2d[wp.vec3]`` [m],
            each of shape ``[world_count, neq]``.
        """
        world_count = mjc_eq_to_newton_jnt.shape[0]
        jnt_connect_constraint_q_rel = wp.zeros((world_count, mjw_neq), dtype=wp.quat, device=model.device)
        jnt_connect_constraint_t_rel = wp.zeros((world_count, mjw_neq), dtype=wp.vec3, device=model.device)

        wp.launch(
            update_jnt_connect_constraint_rel_body_poses_at_qref_kernel,
            dim=(world_count, mjw_neq),
            inputs=[
                mjc_eq_to_newton_jnt,
                model.joint_parent,
                model.joint_child,
                ref_body_q,
            ],
            outputs=[
                jnt_connect_constraint_q_rel,
                jnt_connect_constraint_t_rel,
            ],
            device=model.device,
        )

        return jnt_connect_constraint_q_rel, jnt_connect_constraint_t_rel

    @staticmethod
    def _update_jnt_connect_constraint_anchors(
        model: Model,
        mjw_model: MjWarpModel,
        mjc_eq_to_newton_jnt: wp.array2d[wp.int32],
        jnt_eq_anchor1: wp.array2d[wp.vec3],
        jnt_connect_constraint_q_rel: wp.array2d[wp.quat],
        jnt_connect_constraint_t_rel: wp.array2d[wp.vec3],
    ):
        """Write joint-synthesized CONNECT constraint anchors into MuJoCo Warp model.

        Launches :func:`update_jnt_connect_constraint_anchors_kernel` to copy
        ``anchor1`` [m] and compute
        ``anchor2 = quat_rotate(q_rel, anchor1) + t_rel`` [m] into
        ``mjw_model.eq_data``. Skips immediately when ``mjw_model.neq == 0``.

        Args:
            model: The Newton :class:`Model`.
            mjw_model: The MuJoCo Warp model (target for ``eq_data`` output).
            mjc_eq_to_newton_jnt: Mapping from MuJoCo ``[world, eq]`` to
                Newton joint index, ``wp.array2d[wp.int32]``,
                shape ``[world_count, neq]``.
            jnt_eq_anchor1: Body1-local anchor [m] per ``[world, eq]``,
                ``wp.array2d[wp.vec3]``, shape ``[world_count, neq]``.
            jnt_connect_constraint_q_rel: Relative rotation per ``[world, eq]``,
                ``wp.array2d[wp.quat]``, shape ``[world_count, neq]``.
            jnt_connect_constraint_t_rel: Relative translation [m] per ``[world, eq]``,
                ``wp.array2d[wp.vec3]``, shape ``[world_count, neq]``.
        """
        if mjw_model.neq == 0:
            return

        world_count = mjc_eq_to_newton_jnt.shape[0]

        wp.launch(
            update_jnt_connect_constraint_anchors_kernel,
            dim=(world_count, mjw_model.neq),
            inputs=[
                mjc_eq_to_newton_jnt,
                jnt_eq_anchor1,
                jnt_connect_constraint_q_rel,
                jnt_connect_constraint_t_rel,
            ],
            outputs=[
                mjw_model.eq_data,
            ],
            device=model.device,
        )

    def _update_geom_properties(self):
        """Update geom properties including collision radius, friction, and contact parameters in the MuJoCo model."""

        # Get number of geoms and worlds from MuJoCo model
        num_geoms = self.mj_model.ngeom
        if num_geoms == 0:
            return

        world_count = self.mjc_geom_to_newton_shape.shape[0]

        # Get custom attribute for geom_solimp and geom_solmix
        mujoco_attrs = getattr(self.model, "mujoco", None)
        shape_geom_solimp = getattr(mujoco_attrs, "geom_solimp", None) if mujoco_attrs is not None else None
        shape_geom_solmix = getattr(mujoco_attrs, "geom_solmix", None) if mujoco_attrs is not None else None
        shape_mjc_solref = getattr(mujoco_attrs, "solref", None) if mujoco_attrs is not None else None
        shape_mjc_solref_mode = getattr(mujoco_attrs, "solref_mode", None) if mujoco_attrs is not None else None

        # Shape-material force-space scaling is strictly opt-in (no
        # auto-promote, unlike joint limits from PR #2610): per-example
        # ``default_shape_cfg.ke``/``kd`` overrides are too common for a
        # ke-drift heuristic to be reliable. Set
        # ``model.mujoco.solref_mode[shape] = SOLREF_MODE_FORCE_SPACE``
        # explicitly to enable per-contact ``body_invweight0`` scaling.

        wp.launch(
            update_geom_properties_kernel,
            dim=(world_count, num_geoms),
            inputs=[
                self.model.shape_material_mu,
                self.model.shape_material_ke,
                self.model.shape_material_kd,
                self.model.shape_scale,
                self.model.shape_transform,
                self.mjc_geom_to_newton_shape,
                self.mjw_model.geom_type,
                self._mujoco.mjtGeom.mjGEOM_MESH,
                self.mjw_model.geom_dataid,
                self.mjw_model.mesh_pos,
                self.mjw_model.mesh_quat,
                self.model.shape_material_mu_torsional,
                self.model.shape_material_mu_rolling,
                shape_geom_solimp,
                shape_geom_solmix,
                shape_mjc_solref,
                shape_mjc_solref_mode,
                self.model.shape_margin,
                self.model.shape_gap,
                int(self._use_mujoco_contacts and self._zero_margins_for_native_ccd),
            ],
            outputs=[
                self.mjw_model.geom_friction,
                self.mjw_model.geom_solref,
                self.mjw_model.geom_size,
                self.mjw_model.geom_pos,
                self.mjw_model.geom_quat,
                self.mjw_model.geom_solimp,
                self.mjw_model.geom_solmix,
                self.mjw_model.geom_gap,
                self.mjw_model.geom_margin,
            ],
            device=self.model.device,
        )

    def _update_site_properties(self) -> None:
        """Update MuJoCo site poses and sizes from Newton shape properties.

        ``site_size`` is unbatched in mujoco_warp, so sizes are synced from
        the first world; per-world scale differences are not supported.
        """
        if self.mj_model.nsite == 0:
            return

        wp.launch(
            update_site_properties_kernel,
            dim=(self.mjw_data.nworld, self.mj_model.nsite),
            inputs=[
                self.model.shape_transform,
                self.model.shape_scale,
                self._mjc_site_shape_index,
                self._mjc_site_is_global,
                self._shapes_per_world,
                self._first_env_shape_base,
            ],
            outputs=[
                self.mjw_model.site_pos,
                self.mjw_model.site_quat,
                self.mjw_model.site_size,
            ],
            device=self.model.device,
        )

    def _update_solref_from_invweight0(self):
        """Scale joint-limit ``jnt_solref`` using ``dof_invweight0`` and ``jnt_solimp``.

        MuJoCo's limit-constraint solver computes an effective stiffness
        ``k_eff = k / (invweight * (1 - dmax))`` where ``invweight`` is the
        owning DOF's ``dof_invweight0`` and ``dmax = solimp[1]``. Newton's
        user-facing ``joint_limit_ke``/``joint_limit_kd`` are force-space
        quantities, so ``jnt_solref`` has to be pre-scaled by
        ``dof_invweight0 * (1 - dmax)`` for the downstream ``k_eff`` to
        match the user's configured force-space stiffness and damping.

        MJCF import stores authored ``solreflimit`` values separately in
        ``mujoco.solreflimit``. When present, those raw MuJoCo values are
        forwarded unchanged so imported MuJoCo assets keep native dynamics.
        Joints that rely on MuJoCo's implicit default ``(0.02, 1.0)`` keep that
        native default until ``joint_limit_ke`` / ``joint_limit_kd`` are changed.

        This must run **after** ``_update_joint_dof_properties`` writes the
        current ``jnt_solimp`` values and after MuJoCo refreshes
        ``dof_invweight0`` via ``set_const_0`` / ``mj_setConst`` on the
        current ``ModelBuilder`` / ``notify_model_changed`` cycle (and once
        right after ``put_model`` during initialisation).

        ``geom_solref`` is **not** scaled the same way: MuJoCo mixes the
        two contacting geoms' ``solref`` linearly in ``(timeconst,
        dampratio)`` space, which is non-linear in ``(ke, kd)``. Pre-scaling
        by ``body_invweight0 * (1 - dmax)`` works for a single dynamic
        geom but destroys the stiffness of dynamic-vs-static contacts,
        because the static geom keeps ``factor = 1`` and the mixed
        stiffness collapses. Shape-material contact stiffness therefore
        stays on MuJoCo's existing (unscaled) pathway.
        """
        njnt = self.mjc_jnt_to_newton_dof.shape[1]
        if njnt == 0 or self.model.joint_limit_ke is None:
            return

        mujoco_attrs = getattr(self.model, "mujoco", None)
        joint_limit_solref = getattr(mujoco_attrs, "solreflimit", None) if mujoco_attrs is not None else None
        joint_limit_solref_mode = getattr(mujoco_attrs, "solreflimit_mode", None) if mujoco_attrs is not None else None

        if joint_limit_solref_mode is not None:
            solref_mode_np = joint_limit_solref_mode.numpy()
            mjcf_default = solref_mode_np == SOLREF_MODE_MJCF_DEFAULT
            if np.any(mjcf_default):
                joint_limit_ke_np = self.model.joint_limit_ke.numpy()
                joint_limit_kd_np = self.model.joint_limit_kd.numpy()
                edited = mjcf_default & (
                    ~np.isclose(joint_limit_ke_np, DEFAULT_LIMIT_KE, rtol=DEFAULT_LIMIT_GAIN_RTOL, atol=0.0)
                    | ~np.isclose(joint_limit_kd_np, DEFAULT_LIMIT_KD, rtol=DEFAULT_LIMIT_GAIN_RTOL, atol=0.0)
                )
                if np.any(edited):
                    solref_mode_np = np.array(solref_mode_np, copy=True)
                    solref_mode_np[edited] = SOLREF_MODE_FORCE_SPACE
                    joint_limit_solref_mode.assign(solref_mode_np.astype(np.int32, copy=False))

        # Validate authored RAW ``mujoco.solreflimit`` values once per notify.
        # MuJoCo's solref domain is ``(timeconst > 0, dampratio > 0)`` for the
        # standard interpretation or ``(< 0, < 0)`` for the direct
        # stiffness/damping mode; mixed signs or a single zero component
        # silently disable the limit or trigger divide-by-zero in MuJoCo's
        # ``k_eff = 1/(τ²·ζ²)``. Surface the misconfiguration once via a
        # warning so users can correct the authored value; the kernel cannot
        # warn from inside Warp. The all-zero ``(0, 0)`` solref is the
        # documented MuJoCo "inherit the model default" sentinel and is
        # forwarded verbatim, so it is intentionally excluded here -- matching
        # the MJCF importer, which preserves it without warning.
        if (
            joint_limit_solref_mode is not None
            and joint_limit_solref is not None
            and not self._raw_solreflimit_validated
        ):
            mode_np = joint_limit_solref_mode.numpy()
            raw_np = joint_limit_solref.numpy()
            raw_mask = mode_np == SOLREF_MODE_RAW
            if np.any(raw_mask):
                tc = raw_np[raw_mask, 0]
                dr = raw_np[raw_mask, 1]
                # ``(0, 0)`` is the MuJoCo inherit-default sentinel, not a
                # misconfiguration; flag only a single zero or mixed signs.
                both_zero = (tc == 0.0) & (dr == 0.0)
                invalid = ((tc == 0.0) | (dr == 0.0) | (np.sign(tc) != np.sign(dr))) & ~both_zero
                if np.any(invalid):
                    bad = np.flatnonzero(raw_mask)[invalid]
                    warnings.warn(
                        f"Authored mujoco.solreflimit has invalid components at DOF indices "
                        f"{bad.tolist()}: expected two same-sign non-zero values; MuJoCo will "
                        "silently misbehave (divide-by-zero or disabled limit) until corrected.",
                        stacklevel=2,
                    )
            # One-shot guard: avoids warning every step for the steady-state
            # callers. The flag is re-armed only by ``notify_model_changed``
            # under ``ModelFlags.JOINT_DOF_PROPERTIES``, which is where
            # ``mujoco.solreflimit`` reassignments arrive; other
            # ``need_const_0`` notifies (BODY_INERTIAL_PROPERTIES, etc.) do
            # not reset it because they cannot change the authored solreflimit
            # values themselves.
            self._raw_solreflimit_validated = True

        if self.use_mujoco_cpu:
            joint_limit_ke = self.model.joint_limit_ke.numpy()
            joint_limit_kd = self.model.joint_limit_kd.numpy()
            joint_limit_solref_np = joint_limit_solref.numpy() if joint_limit_solref is not None else None
            joint_limit_solref_mode_np = (
                joint_limit_solref_mode.numpy() if joint_limit_solref_mode is not None else None
            )
            jnt_to_newton_dof = self.mjc_jnt_to_newton_dof.numpy()[0]
            jnt_solref = np.array(self.mj_model.jnt_solref, dtype=np.float64, copy=True)

            mode_present = joint_limit_solref_mode_np is not None
            for mjc_jnt, newton_dof in enumerate(jnt_to_newton_dof):
                if newton_dof < 0:
                    continue

                solref_mode = int(joint_limit_solref_mode_np[newton_dof]) if mode_present else SOLREF_MODE_FORCE_SPACE
                if joint_limit_solref_np is not None:
                    raw_solref = joint_limit_solref_np[newton_dof]
                    if mode_present:
                        if solref_mode == SOLREF_MODE_RAW:
                            jnt_solref[mjc_jnt] = raw_solref
                            continue
                    else:
                        if np.any(raw_solref != 0.0):
                            jnt_solref[mjc_jnt] = raw_solref
                            continue

                ke = float(joint_limit_ke[newton_dof])
                kd = float(joint_limit_kd[newton_dof])
                if (
                    solref_mode == SOLREF_MODE_MJCF_DEFAULT
                    and np.isclose(ke, DEFAULT_LIMIT_KE, rtol=DEFAULT_LIMIT_GAIN_RTOL, atol=0.0)
                    and np.isclose(kd, DEFAULT_LIMIT_KD, rtol=DEFAULT_LIMIT_GAIN_RTOL, atol=0.0)
                ):
                    jnt_solref[mjc_jnt] = DEFAULT_LIMIT_SOLREF
                    continue

                if ke <= 0.0 or kd <= 0.0:
                    # Restore MuJoCo's compiled default so a zero-gain
                    # configuration matches a fresh model with no authored
                    # ``solreflimit``. A ``(ke>0, kd=0)`` pair would otherwise
                    # produce an infinite time constant in the positive
                    # solref conversion.
                    jnt_solref[mjc_jnt] = DEFAULT_LIMIT_SOLREF
                    continue

                dof_idx = int(self.mj_model.jnt_dofadr[mjc_jnt])
                invw = float(self.mj_model.dof_invweight0[dof_idx])
                dmax = float(self.mj_model.jnt_solimp[mjc_jnt][1])
                factor = invw * (1.0 - dmax) if invw > 0.0 and dmax < 1.0 else 1.0
                direct_stiffness = max(ke * factor, MJ_MINVAL)
                direct_damping = max(kd * factor, MJ_MINVAL)
                solref = convert_solref(direct_stiffness, direct_damping, 1.0, 1.0)
                jnt_solref[mjc_jnt] = (float(solref[0]), float(solref[1]))

            self.mj_model.jnt_solref[:] = jnt_solref
            self.mjw_model.jnt_solref.assign(jnt_solref.reshape(1, njnt, 2))
            return

        nworld = self.mjc_jnt_to_newton_dof.shape[0]
        wp.launch(
            update_jnt_solref_from_invweight0_kernel,
            dim=(nworld, njnt),
            inputs=[
                self.mjc_jnt_to_newton_dof,
                self.model.joint_limit_ke,
                self.model.joint_limit_kd,
                joint_limit_solref,
                joint_limit_solref_mode,
                self.mjw_model.jnt_dofadr,
                self.mjw_model.dof_invweight0,
                self.mjw_model.jnt_solimp,
            ],
            outputs=[self.mjw_model.jnt_solref],
            device=self.model.device,
        )
        self.mj_model.jnt_solref[:] = self.mjw_model.jnt_solref.numpy()[0]

    def _update_pair_properties(self):
        """Update MuJoCo contact pair properties from Newton custom attributes.

        Updates the randomizable pair properties (solref, solreffriction, solimp,
        margin, gap, friction) for explicit contact pairs defined in the model.
        """
        if self.use_mujoco_cpu:
            return  # CPU mode not supported for pair runtime updates

        npair = self.mj_model.npair
        if npair == 0:
            return

        # Get custom attributes for pair properties
        mujoco_attrs = getattr(self.model, "mujoco", None)
        if mujoco_attrs is None:
            return

        pair_solref = getattr(mujoco_attrs, "pair_solref", None)
        pair_solreffriction = getattr(mujoco_attrs, "pair_solreffriction", None)
        pair_solimp = getattr(mujoco_attrs, "pair_solimp", None)
        # Restore pair margin/gap at runtime: the spec carries only template-world
        # values, so per-world variance must be reapplied. margin is suppressed only
        # under NATIVECCD/MULTICCD (#2106); gap is always forwarded (MuJoCo 3.9).
        pair_margin = (
            None
            if (self._use_mujoco_contacts and self._zero_margins_for_native_ccd)
            else getattr(mujoco_attrs, "pair_margin", None)
        )
        pair_gap = getattr(mujoco_attrs, "pair_gap", None)
        pair_friction = getattr(mujoco_attrs, "pair_friction", None)

        # Only launch kernel if at least one attribute is defined
        if any(
            attr is not None
            for attr in [pair_solref, pair_solreffriction, pair_solimp, pair_margin, pair_gap, pair_friction]
        ):
            # Compute pairs_per_world from Newton custom attributes
            pair_world_attr = getattr(mujoco_attrs, "pair_world", None)
            if pair_world_attr is not None:
                total_pairs = len(pair_world_attr)
                pairs_per_world = total_pairs // self.model.world_count
            else:
                pairs_per_world = npair

            world_count = self.mjw_data.nworld

            wp.launch(
                update_pair_properties_kernel,
                dim=(world_count, npair),
                inputs=[
                    pairs_per_world,
                    pair_solref,
                    pair_solreffriction,
                    pair_solimp,
                    pair_margin,
                    pair_gap,
                    pair_friction,
                ],
                outputs=[
                    self.mjw_model.pair_solref,
                    self.mjw_model.pair_solreffriction,
                    self.mjw_model.pair_solimp,
                    self.mjw_model.pair_margin,
                    self.mjw_model.pair_gap,
                    self.mjw_model.pair_friction,
                ],
                device=self.model.device,
            )

    def _update_model_properties(self):
        """Update model properties including gravity in the MuJoCo model."""
        if self.use_mujoco_cpu:
            self.mj_model.opt.gravity[:] = np.array([*self.model.gravity.numpy()[0]])
        else:
            if hasattr(self, "mjw_data"):
                wp.launch(
                    kernel=update_model_properties_kernel,
                    dim=self.mjw_data.nworld,
                    inputs=[
                        self.model.gravity,
                    ],
                    outputs=[
                        self.mjw_model.opt.gravity,
                    ],
                    device=self.model.device,
                )

    def _update_eq_properties(self):
        """Update equality constraint properties in the MuJoCo model.

        Updates:

        - eq_solref/eq_solimp from MuJoCo custom attributes (if set)
        - eq_data from model.mujoco equality_constraint_anchor, equality_constraint_relpose,
          equality_constraint_polycoef, equality_constraint_torquescale
        - eq_active from model.mujoco equality_constraint_enabled

        .. note::

            This update affects Newton equality rows, including MuJoCo equalities
            that were projected to loop joints or mimic constraints during import.
            Generic loop closures synthesized directly from loop joints are updated
            by the joint-connect path."""
        if self.model.mujoco.equality_constraint_count == 0:
            return

        neq = self.mj_model.neq
        if neq == 0:
            return

        world_count = self.mjc_eq_to_newton_eq.shape[0]

        # Get custom attributes for eq_solref and eq_solimp
        mujoco_attrs = getattr(self.model, "mujoco", None)
        eq_solref = getattr(mujoco_attrs, "eq_solref", None) if mujoco_attrs is not None else None
        eq_solimp = getattr(mujoco_attrs, "eq_solimp", None) if mujoco_attrs is not None else None

        if eq_solref is not None or eq_solimp is not None:
            wp.launch(
                update_eq_properties_kernel,
                dim=(world_count, neq),
                inputs=[
                    self.mjc_eq_to_newton_eq,
                    eq_solref,
                    eq_solimp,
                ],
                outputs=[
                    self.mjw_model.eq_solref,
                    self.mjw_model.eq_solimp,
                ],
                device=self.model.device,
            )

        # Update eq_data and eq_active from namespaced Newton equality constraint properties
        wp.launch(
            update_eq_data_and_active_kernel,
            dim=(world_count, neq),
            inputs=[
                self.mjc_eq_to_newton_eq,
                self.model.mujoco.equality_constraint_type,
                self.model.mujoco.equality_constraint_anchor,
                self.model.mujoco.equality_constraint_relpose,
                self.model.mujoco.equality_constraint_polycoef,
                self.model.mujoco.equality_constraint_torquescale,
                self.model.mujoco.equality_constraint_enabled,
            ],
            outputs=[
                self.mjw_model.eq_data,
                self.mjw_data.eq_active,
            ],
            device=self.model.device,
        )

    def _update_mimic_eq_properties(self):
        """Update mimic constraint properties in the MuJoCo model.

        Updates:

        - eq_data from Newton's constraint_mimic_coef0, constraint_mimic_coef1
        - eq_active from Newton's constraint_mimic_enabled

        Maps mimic constraints to MuJoCo mjEQ_JOINT equality constraints
        using the polycoef representation: q1 = coef0 + coef1 * q2.
        """
        if self.model.constraint_mimic_count == 0 or self.mjc_eq_to_newton_mimic is None:
            return

        neq = self.mj_model.neq
        if neq == 0:
            return

        world_count = self.mjc_eq_to_newton_mimic.shape[0]

        wp.launch(
            update_mimic_eq_data_and_active_kernel,
            dim=(world_count, neq),
            inputs=[
                self.mjc_eq_to_newton_mimic,
                self.model.constraint_mimic_coef0,
                self.model.constraint_mimic_coef1,
                self.model.constraint_mimic_enabled,
            ],
            outputs=[
                self.mjw_model.eq_data,
                self.mjw_data.eq_active,
            ],
            device=self.model.device,
        )

    def _update_tendon_properties(self):
        """Update fixed tendon properties in the MuJoCo model.

        Updates tendon stiffness, damping, frictionloss, range, margin, solref, solimp,
        armature, and actfrcrange from Newton custom attributes.
        """
        if self.mjc_tendon_to_newton_tendon is None:
            return

        ntendon = self.mj_model.ntendon
        if ntendon == 0:
            return

        world_count = self.mjc_tendon_to_newton_tendon.shape[0]

        # Get custom attributes for tendons
        mujoco_attrs = getattr(self.model, "mujoco", None)
        if mujoco_attrs is None:
            return

        # Get tendon custom attributes (may be None if not defined)
        # Note: tendon_springlength is NOT updated at runtime because it has special
        # initialization semantics in MuJoCo (value -1.0 means auto-compute from initial state).
        tendon_stiffness = getattr(mujoco_attrs, "tendon_stiffness", None)
        tendon_damping = getattr(mujoco_attrs, "tendon_damping", None)
        tendon_frictionloss = getattr(mujoco_attrs, "tendon_frictionloss", None)
        tendon_range = getattr(mujoco_attrs, "tendon_range", None)
        tendon_margin = getattr(mujoco_attrs, "tendon_margin", None)
        tendon_solref_limit = getattr(mujoco_attrs, "tendon_solref_limit", None)
        tendon_solimp_limit = getattr(mujoco_attrs, "tendon_solimp_limit", None)
        tendon_solref_friction = getattr(mujoco_attrs, "tendon_solref_friction", None)
        tendon_solimp_friction = getattr(mujoco_attrs, "tendon_solimp_friction", None)
        tendon_armature = getattr(mujoco_attrs, "tendon_armature", None)
        tendon_actfrcrange = getattr(mujoco_attrs, "tendon_actuator_force_range", None)

        wp.launch(
            update_tendon_properties_kernel,
            dim=(world_count, ntendon),
            inputs=[
                self.mjc_tendon_to_newton_tendon,
                tendon_stiffness,
                tendon_damping,
                tendon_frictionloss,
                tendon_range,
                tendon_margin,
                tendon_solref_limit,
                tendon_solimp_limit,
                tendon_solref_friction,
                tendon_solimp_friction,
                tendon_armature,
                tendon_actfrcrange,
            ],
            outputs=[
                self.mjw_model.tendon_stiffness,
                self.mjw_model.tendon_damping,
                self.mjw_model.tendon_frictionloss,
                self.mjw_model.tendon_range,
                self.mjw_model.tendon_margin,
                self.mjw_model.tendon_solref_lim,
                self.mjw_model.tendon_solimp_lim,
                self.mjw_model.tendon_solref_fri,
                self.mjw_model.tendon_solimp_fri,
                self.mjw_model.tendon_armature,
                self.mjw_model.tendon_actfrcrange,
            ],
            device=self.model.device,
        )

    def _update_actuator_properties(self):
        """Update CTRL_DIRECT actuator properties in the MuJoCo model.

        Only updates actuators that use CTRL_DIRECT mode. JOINT_TARGET actuators are
        updated via _update_joint_dof_properties() using joint_target_ke/kd.
        """
        if self.mjc_actuator_ctrl_source is None or self.mjc_actuator_to_newton_idx is None:
            return

        nu = self.mjc_actuator_ctrl_source.shape[0]
        if nu == 0:
            return

        mujoco_attrs = getattr(self.model, "mujoco", None)
        if mujoco_attrs is None:
            return

        actuator_gainprm = getattr(mujoco_attrs, "actuator_gainprm", None)
        actuator_biasprm = getattr(mujoco_attrs, "actuator_biasprm", None)
        actuator_dynprm = getattr(mujoco_attrs, "actuator_dynprm", None)
        actuator_ctrlrange = getattr(mujoco_attrs, "actuator_ctrlrange", None)
        actuator_forcerange = getattr(mujoco_attrs, "actuator_forcerange", None)
        actuator_actrange = getattr(mujoco_attrs, "actuator_actrange", None)
        actuator_gear = getattr(mujoco_attrs, "actuator_gear", None)
        actuator_cranklength = getattr(mujoco_attrs, "actuator_cranklength", None)
        if (
            actuator_gainprm is None
            or actuator_biasprm is None
            or actuator_dynprm is None
            or actuator_ctrlrange is None
            or actuator_forcerange is None
            or actuator_actrange is None
            or actuator_gear is None
            or actuator_cranklength is None
        ):
            return

        nworld = self.mjw_model.actuator_biasprm.shape[0]
        actuators_per_world = actuator_gainprm.shape[0] // nworld if nworld > 0 else actuator_gainprm.shape[0]

        wp.launch(
            update_ctrl_direct_actuator_properties_kernel,
            dim=(nworld, nu),
            inputs=[
                self.mjc_actuator_ctrl_source,
                self.mjc_actuator_to_newton_idx,
                actuator_gainprm,
                actuator_biasprm,
                actuator_dynprm,
                actuator_ctrlrange,
                actuator_forcerange,
                actuator_actrange,
                actuator_gear,
                actuator_cranklength,
                actuators_per_world,
            ],
            outputs=[
                self.mjw_model.actuator_gainprm,
                self.mjw_model.actuator_biasprm,
                self.mjw_model.actuator_dynprm,
                self.mjw_model.actuator_ctrlrange,
                self.mjw_model.actuator_forcerange,
                self.mjw_model.actuator_actrange,
                self.mjw_model.actuator_gear,
                self.mjw_model.actuator_cranklength,
            ],
            device=self.model.device,
        )

    def _validate_model_for_separate_worlds(self, model: Model) -> None:
        """Validate that the Newton model is compatible with MuJoCo's separate_worlds mode.

        MuJoCo's separate_worlds mode creates identical copies of a single MuJoCo model
        for each Newton world. This requires:
        1. All worlds have the same number of bodies, joints, shapes, and equality constraints
        2. Entity types match across corresponding entities in each world
        3. Global world (-1) only contains static shapes (no bodies, joints, or constraints)

        Args:
            model: The Newton model to validate.

        Raises:
            ValueError: If the model is not compatible with separate_worlds mode.
        """
        world_count = model.world_count

        # Check that we have at least one world
        if world_count == 0:
            raise ValueError(
                "SolverMuJoCo with separate_worlds=True requires at least one world (world_count >= 1). "
                "Found world_count=0 (all entities in global world -1)."
            )

        body_world = model.body_world.numpy()
        joint_world = model.joint_world.numpy()
        shape_world = model.shape_world.numpy()
        # finalize() materializes this as an empty array at zero rows; guard on the count anyway
        # so models assembled without the standard pipeline still work.
        eq_constraint_world = (
            model.mujoco.equality_constraint_world.numpy()
            if model.mujoco.equality_constraint_count > 0
            else np.empty(0, dtype=np.int32)
        )

        # --- Check global world restrictions (always, regardless of world_count) ---
        # No bodies in global world
        global_bodies = np.where(body_world == -1)[0]
        if len(global_bodies) > 0:
            body_names = [model.body_label[i] for i in global_bodies[:3]]
            msg = f"Global world (-1) cannot contain bodies. Found {len(global_bodies)} body(ies) with world == -1"
            if body_names:
                msg += f": {body_names}"
            raise ValueError(msg)

        # No joints in global world
        global_joints = np.where(joint_world == -1)[0]
        if len(global_joints) > 0:
            joint_names = [model.joint_label[i] for i in global_joints[:3]]
            msg = f"Global world (-1) cannot contain joints. Found {len(global_joints)} joint(s) with world == -1"
            if joint_names:
                msg += f": {joint_names}"
            raise ValueError(msg)

        # No equality constraints in global world
        global_constraints = np.where(eq_constraint_world == -1)[0]
        if len(global_constraints) > 0:
            msg = f"Global world (-1) cannot contain equality constraints. Found {len(global_constraints)} constraint(s) with world == -1"
            raise ValueError(msg)

        # No mimic constraints in global world
        mimic_world = model.constraint_mimic_world.numpy()
        global_mimic = np.where(mimic_world == -1)[0]
        if len(global_mimic) > 0:
            msg = f"Global world (-1) cannot contain mimic constraints. Found {len(global_mimic)} constraint(s) with world == -1"
            raise ValueError(msg)

        # Skip homogeneity checks for single-world models
        if world_count <= 1:
            return

        # --- Check entity count homogeneity ---
        # Count entities per world (excluding global shapes)
        non_global_shapes = shape_world[shape_world >= 0]

        for entity_name, world_arr in [
            ("bodies", body_world),
            ("joints", joint_world),
            ("shapes", non_global_shapes),
            ("equality constraints", eq_constraint_world),
            ("mimic constraints", mimic_world),
        ]:
            # Use bincount for O(n) counting instead of O(n * world_count) loop
            if len(world_arr) == 0:
                continue
            counts = np.bincount(world_arr.astype(np.int64), minlength=world_count)
            # Vectorized check: all counts must equal the first
            if not np.all(counts == counts[0]):
                # Find first mismatch for error message (only on failure path)
                expected = counts[0]
                mismatched = np.where(counts != expected)[0]
                w = mismatched[0]
                raise ValueError(
                    f"SolverMuJoCo requires homogeneous worlds. "
                    f"World 0 has {expected} {entity_name}, but world {w} has {counts[w]}."
                )

        # --- Check type matching across worlds (vectorized) ---
        # Load type arrays lazily - only when needed for validation
        joints_per_world = model.joint_count // world_count
        if joints_per_world > 0:
            joint_type = model.joint_type.numpy()
            joint_types_2d = joint_type.reshape(world_count, joints_per_world)
            # Vectorized mismatch check: compare all rows to first row
            mismatches = joint_types_2d != joint_types_2d[0]
            if np.any(mismatches):
                # Find first mismatch position using vectorized operations
                j = np.argmax(np.any(mismatches, axis=0))
                types = joint_types_2d[:, j]
                raise ValueError(
                    f"SolverMuJoCo requires homogeneous worlds. "
                    f"Joint types mismatch at position {j}: world 0 has type {types[0]}, "
                    f"but other worlds have types {types[1:].tolist()}."
                )

        # Only check non-global shapes
        shapes_per_world = len(non_global_shapes) // world_count if world_count > 0 else 0
        if shapes_per_world > 0:
            shape_type = model.shape_type.numpy()
            # Get shape types for non-global shapes only
            non_global_shape_types = shape_type[shape_world >= 0]
            shape_types_2d = non_global_shape_types.reshape(world_count, shapes_per_world)
            # Vectorized mismatch check
            mismatches = shape_types_2d != shape_types_2d[0]
            if np.any(mismatches):
                s = np.argmax(np.any(mismatches, axis=0))
                types = shape_types_2d[:, s]
                raise ValueError(
                    f"SolverMuJoCo requires homogeneous worlds. "
                    f"Shape types mismatch at position {s}: world 0 has type {types[0]}, "
                    f"but other worlds have types {types[1:].tolist()}."
                )

        constraints_per_world = (model.mujoco.equality_constraint_count // world_count) if world_count > 0 else 0
        if constraints_per_world > 0:
            eq_constraint_type = model.mujoco.equality_constraint_type.numpy()
            constraint_types_2d = eq_constraint_type.reshape(world_count, constraints_per_world)
            # Vectorized mismatch check
            mismatches = constraint_types_2d != constraint_types_2d[0]
            if np.any(mismatches):
                c = np.argmax(np.any(mismatches, axis=0))
                types = constraint_types_2d[:, c]
                raise ValueError(
                    f"SolverMuJoCo requires homogeneous worlds. "
                    f"Equality constraint types mismatch at position {c}: world 0 has type {types[0]}, "
                    f"but other worlds have types {types[1:].tolist()}."
                )

    def render_mujoco_viewer(
        self,
        show_left_ui: bool = True,
        show_right_ui: bool = True,
        show_contact_points: bool = True,
        show_contact_forces: bool = False,
        show_transparent_geoms: bool = True,
    ):
        """Create and synchronize the MuJoCo viewer.
        The viewer will be created if it is not already open.

        .. note::

            The MuJoCo viewer only supports rendering Newton models with a single world,
            unless ``use_mujoco_cpu`` is ``True`` or the solver was initialized with
            ``separate_worlds`` set to ``False``.

            The MuJoCo viewer is only meant as a debugging tool.

        Args:
            show_left_ui: Whether to show the left UI.
            show_right_ui: Whether to show the right UI.
            show_contact_points: Whether to show contact points.
            show_contact_forces: Whether to show contact forces.
            show_transparent_geoms: Whether to show transparent geoms.
        """
        if self._viewer is None:
            import mujoco.viewer

            mujoco = self._mujoco

            # make the headlights brighter to improve visibility
            # in the MuJoCo viewer
            self.mj_model.vis.headlight.ambient[:] = [0.3, 0.3, 0.3]
            self.mj_model.vis.headlight.diffuse[:] = [0.7, 0.7, 0.7]
            self.mj_model.vis.headlight.specular[:] = [0.9, 0.9, 0.9]

            self._viewer = mujoco.viewer.launch_passive(
                self.mj_model, self.mj_data, show_left_ui=show_left_ui, show_right_ui=show_right_ui
            )
            # Enter the context manager to keep the viewer alive
            self._viewer.__enter__()

            self._viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = show_contact_points
            self._viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = show_contact_forces
            self._viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = show_transparent_geoms

        if self._viewer.is_running():
            if not self.use_mujoco_cpu:
                with wp.ScopedDevice(self.model.device):
                    self._mujoco_warp.get_data_into(self.mj_data, self.mj_model, self.mjw_data)

            self._viewer.sync()

    def close_mujoco_viewer(self):
        """Close the MuJoCo viewer if it exists."""
        if hasattr(self, "_viewer") and self._viewer is not None:
            try:
                self._viewer.__exit__(None, None, None)
            except Exception:
                pass  # Ignore errors during cleanup
            finally:
                self._viewer = None

    def __del__(self):
        """Cleanup method to close the viewer when the solver is destroyed."""
        self.close_mujoco_viewer()
