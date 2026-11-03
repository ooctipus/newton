# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The tree factor: each articulation's ``P (H + R + K) P^T = L L^T``, the predictor, and the factor's maps of
velocities (BACKBONE §2).

:class:`TreeFactor` owns what the factor reads and writes across a step:

- the augmented state, the step's ``state_aug`` (joint forces, motion axes, body twists and wrenches);
- the kinematic masks and the effective armature R (a kinematic coordinate holds 1e10 armature);
- the drive topology and its rows K, prepared per coordinate apart from inverse dynamics;
- the walk of every held tree (:class:`WalkTrees`, its refresh the factor), the compact diagonal's inverse mass
  and where each body's tree and coordinates live (:class:`.kernels.BodyTrees`);
- the body inertia records, the articulation origins and the passive joint forces;
- the mass-update cadence, its requests and the FK/ID cache.

Its evaluator is the held tree (:mod:`.held_tree`). A step calls :meth:`TreeFactor.factor` (FK/ID, the drives' K and
u0, the mass-update mask, the factor refresh), :meth:`TreeFactor.predict` (``v_hat`` and the drives' saturation),
and :meth:`TreeFactor.finish` (``qdd``, integrate, publish); the walk's rows and leaves read its walk
(:meth:`TreeFactor.plan_walk`).
"""

from __future__ import annotations

import functools
import hashlib
import warnings
from string import Template

import numpy as np
import warp as wp

from ...sim import BodyFlags, JointType, ModelFlags
from . import field_group, host_index, response_builds_rows
from . import tree_scan as ts
from .conditioning import ArmatureFloor, conditioning_armature
from .device_preludes import DevicePrelude, substitute
from .field_group import SolverArrays
from .joint_table import JointTable
from .kernels import (
    BodyTrees,
    apply_free_root_velocity_corrections,
    build_mass_update_mask,
    compute_com_transforms,
    damp_free_root_spin,
    invalidate_articulation_fk_id_cache,
    invalidate_fk_id_cache_for_articulations,
    prescale_joint_velocity_limits,
)
from .row_laws import LINK_ROWS
from .solver_fields import (
    ARTICULATION_ORIGIN,
    ARTICULATION_WORK,
    AUGMENTED_DRIVE_ROWS,
    BODY_INERTIA,
    PREELIMINATION,
    PREELIMINATION_PLAN,
    STATE_AUX,
    DriveRows,
    Preelim,
    hold_drive_row,
)
from .spatial_laws import drive_force, drive_stiffness, spring_damper_stiffness
from .tree_plan import TreePlan
from .tree_scan import (
    WALK_AHEAD,
    WALK_ANGULAR,
    WALK_BOTH,
    WALK_LINEAR,
    WALK_TOUCH,
    WALK_UNROLL,
    WalkClass,
    indent_text,
)

__all__ = [
    "FREE_BODY_WALK",
    "LEAVE_LANES",
    "WALK_COORDINATE",
    "WALK_FRAME",
    "WALK_LINK",
    "WALK_NOW",
    "WALK_PIVOT",
    "WALK_UNPACKED",
    "TreeFactor",
    "prepare_augmented_drives",
    "walk_down_prelude",
    "walk_leave_kernel",
    "walk_paths_prelude",
    "walk_prelude",
    "walk_row_prelude",
]


@wp.kernel
def prepare_augmented_drives(
    drive_row_by_dof: wp.array[int],
    drive_q_index_by_dof: wp.array[int],
    drive_target_index_by_dof: wp.array[int],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_target_pos: wp.array[float],
    joint_target_vel: wp.array[float],
    joint_effort_limit: wp.array[float],
    passive_damping: wp.array[float],
    max_dofs: int,
    dt: float,
    rows: DriveRows,
    drive_tau: wp.array[float],
):
    """Each coordinate's augmented drive, ``K`` (:func:`.spatial_laws.drive_stiffness`, its passive damper's with it) and
    ``u0`` (:func:`.spatial_laws.drive_force`), from the immutable DOF-to-row topology, held on its row
    (:func:`.solver_fields.hold_drive_row`)."""
    dof = wp.tid()
    drive_tau[dof] = 0.0
    row = drive_row_by_dof[dof]
    if row < 0:
        return
    ke = joint_target_ke[dof]
    kd = joint_target_kd[dof]
    K = drive_stiffness(ke, kd, passive_damping[dof], dt)
    if K <= 0.0:
        rows.row_K[row] = 0.0
        rows.drive[dof] = wp.vec4()
        return
    u0 = drive_force(
        ke,
        kd,
        joint_q[drive_q_index_by_dof[dof]],
        joint_qd[dof],
        joint_target_pos[drive_target_index_by_dof[dof]],
        joint_target_vel[dof],
        dt,
    )
    passive = spring_damper_stiffness(0.0, passive_damping[dof], 0.0, dt)
    drive_tau[dof] = hold_drive_row(rows, row, row // max_dofs, dof, u0, K, passive, joint_effort_limit[dof])


class TreeFactor:
    """The tree factor of one solver (see the module docstring); the solver holds it as ``factor``.

    Built in the solver's construction order: the cadence, the cache and the augmented state here, then
    :meth:`kinematic_state` and :meth:`apply_kinematic_state`, :meth:`plan` once the forest plan and the row laws
    exist, :meth:`allocate_common`, :meth:`build_body_trees`,
    :meth:`scatter_armature` and :meth:`install`.

    Args:
        model: The solver's model.
        options: The solver's physics options (:class:`.configuration.Options`).
        nvtx: Whether the stages' timers emit NVTX ranges.
    """

    def __init__(self, model, options, *, nvtx: bool):
        self.model = model
        self.options = options
        self.nvtx = nvtx
        self.forest = None
        self.laws = None
        self._arrays = SolverArrays(self, model.device)
        self.force_mass_update = False
        self.mass_update_requested = wp.zeros(model.articulation_count, dtype=wp.int32, device=model.device)
        # Host mirror of the global mass-update flag the last factor() evaluated; it gates the per-step H memsets.
        self.mass_update_global_flag = True
        self.last_step_dt = None
        # The held tree's cache of next-step dynamics.
        self.fk_id_cache_source_state = None
        self.fk_id_cache_valid = wp.zeros(model.articulation_count, dtype=wp.int32, device=model.device)
        # The augmented state is allocated here rather than on the first step: a wp.zeros() issued inside a CUDA
        # graph capture becomes a memset node that replays every launch and wiped the cached bias forces (body_f_s)
        # at the first substep of each replay.
        if model.body_count:
            self._arrays.allocate(STATE_AUX, dof=model.joint_dof_count, body=model.body_count)
        self.kinematic_joint_mask = wp.zeros(model.joint_count, dtype=wp.int32, device=model.device)
        self.kinematic_dof_mask = wp.zeros(model.joint_dof_count, dtype=wp.int32, device=model.device)
        self.kinematic_dof_mask_host = None
        self.armature = None
        self.armature_floor = None
        # The effective armature by coordinate (scatter_armature): the walk's diagonal and the drives' K-free mass.
        self.dof_armature = wp.zeros(max(1, int(model.joint_dof_count)), dtype=wp.float32, device=model.device)
        self.diagonal_inverse_mass = None
        self.body_trees = None
        self.eliminates = False
        self.block_capacity = 0
        self.block_count = 0
        self.held = None
        self.dynamics_stream = None
        self.walk_arts = None
        self.walk_classes: tuple[WalkClass, ...] = ()
        self.walk_half_host = None
        self.walk_side = True
        self.walk_class = None
        self.walk_leave = None
        self.walk_slip_capacity = 0
        # The walk's step geometry, which the finish publishes where a walk is held (plan_walk sizes it): empty here.
        self.walk_now = wp.zeros(0, dtype=wp.spatial_vector, device=model.device)
        self.walk_half = None
        self.walk_now_link = wp.zeros((0, WALK_LINK), dtype=wp.float32, device=model.device)
        # Each body's centre of mass relative to its articulation's origin, by its joint, as the held tree's finish
        # publishes it (its predictor's wrench shift and the walk's refresh read it).
        self.joint_com_offset = wp.zeros(max(1, int(model.joint_count)), dtype=wp.vec3, device=model.device)
        # The border's status word (tree_kinetic_state.border_solve): bit 0 a tree past its slip capacity, bit 1 a
        # pivot not positive, bit 2 a decided step on a tree the walk does not hold.
        self.walk_release_status = wp.zeros(1, dtype=wp.int32, device=model.device)

    # ---------------------------------------------------------------------------------------------------------------
    # Construction
    # ---------------------------------------------------------------------------------------------------------------

    def kinematic_state(
        self, joint_forest: TreePlan, *, sparse_velocity_limits: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, ArmatureFloor]:
        """Derive the kinematic masks and effective joint armature from the body flags, changing nothing.

        The effective armature also holds the least armature that keeps the mass matrix of a deep
        articulation within float32 factorization (:func:`conditioning_armature`).

        The coordinate mask is the rows' prescribed coordinates (:data:`.joint_table.JOINTS`): none takes a row on
        it alone.

        Args:
            joint_forest: The model's joint forest (:func:`.forest_plan.joint_forest`).
            sparse_velocity_limits: Whether the native sparse factor carries velocity-limit rows.

        Returns:
            ``(armature, joint_mask, dof_mask, floor)`` for :meth:`apply_kinematic_state`.

        Raises:
            RuntimeError: If the flags make a DOF of a sparse split or
                velocity-limit response kinematic, or a DOF of a
                prescribed-response articulation dynamic. Both need a new solver.
        """
        model = self.model
        armature = model.joint_armature.numpy().copy()
        joint_mask = np.zeros(model.joint_count, dtype=np.int32)
        dof_mask = np.zeros(model.joint_dof_count, dtype=np.int32)

        if model.body_count and model.joint_count:
            kinematic_bodies = (model.body_flags.numpy() & int(BodyFlags.KINEMATIC)) != 0
            joint_child = model.joint_child.numpy()
            joint_qd_start = model.joint_qd_start.numpy()
            kinematic_joints = np.flatnonzero(kinematic_bodies[joint_child])
            joint_mask[kinematic_joints] = 1
            dof_start = joint_qd_start[kinematic_joints]
            joint, offset = host_index.expand(np.maximum(joint_qd_start[kinematic_joints + 1] - dof_start, 0))
            dof_mask[dof_start[joint] + offset] = 1
            armature[dof_start[joint] + offset] = 1.0e10

        if sparse_velocity_limits and np.any(dof_mask):
            raise RuntimeError(
                "Sparse response gained prescribed DOFs; reconstruct the solver and recapture its CUDA graph."
            )

        if self.forest is not None:
            selected = np.nonzero(self.forest.prescribed_articulation_host != 0)[0]
            for articulation in selected:
                dof_start = int(self.forest.articulation_dof_start_host[articulation])
                dof_count = int(self.forest.articulation_dof_count_host[articulation])
                if not np.all(dof_mask[dof_start : dof_start + dof_count] != 0):
                    raise RuntimeError(
                        "A prescribed-response articulation changed kinematic membership; "
                        "reconstruct the solver and recapture its CUDA graph."
                    )
        armature, floor = conditioning_armature(model, armature, joint_forest)
        return armature, joint_mask, dof_mask, floor

    def apply_kinematic_state(self, kinematic: tuple[np.ndarray, np.ndarray, np.ndarray, ArmatureFloor]) -> None:
        """Publish derived kinematic masks and armature in place, where captured graphs read them."""
        armature, joint_mask, dof_mask, floor = kinematic
        if floor.coordinates and floor != self.armature_floor:
            warnings.warn(
                f"SolverFeatherPGS adds joint armature, up to {floor.largest:.3g}, to {floor.coordinates} "
                f"coordinate(s) of {floor.articulations} articulation(s) whose mass matrix float32 cannot factor "
                "without it (routes.dynamics.armature_floor). Give these joints armature to choose it yourself.",
                UserWarning,
                stacklevel=3,
            )
        self.armature_floor = floor
        self.armature = armature
        self.kinematic_dof_mask_host = dof_mask.copy()
        self.kinematic_joint_mask.assign(joint_mask)
        self.kinematic_dof_mask.assign(dof_mask)

    def plan(self, forest, laws) -> None:
        """Take the forest plan and the row laws, then select the passive joint forces and the drive topology.

        The held tree's per-coordinate drive rows (each coordinate owns one, prepared apart from inverse dynamics)
        hold wherever each coordinate has one owning drive row.
        """
        self.forest = forest
        self.laws = laws
        self._setup_passive_joint_forces()
        self.setup_drives(forest.articulation_max_dofs > 0)

    def _setup_passive_joint_forces(self) -> None:
        """Select the passive spring/damping arrays applied in the inverse-dynamics pass.

        The toggles resolve here, once, to fixed device arrays (zeros when disabled or
        absent on the model), so inverse-dynamics launches stay CUDA-graph-safe. Springs
        on unsupported DOF types (BALL / FREE / DISTANCE coordinates) warn and are masked
        off — only REVOLUTE, PRISMATIC, and D6 DOFs receive the explicit spring torque.
        """
        model = self.model
        device = model.device
        n_dofs = int(model.joint_dof_count)

        def _zeros():
            return self._arrays.placeholder(wp.float32, (n_dofs,))

        spring_k = getattr(model, "joint_spring_stiffness", None)
        spring_ref = getattr(model, "joint_spring_ref", None)
        if spring_k is None or spring_ref is None:
            self.passive_spring_stiffness = _zeros()
            self.passive_spring_ref = _zeros()
        else:
            k_np = spring_k.numpy()
            if np.any(k_np != 0.0) and model.joint_count:
                # Mask springs on unsupported joint types (kernel applies them per 1-DoF axis).
                supported = (int(JointType.REVOLUTE), int(JointType.PRISMATIC), int(JointType.D6))
                joint_type = model.joint_type.numpy()
                # joint_qd_start is sentinel-closed (length joint_count + 1).
                dof_joint, _ = host_index.expand(np.diff(model.joint_qd_start.numpy()))
                bad = (k_np != 0.0) & ~np.isin(joint_type[dof_joint], supported)
                if np.any(bad):
                    warnings.warn(
                        f"SolverFeatherPGS: passive joint springs on {int(bad.sum())} DOF(s) of "
                        "unsupported joint types (only REVOLUTE/PRISMATIC/D6 are supported); "
                        "those springs are ignored.",
                        stacklevel=3,
                    )
                    k_np = k_np.copy()
                    k_np[bad] = 0.0
                    spring_k = wp.array(k_np, dtype=wp.float32, device=device)
            self.passive_spring_stiffness = spring_k
            self.passive_spring_ref = spring_ref

        damping = getattr(model, "joint_damping", None)
        if damping is None:
            self.passive_joint_damping = _zeros()
        else:
            self.passive_joint_damping = damping

    def plan_blocks(self, forest, laws) -> None:
        """The bilateral blocks' tables (the closure, :func:`.response_builds_rows.close_block`).

        A block holds a linked component's constraints: its trees' mimics and its links' six rows each. Its buffers
        are sized once from the model's largest block (CUDA-graph-safe); its slots and its factor are rebuilt every
        step on the device, since the rows' slots and ``J_B`` depend on the state.
        """
        cut = forest.cut
        links = int(cut.primary.size)
        if not links and not (self.options.enable_bilateral_preelimination and laws.mimic_count):
            return
        model, arts = self.model, self.model.articulation_count
        component = np.arange(arts) if cut.component is None else cut.component
        mimic_art, mimics = np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        if laws.mimic_count:
            mimic_art, _ = host_index.expand(np.diff(laws.mimic_art_start_np))
            mimics = laws._mimic_art_list.numpy().astype(np.int64)
        owner = np.concatenate((component[mimic_art], component[cut.tree_shadow]))
        if not owner.size:
            return
        entries = np.concatenate((mimics, -np.arange(links, dtype=np.int64) - 1))
        rows = np.concatenate((np.ones(mimics.size, dtype=np.int64), np.full(links, LINK_ROWS, dtype=np.int64)))
        heads, block = np.unique(owner, return_inverse=True)
        # Blocks ordered by world: the one writer's lanes of a world close its blocks (each world's range, CSR).
        world_of = forest.articulation_world[heads]
        by_world = np.argsort(world_of, kind="stable")
        renumber = np.empty_like(by_world)
        renumber[by_world] = np.arange(by_world.size)
        block = renumber[block]
        bounds, order = host_index.group_by(block, heads.size)
        capacity = int(np.bincount(block, weights=rows).max())
        self.block_count = n = int(heads.size)
        self.block_capacity = capacity
        self.eliminates = True
        art_to_block = np.full(arts, -1, dtype=np.int32)
        held = np.isin(component, heads)
        art_to_block[held] = renumber[np.searchsorted(heads, component[held])]
        self._preelim_art_to_idx = wp.array(art_to_block, dtype=wp.int32, device=model.device)
        self._arrays.allocate(PREELIMINATION, preelim=n, preelim_row=n * capacity)
        world_blocks = np.bincount(world_of, minlength=forest.world_count)
        self._arrays.upload(
            PREELIMINATION_PLAN,
            _preelim_world=world_of[by_world],
            _preelim_start=bounds,
            _preelim_entry=entries[order],
            _preelim_world_start=host_index.offsets(world_blocks, dtype=np.int32),
        )

    def blocks(self):
        """The bilateral blocks (:data:`.solver_fields.Preelim`) the one writer's block phase closes."""
        return field_group.bind(Preelim, factor=self)

    def seed_blocks(self, *, first, rhs, impulses) -> None:
        """The closure's seed, once the warm start has gathered the rows' impulses: each block's kept rows take
        ``-r_Q`` (:func:`.response_builds_rows.seed_blocks_kernel`); the one writer closed the blocks and corrected the
        other rows."""
        wp.launch(
            response_builds_rows.seed_blocks_kernel,
            dim=self.block_count,
            inputs=[first, self.block_capacity, self.blocks(), rhs],
            outputs=[impulses],
            device=self.model.device,
        )

    def setup_drives(self, eligible: bool) -> None:
        """Allocate the augmented drive rows and separate immutable per-coordinate row ownership from per-step
        coefficients, where ``eligible``."""
        model, forest = self.model, self.forest
        if model.articulation_count and model.joint_count and forest.articulation_max_dofs:
            self._arrays.allocate(
                AUGMENTED_DRIVE_ROWS,
                articulation=model.articulation_count,
                drive_row=model.articulation_count * forest.articulation_max_dofs,
                dof=model.joint_dof_count,
            )
        else:
            self._arrays.release(AUGMENTED_DRIVE_ROWS)
        self.parallel_drives = False
        self.drive_row_by_dof = wp.full((max(1, int(model.joint_dof_count)),), -1, dtype=wp.int32, device=model.device)
        self.drive_q_index_by_dof = None
        self.drive_target_index_by_dof = None
        if not eligible or not model.joint_dof_count:
            return

        dof_count = int(model.joint_dof_count)
        # No DOF lies in two articulations' DOF ranges. A joint outside every articulation's lie in none.
        range_start = forest.articulation_dof_start_host.astype(np.int64)
        owner, local_dof = host_index.expand(forest.articulation_dof_count_host)
        owned_dof = range_start[owner] + local_dof
        if np.any(np.bincount(owned_dof[owned_dof < dof_count], minlength=dof_count)[:dof_count] > 1):
            return

        joint_type = model.joint_type.numpy()
        joint_q_start = model.joint_q_start.numpy()
        joint_target_q_start = model.joint_target_q_start.numpy()
        joint_qd_start = model.joint_qd_start.numpy()
        joint_dof_dim = model.joint_dof_dim.numpy()
        articulation_start = model.articulation_start.numpy()
        max_dofs = forest.articulation_max_dofs

        drive_row_by_dof = np.full(dof_count, -1, dtype=np.int32)
        drive_q_index_by_dof = np.full(dof_count, -1, dtype=np.int32)
        drive_target_index_by_dof = np.full(dof_count, -1, dtype=np.int32)
        row_dof_index = np.zeros(model.articulation_count * max_dofs, dtype=np.int32)
        supported = (int(JointType.PRISMATIC), int(JointType.REVOLUTE), int(JointType.D6))
        # Each articulation's supported joint axes, in joint order, take its first max_dofs row slots.
        first_joint = articulation_start[: model.articulation_count]
        articulation, position = host_index.expand(np.maximum(forest.articulation_joint_end_host - first_joint, 0))
        joint = first_joint[articulation] + position
        driven = np.isin(joint_type[joint], supported)
        articulation, joint = articulation[driven], joint[driven]
        axis_joint, axis = host_index.expand(joint_dof_dim[joint, 0] + joint_dof_dim[joint, 1])
        articulation, joint = articulation[axis_joint], joint[axis_joint]
        axis_count = np.bincount(articulation, minlength=model.articulation_count)
        slot = host_index.segmented_rank(articulation)
        slotted = slot < max_dofs
        articulation, joint, axis, slot = articulation[slotted], joint[slotted], axis[slotted], slot[slotted]
        dof = joint_qd_start[joint] + axis
        row = articulation * max_dofs + slot
        drive_row_by_dof[dof] = row
        drive_q_index_by_dof[dof] = joint_q_start[joint] + axis
        drive_target_index_by_dof[dof] = joint_target_q_start[joint] + axis
        row_dof_index[row] = dof
        row_counts = np.minimum(axis_count, max_dofs).astype(np.int32)

        self.drive_row_by_dof = wp.array(drive_row_by_dof, dtype=wp.int32, device=model.device)
        self.drive_q_index_by_dof = wp.array(drive_q_index_by_dof, dtype=wp.int32, device=model.device)
        self.drive_target_index_by_dof = wp.array(drive_target_index_by_dof, dtype=wp.int32, device=model.device)
        self.aug_row_counts.assign(row_counts)
        self.aug_row_dof_index.assign(row_dof_index)
        self.parallel_drives = True

    def allocate_common(self) -> None:
        """Allocate the mass-update mask, the compact diagonal's inverse mass, the body inertia records (with the COM
        frames) and the articulation origins."""
        model, forest, arrays = self.model, self.forest, self._arrays
        if model.joint_count:
            arrays.allocate(ARTICULATION_WORK, articulation=model.articulation_count)
        else:
            arrays.release(ARTICULATION_WORK)
        self.diagonal_inverse_mass = (
            wp.zeros_like(model.joint_qd) if forest.diagonal_mass_sizes else arrays.placeholder(wp.float32)
        )
        if model.body_count:
            arrays.allocate(BODY_INERTIA, body=model.body_count)
            wp.launch(
                compute_com_transforms,
                model.body_count,
                inputs=[model.body_com],
                outputs=[self.body_X_com],
                device=model.device,
            )
        else:
            arrays.release(BODY_INERTIA)

        if not model.articulation_count or not model.joint_count:
            self.articulation_origin = None
            return

        arrays.allocate(ARTICULATION_ORIGIN, articulation=model.articulation_count)

    def allocate_sparse(self, plan) -> None:
        """The sparse class's pattern ``plan`` and its coordinates' ancestor masks (the support writer's): the walk is
        its factor (Z4.3), so it holds no factor array."""
        self.sparse_pattern = plan
        self.sparse_ancestor_mask = wp.array(plan.ancestor_mask, dtype=wp.uint64, device=self.model.device)

    def scatter_armature(self) -> None:
        """Copy the effective armature (DOF-ordered) to the coordinates' (:attr:`dof_armature`)."""
        if self.armature is not None and self.armature.size:
            self.dof_armature.assign(self.armature.astype(np.float32))

    def build_body_trees(self) -> None:
        """Build where each body's tree and coordinates live (:class:`.kernels.BodyTrees`): each body's articulation,
        each articulation's first coordinate, and the articulation whose coordinates start at each global coordinate.
        Built once with the buffers: a captured step allocates and copies nothing."""
        model, forest = self.model, self.forest
        device = model.device
        first_art = np.full(max(int(model.joint_dof_count), 1), -1, dtype=np.int32)
        if int(model.articulation_count) > 0:
            starts = forest.articulation_dof_start.numpy()
            responding = np.flatnonzero(forest.articulation_response_dof_count.numpy() > 0)
            first_art[starts[responding]] = responding
        body_trees = BodyTrees()
        body_trees.body_to_articulation = forest.body_to_articulation
        body_trees.dof_start = forest.articulation_dof_start
        body_trees.first_art = wp.array(first_art, dtype=wp.int32, device=device)
        self.body_trees = body_trees

    def install(self, solver) -> None:
        """Hold the dynamics (:func:`.held_tree.install`); the drive rows' preparation overlaps FK/ID on a stream of
        its own."""
        from . import held_tree  # noqa: PLC0415

        self.held = held_tree.install(solver)
        model = self.model
        self.dynamics_stream = wp.Stream(model.device) if self.parallel_drives and model.device.is_cuda else None

    # ---------------------------------------------------------------------------------------------------------------
    # Notifications
    # ---------------------------------------------------------------------------------------------------------------

    def forget_repair_continuity(self) -> None:
        """Launch the held tree's next repair after a host call that changes its validity or inputs."""
        self.held.forget_continuity()

    def track_step_dt(self, dt: float) -> None:
        """Record the step duration; a change refreshes mass and invalidates the FK/ID cache of every articulation."""
        if self.last_step_dt is not None and abs(self.last_step_dt - dt) > 1.0e-8:
            self.force_mass_update = True
            self.fk_id_cache_valid.zero_()
            self.forget_repair_continuity()
        self.last_step_dt = dt

    def notify_state_changed(self, articulations: wp.array | None = None) -> None:
        """Invalidate the cached kinematics of ``articulations`` (``None``: every articulation)."""
        self.forget_repair_continuity()
        if articulations is None:
            self.fk_id_cache_valid.zero_()
            return
        wp.launch(
            invalidate_fk_id_cache_for_articulations,
            dim=articulations.shape[0],
            inputs=[articulations],
            outputs=[self.fk_id_cache_valid],
            device=self.model.device,
        )

    def notify_model_changed(self, flags, kinematic) -> None:
        """Take a model change once every check that can refuse it passed: prove the held tree's frames, invalidate
        the caches the changed inputs feed, publish ``kinematic`` (:meth:`kinematic_state`, or ``None``) and re-derive
        the COM frames on an inertial change."""
        # The held tree's proof stores a moved joint frame as its new reference, so it runs after every check that
        # can still refuse the notification.
        held = self.held
        frames_proven = held.validate_model(flags)
        self.forget_repair_continuity()
        cache_inputs = int(flags) & int(
            ModelFlags.JOINT_PROPERTIES
            | ModelFlags.JOINT_DOF_PROPERTIES
            | ModelFlags.BODY_PROPERTIES
            | ModelFlags.BODY_INERTIAL_PROPERTIES
            | ModelFlags.MODEL_PROPERTIES
        )
        if frames_proven and cache_inputs == int(ModelFlags.JOINT_PROPERTIES):
            # The held tree's device checksums of every joint frame and axis just matched (an exact host fingerprint
            # on CPU), and stepping never reads Model.joint_q: every cache input is as it was. Isaac Lab notifies so
            # on each fixed-root pose write. A moved frame leaves the flag in place and invalidates every cache below.
            cache_inputs = 0
        elif not frames_proven:
            from .tree_kinetic_owner import proof_runs  # noqa: PLC0415

            if proof_runs(flags):
                # The proof ran and found a moved joint frame (a changed axis raises in every group's proof):
                # invalidate every cache, whichever flag announced the move.
                cache_inputs |= int(ModelFlags.JOINT_PROPERTIES)
        if cache_inputs:
            held.invalidate()
            self.fk_id_cache_valid.zero_()
        if kinematic is not None:
            self.apply_kinematic_state(kinematic)
            self.scatter_armature()
            self.mass_update_requested.fill_(1)
        if flags & ModelFlags.BODY_INERTIAL_PROPERTIES and self.model.body_count:
            # Re-derive the COM frames baked from body_com in allocate_common so runtime CoM randomization reaches
            # FK; mass and inertia are read live.
            wp.launch(
                compute_com_transforms,
                self.model.body_count,
                inputs=[self.model.body_com],
                outputs=[self.body_X_com],
                device=self.model.device,
            )
            self.mass_update_requested.fill_(1)

    def reset_cache(self, state, world_mask: wp.array | None) -> None:
        """Invalidate the FK/ID cache of the worlds ``world_mask`` selects, and of every world when ``state`` is not
        the state the cache was derived from (unless the held tree compares input banks)."""
        wp.launch(
            invalidate_articulation_fk_id_cache,
            dim=self.model.articulation_count,
            inputs=[world_mask, self.forest.art_to_world],
            outputs=[self.fk_id_cache_valid],
            device=self.model.device,
        )
        if (
            self.fk_id_cache_source_state is not None
            and state is not self.fk_id_cache_source_state
            and not self.held.bank_compare
        ):
            self.fk_id_cache_valid.zero_()
            self.fk_id_cache_source_state = state

    # ---------------------------------------------------------------------------------------------------------------
    # The step
    # ---------------------------------------------------------------------------------------------------------------

    def factor(
        self, state_in, control, state_out, dt: float, step: int, qd_work: wp.array, v_hat: wp.array, v_out: wp.array
    ):
        """FK/ID, the drives' K and u0, the mass-update mask and the factor refresh.

        Returns:
            ``(qd, ready)``: the predictor's velocity (``qd_work`` prescaled when velocities are limited) and the event
            the predictor waits for, if any.
        """
        model = self.model
        with wp.ScopedTimer("S1_FK_ID_CRBA", print=False, use_nvtx=self.nvtx, synchronize=False):
            drive_rows_ready = self.prepare_drives(state_in, control, dt, v_hat, v_out)
            qd = self.fk_id(state_in, state_out, step, qd_work)
            self.refresh(drive_rows_ready, step)
            self.walk_step(state_in)
        # joint_tau holds only u0: the predictor adds the current external, control and bias contributions after
        # waiting for the drives.
        return qd, drive_rows_ready if model.articulation_count else None

    def prescale_velocity(self, joint_qd: wp.array, prescaled_joint_qd: wp.array) -> None:
        """Write every DOF's PhysX-prescaled joint velocity in one launch, without a separate copy."""
        model = self.model
        if not model.articulation_count:
            wp.copy(prescaled_joint_qd, joint_qd)
            return
        wp.launch(
            prescale_joint_velocity_limits,
            dim=self.laws.velocity_prescale_threads,
            inputs=[
                field_group.bind(JointTable, model=model, factor=self, forest=self.forest),
                model.articulation_count,
                self.laws.velocity_prescale_lanes,
                joint_qd,
            ],
            outputs=[prescaled_joint_qd],
            device=model.device,
        )

    def fk_id(self, state_in, state_out, step: int, qd_work: wp.array) -> wp.array:
        """Forward kinematics and inverse-dynamics terms; returns the predictor's velocity (prescaled when limited)."""
        if not self.held.bank_compare:
            # Held trees that compare input banks validate a changed bank on the device.
            if self.fk_id_cache_source_state is not None and state_in is not self.fk_id_cache_source_state:
                self.fk_id_cache_valid.zero_()
                self.forget_repair_continuity()
        self.fk_id_cache_source_state = state_in
        if self.options.enable_joint_velocity_limits:
            self.prescale_velocity(state_in.joint_qd, qd_work)
            qd = qd_work
        else:
            qd = state_in.joint_qd

        refresh_composite = (step % self.options.update_mass_matrix_interval) == 0 or self.force_mass_update
        self.held.begin(state_in, self, qd, self.last_step_dt, refresh_composite, state_out=state_out)
        return qd

    def prepare_drives(self, state_in, control, dt: float, v_hat: wp.array, v_out: wp.array) -> wp.Event | None:
        """Launch the per-coordinate augmented-drive preparation independently of FK/ID; ``v_hat`` and ``v_out`` are last
        step's."""
        if not self.parallel_drives:
            return None

        model = self.model
        dynamics_stream = self.dynamics_stream
        if dynamics_stream is not None:
            dynamics_stream.wait_event(wp.get_stream(model.device).record_event())
        with wp.ScopedStream(dynamics_stream, sync_enter=False):
            scalar = getattr(self.held, "scalar", None)
            if scalar is not None:
                # Scalar branches evaluate their drives in their predict.
                scalar.launch_other_drives(state_in, self, control, dt)
            else:
                wp.launch(
                    prepare_augmented_drives,
                    dim=model.joint_dof_count,
                    inputs=[
                        self.drive_row_by_dof,
                        self.drive_q_index_by_dof,
                        self.drive_target_index_by_dof,
                        state_in.joint_q,
                        state_in.joint_qd,
                        model.joint_target_ke,
                        model.joint_target_kd,
                        control.joint_target_q,
                        control.joint_target_qd,
                        model.joint_effort_limit,
                        self.passive_joint_damping,
                        self.forest.articulation_max_dofs,
                        dt,
                    ],
                    outputs=[self.drive_rows(v_hat, v_out), self.joint_tau],
                    block_dim=256,
                    device=model.device,
                )
        return None if dynamics_stream is None else dynamics_stream.record_event()

    def drive_rows(self, v_hat: wp.array, v_out: wp.array) -> DriveRows:
        """What a step's drives write (:func:`.solver_fields.hold_drive_row`); ``v_hat`` and ``v_out`` are last step's."""
        return field_group.bind(DriveRows, factor=self, v_hat=v_hat, v_out=v_out)

    def refresh(self, drive_rows_ready: wp.Event | None, step: int) -> None:
        """The mass-update mask, which consumes the requests, then the held tree's factor refresh."""
        model = self.model
        global_flag = 1 if ((step % self.options.update_mass_matrix_interval) == 0 or self.force_mass_update) else 0
        # NOTE: this host-evaluated flag is BAKED into CUDA graphs at capture
        # time. With K physics steps captured per graph, replays repeat the
        # K-step flag pattern captured starting at the capture-time step, so
        # only intervals that divide the captured pattern behave as requested
        # (e.g. a 2-step graph captured at step 0 bakes [1, 0]: interval=2 is
        # exact, interval>=3 silently degrades to an effective interval of 2).
        # The same flag gates the H memsets at the end of the step so the
        # baked memset cadence always matches the baked update cadence.
        self.mass_update_global_flag = bool(global_flag)

        # The mask kernel reads what the drives requested.
        if drive_rows_ready is not None:
            wp.get_stream(model.device).wait_event(drive_rows_ready)
        wp.launch(
            build_mass_update_mask,
            dim=model.articulation_count,
            inputs=[global_flag],
            outputs=[self.mass_update_requested, self.mass_update_mask],
            device=model.device,
        )

        self.held.refresh(self)
        self.force_mass_update = False

    def predict(self, state_in, control, qd: wp.array, ready: wp.Event | None, dt: float, v_hat: wp.array) -> None:
        """Stage 3: ``v_hat = qd + h H^-1 (tau - c)``, the free-root correction included. Each tree's launch holds the
        drives' saturation decision and, where it moved a drive, the decided step by the border on the walk
        (:data:`.tree_kinetic_state.SATURATE_DRIVES`, :func:`.tree_kinetic_state.border_solve`); the factor keeps every
        ``K``, so no saturation ever asks the refresh for a factor."""
        with wp.ScopedTimer("S3_Trisolve_Vhat", print=False, use_nvtx=self.nvtx, synchronize=False):
            if ready is not None:
                wp.get_stream(self.model.device).wait_event(ready)
            self.held.predict(state_in, self, control, qd, dt)
            self.correct_free_root_velocity(state_in, qd, dt, v_hat)

    def correct_free_root_velocity(self, state_in, qd: wp.array, dt: float, v_hat: wp.array) -> None:
        """Apply canonical gyro physics and the free roots' spin damping after the held prediction."""
        model = self.model
        forest = self.forest
        # Keep v_hat on jcalc_integrate's free-root convention. The compact launch excludes every non-root joint; a
        # live device mask handles runtime kinematic changes without changing graph topology.
        if forest.free_root_joint_count and forest.free_rigid_body_count:
            wp.launch(
                apply_free_root_velocity_corrections,
                dim=forest.free_root_joint_count,
                inputs=[
                    forest.free_root_joint_indices,
                    model.joint_qd_start,
                    model.joint_child,
                    forest.body_to_articulation,
                    forest.is_free_rigid,
                    self.kinematic_joint_mask,
                    state_in.body_q,
                    model.body_inertia,
                    self.walk_data,
                    qd,
                    dt,
                ],
                outputs=[v_hat],
                device=model.device,
            )
        if self.options.angular_damping != 0.0:
            wp.launch(
                damp_free_root_spin,
                dim=forest.free_root_joint_count,
                inputs=[
                    forest.free_root_joint_indices,
                    model.joint_qd_start,
                    self.kinematic_joint_mask,
                    1.0 - self.options.angular_damping * dt,
                ],
                outputs=[v_hat],
                device=model.device,
            )

    def finish(self, state_in, state_out, dt: float, step: int) -> None:
        """Stage 7: ``qdd`` from the solved velocity, integrate the coordinates and publish the output state."""
        with wp.ScopedTimer("S7_Integrate", print=False, use_nvtx=self.nvtx, synchronize=False):
            next_refresh = ((step + 1) % self.options.update_mass_matrix_interval) == 0
            self.held.finish(state_in, self, state_out, dt, next_refresh)
            self.fk_id_cache_source_state = state_out

    # ---------------------------------------------------------------------------------------------------------------
    # The walk on the device (FL3.3): the converted class's factor as the one writer and the leave read it
    # ---------------------------------------------------------------------------------------------------------------

    def plan_walk(self, articulations: np.ndarray) -> None:
        """Hold the walk's data for ``articulations`` (the converted class's, every responding tree): each coordinate's
        :data:`WALK_COORDINATE` floats, each body's :data:`WALK_LINK` floats and a twist of scratch, each body's path
        entry (its parent and first coordinate within its tree, :func:`walk_paths_prelude`) and its class's join slot
        (:attr:`.tree_scan.WalkClass.slots`), the trees' lane group (:func:`walk_group`), and the step's geometry the
        finish publishes (:attr:`WalkTrees.now`, :attr:`WalkTrees.now_link`): the rows' seeds enter there. One launch on the factor's own refreshes
        (:meth:`walk_step`) takes that geometry as the factor's frames and runs the inward pass, so the walk is the
        factor ``L`` refreshes, to rounding.

        Raises:
            ValueError: For a tree whose bodies do not come after their parents, or that holds more than
                :data:`WALK_JOIN_SLOTS` joins at once.
        """
        model, forest = self.model, self.forest
        device = model.device
        free = forest.is_free_rigid_host if forest.is_free_rigid is not None else np.zeros(1, dtype=np.int32)
        arts = np.asarray(articulations, dtype=np.int32)
        if forest.is_free_rigid is not None:
            # The free rigid bodies (one-body trees) after the articulated trees, so no group of an articulated
            # tree shares its warp with one.
            arts = np.concatenate([arts[free[arts] == 0], arts[free[arts] != 0]]).astype(np.int32)
        start = model.articulation_start.numpy()
        end = forest.articulation_joint_end_host
        qd_start = model.joint_qd_start.numpy()
        joint_parent, joint_child = model.joint_parent.numpy(), model.joint_child.numpy()
        joint_of = np.full(max(int(model.body_count), 1), -1, dtype=np.int64)
        joint_of[joint_child[joint_child >= 0]] = np.flatnonzero(joint_child >= 0)
        path = np.zeros((max(int(model.joint_count), 1), 2), dtype=np.int32)
        dof_body = np.full(max(int(model.joint_dof_count), 1), -1, dtype=np.int32)
        for art in arts.tolist():
            j0, j1 = int(start[art]), int(end[art])
            for j in range(j0, j1):
                up = int(joint_of[joint_parent[j]]) if joint_parent[j] >= 0 else -1
                parent = up - j0 if j0 <= up < j1 else -1
                if parent >= j - j0:
                    raise ValueError(f"The walk needs every parent before its children (articulation {art})")
                path[j] = (parent, int(qd_start[j] - qd_start[j0]))
                dof_body[qd_start[j] : qd_start[j + 1]] = j - j0
        joints, dofs = max(int(model.joint_count), 1), max(int(model.joint_dof_count), 1)
        self.walk_arts = wp.array(arts, dtype=wp.int32, device=device)
        # Each coordinate's column half at its body's frame, from its joint's kind (WALK_COORDINATE): a joint's
        # sliding coordinates the linear half, its turning ones the angular half, but a D6 joint's turns after its
        # slides and a free joint's turns below a root, which can have both (WALK_BOTH): their linear halves sit in the
        # records' side half.
        kinds, dims = model.joint_type.numpy(), model.joint_dof_dim.numpy()
        self.walk_half_host = np.full(dofs, WALK_BOTH, dtype=np.int32)
        walked = np.zeros(max(int(model.joint_count), 1), dtype=bool)
        for art in arts.tolist():
            walked[int(start[art]) : int(end[art])] = True
        for j in range(kinds.size):
            lin, ang = int(dims[j, 0]), int(dims[j, 1])
            two = ang > 0 and (
                (kinds[j] == int(JointType.D6) and lin > 0)
                or (kinds[j] in (int(JointType.FREE), int(JointType.DISTANCE)) and path[j, 0] >= 0)
            )
            c = int(qd_start[j])
            self.walk_half_host[c : c + lin] = WALK_LINEAR
            self.walk_half_host[c + lin : c + lin + ang] = WALK_BOTH if two else WALK_ANGULAR
        both = self.two_halves(np.flatnonzero(np.repeat(walked[: kinds.size], np.diff(qd_start[: kinds.size + 1]))))
        # The walking classes: each held tree's topology, which the step's one leave (walk_leave_kernel) and the
        # predictor's solve emit as its walk, and each articulation's class among them.
        classes: dict[WalkClass, int] = {}
        walk_class = np.full(max(int(model.articulation_count), 1), -1, dtype=np.int32)
        for art in arts.tolist():
            j0, j1 = int(start[art]), int(end[art])
            table = WalkClass(
                tuple(int(p) for p in path[j0:j1, 0]),
                (*(int(f) for f in path[j0:j1, 1]), int(qd_start[j1] - qd_start[j0])),
                tuple(int(h) for h in self.walk_half_host[int(qd_start[j0]) : int(qd_start[j1])]),
            )
            walk_class[art] = classes.setdefault(table, len(classes))
        self.walk_classes = tuple(classes)
        self.walk_class = wp.array(walk_class, dtype=wp.int32, device=device)
        # The refresh's join slots (walk_inward) are its class's (WalkClass.slots): a parent waits in one for the
        # lifted inertia of each child other than the joint after it.
        joins = np.full(max(int(model.joint_count), 1), -1, dtype=np.int32)
        for art in arts.tolist():
            j0, j1 = int(start[art]), int(end[art])
            joins[j0:j1] = self.walk_classes[walk_class[art]].slots[0]
        held = max((walk.slots[1] for walk in classes), default=0)
        if held > WALK_JOIN_SLOTS:
            raise ValueError(f"The walk holds at most {WALK_JOIN_SLOTS} joins of a tree at once, a tree needs {held}")
        # A tree's group in the refresh (walk_group): on CUDA a lane a body of a batch and a join slot, the trees in
        # flight those with a chain to walk (a free rigid body leaves its group after the batch's first body); one
        # thread a tree on the CPU.
        bodies = max([1, *(len(walk.parent) for walk in classes)])
        sizes = np.array([int(end[a]) - int(start[a]) for a in arts.tolist()], dtype=np.int64)
        walked = int(np.count_nonzero(sizes > 1)) or int(arts.size)
        self.walk_lanes = walk_group(bodies, held, walked, device.sm_count) if device.is_cuda else 1
        self.walk_path = wp.array(path, dtype=wp.vec2i, device=device)
        self.walk_joins = wp.array(joins, dtype=wp.int32, device=device)
        self.walk_path_host = path
        # The one flag every walk reader's text is chosen by: whether a record (and a step column) can have a side.
        self.walk_side = both
        self.walk_data = wp.zeros(((2 if both else 1) * dofs, WALK_COORDINATE), dtype=wp.float32, device=device)
        self.walk_link = wp.zeros((joints, WALK_LINK), dtype=wp.float32, device=device)
        # The step's geometry, which the held tree's finish publishes with its motion columns (every step, no launch
        # of the walk's): each coordinate's column at its body's frame and each body's link there. A seed enters as J
        # there, and a refresh takes it as the factor's frames.
        # Each coordinate's column half by its joint's kind on the device (the refresh's pick of a column's half).
        self.walk_half = wp.array(self.walk_half_host, dtype=wp.int32, device=device)
        # A model with a column of two halves keeps the finish's columns at each body's frame; else the step's columns
        # are S's own (WALK_NOW).
        self.walk_now = wp.zeros(dofs if both else 0, dtype=wp.spatial_vector, device=device)
        self.walk_now_link = wp.zeros((joints, WALK_LINK), dtype=wp.float32, device=device)
        # Each articulation's world origin at the factor's frames (the refresh's copy of the step's): the frame points
        # are the articulation's own, relative to its origin, so a world point meets them through it.
        self.walk_origin = wp.zeros(max(int(model.articulation_count), 1), dtype=wp.vec3, device=device)
        view = WalkTrees()
        view.data, view.link, view.path = self.walk_data, self.walk_link, self.walk_path
        view.now = self.walk_now if both else self.joint_S_s
        view.now_link, view.now_com = self.walk_now_link, self.joint_com_offset
        view.origin, view.now_origin = self.walk_origin, self.articulation_origin
        view.joint_art = model.joint_articulation
        view.scratch = wp.zeros(joints, dtype=wp.spatial_vector, device=device)
        view.joint_start = model.articulation_start
        view.joint_end = forest.articulation_joint_end
        view.dof_start = forest.articulation_dof_start
        view.dofs = forest.articulation_dof_count
        view.dof_body = wp.array(dof_body, dtype=wp.int32, device=device)
        body_joint = np.full(max(int(model.body_count), 1), -1, dtype=np.int32)
        body_joint[: joint_of.size] = joint_of[: body_joint.size]
        view.body_joint = wp.array(body_joint, dtype=wp.int32, device=device)
        # The release of each tree's slips, which the predictor's border writes (tree_kinetic_state.border_solve): as
        # many slips a tree as it has limited drives (at most RELEASE_SLIPS), each as deep as the deepest root path.
        self.walk_slip_capacity = self.slip_capacity(arts)
        depth = max([1, *(len(walk.root_path(c)) for walk in classes for c in range(walk.coordinates))])
        view.du = wp.zeros(dofs, dtype=wp.float32, device=device)
        view.release = release_view(
            max(int(model.articulation_count), 1),
            self.walk_slip_capacity,
            depth if self.walk_slip_capacity else 1,
            device,
        )
        self.walk_trees = view
        self.plan_leave()

    def plan_leave(self, nodes: tuple[tuple[int, ...], ...] | None = None) -> None:
        """The step's one leave of every walking tree (:func:`walk_leave_kernel`), the release compiled where a tree
        can slip, reading each class's coordinate ``c`` at ``node(c)`` of its tree's changes: ``nodes`` (each class's,
        a sweep's own layout), or the walk's own coordinates."""
        classes = getattr(self, "walk_classes", ())
        self.walk_leave = walk_leave_kernel(classes, self.walk_slip_capacity > 0, nodes) if classes else None

    def leave(self, v_out: wp.array, du: wp.array | None = None, du_base: wp.array | None = None) -> None:
        """The step's one leave (:func:`walk_leave_kernel`): every walking tree's velocity change walked down onto
        ``v_out`` (``v += P^T L^-T R^T du``): coordinate ``c``'s change is ``du[du_base[art] + node(c)]``, by default
        the walk's own :attr:`WalkTrees.du` at the tree's coordinates, which the sweeps add into from zero each
        step."""
        if self.walk_leave is None:
            return
        lanes = LEAVE_LANES if self.model.device.is_cuda else 1
        wp.launch(
            self.walk_leave,
            dim=self.walk_arts.shape[0] * lanes,
            inputs=[
                self.walk_arts,
                lanes,
                self.walk_trees,
                self.walk_class,
                self.walk_trees.du if du is None else du,
                self.walk_trees.dof_start if du_base is None else du_base,
                v_out,
            ],
            device=self.model.device,
        )

    def slips_by_tree(self, articulations) -> np.ndarray:
        """Each of ``articulations``' drives that can slip: its driven coordinates with a gain and a finite effort
        limit (the model's data at construction)."""
        model, forest = self.model, self.forest
        arts = np.asarray(articulations, dtype=np.int64)
        if not self.parallel_drives or not model.joint_dof_count:
            return np.zeros(arts.size, dtype=np.int64)
        ke, kd = model.joint_target_ke.numpy(), model.joint_target_kd.numpy()
        damping, limit = self.passive_joint_damping.numpy(), model.joint_effort_limit.numpy()
        driven = self.drive_row_by_dof.numpy() >= 0
        limited = driven & ((ke > 0) | (kd > 0) | (damping > 0)) & (limit > 0) & np.isfinite(limit)
        starts, counts = forest.articulation_dof_start_host, forest.articulation_dof_count_host
        return np.array(
            [int(np.count_nonzero(limited[starts[a] : starts[a] + counts[a]])) for a in arts.tolist()], dtype=np.int64
        )

    def slip_capacity(self, articulations) -> int:
        """The most drives of one of ``articulations`` that can slip (:meth:`slips_by_tree`), at most
        :data:`RELEASE_SLIPS`. A tree's border holds that many; 0 where none can slip, so a consumer compiles no
        release."""
        slips = self.slips_by_tree(articulations)
        return min(int(slips.max(initial=0)), RELEASE_SLIPS)

    def walk_view(self) -> WalkTrees:
        """The walk's view (:class:`WalkTrees`) the one writer, the leave and the audit read."""
        return self.walk_trees

    def walk_step(self, state) -> None:
        """The walk's refresh (:func:`walk_step_kernel`, a lane group per held tree), at the poses the held tree
        published in ``state.body_q``, on the trees the refresh just factored (``mass_update_mask``, which the device
        may set on any step): the step's geometry the finish published (:attr:`WalkTrees.now`) becomes the factor's
        frames and the group runs the inward pass. A free rigid body is a one-body tree: its walk is its six
        coordinates' factor. A tree that does not refresh takes nothing here."""
        if self.walk_arts is None:
            return
        model = self.model
        lanes = self.walk_lanes
        wp.launch(
            walk_step_kernel,
            dim=(self.walk_arts.shape[0], lanes),
            inputs=[
                self.walk_arts,
                lanes,
                self.mass_update_mask,
                model.articulation_start,
                self.forest.articulation_joint_end,
                self.walk_path,
                self.walk_joins,
                model.joint_child,
                model.joint_qd_start,
                state.body_q,
                model.body_mass,
                model.body_inertia,
                self.dof_armature,
                self.drive_row_by_dof,
                self.aug_row_K if self.parallel_drives else self._arrays.placeholder(wp.float32),
                self.walk_trees.now,
                self.walk_half,
                self.walk_now_link,
                self.joint_com_offset,
                self.articulation_origin,
            ],
            outputs=[self.walk_data, self.walk_link, self.walk_origin],
            block_dim=max(WALK_BLOCK, lanes),
            device=model.device,
        )

    def two_halves(self, coordinates) -> bool:
        """Whether a coordinate of ``coordinates`` has a step column of two halves at its frame (:data:`WALK_BOTH`, from
        its joint's kind): the one rule a walk reader's text is chosen by. :attr:`walk_side` is it over the walked
        coordinates; a held group's finish reads it over its own (:class:`.tree_kinetic_owner.TreeKineticState`)."""
        if self.walk_half_host is None:
            return False
        return bool(np.any(self.walk_half_host[np.asarray(coordinates, dtype=np.int64)] == WALK_BOTH))

    def diagonal(self, articulation: int) -> np.ndarray:
        """What this factor adds to ``articulation``'s mass matrix: each coordinate's effective armature ``R``
        (conditioning and kinematic coordinates included) and the drive stiffness ``K`` it last refreshed with."""
        qd_start = self.model.joint_qd_start.numpy()
        joints = self.model.articulation_start.numpy()[articulation : articulation + 2]
        dofs = np.arange(qd_start[joints[0]], qd_start[joints[1]])
        added = self.armature[dofs].astype(np.float64)
        rows = self.drive_row_by_dof.numpy()[dofs] if dofs.size else dofs
        if np.any(rows >= 0):
            stiffness = self.aug_row_K.numpy().astype(np.float64)
            added += np.where(rows >= 0, np.maximum(stiffness[np.maximum(rows, 0)], 0.0), 0.0)
        return added

    def walk(self, joint_q: np.ndarray, articulation: int, *, dtype=np.float64) -> TreeWalk:
        """The walk of ``articulation``'s factor ``P (H + R + K) P^T`` at the coordinates ``joint_q``
        (:class:`TreeWalk`): its local frames from the model in float64 (:func:`local_frames`), its diagonal this
        factor's (:meth:`diagonal`), its inward pass in ``dtype``. No step calls it: the tests hold it to the factor
        (Z_UNION §6, Z0)."""
        frames = local_frames(self.model, joint_q, articulation)
        return TreeWalk(*frames, self.diagonal(articulation), dtype=dtype)


# ---------------------------------------------------------------------------------------------------------------------
# The walk (Z_UNION §1.2): the factor as a six-number state's steps along the tree, in local frames
# ---------------------------------------------------------------------------------------------------------------------


def _cross(d: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``d x v`` for a 3-vector ``d`` and ``v`` of shape ``(3, ...)``, in their dtype."""
    return np.stack((d[1] * v[2] - d[2] * v[1], d[2] * v[0] - d[0] * v[2], d[0] * v[1] - d[1] * v[0]))


def _rotation(quaternion) -> np.ndarray:
    """The rotation matrix of a unit quaternion ``(x, y, z, w)``, in float64."""
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _turn(axis, angle) -> np.ndarray:
    """The rotation of ``angle`` about the unit ``axis``, in float64."""
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * k @ k


def local_frames(model, joint_q: np.ndarray, articulation: int, poses: np.ndarray | None = None) -> tuple:
    """One articulation's tree in local frames at the coordinates ``joint_q``, in float64 from the model's data.

    A body's frame sits at its joint's anchor (the parent's joint frame), with world axes; a free body's at its centre
    of mass. Its coordinates' columns are the factor's (:func:`.spatial_laws.joint_motion`: a free joint's turn about
    the articulation's origin, its first root's centre of mass) moved to that frame, and its spatial inertia is
    taken there, in Newton's ``(v, w)`` layout. Nothing is held farther than one link from its frame.

    Args:
        model: The model.
        joint_q: The coordinates.
        articulation: The articulation.
        poses: The bodies' world poses ``[bodies, 7]`` to read instead of composing them in float64 from ``joint_q``
            (a float32 forward kinematics', say).

    Returns:
        ``(parent, first, columns, inertia, offset)`` for :class:`TreeWalk`: each body's parent (-1: the world), each
        body's first coordinate (and the count after the last), each coordinate's column ``[n, 6]``, each body's
        inertia ``[bodies, 6, 6]`` and each body's parent's frame less its own ``[bodies, 3]`` (zero at a root).
    """
    start = model.articulation_start.numpy()
    joints = np.arange(start[articulation], start[articulation + 1])
    kinds, dims = model.joint_type.numpy()[joints], model.joint_dof_dim.numpy()[joints]
    parent_body, child_body = model.joint_parent.numpy()[joints], model.joint_child.numpy()[joints]
    q_start, qd_start = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
    X_p, X_c = model.joint_X_p.numpy().astype(np.float64), model.joint_X_c.numpy().astype(np.float64)
    axes = model.joint_axis.numpy().astype(np.float64)
    mass, com = model.body_mass.numpy().astype(np.float64), model.body_com.numpy().astype(np.float64)
    body_inertia = model.body_inertia.numpy().astype(np.float64)
    local = {int(body): index for index, body in enumerate(child_body)}
    if len(local) < len(joints) or any(body >= 0 and int(body) not in local for body in parent_body):
        raise ValueError("The articulation's joints are not a tree: walk the solver's model, whose loops are cut")
    parent = np.array([local[int(body)] if body >= 0 else -1 for body in parent_body], dtype=np.int64)
    first = np.append(qd_start[joints], qd_start[joints[-1] + 1]) - qd_start[joints[0]]
    count = len(joints)
    rotation, position, frame = np.zeros((count, 3, 3)), np.zeros((count, 3)), np.zeros((count, 3))
    columns, inertia = np.zeros((int(first[-1]), 6)), np.zeros((count, 6, 6))
    origin = None
    for b, joint in enumerate(joints):
        p = parent[b]
        R_p, t_p = (rotation[p], position[p]) if p >= 0 else (np.eye(3), np.zeros(3))
        R_a, t_a = R_p @ _rotation(X_p[joint, 3:]), t_p + R_p @ X_p[joint, :3]
        q = np.asarray(joint_q[q_start[joint] : q_start[joint + 1]], dtype=np.float64)
        lin, ang = int(dims[b, 0]), int(dims[b, 1])
        axis = axes[qd_start[joint] : qd_start[joint + 1]]
        kind = int(kinds[b])
        # The joint's motion (rotation, translation) and its columns at the anchor, world axes.
        own = []
        if kind == int(JointType.REVOLUTE):
            motion = (_turn(axis[0], q[0]), np.zeros(3))
            own.append(np.r_[0.0, 0.0, 0.0, R_a @ axis[0]])
        elif kind == int(JointType.PRISMATIC):
            motion = (np.eye(3), axis[0] * q[0])
            own.append(np.r_[R_a @ axis[0], 0.0, 0.0, 0.0])
        elif kind == int(JointType.BALL):
            motion = (_rotation(q[:4]), np.zeros(3))
            own += [np.r_[0.0, 0.0, 0.0, R_a @ e] for e in np.eye(3)]
        elif kind in (int(JointType.FREE), int(JointType.DISTANCE)):
            motion = (_rotation(q[3:7]), q[:3])
            own += [np.r_[R_a @ e, 0.0, 0.0, 0.0] for e in np.eye(3)]
            own += [np.r_[0.0, 0.0, 0.0, R_a @ e] for e in np.eye(3)]
        elif kind == int(JointType.D6):
            slide = sum((axis[k] * q[k] for k in range(lin)), np.zeros(3))
            own += [np.r_[R_a @ axis[k], 0.0, 0.0, 0.0] for k in range(lin)]
            turned = np.eye(3)
            for k in range(ang):
                w = R_a @ (turned @ axis[lin + k])
                # A turn about the pivot the slides moved, at the anchor: v = (R_a slide) x w.
                own.append(np.r_[np.cross(R_a @ slide, w), w])
                turned = _turn(turned @ axis[lin + k], q[lin + k]) @ turned
            motion = (turned, slide)
        elif kind == int(JointType.FIXED):
            motion = (np.eye(3), np.zeros(3))
        else:
            raise ValueError(f"No walk of joint type {kind}")
        R_c = R_a @ motion[0] @ _rotation(X_c[joint, 3:]).T
        t_c = t_a + R_a @ motion[1] - R_c @ X_c[joint, :3]
        if poses is not None:
            pose = np.asarray(poses[child_body[b]], dtype=np.float64)
            R_c, t_c = _rotation(pose[3:]), pose[:3]
        rotation[b], position[b] = R_c, t_c
        centre = t_c + R_c @ com[child_body[b]]
        if origin is None:
            origin = centre
        free = kind in (int(JointType.FREE), int(JointType.DISTANCE))
        frame[b] = centre if free else t_a
        if free:
            # The factor turns a free joint about the articulation's origin: the same twists at the body's frame.
            for column in own[3:]:
                column[:3] += np.cross(column[3:], frame[b] - origin)
        if own:
            columns[first[b] : first[b + 1]] = np.stack(own)
        r = centre - frame[b]
        cross = np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])
        m = mass[child_body[b]]
        inertia[b, :3, :3] = m * np.eye(3)
        inertia[b, :3, 3:] = -m * cross
        inertia[b, 3:, :3] = m * cross
        inertia[b, 3:, 3:] = R_c @ body_inertia[child_body[b]] @ R_c.T - m * cross @ cross
    offset = np.where(parent[:, None] >= 0, frame[np.maximum(parent, 0)] - frame, 0.0)
    return parent, first, columns, inertia, offset


class TreeWalk:
    """One articulation's tree factor as a walk (Z_UNION §1.2, Theorem A), in local frames (:func:`local_frames`).

    The inward pass eliminates the coordinates leaf first, one at a time, from each body's own inertia plus what its
    children pass up,

        ell_c = (s_c^T I s_c + D_c)^1/2,    ut_c = I s_c / ell_c,    I <- I - ut_c ut_c^T,

    ``D`` the coordinates' armature and drive stiffness, and then moves the body's remaining inertia one link up. Its
    data ``(s, ut, ell)`` is the leaf-first factor ``P (H + D) P^T = L L^T`` (``L_cc = ell_c``, and ``L`` is
    ``s_a^T ut_c`` for ``a`` an ancestor coordinate of ``c``), so:

    - :meth:`up` is ``Z = L^-1 P J^T``, a row's footprints: its seed, a wrench at a body or a covector on the
      coordinates, walked up to the root, each coordinate taking ``e_c^T p`` of the state ``p`` and passing
      ``p - ut_c z_c``, the wrench it cannot feel;
    - :meth:`down` is the leave ``P^T L^-T u``, the walk run back down; its state at a body is the body's twist;
    - :meth:`meet` is ``Z_k^T Z_l``: two walks carried to their lowest common ancestor, met through its Gram.

    A footprint ``z_c`` sits at its coordinate ``c`` (the kit's entry ``c`` of a side). The arithmetic runs in
    ``dtype``; a negative pivot leaves NaN, as the factor would. The step calls none of it.

    Args:
        parent: Each body's parent (-1: the world), every parent before its children.
        first: Each body's first coordinate, and the count after the last.
        columns: Each coordinate's motion column at its body's frame, ``[n, 6]``.
        inertia: Each body's spatial inertia at its frame, ``[bodies, 6, 6]``.
        offset: Each body's parent's frame less its own, ``[bodies, 3]``.
        diagonal: Each coordinate's armature and drive stiffness.
        dtype: The walk's arithmetic.
    """

    def __init__(self, parent, first, columns, inertia, offset, diagonal, *, dtype=np.float64):
        self.dtype = np.dtype(dtype)
        self.parent = np.asarray(parent, dtype=np.int64)
        self.first = np.asarray(first, dtype=np.int64)
        if np.any(self.parent >= np.arange(self.parent.size)):
            raise ValueError("The walk needs every parent before its children")
        self.s = np.asarray(columns, dtype=self.dtype)
        self.offset = np.asarray(offset, dtype=self.dtype)
        held = np.asarray(inertia, dtype=self.dtype).copy()
        added = np.asarray(diagonal, dtype=self.dtype)
        self.ell = np.zeros(self.s.shape[0], dtype=self.dtype)
        self.ut = np.zeros_like(self.s)
        self.own = np.zeros(self.s.shape[0], dtype=self.dtype)
        with np.errstate(invalid="ignore", divide="ignore"):
            for b in range(self.parent.size - 1, -1, -1):
                I = held[b]
                for c in range(self.first[b + 1] - 1, self.first[b] - 1, -1):
                    Is = I @ self.s[c]
                    self.own[c] = self.s[c] @ Is
                    self.ell[c] = np.sqrt(self.own[c] + added[c])
                    self.ut[c] = Is / self.ell[c]
                    I = I - np.outer(self.ut[c], self.ut[c])
                if self.parent[b] >= 0:
                    held[self.parent[b]] += self._lift(I, self.offset[b])

    @property
    def pivots(self) -> np.ndarray:
        """Each coordinate's pivot ``ell_c^2``."""
        return self.ell * self.ell

    @property
    def own_pivots(self) -> np.ndarray:
        """Each coordinate's pivot before its own added diagonal, ``s_c^T I s_c`` (the split pivot's, SAT_PHYS §1.2:
        formed before a drive's ``K`` is added, so no cancellation takes it back out)."""
        return self.own

    def _lift(self, inertia: np.ndarray, d: np.ndarray) -> np.ndarray:
        """``inertia`` at a body's frame, taken at its parent's (``d`` = the parent's frame less the body's)."""
        D = np.array([[0.0, -d[2], d[1]], [d[2], 0.0, -d[0]], [-d[1], d[0], 0.0]], dtype=self.dtype)
        to = np.eye(6, dtype=self.dtype)
        to[:3, 3:] = D  # a twist at the parent's frame, taken at the body's: v + d x w
        wrench = np.eye(6, dtype=self.dtype)
        wrench[3:, :3] = -D  # a wrench at the body's frame, taken at the parent's: t - d x f
        return wrench @ inertia @ to

    def _step(self, b: int, p: np.ndarray, x: np.ndarray | None, z: np.ndarray | None) -> np.ndarray:
        """Walk body ``b``'s coordinates, leaf first, from the state ``p`` at its frame: store the footprints in
        ``z`` (when given), with ``x`` a covector's own entries, and return the state its joint passes."""
        for c in range(self.first[b + 1] - 1, self.first[b] - 1, -1):
            footprint = self.s[c] @ p if x is None else x[c] + self.s[c] @ p
            footprint = footprint / self.ell[c]
            if z is not None:
                z[c] = footprint
            p = p - np.multiply.outer(self.ut[c], footprint)
        return p

    def _carry(self, p: np.ndarray, b: int) -> np.ndarray:
        """The state ``p`` passed by body ``b``'s joint, at the parent's frame: ``t - d x f``."""
        d = self.offset[b]
        return np.concatenate((p[:3], p[3:] - _cross(d, p[:3])))

    def up(self, body=None, wrench=None, covector=None) -> tuple[np.ndarray, np.ndarray]:
        """Footprints ``Z = L^-1 P J^T`` of rows seeded by ``wrench`` at ``body`` (a wrench ``(f, t)`` at the body's
        frame, ``[6]`` or one column per row ``[6, m]``, with ``body`` one body or one per row) and by ``covector`` on
        the coordinates (``[n]`` or ``[n, m]``): a limit's, a mimic's, or any row's ``J``.

        Returns:
            The footprints ``[n]`` or ``[n, m]`` by coordinate, and the state each root passes to the world
            ``[bodies, 6, ...]`` (zero above a free root).
        """
        batch = () if wrench is None and covector is None else None
        if wrench is not None:
            wrench = np.asarray(wrench, dtype=self.dtype)
            batch = wrench.shape[1:]
        if covector is not None:
            covector = np.asarray(covector, dtype=self.dtype)
            batch = covector.shape[1:]
        count = self.parent.size
        state = np.zeros((count, 6, *batch), dtype=self.dtype)
        active = np.zeros(count, dtype=bool)
        if wrench is not None:
            if np.ndim(body) == 0:
                state[int(body)] += wrench
                active[int(body)] = True
            else:
                for row, b in enumerate(np.asarray(body).tolist()):
                    state[b, :, row] += wrench[:, row]
                    active[b] = True
        if covector is not None:
            touched = np.flatnonzero(np.any(covector.reshape(covector.shape[0], -1) != 0.0, axis=1))
            for c in touched:
                active[np.searchsorted(self.first, c, side="right") - 1] = True
        z = np.zeros((self.s.shape[0], *batch), dtype=self.dtype)
        for b in range(count - 1, -1, -1):
            if not active[b]:
                continue
            state[b] = self._step(b, state[b], covector, z)
            if self.parent[b] >= 0:
                state[self.parent[b]] += self._carry(state[b], b)
                active[self.parent[b]] = True
                state[b] = 0.0
        return z, state

    def down(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The leave ``P^T L^-T u`` of ``u`` (``[n]`` or ``[n, m]``, by coordinate as :meth:`up` returns them).

        Returns:
            The joint velocities ``[n, ...]`` and each body's twist ``(v, w)`` at its frame ``[bodies, 6, ...]``.
        """
        u = np.asarray(u, dtype=self.dtype)
        count = self.parent.size
        twist = np.zeros((count, 6, *u.shape[1:]), dtype=self.dtype)
        qd = np.zeros_like(u)
        for b in range(count):
            p = self.parent[b]
            if p >= 0:
                # The parent's twist at this body's frame: v + d x w, d = the parent's frame less this one.
                V = twist[p]
                V = np.concatenate((V[:3] + _cross(self.offset[b], V[3:]), V[3:]))
            else:
                V = twist[b]
            for c in range(self.first[b], self.first[b + 1]):
                qd[c] = (u[c] - self.ut[c] @ V) / self.ell[c]
                V = V + np.multiply.outer(self.s[c], qd[c])
            twist[b] = V
        return qd, twist

    def ancestors(self, b: int) -> list[int]:
        """``b`` and its ancestors, ``b`` first."""
        path = []
        while b >= 0:
            path.append(int(b))
            b = int(self.parent[b])
        return path

    def carry(self, body: int, wrench: np.ndarray, to: int) -> np.ndarray:
        """``wrench`` at ``body`` walked up to its ancestor ``to``: the state that reaches ``to``'s frame, before its
        coordinates (``Psi`` of Z_UNION §1.2)."""
        p = np.asarray(wrench, dtype=self.dtype)
        b = body
        while b != to:
            if b < 0:
                raise ValueError(f"Body {to} is not an ancestor of body {body}")
            p = self._carry(self._step(b, p, None, None), b)
            b = int(self.parent[b])
        return p

    def meet(self, k: int, w_k: np.ndarray, l: int, w_l: np.ndarray) -> np.ndarray:
        """``Z_k^T Z_l`` of the rows seeded by ``w_k`` at ``k`` and ``w_l`` at ``l`` (``[6]`` or ``[6, m]``): both
        walks carried to their lowest common ancestor ``c``, met through ``Gamma_c^T Gamma_c``, the Gram of the
        footprints of ``c``'s six unit wrenches (never the 6x6 recurrence, which loses float32 on deep chains)."""
        path = set(self.ancestors(l))
        c = next((b for b in self.ancestors(k) if b in path), -1)
        w_k, w_l = np.asarray(w_k, dtype=self.dtype), np.asarray(w_l, dtype=self.dtype)
        if c < 0:
            return np.zeros((*w_k.shape[1:], *w_l.shape[1:]), dtype=self.dtype)
        gamma, _ = self.up(c, np.eye(6, dtype=self.dtype))
        return self.carry(k, w_k, c).T @ (gamma.T @ gamma) @ self.carry(l, w_l, c)

    def device_data(self) -> tuple[np.ndarray, np.ndarray]:
        """The walk's data as the device walks it (:func:`walk_prelude`), float32: each coordinate's record
        (:func:`pack_walk_records` of ``s``, ``ut`` and ``1 / ell``), and each body's link (:data:`WALK_LINK`), its
        frame point (:meth:`frame_points`)."""
        data = pack_walk_records(self.s, self.ut, 1.0 / self.ell, np.zeros(self.s.shape[0]))
        link = np.zeros((self.parent.size, WALK_LINK), dtype=np.float32)
        link[:, 0:3] = self.frame_points()
        return data, link

    def frame_points(self) -> np.ndarray:
        """Each body's frame point with the roots at the origin: its parent's less its offset (``[bodies, 3]``)."""
        frames = np.zeros((self.parent.size, 3))
        for b, up in enumerate(self.parent):
            if up >= 0:
                frames[b] = frames[up] - np.asarray(self.offset[b], dtype=np.float64)
        return frames

    @property
    def path_table(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """The tree's path table (:func:`walk_prelude`): each body's parent, and each body's first coordinate with the
        count after the last."""
        return tuple(int(b) for b in self.parent), tuple(int(c) for c in self.first)


# ---------------------------------------------------------------------------------------------------------------------
# The walk on the device (FL3.1): a lane per seed up, a lane per tree down, the path table in constant memory
# ---------------------------------------------------------------------------------------------------------------------

WALK_COORDINATE = 12
"""Floats per coordinate of the device walk's record, three aligned ``float4`` (48 B): its column's non-zero half and
``1 / ell``; ``ut`` 0..3; ``ut`` 4..5, the pivot without any drive's ``K`` (``P0 = s^T I s + R``, the split pivot's,
SAT_PHYS §1.2) and the side offset. A column at its body's frame has one non-zero half for every joint but a D6 joint's
turns after its slides and a free joint's turns below a root: the record names its half by ``1 / ell``'s sign bit (set:
the angular half), and a column with both halves keeps its linear half in a side ``float4`` whose offset in floats from
the record its last word holds (zero: none; :func:`pack_walk_records`). Readers take it unpacked, :data:`WALK_UNPACKED`
floats (``s`` 6, ``ut`` 6, ``1 / ell``, ``P0``), through ``WALK_READ``."""

WALK_UNPACKED = 14
"""Floats of a coordinate's record unpacked (``WALK_READ``): ``s`` 6, ``ut`` 6, ``1 / ell``, ``P0``."""

WALK_LINK = 4
"""Floats per body of the device walk's data: its frame's world point and a pad, one aligned ``float4``. A walk forms a
body's offset to its parent's frame from the two frame points (``WALK_READ_LINK``, ``WALK_LINK_IN``), the float32
difference the refresh takes (``WALK_OFFSET``); the step's links (:attr:`WalkTrees.now_link`) take the same layout."""

WALK_PIVOT = 13
"""Where a coordinate's unpacked record holds its pivot without ``K`` (:data:`WALK_UNPACKED`)."""


def pack_walk_records(s: np.ndarray, ut: np.ndarray, inverse: np.ndarray, pivot: np.ndarray) -> np.ndarray:
    """Coordinates' records as the refresh stores them (:data:`WALK_COORDINATE`), float32: each column ``s`` (``[n,
    6]``) by its non-zero half, ``ut``, ``1 / ell`` and ``P0``; with any column of two halves the rows double, the side
    ``float4`` of coordinate ``c`` at row ``n + c`` (its offset ``12 n`` floats)."""
    s = np.asarray(s, dtype=np.float32).reshape(-1, 6)
    n = s.shape[0]
    turns = np.any(s[:, 3:6] != 0.0, axis=1)
    both = turns & np.any(s[:, 0:3] != 0.0, axis=1)
    data = np.zeros((2 * n if both.any() else n, WALK_COORDINATE), dtype=np.float32)
    data[:n, 0:3] = np.where(turns[:, None], s[:, 3:6], s[:, 0:3])
    data[:n, 3] = np.where(turns, -1.0, 1.0) * np.asarray(inverse, dtype=np.float32).reshape(n)
    data[:n, 4:10] = np.asarray(ut, dtype=np.float32).reshape(n, 6)
    data[:n, 10] = np.asarray(pivot, dtype=np.float32).reshape(n)
    if both.any():
        data[:n, 11] = np.where(both, np.int32(WALK_COORDINATE * n), np.int32(0)).view(np.float32)
        data[n:, 0:3] = np.where(both[:, None], s[:, 0:3], 0.0)
    return data


def unpack_walk_records(data: np.ndarray, n: int) -> np.ndarray:
    """The first ``n`` records of ``data`` (:func:`pack_walk_records`) unpacked, :data:`WALK_UNPACKED` floats a
    coordinate (``s``, ``ut``, ``1 / ell``, ``P0``), as ``WALK_READ`` reads them."""
    data = np.asarray(data, dtype=np.float32)
    flat = data.reshape(-1)
    out = np.zeros((n, WALK_UNPACKED), dtype=np.float32)
    turns = np.signbit(data[:n, 3])
    side = data[:n, 11].view(np.int32)
    lin = np.zeros((n, 3), dtype=np.float32)
    for c in np.flatnonzero(side):
        at = WALK_COORDINATE * c + int(side[c])
        lin[c] = flat[at : at + 3]
    out[:, 0:3] = np.where(turns[:, None], lin, data[:n, 0:3])
    out[:, 3:6] = np.where(turns[:, None], data[:n, 0:3], 0.0)
    out[:, 6:12] = data[:n, 4:10]
    out[:, 12] = np.abs(data[:n, 3])
    out[:, 13] = data[:n, 10]
    return out


WALK_SLIP_BATCH = 8
"""The steps of a slip's transposed release a walk loads together (``WALK_SLIP_BACK``: the leave's, the audit's): one
global round trip a batch of its path, in place of one a step."""

WALK_FRAME = 0
"""Where a body's link holds its frame's world point (:data:`WALK_LINK`), at the refresh's poses: the factor's
inertias are taken there."""

WALK_NOW = 6
"""Floats per coordinate of the step's columns (:attr:`WalkTrees.now`, a ``wp.spatial_vector`` each, ``S``'s layout):
in a model without a column of two halves ``S`` itself (``joint_S_s``, which the finish stores anyway), whose half at the
body's frame the record names (its other half the lever about the origin a turn carries, which the frame drops); with
one, the finish's columns at each body's frame, both halves (``WALK_READ_COLUMN``). Its links,
:attr:`WalkTrees.now_link`, take :data:`WALK_LINK`'s layout."""


_WALK_TABLE = """    static __constant__ int WALK_PARENT[$bodies] = {$parent};
    static __constant__ int WALK_FIRST[$ends] = {$first};
    const int WALK_BODIES = $bodies;
    const auto WALK_PARENT_OF = [&](const int b) -> int { return WALK_PARENT[b]; };
    const auto WALK_FIRST_OF = [&](const int b) -> int { return WALK_FIRST[b]; };"""

_WALK_TABLE_CPU = """    static const int WALK_PARENT[$bodies] = {$parent};
    static const int WALK_FIRST[$ends] = {$first};
    const int WALK_BODIES = $bodies;
    const auto WALK_PARENT_OF = [&](const int b) -> int { return WALK_PARENT[b]; };
    const auto WALK_FIRST_OF = [&](const int b) -> int { return WALK_FIRST[b]; };"""

_WALK_PATHS = """    // The tree a lane walks (WALK_TREE): its path table, each body's parent and first coordinate as int pairs
    // (TreeFactor.walk_path), and its bodies and coordinates. Set before each tree's walk.
    const int* WALK_PATH = nullptr;
    int WALK_BODIES = 0, WALK_COORDINATES = 0;
    const auto WALK_TREE = [&](const int* path, const int bodies, const int coordinates) {
        WALK_PATH = path;
        WALK_BODIES = bodies;
        WALK_COORDINATES = coordinates;
    };
    const auto WALK_PARENT_OF = [&](const int b) -> int { return WALK_PATH[2 * b]; };
    const auto WALK_FIRST_OF = [&](const int b) -> int {
        return b < WALK_BODIES ? WALK_PATH[2 * b + 1] : WALK_COORDINATES;
    };"""

_WALK = """    // The walk of a tree factor (Z_UNION section 1.2, tree_factor.TreeWalk) over the trees of a path table above
    // (WALK_PARENT_OF, WALK_FIRST_OF, WALK_BODIES): each body's parent (-1: the world, every parent before its
    // children) and first coordinate. data holds each coordinate's s, ut ((v, w), six each) and 1 / ell
    // (WALK_COORDINATE floats), link each body's offset to its parent's frame (WALK_LINK floats), as
    // TreeWalk.device_data and TreeFactor.walk_step lay them out.
    //
    // WALK_UP(data, link, now, now_link, shift, body, top, p, w, sink): the states p of $rows rows, p[6 r + k] the wrench
    // (f, t) of row r at body's frame, walked up to the root from coordinate top of body (its last for a wrench seed):
    // each coordinate c takes the footprint z = (s_c . p) / ell_c, given to sink(r, c, z, j), and passes p - ut_c z; a
    // joint passes t - d x f to its parent. now (null for rows seeded by covectors alone): the step's geometry the
    // finish published, each coordinate's column N_c at its body's frame ($now floats a coordinate) and each body's
    // frame point in now_link (WALK_LINK floats a body, the point at WALK_FRAME), with w[6 r + k] row r's seed wrench at
    // the current body's frame (link's frame point, the factor's). The frame points are the articulation's, relative
    // to its origin: shift is the factor's origin less the step's (TreeFactor's walk_origin, zero on a refresh step).
    // A seed enters as the step's J and the factor whitens
    // it (a held factor between refreshes, L^-1 P J^T as the dense rows do): the step's column of c at the factor's
    // frame is s_now_c = (N_v + N_w x (frame - frame_now), N_w), the J entry j = s_now_c . w (the point is any one: w
    // is at the same), and p is the seed's wrench less what the factor has taken, so the footprint takes
    // (s_now_c - s_c) . w more; w passes each joint as p does. The shift is the body's motion since the refresh, small,
    // and zero on a refresh step, where the two geometries are one: one residual state keeps FL3.1's precision with no
    // lever about the articulation's origin. The sink's j is the step's J entry (zero without now), so a row's incident
    // J v is the sum of j v_c along its walk.
    const auto WALK_UP = [&](const float* data, const float* link, const float* now, const float* now_link,
                             const float* shift, int body, int top, float* p, float* w, auto&& sink) {
$up_head
$record_c
                // The step's column at the factor's frame: its linear part shifted there from the step's frame, its
                // angular part as is (the frames read at each coordinate, so no register holds them across the walk;
                // once for every row).
$column_c
                float e0 = 0.0f, e1 = 0.0f, e2 = 0.0f;
                if (now != nullptr) {
$point_b
                    const float o0 = (l[$frame] - f[0]) + shift[0], o1 = (l[$frame + 1] - f[1]) + shift[1],
                                o2 = (l[$frame + 2] - f[2]) + shift[2];
                    e0 = S[0] + (S[4] * o2 - S[5] * o1);
                    e1 = S[1] + (S[5] * o0 - S[3] * o2);
                    e2 = S[2] + (S[3] * o1 - S[4] * o0);
                }
                for (int r = 0; r < $rows; ++r) {
                    float* q = p + 6 * r;
                    float x = $dot_q;
                    float j = 0.0f;
                    if (now != nullptr) {
                        const float* v = w + 6 * r;
                        x += $dot_pose;
                        j = $dot_j;
                    }
                    $step_q
                    sink(r, c, z, j);
                }
$up_tail
    };
    // WALK_SEED(data, c, x, p, sink): the first step of rows seeded by the covector x[r] at coordinate c (a limit's, a
    // velocity limit's), on the states p: z = (x + s_c . p) / ell_c, its J entry x[r]. WALK_UP continues from c - 1 of
    // c's body.
    const auto WALK_SEED = [&](const float* data, int c, const float* x, float* p, auto&& sink) {
$record_seed
        for (int r = 0; r < $rows; ++r) {
            float* q = p + 6 * r;
            $seed_q
            sink(r, c, z, x[r]);
        }
    };
    // WALK_PARENT_TWIST(link, twist, b, V): body b's parent's twist (v, w) at b's frame, v + d x w (zero at a root).
    const auto WALK_PARENT_TWIST = [&](const float* link, const float* twist, int b, float* V) {
        const int up = WALK_PARENT_OF(b);
        if (up < 0) {
            for (int k = 0; k < 6; ++k) V[k] = 0.0f;
            return;
        }
        const float* W = twist + 6 * up;
$link_twist
        $edge_twist
    };
    // WALK_ENTER(data, link, qd, u, twist): the walk's coordinates of known velocities qd, u = L^T P qd (the leave
    // inverted), root first: u_c = ell_c qd_c + ut_c . V, then V += s_c qd_c, V its body's twist (the audit's).
    const auto WALK_ENTER = [&](const float* data, const float* link, const float* qd, float* u, float* twist) {
        for (int b = 0; b < WALK_BODIES; ++b) {
            float V[6];
            WALK_PARENT_TWIST(link, twist, b, V);
            for (int c = WALK_FIRST_OF(b); c < WALK_FIRST_OF(b + 1); ++c) {
$record_enter
                $enter_V
            }
            for (int k = 0; k < 6; ++k) twist[6 * b + k] = V[k];
        }
    };
    // The slips' release of tree art (WalkRelease, written by the predictor's border; W9's verdict: R = L_F^-1 L_K,
    // tree-lower, so every row keeps its support). Slip j's steps along its root path, leaf first, start at
    // at = (art * cap + j) * steps of release.coordinate, .b, .a, .g; a coordinate of -1 ends the path.
    // WALK_SLIP_ON(release, at, z, rows, stride, place): z <- R_j z on rows rows (at most $rows) of a footprint whose
    // coordinate c sits at z[place(c) + r * stride] (place(c) < 0 where the footprint's support does not hold c: zero
    // there, and every coordinate of the path below the support's is off it), leaf first, a step at a time: its
    // readers are row kernels at their register line (the support writer, the local owners' packets), where a batch's
    // registers cost more than its round trips save (measured, HANDOFF_ZW2 §3).
    // The transposed forms (WALK_SLIP_BACK: the leave, a thread a tree, the audit) walk a slip's steps in batches of
    // $batch: the batch's coordinates and coefficients loaded together, then its entries, then the recurrence in
    // registers, then its stores. A path's coordinates are distinct, so no step's load waits on another step's store
    // (one global round trip a batch, not a step).
    // WALK_SLIP_T(release, at, u): u <- R_j^T u, root first. WALK_SLIP_IT(release, at, u): u <- R_j^-T u, root first.
    // WALK_RELEASE_ON(release, art, z, rows, stride, place): a row's footprints z <- R z, the first slip first;
    // WALK_RELEASE(release, art, z, stride) the same on one row whose coordinate c sits at z[c * stride].
    // WALK_RELEASE_T(release, art, u): the leave's u <- R^T u, the last slip first.
    // WALK_RELEASE_IT(release, art, u): the audit's coordinates u <- R^-T u, the first slip first.
    const auto WALK_SLIP_ON = [&](const auto& release, const size_t at, float* z, const int rows, const int stride,
                                  auto&& place) {
        const int steps = release.b.shape[2];
        const int* coordinate = release.coordinate.data + at;
        const float *b = release.b.data + at, *a = release.a.data + at, *g = release.g.data + at;
        float s[$rows];
        for (int r = 0; r < $rows; ++r) s[r] = 0.0f;
        for (int k = 0; k < steps; ++k) {
            const int c = coordinate[k];
            if (c < 0) break;
            const int e = place(c);
            if (e < 0) continue;
            for (int r = 0; r < rows; ++r) {
                $slip_on
            }
        }
    };
    // The transposed forms, root first: a batch of the steps above the last one taken, the path's end (-1) skipped.
    const auto WALK_SLIP_BACK = [&](const auto& release, const size_t at, float* u, auto&& step) {
        const int steps = release.b.shape[2];
        const int* coordinate = release.coordinate.data + at;
        const float *b = release.b.data + at, *a = release.a.data + at, *g = release.g.data + at;
        float t = 0.0f;
        for (int k1 = steps - 1; k1 >= 0; k1 -= $batch) {
            int cq[$batch];
            float bk[$batch], ak[$batch], gk[$batch], x[$batch];
#pragma unroll
            for (int q = 0; q < $batch; ++q) {
                const int k = k1 - q;
                cq[q] = k >= 0 ? coordinate[k] : -1;
                bk[q] = k >= 0 ? b[k] : 0.0f;
                ak[q] = k >= 0 ? a[k] : 0.0f;
                gk[q] = k >= 0 ? g[k] : 0.0f;
            }
#pragma unroll
            for (int q = 0; q < $batch; ++q) x[q] = cq[q] >= 0 ? u[cq[q]] : 0.0f;
#pragma unroll
            for (int q = 0; q < $batch; ++q)
                if (cq[q] >= 0) x[q] = step(x[q], t, bk[q], ak[q], gk[q]);
#pragma unroll
            for (int q = 0; q < $batch; ++q)
                if (cq[q] >= 0) u[cq[q]] = x[q];
        }
    };
    const auto WALK_SLIP_T = [&](const auto& release, const size_t at, float* u) {
        WALK_SLIP_BACK(release, at, u, [](const float v, float& t, const float b, const float a, const float g) {
            $slip_t
        });
    };
    const auto WALK_SLIP_IT = [&](const auto& release, const size_t at, float* u) {
        WALK_SLIP_BACK(release, at, u, [](const float v, float& t, const float b, const float a, const float g) {
            $slip_it
        });
    };
    const auto WALK_RELEASE_ON = [&](const auto& release, const int art, float* z, const int rows, const int stride,
                                     auto&& place) {
        const int cap = release.b.shape[1], steps = release.b.shape[2];
        for (int j = 0; j < release.count.data[art]; ++j)
            WALK_SLIP_ON(release, ((size_t)art * cap + j) * steps, z, rows, stride, place);
    };
    const auto WALK_RELEASE = [&](const auto& release, const int art, float* z, const int stride = 1) {
        WALK_RELEASE_ON(release, art, z, 1, 0, [&](const int c) { return c * stride; });
    };
    const auto WALK_RELEASE_T = [&](const auto& release, const int art, float* u) {
        const int cap = release.b.shape[1], steps = release.b.shape[2];
        for (int j = release.count.data[art] - 1; j >= 0; --j) WALK_SLIP_T(release, ((size_t)art * cap + j) * steps, u);
    };
    const auto WALK_RELEASE_IT = [&](const auto& release, const int art, float* u) {
        const int cap = release.b.shape[1], steps = release.b.shape[2];
        for (int j = 0; j < release.count.data[art]; ++j) WALK_SLIP_IT(release, ((size_t)art * cap + j) * steps, u);
    };"""


_UP_HEAD, _UP_TAIL = ts.root_path(
    "body",
    "top",
    begin=["$link_b"],
    joint=[
        "for (int r = 0; r < $rows; ++r) {",
        "    float* q = p + 6 * r;",
        "    $edge_q",
        "}",
        "if (now != nullptr) {",
        "    for (int r = 0; r < $rows; ++r) {",
        "        float* v = w + 6 * r;",
        "        $edge_v",
        "    }",
        "}",
    ],
)
_WALK = Template(_WALK).safe_substitute(up_head=ts.indent_text(_UP_HEAD, 8), up_tail=ts.indent_text(_UP_TAIL, 8))
"""``WALK_UP``'s root path, leaf first (:func:`.tree_scan.root_path`): each body's link where the path opens it, its
coordinates, the rows' and the seed wrenches' joint to its parent."""

_POSE = ("(e0 - d[0])", "(e1 - d[1])", "(e2 - d[2])", "(S[3] - d[3])", "(S[4] - d[4])", "(S[5] - d[5])")
_WALK_TEXTS = {
    "dot_q": ts.dot(ts.LOADED.s, ts.channels("q")),
    "dot_pose": ts.dot(_POSE, ts.channels("v")),
    "dot_j": ts.dot(("e0", "e1", "e2", "S[3]", "S[4]", "S[5]"), ts.channels("v")),
    "step_q": ts.statements(ts.step(ts.Direction.UP, ts.LOADED, "q", value="x"), 20),
    "seed_q": ts.statements(ts.step(ts.Direction.UP, ts.LOADED, "q", source="x[r]"), 12),
    "edge_q": ts.statements(ts.edge(ts.Direction.UP, ts.LOADED.lever, "q"), 20),
    "edge_v": ts.statements(ts.edge(ts.Direction.UP, ts.LOADED.lever, "v"), 24),
    "edge_twist": ts.statements(ts.edge(ts.Direction.DOWN, ts.LOADED.lever, "W", into="V"), 8),
    "enter_V": ts.statements(ts.enter(ts.LOADED, "V", "qd[c]", "u[c]"), 16),
    "slip_on": ts.statements(
        ts.slip("R", "z[e + r * stride]", "s[r]", "b[k]", "a[k]", "g[k]", "z[e + r * stride] = {};"), 16
    ),
    "slip_t": ts.statements(ts.slip("RT", "v", "t", "b", "a", "g", "return {};"), 12),
    "slip_it": ts.statements(ts.slip("RIT", "v", "t", "b", "a", "g", "return {};"), 12),
}
"""The path walk's arithmetic (:data:`_WALK`): the rows' steps up (the held pose a source of statements, ``x``), a
covector seed's first step, the joints of the rows' and the seed wrenches' up and of a parent's twist down, and the
audit's step in (``ENTER``)."""


_WALK_LOADS = """#ifndef WALK_READ
// The walk's field groups and their one access, the one place the device layout lives (WALK_COORDINATE, WALK_LINK,
// WALK_NOW), one text for both devices: a coordinate's record, three aligned quads (its column's non-zero half and
// 1 / ell; ut 0..3; ut 4..5, P0 and the side offset); a body's link, its frame point, one quad, read with its parent's;
// a step column, six floats (WALK_NOW, S's layout); a step link's frame point, one quad. Each read
// takes its load kind, the reader's data: WALK_LDG the read-only path (no walk writes them while a reader runs; ptxas
// may move it down to its use), WALK_PLAIN a coherent load issued where the text places it (the class emitter's, a
// store of the chain after it keeps it there). A kernel's staged copy is read by pointer (tree_factor._walk_loads).
// WALK_READ(load, d, p, half): record p as the walk's WALK_UNPACKED floats d: s, ut, 1 / ell, P0. half is the
// column's half as the reader knows it, its class's data: 0 linear, 1 angular (compiled: no flag, no side load),
// else named by the record (1 / ell's sign bit set: the angular half), 3 (WALK_EITHER) in a model without a column
// of two halves, 2 (WALK_BOTH) with the side quad of one (the linear half; its offset in floats the record's last
// word). The half not stored is zero: the same products as a stored zero's, up to the sign of a zero.
#define WALK_READ(load, d, p, half)                                                                             \
    {                                                                                                          \
        const float* walk_p_ = (p);                                                                            \
        const WALK_QUAD walk_a_ = load(walk_p_), walk_b_ = load(walk_p_ + 4), walk_e_ = load(walk_p_ + 8);    \
        WALK_QUAD walk_v_ = walk_a_;                                                                           \
        walk_v_.x = walk_v_.y = walk_v_.z = 0.0f;                                                              \
        bool walk_t_ = (half) == 1;                                                                            \
        if ((half) > 1) {                                                                                      \
            const int walk_s_ = WALK_BITS(walk_e_.w);                                                          \
            if ((half) == 2 && walk_s_ != 0) walk_v_ = load(walk_p_ + walk_s_);                                \
            walk_t_ = WALK_BITS(walk_a_.w) < 0;                                                                \
        }                                                                                                      \
        d[0] = walk_t_ ? walk_v_.x : walk_a_.x; d[1] = walk_t_ ? walk_v_.y : walk_a_.y;                        \
        d[2] = walk_t_ ? walk_v_.z : walk_a_.z; d[3] = walk_t_ ? walk_a_.x : 0.0f;                             \
        d[4] = walk_t_ ? walk_a_.y : 0.0f; d[5] = walk_t_ ? walk_a_.z : 0.0f;                                  \
        d[6] = walk_b_.x; d[7] = walk_b_.y; d[8] = walk_b_.z; d[9] = walk_b_.w; d[10] = walk_e_.x;             \
        d[11] = walk_e_.y; d[12] = WALK_ABS(walk_a_.w); d[13] = walk_e_.z;                                     \
    }
// WALK_READ_COLUMN(load, S, p, r, half): a coordinate's step column at its body's frame as six floats S, from its row p
// of the step's columns (WALK_NOW: S's own, or the finish's at the frame) and its record r: the half WALK_READ's names
// (its pair loads, load_PAIR), the other zero; with WALK_BOTH a column of two halves (the record's side) whole.
#define WALK_READ_COLUMN(load, S, p, r, half)                                                                   \
    {                                                                                                          \
        const float* walk_p_ = (p);                                                                            \
        const auto walk_x_ = load##_PAIR(walk_p_), walk_y_ = load##_PAIR(walk_p_ + 2),                         \
                   walk_z_ = load##_PAIR(walk_p_ + 4);                                                         \
        bool walk_l_ = (half) == 0, walk_t_ = (half) == 1;                                                     \
        if ((half) > 1) {                                                                                      \
            walk_t_ = WALK_BITS(load(r).w) < 0;                                                                \
            walk_l_ = !walk_t_ || ((half) == 2 && WALK_BITS(load((r) + 8).w) != 0);                            \
        }                                                                                                      \
        S[0] = walk_l_ ? walk_x_.x : 0.0f; S[1] = walk_l_ ? walk_x_.y : 0.0f; S[2] = walk_l_ ? walk_y_.x : 0.0f; \
        S[3] = walk_t_ ? walk_y_.y : 0.0f; S[4] = walk_t_ ? walk_z_.x : 0.0f; S[5] = walk_t_ ? walk_z_.y : 0.0f; \
    }
// WALK_READ_LINK(load, l, p, b, up): body b's link as l, its offset to its parent up's frame (zero at a root, up < 0),
// the difference of the two frame points, the refresh's WALK_OFFSET, the same float32 subtraction; then its frame
// point.
#define WALK_READ_LINK(load, l, p, b, up)                                                                       \
    {                                                                                                          \
        const WALK_QUAD walk_f_ = load((p) + $link * (b));                                                     \
        l[0] = l[1] = l[2] = 0.0f;                                                                             \
        if ((up) >= 0) {                                                                                       \
            const WALK_QUAD walk_u_ = load((p) + $link * (up));                                                \
            l[0] = walk_u_.x - walk_f_.x; l[1] = walk_u_.y - walk_f_.y; l[2] = walk_u_.z - walk_f_.z;          \
        }                                                                                                      \
        l[3] = walk_f_.x; l[4] = walk_f_.y; l[5] = walk_f_.z;                                                  \
    }
// WALK_READ_POINT(load, f, p): a step link's frame point and pad as f.
#define WALK_READ_POINT(load, f, p)                                                                             \
    {                                                                                                          \
        const WALK_QUAD walk_a_ = load(p);                                                                     \
        f[0] = walk_a_.x; f[1] = walk_a_.y; f[2] = walk_a_.z; f[3] = walk_a_.w;                                \
    }
// WALK_SHIFT(w, a, s): articulation a's factor origin less its step origin (WalkTrees.origin, .now_origin), the shift
// a factor's frame point takes to meet the step's (both the articulation's, relative to its origin).
#define WALK_SHIFT(w, a, s)                                                                                     \
    {                                                                                                          \
        const wp::vec3 walk_o_ = (w).origin.data[a], walk_n_ = (w).now_origin.data[a];                         \
        for (int walk_k_ = 0; walk_k_ < 3; ++walk_k_) s[walk_k_] = walk_o_[walk_k_] - walk_n_[walk_k_];        \
    }
// WALK_MUL(a, b): a walk step's output, a product rounded on its own: the chain's coordinate is a float32 value before
// any consumer adds it, whatever block the consumer sits in.
$device
#endif"""

_WALK_DEVICE = {
    True: """#define WALK_QUAD float4
#define WALK_LDG(p) __ldg(reinterpret_cast<const float4*>(p))
#define WALK_PLAIN(p) (*reinterpret_cast<const float4*>(p))
#define WALK_LDG_PAIR(p) __ldg(reinterpret_cast<const float2*>(p))
#define WALK_PLAIN_PAIR(p) (*reinterpret_cast<const float2*>(p))
#define WALK_LOAD_PIVOT(p) __ldg((p) + 10)
#define WALK_BITS(x) __float_as_int(x)
#define WALK_ABS(x) fabsf(x)
#define WALK_MUL(a, b) __fmul_rn((a), (b))""",
    False: """// The CPU's quad is Warp's (four named floats); its loads are plain. Its WALK_MUL is a function boundary, as CUDA's
// __fmul_rn: a product inside the consumer's expression could be contracted into it.
#define WALK_QUAD wp::quatf
#define WALK_LDG(p) (wp::quatf((p)[0], (p)[1], (p)[2], (p)[3]))
#define WALK_PLAIN(p) WALK_LDG(p)
#define WALK_LDG_PAIR(p) (wp::quatf((p)[0], (p)[1], 0.0f, 0.0f))
#define WALK_PLAIN_PAIR(p) WALK_LDG_PAIR(p)
#define WALK_LOAD_PIVOT(p) ((p)[10])
#define WALK_BITS(x) __builtin_bit_cast(int, x)
#define WALK_ABS(x) __builtin_fabsf(x)
#define WALK_MUL(a, b) ([](const float walk_a_, const float walk_b_) -> float { return walk_a_ * walk_b_; }((a), (b)))""",
}
"""The device's words of :data:`_WALK_LOADS`: its quad, its load kinds, the record's pivot, the bit and absolute
value, the step's pin. Everything else of the access is one text."""

WALK_LINK_FRAME = 3
"""Where a body's link, as a walk reads it (``WALK_READ_LINK``'s registers, or a kernel's staged copy), holds its frame
point: after its offset to its parent's frame."""


def _walk_loads_text(cuda: bool) -> str:
    """The walk's one access (:data:`_WALK_LOADS`) and the step's pin ``WALK_MUL`` at the device layout, for a device,
    under its one guard: every walk prelude carries it."""
    return substitute(_WALK_LOADS, link=WALK_LINK, device=_WALK_DEVICE[cuda])


@functools.cache
def walk_loads_prelude() -> DevicePrelude:
    """The walk's one access alone (:func:`_walk_loads_text`), for both devices: a reader of the records without a
    walk (the predictor's border) declares it."""
    return DevicePrelude("walk_loads", _walk_loads_text(True), cpu=_walk_loads_text(False))


def _walk_loads(coordinate: int, link: int, now: int, half: int = WALK_BOTH) -> dict[str, str]:
    """The shared walk text's reads (``_WALK``'s ``$record_*``, ``$link_*``, ``$column_c``, ``$point_b`` slots) for a
    layout: the device layout by the one access (:data:`_WALK_LOADS`, the read-only load kind, its records and columns
    of ``half``), a kernel's staged copy of it (another stride, in shared memory, unpacked) by pointer."""
    device = (coordinate, link, now) == (WALK_COORDINATE, WALK_LINK, WALK_NOW)

    def record(indent):
        pad = " " * indent
        if device:
            return f"{pad}float d[{WALK_UNPACKED}];\n{pad}WALK_READ(WALK_LDG, d, data + {WALK_COORDINATE} * c, {half});"
        return f"{pad}const float* d = data + {coordinate} * c;"

    def body_link(indent, up):
        pad = " " * indent
        if link == WALK_LINK:
            return f"{pad}float l[6];\n{pad}WALK_READ_LINK(WALK_LDG, l, link, b, {up});"
        return f"{pad}const float* l = link + {link} * b;"

    def path_link():
        # A body's link where the root path opens it (ts.root_path's begin: one line, the path's indent).
        return " ".join(line.strip() for line in body_link(0, "WALK_PARENT_OF(b)").split("\n"))

    if device:
        column = (
            "                float S[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};\n"
            f"                if (now != nullptr)\n"
            f"                    WALK_READ_COLUMN(WALK_LDG, S, now + {WALK_NOW} * c, data + {WALK_COORDINATE} * c, {half});"
        )
    else:
        column = f"                const float* S = now != nullptr ? now + {now} * c : d;"
    point = (
        "                    float f[4];\n"
        f"                    WALK_READ_POINT(WALK_LDG, f, now_link + {WALK_LINK} * b + {WALK_FRAME});"
    )
    return {
        "record_c": record(16),
        "record_seed": record(8),
        "record_enter": record(16),
        "link_b": path_link(),
        "link_twist": body_link(8, "up"),
        "column_c": column,
        "point_b": point,
    }


@functools.cache
def walk_prelude(parent: tuple[int, ...], first: tuple[int, ...], rows: int, half: int = WALK_BOTH) -> DevicePrelude:
    """The device walk of a class of trees (one tree, or a world's forest), for both devices: a prelude declaring

    - the class's path table in constant memory (``WALK_PARENT``, ``WALK_FIRST``): topology, one table per class, so a
      lane walks a root path without a global load in front of its steps (GPU_WALK §8, variant C);
    - ``WALK_UP`` and ``WALK_SEED``, the rows: a lane per seed walks its ``rows`` rows up the seed's root path, every
      load shared (Z_UNION Theorem B), and gives each footprint, with the step's ``J`` entry it whitens, to the
      caller's sink, the placement's store;
    - ``WALK_ENTER``, the leave's inverse: a lane per tree, root first (the leave is the class's emitted walk down,
      :func:`walk_down_prelude`, in the step's one leave, :func:`walk_leave_kernel`).

    No step launches it (FL3.1). The data is :meth:`TreeWalk.device_data`'s.

    Args:
        parent: Each body's parent (-1: the world), every parent before its children.
        first: Each body's first coordinate, and the count after the last.
        rows: The rows a lane walks together: the seed's unit (1 a limit, 2 a velocity limit, 3 a point contact, 4
            with spin).
        half: The class's records' half as its walk reads them (``WALK_READ``, :func:`.tree_scan.classes_half`).
    """
    if len(first) != len(parent) + 1 or any(p >= b for b, p in enumerate(parent)):
        raise ValueError("a walk's path table needs every parent before its children and a first coordinate per body")
    if rows < 1:
        raise ValueError(f"a lane walks at least one row, got {rows}")
    table = {
        "bodies": len(parent),
        "ends": len(first),
        "parent": ", ".join(str(b) for b in parent),
        "first": ", ".join(str(c) for c in first),
    }
    walk = substitute(
        _WALK,
        rows=rows,
        **_WALK_TEXTS,
        now=WALK_NOW,
        frame=WALK_LINK_FRAME,
        batch=WALK_SLIP_BATCH,
        **_walk_loads(WALK_COORDINATE, WALK_LINK, WALK_NOW, half),
    )
    return DevicePrelude(
        f"walk_{rows}_{len(parent)}" + ("" if half == WALK_BOTH else f"_{half}"),
        _walk_loads_text(cuda=True) + "\n" + substitute(_WALK_TABLE, table) + "\n" + walk,
        cpu=_walk_loads_text(cuda=False) + "\n" + substitute(_WALK_TABLE_CPU, table) + "\n" + walk,
    )


@functools.cache
def walk_paths_prelude(
    rows: int, *, coordinate: int = WALK_COORDINATE, link: int = WALK_LINK, now: int = WALK_NOW, half: int = WALK_BOTH
) -> DevicePrelude:
    """The device walk of any tree, for both devices: :func:`walk_prelude`'s functions over a path table in global
    memory, one tree at a time (``WALK_TREE(path, bodies, coordinates)`` before its walks), the table
    :attr:`TreeFactor.walk_path` holds for every articulation. The general placement's worlds hold any forest, so its
    one writer and leave read each tree's table where a class's kernel bakes it into constant memory.

    Args:
        rows: The rows a lane walks together (:func:`walk_prelude`).
        coordinate: Floats a coordinate's record, ``link`` a body's and ``now`` a step column's: the device walk's
            layout (:data:`WALK_COORDINATE`, :data:`WALK_LINK`, :data:`WALK_NOW`), or a kernel's staged copy of it, a
            body's link unpacked (its offset, then its frame point at :data:`WALK_LINK_FRAME`)
            (:class:`.response_operator.LocalBlock`: a coordinate's step column interleaved after its
            :data:`WALK_STAGED` floats, so ``now == coordinate``).
        half: The records' and step columns' half as the reader knows it (``WALK_READ``): :data:`.tree_scan.WALK_BOTH`,
            or :data:`.tree_scan.WALK_EITHER` where no column has two halves (:attr:`TreeFactor.walk_side` false,
            :func:`.tree_scan.record_half`). The step's frame points (``now_link``) are read at :data:`WALK_LINK`'s
            layout wherever the rest is.
    """
    if rows < 1:
        raise ValueError(f"a lane walks at least one row, got {rows}")
    walk = (
        _WALK_PATHS
        + "\n"
        + substitute(
            _WALK,
            rows=rows,
            **_WALK_TEXTS,
            now=now,
            frame=WALK_LINK_FRAME,
            batch=WALK_SLIP_BATCH,
            **_walk_loads(coordinate, link, now, half),
        )
    )
    layout = (coordinate, link, now)
    name = f"walk_paths_{rows}" + (
        "" if layout == (WALK_COORDINATE, WALK_LINK, WALK_NOW) else f"_{coordinate}_{link}_{now}"
    )
    name += "" if half == WALK_BOTH else f"_{half}"
    return DevicePrelude(name, _walk_loads_text(True) + "\n" + walk, cpu=_walk_loads_text(False) + "\n" + walk)


_WALK_ROW = """    // WALK_WRENCH(q, sign, d, at, origin, o, torque): a row's seed at a body's frame (the native twin of
    // contact_laws.contact_seed and spin_seed): the force f = sign d at the world point at, its wrench (f, l x f) with
    // l = (at - origin) - o the lever from the body's frame point o (the articulation's, relative to its origin), or
    // the torque (0, f).
    const auto WALK_WRENCH = [](float* q, const float sign, const float* d, const float* at, const float* origin,
                                const float* o, const bool torque) {
        const float f[3] = {sign * d[0], sign * d[1], sign * d[2]};
        const float l[3] = {(at[0] - origin[0]) - o[0], (at[1] - origin[1]) - o[1], (at[2] - origin[2]) - o[2]};
        q[0] = torque ? 0.0f : f[0];
        q[1] = torque ? 0.0f : f[1];
        q[2] = torque ? 0.0f : f[2];
        q[3] = torque ? f[0] : l[1] * f[2] - l[2] * f[1];
        q[4] = torque ? f[1] : l[2] * f[0] - l[0] * f[2];
        q[5] = torque ? f[2] : l[0] * f[1] - l[1] * f[0];
    };
    // WALK_ROW(data, link, now, now_link, shift, body, k, seed, sink): every reader's seed walk, after a walk
    // prelude (walk_prelude, walk_paths_prelude). Rows seeded at body walked up its root path into sink: the wrench
    // seed[6 r + i] at body's frame (k < 0; now, now_link and shift its step's geometry, as WALK_UP takes them), or the
    // covector seed[r] e_k at body's coordinate k (WALK_SEED, then WALK_UP from k - 1, without the step's geometry).
    const auto WALK_ROW = [&](const float* data, const float* link, const float* now, const float* now_link,
                              const float* shift, const int body, const int k, const float* seed, auto&& sink) {
        float p[6 * $rows], w[6 * $rows];
        int top = WALK_FIRST_OF(body + 1) - 1;
        for (int e = 0; e < 6 * $rows; ++e) p[e] = w[e] = k < 0 ? seed[e] : 0.0f;
        if (k >= 0) {
            WALK_SEED(data, k, seed, p, sink);
            top = k - 1;
            now = nullptr;
        }
        WALK_UP(data, link, now, now_link, shift, body, top, p, w, sink);
    };"""


@functools.cache
def walk_row_prelude(rows: int) -> DevicePrelude:
    """The seed walk of ``rows`` rows (``WALK_ROW``) and a wrench seed's formation (``WALK_WRENCH``), for both devices:
    every row reader's (the one writer's units and natives, the parallel tiers' rows, the local blocks' seeds) binds its
    sink to it. It follows the walk prelude whose
    ``WALK_UP`` and ``WALK_SEED`` it calls, and only readers that walk seeds declare it."""
    text = substitute(_WALK_ROW, rows=rows)
    return DevicePrelude(f"walk_row_{rows}", text, cpu=text)


FREE_BODY_WALK = WalkClass((-1,), (0, 6), (0, 0, 0, 1, 1, 1))
"""A free rigid body's walking class: one body of six coordinates."""

_WALK_ACCESS = """    // The class emitter's reads, ahead of its chain, by the one access (WALK_READ, WALK_READ_LINK) and the plain load
    // kind: the walk issues each where its text places it, WALK_AHEAD items ahead of the step that reads it
    // (WalkClass.walk), and a store of the chain after it keeps it there; ptxas moves a read-only load down to its use.
    // WALK_IN(d, u, p, half): coordinate record d's step input, its record unpacked (half its class's, WALK_READ's),
    // and the caller's two values: the step's input (u) and the prior value its output updates (p). WALK_HAND(x, u, p):
    // a held input x with the caller's values. WALK_LINK_IN(link, b, up): body b's link (WALK_READ_LINK's).
    struct WalkIn {
        float d[$unpacked];
        float u, p;
    };
    struct WalkLinkIn {
        float l[6];
    };
    const auto WALK_IN = [&](const float* d, const float u, const float p, const int half) -> WalkIn {
        WalkIn in;
        WALK_READ(WALK_PLAIN, in.d, d, half);
        in.u = u;
        in.p = p;
        return in;
    };
    const auto WALK_HAND = [](WalkIn x, const float u, const float p) -> WalkIn {
        x.u = u;
        x.p = p;
        return x;
    };
    const auto WALK_LINK_IN = [&](const float* link, const int b, const int up) -> WalkLinkIn {
        WalkLinkIn in;
        WALK_READ_LINK(WALK_PLAIN, in.l, link, b, up);
        return in;
    };
    const auto WALK_LOAD = [&](const float* p) -> float { return *p; };
$fetch"""

_WALK_FETCH = {
    True: """    // A lane group's fetch of a byte range into L1, one line a lane a trip.
    const auto WALK_FETCH = [&](const void* start, const int bytes, const int lane, const int lanes) {
        const size_t end = reinterpret_cast<size_t>(start) + (size_t)bytes;
        for (size_t line = (reinterpret_cast<size_t>(start) & ~(size_t)127) + (size_t)128 * lane; line < end;
             line += (size_t)128 * lanes)
            asm volatile("prefetch.global.L1 [%0];" ::"l"(line));
    };
    // WALK_TOUCH(p): ask L1 for the line of p, a hint ahead of the load that reads it (no register, no scoreboard).
    const auto WALK_TOUCH = [&](const void* p) { asm volatile("prefetch.global.L1 [%0];" ::"l"(p)); };""",
    False: """    const auto WALK_FETCH = [&](const void*, const int, const int, const int) {};
    const auto WALK_TOUCH = [&](const void*) {};""",
}
"""The device's words of :data:`_WALK_ACCESS`: the L1 hints, which the CPU has none of."""


def _walk_down(classes: tuple[WalkClass, ...], cuda: bool, unroll: int, ahead: int | None) -> str:
    texts = ts.walk_texts(
        ts.Direction.DOWN,
        source="$X.u",
        sink="qd($c, q, $X.p);",
        value="u($c)",
        prior="prior($c)",
        coordinate=WALK_COORDINATE,
        link=WALK_LINK,
        end="twist($b, $V);",
    )
    walks = [walk.walk(**texts, cuda=cuda, unroll=unroll, ahead=ahead) for walk in classes]
    walk = indent_text(ts.by_class(walks), 8)
    return f"""    // WALK_DOWN(cls, data, link, u, prior, qd, twist): the chain of the leave P^T L^-T u of one tree of class cls (an
    // index into the classes the prelude was emitted for), on one lane, root first: each coordinate c takes
    // q = (u_c - ut_c . V) / ell_c, then V += s_c q, V its body's twist (v, w) at its frame, a register, which
    // twist(b, V) receives after the body's coordinates. u(c) reads coordinate c, prior(c) the value its output
    // updates (or 0), and qd(c, q, prior) writes the output. Every coordinate's record, link, u and prior are read
    // WALK_AHEAD items ahead of the chain (WalkClass.walk), so qd may write in place, but not into a coordinate after c.
    const auto WALK_DOWN = [&](const int cls, const float* data, const float* link, auto&& u, auto&& prior, auto&& qd,
                               auto&& twist) {{
        (void)cls;
{walk}
    }};"""


@functools.cache
def walk_down_prelude(
    classes: tuple[WalkClass, ...], unroll: int = WALK_UNROLL, ahead: int | None = None
) -> DevicePrelude:
    """The one walk down (the leave ``P^T L^-T u``) of the walking classes ``classes``, for both devices: a prelude
    declaring ``WALK_DOWN(cls, data, link, u, prior, qd, twist)``, the chain on one lane: each class's walk emitted by
    :meth:`WalkClass.walk` (straight-line code up to ``unroll`` coordinates), its links, records, ``u`` and prior
    outputs read :data:`WALK_AHEAD` items ahead of the chain (``WALK_IN``), every twist a register, handed to
    ``twist(b, V)`` after its body.

    No shared memory and no scratch. The step's one leave takes it (:func:`walk_leave_kernel`), and the general sweep's
    free bodies' speed rows on their one-body class (:data:`FREE_BODY_WALK`).

    Args:
        classes: The walking classes, indexed by ``cls``.
        unroll: Coordinates up to which a class's walk is straight-line code (:data:`WALK_UNROLL`).
        ahead: Items whose inputs the chain's lane holds in registers ahead of it (:meth:`WalkClass.walk`).
    """
    if not classes:
        raise ValueError("a walk down needs a class")
    tag = hashlib.sha256(
        repr((classes, unroll, WALK_AHEAD if ahead is None else ahead, WALK_TOUCH)).encode()
    ).hexdigest()[:8]
    # The loads package carries the step's pin (WALK_MUL), under its one guard.
    texts = {
        cuda: "\n".join(
            (
                _walk_loads_text(cuda),
                substitute(_WALK_ACCESS, unpacked=WALK_UNPACKED, fetch=_WALK_FETCH[cuda]),
                _walk_down(classes, cuda, unroll, ahead),
            )
        )
        for cuda in (True, False)
    }
    return DevicePrelude(f"walk_down_{tag}", texts[True], cpu=texts[False])


LEAVE_LANES = 4
"""Lanes of a tree's group in the step's leave (:func:`walk_leave_kernel`) on CUDA: they fetch the tree's changes and
velocities into L1 in one parallel trip and one of them walks, its records and links asked of L1 :data:`WALK_TOUCH`
items ahead (WALK_MAPPING §4, B5: 4 lanes ran at or under 8 on g1, franka and allegro)."""

_LEAVE = """    const int tree = tid / lanes, lane = tid - tree * lanes;
    if (tree >= arts.shape[0]) return;
    const int art = arts.data[tree];
    const int cls = walk_class.data[art];
    if (cls < 0) return;
    const int j0 = walk.joint_start.data[art], d0 = walk.dof_start.data[art], n = walk.dofs.data[art];
    // Coordinate c's change is change[node(c)]: the walk's du at the tree's coordinates (node(c) = c), or a sweep's
    // own layout (the tree's base there, and its class's node of each coordinate).
    const float* const change = du.data + du_base.data[art];
    float* const v = v_out.data + d0;
    const float* const data = walk.data.data + (size_t)d0 * $coordinate;
    const float* const link = walk.link.data + (size_t)j0 * $link;
    const auto prior = [&](const int c) -> float { return v[c]; };
    const auto qd = [&](const int c, const float q, const float p) { v[c] = p + q; };
    const auto twist = [](const int, const float*) {};
$classes"""

_LEAVE_CLASS = """    // The group's lanes fetch the tree's changes and velocity into L1; lane 0 prepares the change (ready) and walks the
    // tree down (WALK_DOWN: its records and links asked of L1 WALK_TOUCH items ahead, loaded WALK_AHEAD items ahead;
    // a whole tree fetched at the start overflows L1, chain_106's 22 trees an SM hold 230 KB).
    WALK_FETCH(change, $span, lane, lanes);
    WALK_FETCH(v, n * 4, lane, lanes);
$release    if (lane == 0) {
        ready();
        WALK_DOWN($cls, data, link, read, prior, qd, twist);
    }
"""

_LEAVE_PLAIN = """    const auto ready = []() {};
    const auto read = [&](const int c) -> float { return WALK_LOAD(change + $node); };
"""

_LEAVE_RELEASE = """    // The slips' release transposed, R^T du, on a copy where the tree slips (the predictor's border's R): slip j's
    // R_j^T along its root path, root first, the last slip first (WALK_RELEASE_T). $how
    float r[$coordinates];
    bool slipped = false;
    const auto read = [&](const int c) -> float { return slipped ? r[c] : WALK_LOAD(change + $node); };
    const auto ready = [&]() {
        if (walk.release.count.data[art] == 0) return;
$copy
        slipped = true;
    };
"""

_LEAVE_SLIPS = """        const int cap = walk.release.b.shape[1], steps = walk.release.b.shape[2];
        for (int j = walk.release.count.data[art] - 1; j >= 0; --j) {
            const size_t at = ((size_t)art * cap + j) * steps;
            const float *g = walk.release.g.data + at, *a = walk.release.a.data + at, *b = walk.release.b.data + at;
            float t = 0.0f;
            switch (walk.release.coordinate.data[at]) {
$paths
            }
        }"""


def _leave_class(k: int, walk: WalkClass, node: tuple[int, ...] | None, release: bool, unroll: int) -> str:
    """The leave of a tree of class ``k``: its read of the change, the release where the class can slip, the unit."""
    n = walk.coordinates
    at = "c" if node is None else f"LEAVE_NODE_{k}[c]"
    table = ""
    if node is not None:
        table = (
            "#if defined(__CUDA_ARCH__)\n"
            f"    static __constant__ int LEAVE_NODE_{k}[{max(n, 1)}] = {{{', '.join(map(str, node))}}};\n"
            "#else\n"
            f"    static const int LEAVE_NODE_{k}[{max(n, 1)}] = {{{', '.join(map(str, node))}}};\n"
            "#endif\n"
        )
    span = "n * 4" if node is None else str(4 * (max(node, default=-1) + 1))
    if not release:
        head = substitute(_LEAVE_PLAIN, node=at)
    elif n <= unroll:
        # Every index into the copy is a constant (the class's coordinates, each slip's path by a switch on its
        # coordinate), so the copy stays in registers.
        steps = []
        for c in range(n):
            path = walk.root_path(c)
            body = " ".join(
                "{ "
                + " ".join(ts.slip("RT", f"r[{d}]", "t", f"b[{t}]", f"a[{t}]", f"g[{t}]", f"r[{d}] = {{}};"))
                + " }"
                for t, d in reversed(list(enumerate(path)))
            )
            steps.append(f"case {c}: {{ {body} }} break;")
        copy = f"        #pragma unroll\n        for (int c = 0; c < {n}; ++c) r[c] = WALK_LOAD(change + {at});\n"
        copy += substitute(_LEAVE_SLIPS, paths=indent_text("\n".join(steps), 12))
        head = substitute(
            _LEAVE_RELEASE,
            how="Each index a constant: the copy is registers.",
            coordinates=max(n, 1),
            node=at,
            copy=copy,
        )
    else:
        copy = f"        for (int c = 0; c < {n}; ++c) r[c] = WALK_LOAD(change + {at});\n"
        copy += "        WALK_RELEASE_T(walk.release, art, r);"
        head = substitute(
            _LEAVE_RELEASE, how="A loop-form class: the copy on the stack.", coordinates=max(n, 1), node=at, copy=copy
        )
    return table + substitute(_LEAVE_CLASS, span=span, release=head, cls=k)


@functools.cache
def walk_leave_kernel(
    classes: tuple[WalkClass, ...], release: bool, nodes: tuple[tuple[int, ...], ...] | None = None
) -> wp.Kernel:
    """The step's one leave, ``v += P^T L^-T R^T du``, of every walking tree (``arts``), a lane group a tree (``lanes``:
    :data:`LEAVE_LANES` on CUDA, one on the CPU; ``dim`` trees times lanes), by the walk down (``WALK_DOWN`` of
    :func:`walk_down_prelude`, the tree's class at ``walk_class``). Coordinate ``c``'s change is
    ``du[du_base[art] + node(c)]``: every placement's sweep adds its change in the walk's coordinates into
    :attr:`WalkTrees.du` (the base the tree's first coordinate, ``node(c) = c``), the support placement's sparse sweep
    publishes it in its factor coordinates (the base the tree's offset in its world's, ``node`` its class's factor
    coordinates, ``nodes``). Where the class can slip (``release``) the leave takes the slips' release transposed on a
    copy of the tree's changes, in registers for a straight-line class (each slip's root path by a switch on its
    coordinate), and walks each tree down onto ``v_out``, which holds the predictor and the changes of the placements
    that do not walk.

    Args:
        classes: The walking classes, indexed by ``walk_class``.
        release: Whether a tree of the classes can slip (:attr:`TreeFactor.walk_slip_capacity`).
        nodes: Each class's node of each coordinate, or ``None`` for the walk's own coordinates.
    """
    from .kernel_template import THREADS, KernelTemplate  # noqa: PLC0415

    if nodes is not None and (
        len(nodes) != len(classes) or any(len(k) != walk.coordinates for k, walk in zip(nodes, classes, strict=True))
    ):
        raise ValueError("the leave needs a node for each coordinate of each class")
    bodies = [
        _leave_class(k, walk, None if nodes is None else nodes[k], release, WALK_UNROLL)
        for k, walk in enumerate(classes)
    ]
    body = substitute(_LEAVE, classes=indent_text(ts.by_class(bodies), 4), coordinate=WALK_COORDINATE, link=WALK_LINK)
    tag = hashlib.sha256(repr((classes, nodes)).encode()).hexdigest()[:8]
    return KernelTemplate(
        f"walk_leave_{len(classes)}_{tag}" + ("_release" if release else ""),
        (
            ("arts", wp.array[int]),
            ("lanes", int),
            ("walk", WalkTrees),
            ("walk_class", wp.array[int]),
            ("du", wp.array[float]),
            ("du_base", wp.array[int]),
            ("v_out", wp.array[float]),
        ),
        body,
        THREADS,
        preludes=((walk_paths_prelude(1),) if release else ()) + (walk_down_prelude(classes, WALK_UNROLL),),
        dual=True,
        native_name="walk_leave_native",
    ).kernel()


# ---------------------------------------------------------------------------------------------------------------------
# The walk's view (FL3.3): what the one writer, the leave and the audit read of the converted class's trees
# ---------------------------------------------------------------------------------------------------------------------


@wp.struct
class WalkRelease:
    """The release of each tree's slipping drives this step, which the predictor's border writes
    (:func:`.tree_kinetic_state.border_solve`, W9): its slips (``count``, 0 for none), each one's coordinate within its
    tree (``dof``, root-most first) and its root (``root``, its path's root-most coordinate), and at each step of each
    one's root path, leaf first, the step's coordinate (``coordinate``, -1 past the path), the column ``b`` of the
    released inverse and the recurrence's ``a`` (``beta d^1/2``) and ``g`` (``(d / dbar)^1/2``): ``[tree, slip, step]``,
    as deep as the class's deepest root path. Slip ``j``'s ``R_j`` maps a footprint ``z`` along its path, leaf first,
    ``u = z_c - b s``, ``s += a u``, ``z_c = g u``; ``R = R_(s-1) ... R_0 = L_F^-1 L_K``, the decided factor's whitening on
    the all-``K`` walk, which keeps every row's support (``WALK_RELEASE``, ``WALK_RELEASE_T``, ``WALK_RELEASE_IT``). Two slips' paths meet if and only if their roots are equal (their ``R_j`` touch
    disjoint coordinates otherwise, and commute), and a footprint whose support holds every ancestor of its coordinates
    (a row's) meets slip ``j``'s path if and only if it holds ``root[j]``."""

    count: wp.array[int]
    dof: wp.array2d[int]
    root: wp.array2d[int]
    coordinate: wp.array3d[int]
    b: wp.array3d[float]
    a: wp.array3d[float]
    g: wp.array3d[float]


def release_view(articulations: int, capacity: int, depth: int, device) -> WalkRelease:
    """A :class:`WalkRelease` of ``capacity`` slips a tree and ``depth`` steps a slip; no slip room without a capacity."""
    view = WalkRelease()
    trees = articulations if capacity else 0
    view.count = wp.zeros(articulations, dtype=wp.int32, device=device)
    view.dof = wp.zeros((trees, max(capacity, 1)), dtype=wp.int32, device=device)
    view.root = wp.zeros((trees, max(capacity, 1)), dtype=wp.int32, device=device)
    view.coordinate = wp.full((trees, max(capacity, 1), depth), -1, dtype=wp.int32, device=device)
    view.b = wp.zeros((trees, max(capacity, 1), depth), dtype=wp.float32, device=device)
    view.a = wp.zeros((trees, max(capacity, 1), depth), dtype=wp.float32, device=device)
    view.g = wp.zeros((trees, max(capacity, 1), depth), dtype=wp.float32, device=device)
    return view


@wp.struct
class WalkTrees:
    """The walk of every held tree of the converted class (:meth:`TreeFactor.plan_walk`), by global coordinate, body
    (its joint) and articulation:

    - ``data``, ``link``, ``path``: :data:`WALK_COORDINATE` floats a coordinate, :data:`WALK_LINK` floats a body and
      each body's parent and first coordinate within its tree (:func:`walk_paths_prelude`'s table);
    - ``now``, ``now_link``, ``now_com``: each coordinate's column at its body's frame, each body's link
      (:data:`WALK_LINK`'s layout: its frame point) and its centre of mass (a ``vec3`` a joint), at the step's poses,
      which the held tree's finish publishes with its motion columns: a seed enters the walk as ``J`` there (each column
      shifted to its body's frame point in ``link`` by the body's motion since the refresh), so a held factor (a step
      between refreshes) whitens the step's ``J``, and a row's incident ``J v_hat`` is summed along its walk; a refresh
      takes them as the factor's frames;
    - ``origin``, ``now_origin``: each articulation's world origin at the factor's frames and at the step's (the
      finish's ``articulation_origin``). Every frame point is the articulation's own, relative to its origin, so its
      precision does not depend on where the articulation stands; a world point meets one through the origin, and a
      factor's frame meets the step's through the two origins' difference (``WALK_UP``'s ``shift``, zero on a refresh
      step); ``joint_art``: each joint's articulation;
    - ``scratch``: a twist a joint, the audit's (``WALK_ENTER``) and the predictor's border's work;
    - each articulation's first joint, joint end, first coordinate and coordinates (a free rigid body is a one-body
      tree of six coordinates);
    - ``dof_body``: each coordinate's body within its tree; ``body_joint``: each body's joint;
    - ``du``: each coordinate's velocity change in the walk's coordinates, which the sweeps add into from zero each
      step and the step's one leave walks down (:func:`walk_leave_kernel`);
    - ``release``: the release of each tree's slips (the block primitive's first phase, :class:`WalkRelease`), which the
      predictor's border writes (:func:`.tree_kinetic_state.border_solve`, FL3.9): the rows take ``R``, the leave
      ``R^T`` and the audit's coordinates ``R^-T``.
    """

    data: wp.array2d[float]
    link: wp.array2d[float]
    path: wp.array[wp.vec2i]
    now: wp.array[wp.spatial_vector]
    now_link: wp.array2d[float]
    now_com: wp.array[wp.vec3]
    origin: wp.array[wp.vec3]
    now_origin: wp.array[wp.vec3]
    joint_art: wp.array[int]
    scratch: wp.array[wp.spatial_vector]
    joint_start: wp.array[int]
    joint_end: wp.array[int]
    dof_start: wp.array[int]
    dofs: wp.array[int]
    dof_body: wp.array[int]
    body_joint: wp.array[int]
    release: WalkRelease
    du: wp.array[float]


def empty_walk_view(model, forest) -> WalkTrees:
    """A :class:`WalkTrees` that holds no tree: the predictor's border reads it where no tree could slip at construction
    and the walk is not held (:meth:`TreeFactor.plan_walk`), and clears its slip counts. Its coordinate ranges are the
    forest's and its data has no row, so a decided step there (a limit set after construction) finds no tree, keeps its
    trial and flags the border's status (``constraint_capacity_status()["slip_frames"]``), never reading past an array."""
    device = model.device
    view = WalkTrees()
    view.data = wp.zeros((0, WALK_COORDINATE), dtype=wp.float32, device=device)
    view.link = wp.zeros((1, WALK_LINK), dtype=wp.float32, device=device)
    view.path = wp.zeros(1, dtype=wp.vec2i, device=device)
    view.now = wp.zeros(1, dtype=wp.spatial_vector, device=device)
    view.now_link = wp.zeros((1, WALK_LINK), dtype=wp.float32, device=device)
    view.now_com = wp.zeros(1, dtype=wp.vec3, device=device)
    view.origin = view.now_origin = wp.zeros(1, dtype=wp.vec3, device=device)
    view.joint_art = wp.zeros(1, dtype=wp.int32, device=device)
    view.scratch = wp.zeros(1, dtype=wp.spatial_vector, device=device)
    view.joint_start = model.articulation_start
    view.joint_end = forest.articulation_joint_end
    view.dof_start = forest.articulation_dof_start
    view.dofs = forest.articulation_dof_count
    view.dof_body = view.body_joint = wp.zeros(1, dtype=wp.int32, device=device)
    view.du = wp.zeros(1, dtype=wp.float32, device=device)
    view.release = release_view(max(int(model.articulation_count), 1), 0, 1, device)
    return view


_ENTER = """    if (tid >= arts.shape[0]) return;
    const int art = arts.data[tid];
    const int world = art_to_world.data[art], offset = world_offset.data[art];
    if (world < 0 || offset < 0) return;
    const int j0 = walk.joint_start.data[art], d0 = walk.dof_start.data[art];
    WALK_TREE(reinterpret_cast<const int*>(walk.path.data + j0), walk.joint_end.data[art] - j0, walk.dofs.data[art]);
    float* u = coordinates.data + (size_t)world * coordinates.shape[1] + offset;
    WALK_ENTER(walk.data.data + (size_t)d0 * $coordinate, walk.link.data + (size_t)j0 * $link, velocity.data + d0, u,
               reinterpret_cast<float*>(walk.scratch.data + j0));
    // The rows' release: u <- R^-T u (the predictor's border's R).
    WALK_RELEASE_IT(walk.release, art, u);
"""


@functools.cache
def walk_enter_kernel() -> wp.Kernel:
    """Each held tree's coordinates of a velocity in the walk's frame, ``u = L^T P v`` (``WALK_ENTER``), a thread per
    tree, at its world coordinates: what the audit reads the walked rows against (:mod:`.audit`)."""
    from .kernel_template import THREADS, KernelTemplate  # noqa: PLC0415

    return KernelTemplate(
        "walk_enter",
        (
            ("arts", wp.array[int]),
            ("walk", WalkTrees),
            ("art_to_world", wp.array[int]),
            ("world_offset", wp.array[int]),
            ("velocity", wp.array[float]),
            ("coordinates", wp.array2d[float]),
        ),
        substitute(_ENTER, coordinate=WALK_COORDINATE, link=WALK_LINK),
        THREADS,
        preludes=(walk_paths_prelude(1),),
        dual=True,
        native_name="walk_enter_native",
    ).kernel()


_WALK_INTO = """    WALK_TREE(reinterpret_cast<const int*>(walk.path.data + joint0), bodies, coordinates);
    const float* data = walk.data.data + (size_t)dof0 * $coordinate;
    const float* link = walk.link.data + (size_t)joint0 * $link;
    float* row = out.data + ((size_t)line * out.shape[1] + slot) * out.shape[2] + entry;
    const float* v = qd.data + dof0;
    float incident = 0.0f, squares = 0.0f;
    const auto sink = [&](int, int c, float z, float j) {
        if (add > 0)
            row[c] += z;
        else if (add == 0)
            row[c] = z;
        squares += z * z;
        incident += j * v[c];
    };
    // The seed's kind is the caller's constant (covector), so each call site compiles one walk: a covector at local
    // coordinate k takes no step geometry; a wrench enters at the step's geometry, shifted by the articulation's origins.
    if (covector != 0) {
        __builtin_assume(k >= 0);
        WALK_ROW(data, link, nullptr, nullptr, nullptr, body, k, &seed[0], sink);
    } else {
        float shift[3];
        WALK_SHIFT(walk, walk.joint_art.data[joint0], shift);
        WALK_ROW(data, link, reinterpret_cast<const float*>(walk.now.data + dof0), walk.now_link.data + (size_t)joint0 * $link, shift,
                 body, -1, &seed[0], sink);
    }
    return wp::vec2(incident, squares);
"""


@functools.cache
def _walk_into_snippet() -> str:
    from .device_preludes import dual_native  # noqa: PLC0415

    return dual_native(
        substitute(_WALK_INTO, coordinate=WALK_COORDINATE, link=WALK_LINK),
        preludes=(walk_paths_prelude(1), walk_row_prelude(1)),
    )


@wp.func_native(_walk_into_snippet())
def walk_into(
    walk: WalkTrees,
    joint0: int,
    dof0: int,
    bodies: int,
    coordinates: int,
    body: int,
    covector: int,
    k: int,
    seed: wp.spatial_vector,
    qd: wp.array[float],
    add: int,
    out: wp.array3d[float],
    line: int,
    slot: int,
    entry: int,
) -> wp.vec2:
    """Walk one row's seed (``WALK_ROW``) into ``out[line, slot, entry + c]``, on the tree of path table ``joint0``:
    the wrench ``seed`` at local ``body``'s frame (``covector`` 0; the factor's, :attr:`WalkTrees.link`), where the
    step's columns the finish published, shifted to the path's frames, are its ``J``, or (``covector`` 1) the covector
    ``seed[0] e_k`` at ``body``'s local coordinate ``k``. ``covector`` is each call site's constant, so the site
    compiles one walk. Stores the footprints (adds them with ``add`` 1: a second seed on one tree;
    stores none with ``add`` -1) and returns the seed's incident ``J qd`` (the step's ``J`` entries the walk passes,
    against ``qd``; a covector's ``seed[0] qd[k]``) and the sum of its footprints' squares."""
    ...


RELEASE_SLIPS = 16
"""The most slipping drives a tree's release holds (:class:`WalkRelease`, three floats and a coordinate a slip and path
step): a tree past it keeps the rest's ``K`` and ``constraint_capacity_status()["slip_frames"]`` flags it. It bounds the
release's memory until a data-sized arena holds it (DECISIONS 10-24); nothing else reads it."""


# ---------------------------------------------------------------------------------------------------------------------
# The walk's step on the device (FL3.3): the step's geometry every step, the inward pass on the factor's refreshes,
# one launch, a lane group per tree
# ---------------------------------------------------------------------------------------------------------------------

WALK_JOIN_SLOTS = 32
"""The most joins one tree's inward pass holds at once: a parent that waits for the lifted inertia of a child other
than the joint after it (:meth:`TreeFactor.plan_walk` assigns the slots). On CUDA lane ``k`` of the tree's group
holds slot ``k`` in registers, so a group has at least as many lanes as its tree's joins in flight; the CPU's one
thread holds them all."""


def walk_group(bodies: int, joins: int, trees: int, sms: int) -> int:
    """The lane group of a tree in :func:`walk_step_kernel` on CUDA, from the class's data: the bodies a batch of the
    inward pass builds, one a lane, against the trees in flight. The group is the largest power of two with which the
    class's walked trees still fit the device in one wave (:data:`WALK_RESIDENT` threads an SM), within the tree (no
    more lanes than its bodies) and a warp, and never fewer lanes than the joins a tree holds at once (one a lane).

    Measured on both GPUs (``SP/core/FL/HANDOFF_ZR.md``): with more trees than the device holds a group of four
    wins (g1, kuka, franka-reach at 16,384 trees: 4 lanes), with fewer the batch widens (chain_106 at 4,096: 16).

    Args:
        bodies: The most bodies of one of the class's walked trees.
        joins: The most joins one of its trees holds at once (:meth:`TreeFactor.plan_walk`).
        trees: The trees the launch walks.
        sms: The device's multiprocessors.
    """
    fit = max(sms * WALK_RESIDENT // max(trees, 1), 1)
    top = min(bodies, 32)
    lanes = 1
    while lanes * 2 <= fit and lanes < top:
        lanes *= 2
    while lanes < joins:
        lanes *= 2
    return lanes


WALK_RESIDENT = 512
"""Threads of :func:`walk_step_kernel` one multiprocessor holds: its 125-127 registers (``kernel_sass``, sm_120 and
sm_103) leave 16 warps an SM."""

WALK_BLOCK = 64
"""Threads of a block of :func:`walk_step_kernel` on CUDA: whole groups, so a group never straddles a warp."""


_INWARD = """    // The refresh by the tree's lane group (walk_group): the lane feed (FEED, leaves to root) hands the chain one body
    // at a time; a lane builds each body of a batch (BODY, its path entry, its join slots and its last coordinate's
    // column), every lane runs the chain on it, read by shuffle. A body's lifted inertia is carried when its parent is
    // the next joint inward, else added into the parent's join slot (joins[parent], lane k's registers on CUDA, the
    // CPU's one thread every slot) until the chain reaches it. On the CPU one thread is the group.
    SYNC();
    // The refresh's arithmetic on one tree, float32 (TreeWalk's inward pass, on the factor's diagonal), at the step's
    // geometry the finish published, each coordinate's column in now and each body's link and frame point in
    // now_link: on a refresh they become the factor's frames. The lane feed's batches run it leaf first; the chain's
    // state is the articulated inertia h,
    // symmetric (A 00 01 02 11 12 22, B row-major 3x3, D 00 01 02 11 12 22), in registers.
    //   BODY(j, up, r): body j's inertia at its frame [m 1, -m X; m X, R I R^T - m X X] (X the skew of c, its centre of
    //     mass less its frame, both as the finish published them in now_com and now_link) as words m, m c, rot, and
    //     its link offset d
    //     (WALK_OFFSET from its parent's frame point, up's, and its own); its frame point becomes the factor's link
    //     (every walk forms d from the two points). Nothing here reads a world position: the frame points are the
    //     articulation's, relative to its origin.
    //   ADD(h, r): h += body r's inertia.
    //   STEP(h, s, c): coordinate c of column s (s 6, R, R + K) eliminated (tree_scan.inward_step); its record
    //     (s by its half, ut, 1/ell, P0 = own + R) leaves whole, once.
    //   LIFT(h, d): h to the parent's frame (tree_scan.inward_edge).
    const float* Lk = now_link.data;
    const float* Lc = reinterpret_cast<const float*>(now_com.data);
    // A column of two halves keeps its linear half WALK_SIDE floats past its record, in data's second half (its rows
    // doubled where a joint's column can have two halves: TreeFactor's walk_data).
    const int WALK_SIDE = data.shape[0] > armature.shape[0] ? $coordinate * armature.shape[0] : 0;
    const float* Nw = reinterpret_cast<const float*>(now.data);
    const int* Hf = half.data;
    float* Lo = link.data;
    float* Dt = data.data;
    const float* Ra = armature.data;
    const int* Dr = drive_row_by_dof.data;
    const float* Kr = row_K.data;
    const int* Ct = joint_child.data;
    const float* Qb = reinterpret_cast<const float*>(body_q.data);
    const float* Mb = body_mass.data;
    const float* Ib = reinterpret_cast<const float*>(body_inertia.data);
#if defined(__CUDA_ARCH__)
    const auto PREFETCH = [](const void* p) { asm volatile("prefetch.global.L1 [%0];" ::"l"(p)); };
#else
    const auto PREFETCH = [](const void* p) {};
#endif
    const auto SYM = [](int a, int b) -> int { return a <= b ? (a == 0 ? b : (a == 1 ? 2 + b : 5)) : (b == 0 ? a : (b == 1 ? 2 + a : 5)); };
    enum { M = 0, MC = 1, ROT = 4, OFFSET = 10, INERTIA = 13 };
    // A coordinate's column at the step's poses and what the factor adds to its diagonal, R and R + K.
    const auto COLUMN = [&](int c, float* x) {
        // The step column (WALK_NOW): the half its joint's kind names (Hf: 0 linear, 1 angular, 2 both, the finish's
        // column at the frame whole), the other zero.
        const float* n = Nw + $now * c;
        const int h = Hf[c];
        for (int k = 0; k < 3; ++k) {
            x[k] = h != 1 ? n[k] : 0.0f;
            x[3 + k] = h != 0 ? n[3 + k] : 0.0f;
        }
        float added = Ra[c];
        x[6] = added;
        const int row = Dr[c];
        if (row >= 0) {
            if (Kr[row] > 0.0f) added += Kr[row];
        }
        x[7] = added;
    };
    // Every line one body's inertia and link read.
    const auto AHEAD = [&](int j) {
        const int b = Ct[j];
        PREFETCH(Qb + 7 * b + 3);
        PREFETCH(Qb + 7 * b + 6);
        PREFETCH(Mb + b);
        PREFETCH(Ib + 9 * b);
        PREFETCH(Ib + 9 * b + 8);
        PREFETCH(Lk + $link * j);
        PREFETCH(Lc + 3 * j);
    };
    // WALK_OFFSET(p, f, d): a body's link offset d, its parent's frame point p (null at a root) less its own f: the
    // one place a walk forms it from the frame points the finish published.
    const auto WALK_OFFSET = [](const float* p, const float* f, float* d) {
        for (int k = 0; k < 3; ++k) d[k] = p != nullptr ? p[k] - f[k] : 0.0f;
    };
    const auto BODY = [&](int j, int up, float* r) {
        const float* Lr = Lk + $link * j;
        float offset[3];
        WALK_OFFSET(up >= 0 ? Lk + $link * up + $frame : nullptr, Lr + $frame, offset);
        for (int k = 0; k < 3; ++k) r[OFFSET + k] = offset[k];
#if defined(__CUDA_ARCH__)
        reinterpret_cast<float4*>(Lo + $link * j)[0] = reinterpret_cast<const float4*>(Lr)[0];
#else
        for (int k = 0; k < $link; ++k) Lo[$link * j + k] = Lr[k];
#endif
        const int b = Ct[j];
        const float* q = Qb + 7 * b;
        const float* I = Ib + 9 * b;
        const float m = Mb[b];
        // The body's rotation and its centre of mass less its frame, c.
        const float x = q[3], y = q[4], z = q[5], w = q[6];
        const float R[3][3] = {
            {1.0f - 2.0f * (y * y + z * z), 2.0f * (x * y - z * w), 2.0f * (x * z + y * w)},
            {2.0f * (x * y + z * w), 1.0f - 2.0f * (x * x + z * z), 2.0f * (y * z - x * w)},
            {2.0f * (x * z - y * w), 2.0f * (y * z + x * w), 1.0f - 2.0f * (x * x + y * y)}};
        float c[3], mc[3];
        for (int a = 0; a < 3; ++a) c[a] = Lc[3 * j + a] - Lr[$frame + a];
        for (int a = 0; a < 3; ++a) mc[a] = m * c[a];
        // R I R^T - m X X, X the skew of c: -m X X = m (|c|^2 1 - c c^T).
        float RI[3][3];
        for (int a = 0; a < 3; ++a)
            for (int e = 0; e < 3; ++e) RI[a][e] = R[a][0] * I[e] + R[a][1] * I[3 + e] + R[a][2] * I[6 + e];
        const float cc = c[0] * c[0] + c[1] * c[1] + c[2] * c[2];
        r[M] = m;
        for (int a = 0; a < 3; ++a) r[MC + a] = mc[a];
        for (int a = 0; a < 3; ++a)
            for (int e = a; e < 3; ++e)
                r[ROT + SYM(a, e)] = (RI[a][0] * R[e][0] + RI[a][1] * R[e][1] + RI[a][2] * R[e][2]) + mc[a] * -c[e] + (a == e ? m * cc : 0.0f);
    };
    // h = body + h (carrying) or h = body.
    const auto ADD = [&](float* h, const float* r, bool carrying) {
        const float bm = r[M];
        const float* bc = r + MC;
        const float* br = r + ROT;
        const float body[21] = {bm, 0.0f, 0.0f, bm, 0.0f, bm,
                                0.0f, bc[2], -bc[1], -bc[2], 0.0f, bc[0], bc[1], -bc[0], 0.0f,
                                br[0], br[1], br[2], br[3], br[4], br[5]};
        for (int k = 0; k < 21; ++k) h[k] = carrying ? body[k] + h[k] : body[k];
    };
    const auto STEP = [&](float* h, const float* s, int dof) {
$step_head
        if (LANE == 0) {
            // The record (WALK_COORDINATE): the column's non-zero half (the angular one where it turns, 1 / ell's
            // sign bit set), and a column of two halves its linear half in the side float4 (WALK_SIDE floats on).
            float* o = Dt + $coordinate * dof;
            const bool turns = s[3] != 0.0f || s[4] != 0.0f || s[5] != 0.0f;
            const bool both = WALK_SIDE != 0 && turns && (s[0] != 0.0f || s[1] != 0.0f || s[2] != 0.0f);
            const int side = both ? WALK_SIDE : 0;
            const float half[3] = {turns ? s[3] : s[0], turns ? s[4] : s[1], turns ? s[5] : s[2]};
#if defined(__CUDA_ARCH__)
            float4* o4 = reinterpret_cast<float4*>(o);
            o4[0] = make_float4(half[0], half[1], half[2], turns ? -inv : inv);
            o4[1] = make_float4(ut[0], ut[1], ut[2], ut[3]);
            o4[2] = make_float4(ut[4], ut[5], own + s[6], __int_as_float(side));
            if (both) reinterpret_cast<float4*>(o + side)[0] = make_float4(s[0], s[1], s[2], 0.0f);
#else
            for (int k = 0; k < 3; ++k) o[k] = half[k];
            o[3] = turns ? -inv : inv;
            for (int k = 0; k < 6; ++k) o[4 + k] = ut[k];
            o[10] = own + s[6];
            o[11] = __builtin_bit_cast(float, side);
            if (both) {
                for (int k = 0; k < 3; ++k) o[side + k] = s[k];
                o[side + 3] = 0.0f;
            }
#endif
        }
$step_update
    };
    const auto LIFT = [&](float* h, const float* bd) {
$lift
    };
    const int* Pt = reinterpret_cast<const int*>(path.data);
    const int* Jt = joins.data;
#if defined(__CUDA_ARCH__)
    constexpr int HELD = 1;
#else
    constexpr int HELD = $slots;
#endif
    for (int c = dof0 + LANE; c < dof1; c += LANES) {
        PREFETCH(Nw + $now * c);
        PREFETCH(Hf + c);
        PREFETCH(Ra + c);
        PREFETCH(Dr + c);
    }
    float held[HELD][21];
    for (int s = 0; s < HELD; ++s)
        for (int k = 0; k < 21; ++k) held[s][k] = 0.0f;
    const auto OWNS = [&](int slot) -> bool { return HELD > 1 || LANE == slot; };
    const auto AT = [&](int slot) -> int { return HELD > 1 ? slot : 0; };
    float h[21];
    for (int k = 0; k < 21; ++k) h[k] = 0.0f;
    bool carrying = false;
    // A body as the feed hands it to the chain: BODY's words, its last coordinate's column (s 6, R, R + K), and as
    // bits its parent, coordinates [lo, hi), its own join slot and its parent's.
    struct Feed { float w[26]; };
    enum { RECORD = INERTIA, UP = RECORD + 8, LO = UP + 1, HI = LO + 1, SLOT = HI + 1, INTO = SLOT + 1 };
    FEED(j0, j1 - 1, -1,
        [&](int j) {
            AHEAD(j);
            PREFETCH(Pt + 2 * j);
            PREFETCH(Jt + j);
        },
        [&](int mine, bool valid) -> Feed {
            Feed r;
            for (int k = 0; k < 26; ++k) r.w[k] = 0.0f;
            int up = -1, lo = 0, hi = 0, slot = -1, into = -1;
            if (valid) {
                up = Pt[2 * mine];
                lo = dof0 + Pt[2 * mine + 1];
                hi = mine + 1 < j1 ? dof0 + Pt[2 * (mine + 1) + 1] : dof1;
                slot = Jt[mine];
                into = up >= 0 ? Jt[j0 + up] : -1;
                BODY(mine, up >= 0 ? j0 + up : -1, r.w);
                if (hi > lo) COLUMN(hi - 1, r.w + RECORD);
            }
            r.w[UP] = __int_as_float(up);
            r.w[LO] = __int_as_float(lo);
            r.w[HI] = __int_as_float(hi);
            r.w[SLOT] = __int_as_float(slot);
            r.w[INTO] = __int_as_float(into);
            return r;
        },
        [&](int b, const Feed& r) {
            float s[8];
            for (int k = 0; k < 8; ++k) s[k] = r.w[RECORD + k];
            const int b_up = __float_as_int(r.w[UP]), b_lo = __float_as_int(r.w[LO]), b_hi = __float_as_int(r.w[HI]);
            const int b_slot = __float_as_int(r.w[SLOT]), b_into = __float_as_int(r.w[INTO]);
            ADD(h, r.w, carrying);
            if (b_slot >= 0) {
                for (int k = 0; k < 21; ++k) h[k] += READ(held[AT(b_slot)][k], b_slot);
                if (OWNS(b_slot))
                    for (int k = 0; k < 21; ++k) held[AT(b_slot)][k] = 0.0f;
            }
            for (int dof = b_hi - 1; dof >= b_lo; --dof) {
                // The next coordinate's column, loaded before this one stores (a body with several coordinates).
                float next[8];
                if (dof > b_lo) COLUMN(dof - 1, next);
                STEP(h, s, dof);
                if (dof > b_lo)
                    for (int k = 0; k < 8; ++k) s[k] = next[k];
            }
            carrying = false;
            if (b_up >= 0) {
                LIFT(h, r.w + OFFSET);
                if (b_up == b - j0 - 1) {
                    carrying = true;
                } else if (OWNS(b_into)) {
                    for (int k = 0; k < 21; ++k) held[AT(b_into)][k] += h[k];
                }
            }
        });
"""


@functools.cache
def _walk_inward_snippet() -> str:
    from .device_preludes import CPU_MATH, LANE_FEED, dual_native, lane_group  # noqa: PLC0415

    step_head, step_update = ts.inward_step("h", "s")
    return dual_native(
        substitute(
            _INWARD,
            step_head=ts.indent_text(step_head, 8),
            step_update=ts.indent_text(step_update, 8),
            lift=ts.indent_text(ts.inward_edge(("bd[0]", "bd[1]", "bd[2]"), "h"), 8),
            coordinate=WALK_COORDINATE,
            link=WALK_LINK,
            now=WALK_NOW,
            frame=WALK_FRAME,
            slots=WALK_JOIN_SLOTS,
        ),
        preludes=(lane_group("lanes"), CPU_MATH, LANE_FEED),
    )


@wp.func_native(_walk_inward_snippet())
def walk_inward(
    lanes: int,
    j0: int,
    j1: int,
    dof0: int,
    dof1: int,
    path: wp.array[wp.vec2i],
    joins: wp.array[int],
    now: wp.array[wp.spatial_vector],
    half: wp.array[int],
    now_link: wp.array2d[float],
    now_com: wp.array[wp.vec3],
    armature: wp.array[float],
    drive_row_by_dof: wp.array[int],
    row_K: wp.array[float],
    joint_child: wp.array[int],
    body_q: wp.array[wp.transform],
    body_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    link: wp.array2d[float],
    data: wp.array2d[float],
):
    """The refresh's inward pass of the tree of joints ``[j0, j1)`` and coordinates ``[dof0, dof1)`` by its group of
    ``lanes`` (one thread on the CPU), at the step's geometry the finish published (``now``, ``now_link``): the factor's
    frames (each body's ``link``) and each coordinate's whole record in ``data`` (:data:`WALK_COORDINATE`), with each body's spatial inertia built in the pass from its pose and mass properties."""
    ...


@wp.kernel(enable_backward=False)
def walk_step_kernel(
    arts: wp.array[int],
    lanes: int,
    mass_update_mask: wp.array[int],
    articulation_start: wp.array[int],
    articulation_joint_end: wp.array[int],
    walk_path: wp.array[wp.vec2i],
    walk_joins: wp.array[int],
    joint_child: wp.array[int],
    joint_qd_start: wp.array[int],
    body_q: wp.array[wp.transform],
    body_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    armature: wp.array[float],
    drive_row_by_dof: wp.array[int],
    row_K: wp.array[float],
    walk_now: wp.array[wp.spatial_vector],
    walk_half: wp.array[int],
    walk_now_link: wp.array2d[float],
    walk_now_com: wp.array[wp.vec3],
    articulation_origin: wp.array[wp.vec3],
    # outputs
    walk_data: wp.array2d[float],
    walk_link: wp.array2d[float],
    walk_origin: wp.array[wp.vec3],
):
    """The walk's refresh, a group of ``lanes`` threads per tree (:meth:`TreeFactor.walk_step`), float32, at the poses
    ``body_q`` holds, on the trees the refresh just factored (``mass_update_mask``; every other tree returns): the
    step's geometry the finish published (``walk_now``, ``walk_now_link``, ``walk_now_com``) becomes the factor's frames,
    with the articulation's origin they are relative to, and the group runs the refresh (:func:`walk_inward`, the lane
    feed's batches), so the walk and ``L`` are one factor."""
    tree, lane = wp.tid()
    art = arts[tree]
    if mass_update_mask[art] == 0:
        return
    j0 = articulation_start[art]
    j1 = articulation_joint_end[art]
    walk_inward(
        lanes,
        j0,
        j1,
        joint_qd_start[j0],
        joint_qd_start[j1],
        walk_path,
        walk_joins,
        walk_now,
        walk_half,
        walk_now_link,
        walk_now_com,
        armature,
        drive_row_by_dof,
        row_K,
        joint_child,
        body_q,
        body_mass,
        body_inertia,
        walk_link,
        walk_data,
    )
    if lane == 0:
        walk_origin[art] = articulation_origin[art]
