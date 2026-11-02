# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The tree factor: each articulation's ``P (H + R + K) P^T = L L^T``, the predictor, and the factor's maps of
velocities (BACKBONE §2).

:class:`TreeFactor` owns what the factor reads and writes across a step:

- the augmented state, the step's ``state_aug`` (joint forces, motion axes, body twists and wrenches);
- the kinematic masks and the effective armature R (a kinematic coordinate holds 1e10 armature);
- the drive topology and its rows K, prepared per coordinate apart from inverse dynamics;
- L and R per size group, the compact diagonal's inverse mass and the free bodies' 6x6 factors;
- the body inertia records, the articulation origins and the passive joint forces;
- the mass-update cadence, its requests and the FK/ID cache.

Its evaluator is the held tree (:mod:`.held_tree`). A step calls :meth:`TreeFactor.factor` (FK/ID, the drives' K and
u0, the mass-update mask, the factor refresh), :meth:`TreeFactor.predict` (``v_hat`` and the drives' saturation),
and :meth:`TreeFactor.finish` (``qdd``, integrate, publish); the walk's rows and leaves read its walk
(:meth:`TreeFactor.plan_walk`).
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import warnings
from string import Template

import numpy as np
import warp as wp

from ...sim import BodyFlags, JointType, ModelFlags
from . import field_group, host_index, response_builds_rows
from .conditioning import ArmatureFloor, conditioning_armature
from .device_preludes import DevicePrelude, substitute
from .field_group import SolverArrays
from .joint_table import JointTable
from .kernels import (
    BodyFactors,
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
    SIZE_FACTORS,
    SPARSE_FACTOR,
    STATE_AUX,
    DriveRows,
    Preelim,
    hold_drive_row,
)
from .sparse_mass_matrix import factor_indices
from .spatial_laws import drive_force, drive_stiffness, spring_damper_stiffness
from .tree_plan import TreePlan

__all__ = [
    "FREE_BODY_WALK",
    "LEAVE_LANES",
    "WALK_ADDED",
    "WALK_ARMATURE",
    "WALK_COORDINATE",
    "WALK_FRAME",
    "WALK_LINK",
    "WALK_NOW",
    "WALK_PIVOT",
    "WALK_UNROLL",
    "TreeFactor",
    "WalkClass",
    "prepare_augmented_drives",
    "walk_down_prelude",
    "walk_leave_kernel",
    "walk_paths_prelude",
    "walk_prelude",
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
    exist, :meth:`allocate_common`, :meth:`allocate_size_groups`, :meth:`build_body_factors`,
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
        self.L_by_size: dict[int, wp.array] = {}
        self.R_by_size: dict[int, wp.array] = {}
        self.diagonal_inverse_mass = None
        self.body_factors = None
        self.eliminates = False
        self.block_capacity = 0
        self.block_count = 0
        self.held = None
        self.dynamics_stream = None
        self.walk_arts = None
        self.walk_classes: tuple[WalkClass, ...] = ()
        self.walk_class = None
        self.walk_leave = None
        self.walk_slip_capacity = 0
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

    def allocate_sparse(self, plan, articulations: int) -> None:
        """The native sparse factor of ``articulations`` of one size: its pattern ``plan``, its device indices, its
        inverse entries and its status."""
        self.sparse_mass_matrix_plan = plan
        self.sparse_mass_matrix_indices = factor_indices(plan, self.model.device)
        self._arrays.allocate(SPARSE_FACTOR, group=articulations, nonzero=plan.nonzero_count)

    def allocate_size_groups(self, placeholder_sizes) -> None:
        """Allocate each size group's factor L and armature R; ``placeholder_sizes`` (the compact diagonal and the
        native sparse factor) keep (1, 1, 1) placeholders of L."""
        for size in self.forest.size_groups:
            n_arts = self.forest.n_arts_by_size[size]
            placeholder = size in placeholder_sizes
            factor = 1 if placeholder else 0
            sizes = {"art": n_arts, "size": size, "factor_art": factor or n_arts, "factor_size": factor or size}
            for name, array in self._arrays.allocate(SIZE_FACTORS, owner={}, **sizes).items():
                getattr(self, f"{name}_by_size")[size] = array

    def scatter_armature(self) -> None:
        """Copy the effective armature (DOF-ordered) to the size groups' R, and to the walk's where it is held."""
        if self.walk_arts is not None:
            self.walk_armature.assign(self.armature.astype(np.float32))
        if not self.forest.size_groups:
            return

        armature_np = self.armature
        art_dof_start_np = self.forest.articulation_dof_start_host
        art_H_rows_np = self.forest.articulation_dof_count_host

        # R_by_size is sized to the actual DOF count (as the factor L).
        for size in self.forest.size_groups:
            n_arts = self.forest.n_arts_by_size[size]
            R_np = np.zeros((n_arts, size), dtype=np.float32)

            group_to_art_np = self.forest.group_to_art_host[size]
            for group_idx in range(n_arts):
                art_idx = group_to_art_np[group_idx]
                dof_start = art_dof_start_np[art_idx]
                dof_count = art_H_rows_np[art_idx]
                R_np[group_idx, :dof_count] = armature_np[dof_start : dof_start + dof_count]

            self.R_by_size[size].assign(R_np)

    def build_body_factors(self) -> None:
        """Build each free body's held factor and coordinates (:class:`.kernels.BodyFactors`): the dynamics' size-6
        factors ``L_by_size[6]`` (armature included), which whiten its rows and map its velocity into factor
        coordinates, and the lookup of a body's factor by its first global coordinate. Built once with the buffers: a
        captured step allocates and copies nothing."""
        model, forest = self.model, self.forest
        device = model.device
        dof_factor = np.full(max(int(model.joint_dof_count), 1), -1, dtype=np.int32)
        if forest.has_free_rigid_bodies:
            free = np.flatnonzero(forest.is_free_rigid_host != 0)
            factor_index = forest.art_group_idx.numpy()[free]
            starts = forest.articulation_dof_start.numpy()[free]
            held = factor_index >= 0
            dof_factor[starts[held]] = factor_index[held]
        factors = BodyFactors()
        factors.body_to_articulation = forest.body_to_articulation
        factors.factor_index = forest.art_group_idx
        factors.dof_start = forest.articulation_dof_start
        factors.dof_factor = wp.array(dof_factor, dtype=wp.int32, device=device)
        lower = self.L_by_size.get(6) if forest.has_free_rigid_bodies else None
        factors.L = lower if lower is not None else wp.zeros((1, 6, 6), dtype=wp.float32, device=device)
        # Every articulation's dense factor, by address: the size groups' factors are allocated once, with the buffers.
        art_count = max(int(model.articulation_count), 1)
        first_art = np.full(max(int(model.joint_dof_count), 1), -1, dtype=np.int32)
        sizes = np.zeros(art_count, dtype=np.int32)
        offsets = np.zeros(art_count, dtype=np.int32)
        addresses = np.zeros(art_count, dtype=np.uint64)
        if int(model.articulation_count) > 0:
            starts = forest.articulation_dof_start.numpy()
            counts = forest.articulation_response_dof_count.numpy()
            world_offsets = forest.articulation_world_dof_offset
            offsets[:] = world_offsets.numpy() if world_offsets is not None else 0
            sizes[:] = counts
            responding = np.flatnonzero(counts > 0)
            first_art[starts[responding]] = responding
            for size, lower_group in self.L_by_size.items():
                if lower_group.shape[1] != size or size not in forest.group_to_art:
                    continue  # A compact or sparse size keeps a placeholder: no dense factor.
                arts = forest.group_to_art[size].numpy()
                addresses[arts] = lower_group.ptr + np.arange(len(arts), dtype=np.uint64) * np.uint64(size * size * 4)
        factors.first_art = wp.array(first_art, dtype=wp.int32, device=device)
        factors.size = wp.array(sizes, dtype=wp.int32, device=device)
        factors.world_offset = wp.array(offsets, dtype=wp.int32, device=device)
        factors.address = wp.array(addresses, dtype=wp.uint64, device=device)
        self.body_factors = factors

    def view(self) -> BodyFactors:
        """The operand struct of the factor's whiten device functions: the free bodies' factors and every
        articulation's dense factor by address (:meth:`build_body_factors`)."""
        return self.body_factors

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
                    forest.art_group_idx,
                    self.kinematic_joint_mask,
                    state_in.body_q,
                    model.body_inertia,
                    self.L_by_size[6],
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
        :data:`WALK_COORDINATE` floats, each body's :data:`WALK_LINK` floats and the leave's twists, each body's path
        entry (its parent and first coordinate within its tree, :func:`walk_paths_prelude`) and join slot, and the
        trees' lane group (:func:`walk_group`). One
        launch a step (:meth:`walk_step`) writes the step's geometry (:data:`WALK_NOW` floats a coordinate and
        :data:`WALK_LINK` a body), where the rows' seeds enter, and on the factor's own refreshes runs the inward pass,
        so the walk is the factor ``L`` refreshes, to rounding.

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
        parent_body = np.full(max(int(model.joint_count), 1), -1, dtype=np.int32)
        dof_body = np.full(max(int(model.joint_dof_count), 1), -1, dtype=np.int32)
        for art in arts.tolist():
            j0, j1 = int(start[art]), int(end[art])
            for j in range(j0, j1):
                up = int(joint_of[joint_parent[j]]) if joint_parent[j] >= 0 else -1
                parent = up - j0 if j0 <= up < j1 else -1
                if parent >= j - j0:
                    raise ValueError(f"The walk needs every parent before its children (articulation {art})")
                path[j] = (parent, int(qd_start[j] - qd_start[j0]))
                parent_body[j] = int(joint_child[j0 + parent]) if parent >= 0 else -1
                dof_body[qd_start[j] : qd_start[j + 1]] = j - j0
        # The inward pass's join slots (walk_inward): leaf first, a parent waits in a slot for the lifted inertia of
        # each child other than the joint after it, from the first such child until the pass reaches the parent.
        joins = np.full(max(int(model.joint_count), 1), -1, dtype=np.int32)
        held, bodies = 0, 1
        for art in arts.tolist():
            j0, j1 = int(start[art]), int(end[art])
            bodies = max(bodies, j1 - j0)
            free_slots: list[int] = []
            used = 0
            for j in range(j1 - 1, j0 - 1, -1):
                if joins[j] >= 0:
                    free_slots.append(int(joins[j]))
                parent = int(path[j, 0])
                if parent >= 0 and parent != j - j0 - 1 and joins[j0 + parent] < 0:
                    if free_slots:
                        free_slots.sort()
                        joins[j0 + parent] = free_slots.pop(0)
                    else:
                        joins[j0 + parent] = used
                        used += 1
            held = max(held, used)
        if held > WALK_JOIN_SLOTS:
            raise ValueError(f"The walk holds at most {WALK_JOIN_SLOTS} joins of a tree at once, a tree needs {held}")
        joints, dofs = max(int(model.joint_count), 1), max(int(model.joint_dof_count), 1)
        self.walk_arts = wp.array(arts, dtype=wp.int32, device=device)
        # The walking classes: each held tree's topology, which the step's one leave emits as its walk
        # (walk_leave_kernel), and each articulation's class among them.
        classes: dict[WalkClass, int] = {}
        walk_class = np.full(max(int(model.articulation_count), 1), -1, dtype=np.int32)
        for art in arts.tolist():
            j0, j1 = int(start[art]), int(end[art])
            table = WalkClass(
                tuple(int(p) for p in path[j0:j1, 0]),
                (*(int(f) for f in path[j0:j1, 1]), int(qd_start[j1] - qd_start[j0])),
            )
            walk_class[art] = classes.setdefault(table, len(classes))
        self.walk_classes = tuple(classes)
        self.walk_class = wp.array(walk_class, dtype=wp.int32, device=device)
        # A tree's group in the step's walk kernel (walk_step_kernel, walk_group): on CUDA a lane a body of a batch of
        # the inward pass and a join slot; one thread a tree on the CPU.
        self.walk_lanes = walk_group(bodies, held) if device.is_cuda else 1
        self.walk_block = WALK_BLOCK
        self.walk_path = wp.array(path, dtype=wp.vec2i, device=device)
        self.walk_joins = wp.array(joins, dtype=wp.int32, device=device)
        # Each joint's parent body (-1 at a root): the step's geometry reads its pose in one load.
        self.walk_parent_body = wp.array(parent_body, dtype=wp.int32, device=device)
        self.walk_path_host = path
        self.walk_data = wp.zeros((dofs, WALK_COORDINATE), dtype=wp.float32, device=device)
        self.walk_link = wp.zeros((joints, WALK_LINK), dtype=wp.float32, device=device)
        self.walk_leave_twist = wp.zeros(joints, dtype=wp.spatial_vector, device=device)
        self.walk_armature = wp.zeros(dofs, dtype=wp.float32, device=device)
        if self.armature is not None and self.armature.size:
            self.walk_armature.assign(self.armature.astype(np.float32))
        # The step's geometry (walk_step_kernel, every step): a seed enters as J at the step's poses.
        self.walk_now = wp.zeros((dofs, WALK_NOW), dtype=wp.float32, device=device)
        self.walk_now_link = wp.zeros((joints, WALK_LINK), dtype=wp.float32, device=device)
        view = WalkTrees()
        view.data, view.link, view.path = self.walk_data, self.walk_link, self.walk_path
        view.now, view.now_link = self.walk_now, self.walk_now_link
        view.scratch = self.walk_leave_twist
        view.joint_start = model.articulation_start
        view.joint_end = forest.articulation_joint_end
        view.dof_start = forest.articulation_dof_start
        view.dofs = forest.articulation_dof_count
        view.dof_body = wp.array(dof_body, dtype=wp.int32, device=device)
        body_joint = np.full(max(int(model.body_count), 1), -1, dtype=np.int32)
        body_joint[: joint_of.size] = joint_of[: body_joint.size]
        view.body_joint = wp.array(body_joint, dtype=wp.int32, device=device)
        # The release of each tree's slips, which the predictor's border writes (tree_kinetic_state.border_solve): as
        # many slips a tree as it has limited drives (at most RELEASE_SLIPS), each as deep as the class's deepest root
        # path.
        self.walk_slip_capacity = self.slip_capacity(arts)
        depth = 1
        for art in arts.tolist():
            j0, j1 = int(start[art]), int(end[art])
            reach = np.zeros(j1 - j0, dtype=np.int64)
            for j in range(j0, j1):
                parent = int(path[j][0])
                reach[j - j0] = (reach[parent] if parent >= 0 else 0) + int(qd_start[j + 1] - qd_start[j])
            depth = max(depth, int(reach.max(initial=1)))
        view.du = wp.zeros(dofs, dtype=wp.float32, device=device)
        # The leave reads the walk's du at each coordinate itself.
        self.walk_du_index = wp.array(np.arange(dofs, dtype=np.int32), dtype=wp.int32, device=device)
        view.release = release_view(
            max(int(model.articulation_count), 1),
            self.walk_slip_capacity,
            depth if self.walk_slip_capacity else 1,
            device,
        )
        self.walk_trees = view
        # The step's one leave of every walking tree (walk_leave_kernel), the release compiled where a tree can slip.
        self.walk_leave = walk_leave_kernel(self.walk_classes, self.walk_slip_capacity > 0) if classes else None

    def leave(self, v_out: wp.array, du: wp.array | None = None, du_index: wp.array | None = None) -> None:
        """The step's one leave (:func:`walk_leave_kernel`): every walking tree's velocity change walked down onto
        ``v_out`` (``v += P^T L^-T R^T du``): coordinate ``c``'s change is ``du[du_index[c]]``, by default the walk's
        own :attr:`WalkTrees.du`, which the sweeps add into from zero each step."""
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
                self.walk_du_index if du_index is None else du_index,
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
        """The walk's one launch a step (:func:`walk_step_kernel`, a lane group per held tree), at the poses the held
        tree published in ``state.body_q``: every tree's step geometry, each coordinate's column at its body's frame and
        each body's link and frame point, where the rows' seeds enter as ``J``; and on the trees the refresh just
        factored (``mass_update_mask``), the inward pass, so the step's geometry becomes the factor's frames. A free
        rigid body is a one-body tree: its walk is its six coordinates' factor."""
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
                self.walk_parent_body,
                self.walk_joins,
                model.joint_type,
                model.joint_dof_dim,
                model.joint_child,
                model.joint_q_start,
                model.joint_qd_start,
                model.joint_X_p,
                model.joint_axis,
                state.joint_q,
                state.body_q,
                model.body_mass,
                model.body_com,
                model.body_inertia,
                self.walk_armature,
                self.drive_row_by_dof,
                self.aug_row_K if self.parallel_drives else self._arrays.placeholder(wp.float32),
            ],
            outputs=[self.walk_now, self.walk_now_link, self.walk_data, self.walk_link],
            block_dim=max(self.walk_block, lanes),
            device=model.device,
        )

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
        """The walk's data as the device walks it (:func:`walk_prelude`), float32: each coordinate's
        :data:`WALK_COORDINATE` floats ``s``, ``ut`` and ``1 / ell``, and each body's :data:`WALK_LINK` floats, the
        offset to its parent's frame."""
        data = np.zeros((self.s.shape[0], WALK_COORDINATE), dtype=np.float32)
        data[:, 0:6], data[:, 6:12], data[:, 12] = self.s, self.ut, 1.0 / self.ell
        link = np.zeros((self.parent.size, WALK_LINK), dtype=np.float32)
        link[:, 0:3] = self.offset
        return data, link

    @property
    def path_table(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """The tree's path table (:func:`walk_prelude`): each body's parent, and each body's first coordinate with the
        count after the last."""
        return tuple(int(b) for b in self.parent), tuple(int(c) for c in self.first)


# ---------------------------------------------------------------------------------------------------------------------
# The walk on the device (FL3.1): a lane per seed up, a lane per tree down, the path table in constant memory
# ---------------------------------------------------------------------------------------------------------------------

WALK_STAGED = 13
"""The floats of a coordinate's record the walk's functions read (``s``, ``ut``, ``1 / ell``): what a kernel stages of
:data:`WALK_COORDINATE` in shared memory (:class:`.response_operator.LocalBlock`)."""

WALK_COORDINATE = 16
"""Floats per coordinate of the device walk's data: ``s`` and ``ut`` (``(v, w)``, six each), ``1 / ell``, the pivot
without any drive's ``K`` (``P0 = s^T I s + R``, the split pivot's, SAT_PHYS §1.2), and the diagonal the refresh adds,
``R`` and ``R + K`` (:data:`WALK_ARMATURE`, :data:`WALK_ADDED`), so a coordinate's record is four aligned ``float4``."""

WALK_LINK = 8
"""Floats per body of the device walk's data: its parent's frame less its own, a pad, its frame's world point, a pad."""

WALK_PIVOT = 13
"""Where a coordinate's record holds its pivot without ``K`` (:data:`WALK_COORDINATE`)."""

WALK_FRAME = 4
"""Where a body's record holds its frame's world point (:data:`WALK_LINK`), at the refresh's poses: the factor's
inertias are taken there."""

WALK_NOW = 6
"""Floats per coordinate of the step's columns (:attr:`WalkTrees.now`): ``s`` at its body's frame at the step's poses
(its links, :attr:`WalkTrees.now_link`, take :data:`WALK_LINK`'s layout)."""

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
    // WALK_UP(data, link, now, now_link, body, top, p, w, sink): the states p of $rows rows, p[6 r + k] the wrench
    // (f, t) of row r at body's frame, walked up to the root from coordinate top of body (its last for a wrench seed):
    // each coordinate c takes the footprint z = (s_c . p) / ell_c, given to sink(r, c, z, j), and passes p - ut_c z; a
    // joint passes t - d x f to its parent. now (null for rows seeded by covectors alone): the step's geometry, each
    // coordinate's column at its body's frame at the step's poses (WALK_NOW floats) and each body's link there
    // (now_link, link's layout), with w[6 r + k] row r's seed wrench at the current body's frame there. A seed enters
    // as the step's J and the refresh's factor whitens it (a held factor between refreshes, L^-1 P J^T as the dense rows
    // do): p is the seed's wrench less what the factor has taken, so the footprint takes (s_now_c - s_c) . w and a joint
    // passes (d - d_now) x f_w more. On a refresh step the two geometries are one and both terms are zero. The sink's j
    // is the step's J entry, s_now_c . w (zero without now), so a row's incident J v is the sum of j v_c along its walk.
    const auto WALK_UP = [&](const float* data, const float* link, const float* now, const float* now_link, int body,
                             int top, float* p, float* w, auto&& sink) {
        for (int b = body; b >= 0; b = WALK_PARENT_OF(b)) {
            for (int c = b == body ? top : WALK_FIRST_OF(b + 1) - 1; c >= WALK_FIRST_OF(b); --c) {
                const float* d = data + $coordinate * c;
                const float* e = now != nullptr ? now + $now * c : nullptr;
                for (int r = 0; r < $rows; ++r) {
                    float* q = p + 6 * r;
                    float x = (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]) + (d[3] * q[3] + d[4] * q[4] + d[5] * q[5]);
                    float j = 0.0f;
                    if (e != nullptr) {
                        const float* v = w + 6 * r;
                        x += ((e[0] - d[0]) * v[0] + (e[1] - d[1]) * v[1] + (e[2] - d[2]) * v[2])
                             + ((e[3] - d[3]) * v[3] + (e[4] - d[4]) * v[4] + (e[5] - d[5]) * v[5]);
                        j = (e[0] * v[0] + e[1] * v[1] + e[2] * v[2]) + (e[3] * v[3] + e[4] * v[4] + e[5] * v[5]);
                    }
                    const float z = x * d[12];
                    for (int k = 0; k < 6; ++k) q[k] -= d[6 + k] * z;
                    sink(r, c, z, j);
                }
            }
            if (WALK_PARENT_OF(b) >= 0) {
                const float* l = link + $link * b;
                for (int r = 0; r < $rows; ++r) {
                    float* q = p + 6 * r;
                    const float fx = q[0], fy = q[1], fz = q[2];
                    q[3] -= l[1] * fz - l[2] * fy;
                    q[4] -= l[2] * fx - l[0] * fz;
                    q[5] -= l[0] * fy - l[1] * fx;
                }
                if (now != nullptr) {
                    const float* o = now_link + $link * b;
                    const float g0 = l[0] - o[0], g1 = l[1] - o[1], g2 = l[2] - o[2];
                    for (int r = 0; r < $rows; ++r) {
                        float* q = p + 6 * r;
                        float* v = w + 6 * r;
                        const float fx = v[0], fy = v[1], fz = v[2];
                        q[3] += g1 * fz - g2 * fy;
                        q[4] += g2 * fx - g0 * fz;
                        q[5] += g0 * fy - g1 * fx;
                        v[3] -= o[1] * fz - o[2] * fy;
                        v[4] -= o[2] * fx - o[0] * fz;
                        v[5] -= o[0] * fy - o[1] * fx;
                    }
                }
            }
        }
    };
    // WALK_SEED(data, c, x, p, sink): the first step of rows seeded by the covector x[r] at coordinate c (a limit's, a
    // velocity limit's), on the states p: z = (x + s_c . p) / ell_c, its J entry x[r]. WALK_UP continues from c - 1 of
    // c's body.
    const auto WALK_SEED = [&](const float* data, int c, const float* x, float* p, auto&& sink) {
        const float* d = data + $coordinate * c;
        for (int r = 0; r < $rows; ++r) {
            float* q = p + 6 * r;
            const float z = (x[r] + ((d[0] * q[0] + d[1] * q[1] + d[2] * q[2]) + (d[3] * q[3] + d[4] * q[4] + d[5] * q[5])))
                            * d[12];
            for (int k = 0; k < 6; ++k) q[k] -= d[6 + k] * z;
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
        const float* l = link + $link * b;
        V[0] = W[0] + (l[1] * W[5] - l[2] * W[4]);
        V[1] = W[1] + (l[2] * W[3] - l[0] * W[5]);
        V[2] = W[2] + (l[0] * W[4] - l[1] * W[3]);
        V[3] = W[3];
        V[4] = W[4];
        V[5] = W[5];
    };
    // WALK_ENTER(data, link, qd, u, twist): the walk's coordinates of known velocities qd, u = L^T P qd (the leave
    // inverted), root first: u_c = ell_c qd_c + ut_c . V, then V += s_c qd_c, V its body's twist (the audit's).
    const auto WALK_ENTER = [&](const float* data, const float* link, const float* qd, float* u, float* twist) {
        for (int b = 0; b < WALK_BODIES; ++b) {
            float V[6];
            WALK_PARENT_TWIST(link, twist, b, V);
            for (int c = WALK_FIRST_OF(b); c < WALK_FIRST_OF(b + 1); ++c) {
                const float* d = data + $coordinate * c;
                u[c] = qd[c] / d[12] + ((d[6] * V[0] + d[7] * V[1] + d[8] * V[2])
                                        + (d[9] * V[3] + d[10] * V[4] + d[11] * V[5]));
                for (int k = 0; k < 6; ++k) V[k] += d[k] * qd[c];
            }
            for (int k = 0; k < 6; ++k) twist[6 * b + k] = V[k];
        }
    };
    // The slips' release of tree art (WalkRelease, written by the predictor's border; W9's verdict: R = L_F^-1 L_K,
    // tree-lower, so every row keeps its support). Slip j's steps along its root path, leaf first, start at
    // at = (art * cap + j) * steps of release.coordinate, .b, .a, .g; a coordinate of -1 ends the path.
    // WALK_SLIP(release, at, z, stride): z <- R_j z (coordinate c at z[c * stride]), leaf first.
    // WALK_SLIP_T(release, at, u): u <- R_j^T u, root first. WALK_SLIP_IT(release, at, u): u <- R_j^-T u, root first.
    // WALK_RELEASE(release, art, z, stride): a row's footprints z <- R z, the first slip first.
    // WALK_RELEASE_T(release, art, u): the leave's u <- R^T u, the last slip first.
    // WALK_RELEASE_IT(release, art, u): the audit's coordinates u <- R^-T u, the first slip first.
    const auto WALK_SLIP = [&](const auto& release, const size_t at, float* z, const int stride) {
        const int steps = release.b.shape[2];
        const int* coordinate = release.coordinate.data + at;
        const float *b = release.b.data + at, *a = release.a.data + at, *g = release.g.data + at;
        float s = 0.0f;
        for (int k = 0; k < steps; ++k) {
            const int c = coordinate[k];
            if (c < 0) break;
            const float x = z[c * stride] - b[k] * s;
            s += a[k] * x;
            z[c * stride] = g[k] * x;
        }
    };
    const auto WALK_SLIP_T = [&](const auto& release, const size_t at, float* u) {
        const int steps = release.b.shape[2];
        const int* coordinate = release.coordinate.data + at;
        const float *b = release.b.data + at, *a = release.a.data + at, *g = release.g.data + at;
        float t = 0.0f;
        for (int k = steps - 1; k >= 0; --k) {
            const int c = coordinate[k];
            if (c < 0) continue;
            const float x = g[k] * u[c] + a[k] * t;
            u[c] = x;
            t -= b[k] * x;
        }
    };
    const auto WALK_SLIP_IT = [&](const auto& release, const size_t at, float* u) {
        const int steps = release.b.shape[2];
        const int* coordinate = release.coordinate.data + at;
        const float *b = release.b.data + at, *a = release.a.data + at, *g = release.g.data + at;
        float t = 0.0f;
        for (int k = steps - 1; k >= 0; --k) {
            const int c = coordinate[k];
            if (c < 0) continue;
            const float x = u[c];
            u[c] = (x - a[k] * t) / g[k];
            t -= b[k] * x;
        }
    };
    const auto WALK_RELEASE = [&](const auto& release, const int art, float* z, const int stride = 1) {
        const int cap = release.b.shape[1], steps = release.b.shape[2];
        for (int j = 0; j < release.count.data[art]; ++j) WALK_SLIP(release, ((size_t)art * cap + j) * steps, z, stride);
    };
    const auto WALK_RELEASE_T = [&](const auto& release, const int art, float* u) {
        const int cap = release.b.shape[1], steps = release.b.shape[2];
        for (int j = release.count.data[art] - 1; j >= 0; --j) WALK_SLIP_T(release, ((size_t)art * cap + j) * steps, u);
    };
    const auto WALK_RELEASE_IT = [&](const auto& release, const int art, float* u) {
        const int cap = release.b.shape[1], steps = release.b.shape[2];
        for (int j = 0; j < release.count.data[art]; ++j) WALK_SLIP_IT(release, ((size_t)art * cap + j) * steps, u);
    };"""


@functools.cache
def walk_prelude(parent: tuple[int, ...], first: tuple[int, ...], rows: int) -> DevicePrelude:
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
    walk = substitute(_WALK, rows=rows, coordinate=WALK_COORDINATE, link=WALK_LINK, now=WALK_NOW)
    return DevicePrelude(
        f"walk_{rows}_{len(parent)}",
        substitute(_WALK_TABLE, table) + "\n" + walk,
        cpu=substitute(_WALK_TABLE_CPU, table) + "\n" + walk,
    )


@functools.cache
def walk_paths_prelude(
    rows: int, *, coordinate: int = WALK_COORDINATE, link: int = WALK_LINK, now: int = WALK_NOW
) -> DevicePrelude:
    """The device walk of any tree, for both devices: :func:`walk_prelude`'s functions over a path table in global
    memory, one tree at a time (``WALK_TREE(path, bodies, coordinates)`` before its walks), the table
    :attr:`TreeFactor.walk_path` holds for every articulation. The general placement's worlds hold any forest, so its
    one writer and leave read each tree's table where a class's kernel bakes it into constant memory.

    Args:
        rows: The rows a lane walks together (:func:`walk_prelude`).
        coordinate: Floats a coordinate's record, ``link`` a body's and ``now`` a step column's: the device walk's
            layout (:data:`WALK_COORDINATE`, :data:`WALK_LINK`, :data:`WALK_NOW`), or a kernel's staged copy of it
            (:class:`.response_operator.LocalBlock`: a coordinate's step column interleaved after its
            :data:`WALK_STAGED` floats, so ``now == coordinate``).
    """
    if rows < 1:
        raise ValueError(f"a lane walks at least one row, got {rows}")
    walk = _WALK_PATHS + "\n" + substitute(_WALK, rows=rows, coordinate=coordinate, link=link, now=now)
    layout = (coordinate, link, now)
    name = f"walk_paths_{rows}" + (
        "" if layout == (WALK_COORDINATE, WALK_LINK, WALK_NOW) else f"_{coordinate}_{link}_{now}"
    )
    return DevicePrelude(name, walk, cpu=walk)


# ---------------------------------------------------------------------------------------------------------------------
# The class emitter (ZB): a class's walk as straight-line native code, and the one walk down on it
# ---------------------------------------------------------------------------------------------------------------------

WALK_UNROLL = 64
"""Coordinates of a class up to which :meth:`WalkClass.walk` emits its walk as straight-line code; a larger class takes
the loop form. Chosen from data (WALK_MAPPING §4, §8): unrolled, every load of the walk issues ahead of its serial
chain and every body's state is a register, while the code grows by about 20 instructions a coordinate; a deep chain
gains nothing from straight-line code once its loads leave the chain (its state is one body's), and K1's trees, which
the support masks bound at 64 coordinates, all unroll."""


@functools.cache
def _walk_slots(parent: tuple[int, ...]) -> tuple[tuple[int, ...], int]:
    """Each body's slot for its state where a child other than the next body meets it (a join), else -1, and the slot
    count. A join's slot is live over its bodies from the join to its last child, in either direction of the walk, so
    joins whose spans do not overlap share a slot."""
    last = {}
    for body, up in enumerate(parent):
        if up >= 0:
            last[up] = body
    joins = {up for body, up in enumerate(parent) if up >= 0 and up != body - 1}
    slot, free, held, count = [-1] * len(parent), [], {}, 0
    for body in range(len(parent)):
        for join in [j for j, child in held.items() if child == body]:
            free.append(slot[join])
            del held[join]
        if body in joins:
            if free:
                slot[body] = free.pop()
            else:
                slot[body], count = count, count + 1
            held[body] = last[body]
    return tuple(slot), count


def _fill(text: str, **values) -> str:
    """``text`` with its ``$`` slots filled from ``values``: a walk text names only the slots it reads."""
    template = Template(text)
    return template.substitute({name: str(values[name]) for name in template.get_identifiers()})


def indent_text(text: str, spaces: int) -> str:
    """``text`` with every non-empty line indented by ``spaces``."""
    pad = " " * spaces
    return "\n".join(pad + line if line.strip() else line for line in text.splitlines())


@dataclasses.dataclass(frozen=True)
class WalkClass:
    """A walking class: the topology its trees share, each body's parent (-1: the world, every parent before its
    children) and first coordinate, with the count after the last (:func:`walk_prelude`'s table).

    :meth:`walk` is the class emitter: it turns the table into a walk's native statements, root first (the leave, a
    predictor's solve down) or leaf first (an inward pass, a solve up), so a kernel built for the class walks its tree
    with no table in front of its steps. The walk's own arithmetic is the caller's text."""

    parent: tuple[int, ...]
    first: tuple[int, ...]

    def __post_init__(self):
        if len(self.first) != len(self.parent) + 1 or any(p >= b for b, p in enumerate(self.parent)):
            raise ValueError("a walk's class needs every parent before its children and a first coordinate per body")

    @property
    def coordinates(self) -> int:
        return int(self.first[-1])

    @property
    def tag(self) -> str:
        """A short name of the table, for the class's emitted names and its kernels' names."""
        return hashlib.sha256(repr((self.parent, self.first)).encode()).hexdigest()[:8]

    def walk(
        self,
        joint: str,
        step: str,
        *,
        begin: str = "",
        end: str = "",
        root_first: bool = True,
        width: int = 6,
        cuda: bool = True,
    ) -> str:
        """Native statements that walk the class's tree on a state of ``width`` floats a body, in either direction.

        Root first, a body's state ``$V`` takes its parent's ``$P`` through ``joint`` (a root's starts at zero), then
        ``begin``, its coordinates in order (``step``) and ``end``. Leaf first, a body's state starts at zero and
        gathers what its children pass through ``joint`` (written into ``$P``, the parent's), then ``begin``, its
        coordinates last first and ``end``; then it passes to its parent.

        Up to :data:`WALK_UNROLL` coordinates the walk is straight-line code: each body's state an array of constant
        indices (registers), every load free to issue ahead of the serial chain. Past it, the loop form over the class's
        table in constant memory: a chain's state in registers, a join's in a slot (:func:`_walk_slots`).

        Args:
            joint: Body ``$b``'s joint to its parent: ``$V`` from ``$P`` root first, ``$V`` into ``$P`` leaf first.
            step: Coordinate ``$c`` of body ``$b`` on its state ``$V``.
            begin: Body ``$b``'s statements before its coordinates, on ``$V``.
            end: Body ``$b``'s statements after its coordinates, on ``$V``.
            root_first: The walk's direction.
            width: Floats of a body's state.
            cuda: The CUDA text (its tables in constant memory) or its CPU twin.
        """
        bodies = len(self.parent)
        zero = "{" + ", ".join(["0.0f"] * width) + "}"

        def body_text(b, state, coordinates):
            out = [_fill(begin, b=b, V=state)] if begin else []
            out += [_fill(step, b=b, c=c, V=state) for c in coordinates]
            out += [_fill(end, b=b, V=state)] if end else []
            return out

        if self.coordinates <= WALK_UNROLL:
            lines = []
            if root_first:
                for b, up in enumerate(self.parent):
                    state = f"walk{b}"
                    if up < 0:
                        lines.append(f"float {state}[{width}] = {zero};")
                    else:
                        lines += [f"float {state}[{width}];", _fill(joint, b=b, V=state, P=f"walk{up}")]
                    lines += body_text(b, state, range(self.first[b], self.first[b + 1]))
            else:
                lines.append(f"float {', '.join(f'walk{b}[{width}] = {zero}' for b in range(bodies))};")
                for b in reversed(range(bodies)):
                    up = self.parent[b]
                    lines += body_text(b, f"walk{b}", reversed(range(self.first[b], self.first[b + 1])))
                    if up >= 0:
                        lines.append(_fill(joint, b=b, V=f"walk{b}", P=f"walk{up}"))
            return "\n".join(lines)
        slots, count = _walk_slots(self.parent)
        memory = "__constant__" if cuda else "const"
        name = f"WALK_CLASS_{self.tag}"
        tables = [
            f"static {memory} int {name}_PARENT[{bodies}] = {{{', '.join(map(str, self.parent))}}};",
            f"static {memory} int {name}_FIRST[{bodies + 1}] = {{{', '.join(map(str, self.first))}}};",
        ]
        if count:
            tables.append(f"static {memory} int {name}_SLOT[{bodies}] = {{{', '.join(map(str, slots))}}};")
        each = "for (int k = 0; k < $width; ++k)"
        slot = f"S[{name}_SLOT[$at]]"
        if root_first:
            parent_state = f"{each} P[k] = V[k];"
            if count:
                parent_state = (
                    f"if (up == b - 1) {{\n    {each} P[k] = V[k];\n}} else {{\n"
                    f"    {each} P[k] = {_fill(slot, at='up')}[k];\n}}"
                )
            loop = f"""float V[$width], P[$width];{f" float S[{count}][$width];" if count else ""}
for (int b = 0; b < {bodies}; ++b) {{
    const int up = {name}_PARENT[b];
    if (up < 0) {{
        {each} V[k] = 0.0f;
    }} else {{
{indent_text(parent_state, 8)}
{indent_text(_fill(joint, b="b", V="V", P="P"), 8)}
    }}
{indent_text(_fill(begin, b="b", V="V"), 4) if begin else ""}
    for (int c = {name}_FIRST[b]; c < {name}_FIRST[b + 1]; ++c) {{
{indent_text(_fill(step, b="b", c="c", V="V"), 8)}
    }}
{indent_text(_fill(end, b="b", V="V"), 4) if end else ""}
{indent_text(f"if ({name}_SLOT[b] >= 0)" + chr(10) + f"    {each} {_fill(slot, at='b')}[k] = V[k];", 4) if count else ""}
}}"""
        else:
            gather = f"{each} {{\n    V[k] = N[k];\n    N[k] = 0.0f;\n}}"
            if count:
                gather = (
                    f"if ({name}_SLOT[b] >= 0) {{\n    {each} {{\n        V[k] = {_fill(slot, at='b')}[k];\n"
                    f"        {_fill(slot, at='b')}[k] = 0.0f;\n    }}\n}} else {{\n{indent_text(gather, 4)}\n}}"
                )
            give = _fill(joint, b="b", V="V", P="N")
            if count:
                give = (
                    f"if ({name}_SLOT[up] >= 0) {{\n{indent_text(_fill(joint, b='b', V='V', P=_fill(slot, at='up')), 4)}\n"
                    f"}} else {{\n{indent_text(give, 4)}\n}}"
                )
            loop = f"""float V[$width], N[$width] = {zero};{f" float S[{count}][$width] = {{}};" if count else ""}
for (int b = {bodies - 1}; b >= 0; --b) {{
{indent_text(gather, 4)}
{indent_text(_fill(begin, b="b", V="V"), 4) if begin else ""}
    for (int c = {name}_FIRST[b + 1] - 1; c >= {name}_FIRST[b]; --c) {{
{indent_text(_fill(step, b="b", c="c", V="V"), 8)}
    }}
{indent_text(_fill(end, b="b", V="V"), 4) if end else ""}
    const int up = {name}_PARENT[b];
    if (up >= 0) {{
{indent_text(give, 8)}
    }}
}}"""
        return "\n".join(tables) + "\n" + Template(loop).substitute(width=width)


FREE_BODY_WALK = WalkClass((-1,), (0, 6))
"""A free rigid body's walking class: one body of six coordinates."""

_WALK_ACCESS = """    // The walk's read-only loads (its records and links are never written while a walk runs, so the loads leave the
    // chain), and a lane group's fetch of a byte range into L1, one line a lane a trip.
    const auto WALK_LOAD = [&](const float* p) -> float { return __ldg(p); };
    const auto WALK_RECORD = [&](const float* p) -> float4 { return __ldg(reinterpret_cast<const float4*>(p)); };
    // A product rounded on its own: the chain's coordinate is a float32 value before any consumer adds it.
    const auto WALK_MUL = [&](const float a, const float b) -> float { return __fmul_rn(a, b); };
    const auto WALK_INDEX = [&](const int* p) -> int { return __ldg(p); };
    const auto WALK_FETCH = [&](const void* start, const int bytes, const int lane, const int lanes) {
        const size_t end = reinterpret_cast<size_t>(start) + (size_t)bytes;
        for (size_t line = (reinterpret_cast<size_t>(start) & ~(size_t)127) + (size_t)128 * lane; line < end;
             line += (size_t)128 * lanes)
            asm volatile("prefetch.global.L1 [%0];" ::"l"(line));
    };"""

_WALK_ACCESS_CPU = """    struct WalkRecord {
        float x, y, z, w;
    };
    const auto WALK_LOAD = [&](const float* p) -> float { return *p; };
    const auto WALK_RECORD = [&](const float* p) -> WalkRecord { return WalkRecord{p[0], p[1], p[2], p[3]}; };
    const auto WALK_MUL = [&](const float a, const float b) -> float { return a * b; };
    const auto WALK_INDEX = [&](const int* p) -> int { return *p; };
    const auto WALK_FETCH = [&](const void*, const int, const int, const int) {};"""

_DOWN_JOINT = """{
    const auto l = WALK_RECORD(link + $link * ($b));
    $V[0] = $P[0] + (l.y * $P[5] - l.z * $P[4]);
    $V[1] = $P[1] + (l.z * $P[3] - l.x * $P[5]);
    $V[2] = $P[2] + (l.x * $P[4] - l.y * $P[3]);
    $V[3] = $P[3];
    $V[4] = $P[4];
    $V[5] = $P[5];
}"""

_DOWN_STEP = """{
    const float* d = data + $coordinate * ($c);
    const auto s = WALK_RECORD(d), t = WALK_RECORD(d + 4), w = WALK_RECORD(d + 8);
    const float q = WALK_MUL(
        u($c) - ((t.z * $V[0] + t.w * $V[1] + w.x * $V[2]) + (w.y * $V[3] + w.z * $V[4] + w.w * $V[5])), WALK_LOAD(d + 12));
    qd($c, q);
    $V[0] += s.x * q;
    $V[1] += s.y * q;
    $V[2] += s.z * q;
    $V[3] += s.w * q;
    $V[4] += t.x * q;
    $V[5] += t.y * q;
}"""


def _walk_down(classes: tuple[WalkClass, ...], cuda: bool) -> str:
    joint = Template(_DOWN_JOINT).safe_substitute(link=WALK_LINK)
    step = Template(_DOWN_STEP).safe_substitute(coordinate=WALK_COORDINATE)
    bodies = [walk.walk(joint, step, end="twist($b, $V);", cuda=cuda) for walk in classes]
    if len(bodies) == 1:
        walk = indent_text(bodies[0], 8)
    else:
        cases = (f"case {k}: {{\n{indent_text(body, 4)}\n    break;\n}}" for k, body in enumerate(bodies))
        walk = indent_text("switch (cls) {\n" + "\n".join(cases) + "\n}", 8)
    return f"""    // WALK_DOWN(cls, data, link, u, qd, twist): the chain of the leave P^T L^-T u of one tree of class cls (an index
    // into the classes the prelude was emitted for), on one lane, root first: each coordinate c takes
    // qd_c = (u_c - ut_c . V) / ell_c, then V += s_c qd_c, V its body's twist (v, w) at its frame, a register, which
    // twist(b, V) receives after the body's coordinates. u(c) reads coordinate c and qd(c, q) writes it (in place
    // allowed: u(c) is read before qd(c, q)).
    const auto WALK_DOWN = [&](const int cls, const float* __restrict__ data, const float* __restrict__ link, auto&& u,
                               auto&& qd, auto&& twist) {{
        (void)cls;
{walk}
    }};
    // WALK_LEAVE(cls, data, link, coordinates, bodies, lane, lanes, walker, ready, u, qd, twist): the walk down's unit,
    // one tree by a lane group. Inputs: the tree's class cls, its records (coordinates x {WALK_COORDINATE} floats at
    // data) and links (bodies x {WALK_LINK} floats at link), this thread's lane among the group's lanes, and the lane
    // that walks. The group's lanes fetch the records and links into L1 in one parallel trip, off the chain (no
    // barrier: a fetch is a hint); then lane walker runs ready() (the caller's preparation of u, while the lines
    // arrive) and WALK_DOWN, the serial chain, its state (a body's twist) in registers. The chain is handed u(c), the
    // tree's coordinate c in the factor's frame, and hands back qd(c, q) and twist(b, V). No shared memory, no scratch.
    const auto WALK_LEAVE = [&](const int cls, const float* data, const float* link, const int coordinates,
                                const int bodies, const int lane, const int lanes, const int walker, auto&& ready,
                                auto&& u, auto&& qd, auto&& twist) {{
        WALK_FETCH(data, coordinates * {4 * WALK_COORDINATE}, lane, lanes);
        WALK_FETCH(link, bodies * {4 * WALK_LINK}, lane, lanes);
        if (lane != walker) return;
        ready();
        WALK_DOWN(cls, data, link, u, qd, twist);
    }};"""


@functools.cache
def walk_down_prelude(classes: tuple[WalkClass, ...]) -> DevicePrelude:
    """The one walk down (the leave ``P^T L^-T u``) of the walking classes ``classes``, for both devices: a prelude
    declaring

    - ``WALK_LEAVE(cls, data, link, coordinates, bodies, lane, lanes, walker, ready, u, qd, twist)``, the leave's unit:
      a caller's lane group fetches one tree's records and links into L1 in one parallel trip (``WALK_FETCH``), then
      one of its lanes prepares ``u`` (``ready``) and runs the chain (WALK_MAPPING §4, B5);
    - ``WALK_DOWN(cls, data, link, u, qd, twist)``, the chain: each class's walk emitted by :meth:`WalkClass.walk`
      (straight-line code up to :data:`WALK_UNROLL` coordinates), its records read through the read-only path
      (``WALK_LOAD``, ``WALK_RECORD``), every twist a register, handed to ``twist(b, V)`` after its body.

    No shared memory and no scratch. The step's one leave takes it (:func:`walk_leave_kernel`), and the general sweep's
    free bodies' speed rows on their one-body class (:data:`FREE_BODY_WALK`).

    Args:
        classes: The walking classes, indexed by ``cls``.
    """
    if not classes:
        raise ValueError("a walk down needs a class")
    tag = hashlib.sha256(repr(classes).encode()).hexdigest()[:8]
    return DevicePrelude(
        f"walk_down_{tag}",
        _WALK_ACCESS + "\n" + _walk_down(classes, True),
        cpu=_WALK_ACCESS_CPU + "\n" + _walk_down(classes, False),
    )


LEAVE_LANES = 4
"""Lanes of a tree's group in the step's leave (:func:`walk_leave_kernel`) on CUDA: they fetch the tree's lines into L1
in one parallel trip and one of them walks (WALK_MAPPING §4, B5: 4 lanes ran at or under 8 on g1, franka and
allegro)."""

_LEAVE = """    const int tree = tid / lanes, lane = tid - tree * lanes;
    if (tree >= arts.shape[0]) return;
    const int art = arts.data[tree];
    const int cls = walk_class.data[art];
    if (cls < 0) return;
    const int j0 = walk.joint_start.data[art], d0 = walk.dof_start.data[art], n = walk.dofs.data[art];
    // Coordinate c's change is du[at[c]]: the walk's du at the coordinate itself, or a sweep's own layout.
    const int* const at = du_index.data + d0;
    float* const v = v_out.data + d0;
    // The group's lanes fetch the tree's change indices and velocity beside its records (WALK_LEAVE).
    WALK_FETCH(at, n * 4, lane, lanes);
    WALK_FETCH(v, n * 4, lane, lanes);
$release    WALK_LEAVE(
        cls, walk.data.data + (size_t)d0 * $coordinate, walk.link.data + (size_t)j0 * $link, n,
        walk.joint_end.data[art] - j0, lane, lanes, 0, ready, $read, [&](const int c, const float q) { v[c] += q; },
        [](const int, const float*) {});
"""

_LEAVE_RELEASE = """    // The slips' release transposed, R^T du, on a copy where the tree slips (the predictor's border's R).
    float r[$width];
    bool slipped = false;
    const auto ready = [&]() {
        if (walk.release.count.data[art] == 0) return;
        for (int c = 0; c < n; ++c) r[c] = du.data[at[c]];
        WALK_RELEASE_T(walk.release, art, r);
        slipped = true;
    };
"""


@functools.cache
def walk_leave_kernel(classes: tuple[WalkClass, ...], release: bool) -> wp.Kernel:
    """The step's one leave, ``v += P^T L^-T R^T du``, of every walking tree (``arts``), a lane group a tree (``lanes``:
    :data:`LEAVE_LANES` on CUDA, one on the CPU; ``dim`` trees times lanes), by the walk down's unit (``WALK_LEAVE`` of
    :func:`walk_down_prelude`, the tree's class at ``walk_class``). Each coordinate's change is ``du[du_index[dof]]``:
    every placement's sweep adds its change in the walk's coordinates into :attr:`WalkTrees.du` (the index the
    coordinate itself), the support placement's sparse sweep publishes it in its factor coordinates (the index its
    factor coordinate). This kernel takes the slips' release transposed where the class can slip (``release``) and
    walks each tree down onto ``v_out``, which holds the predictor and the changes of the placements that do not walk.

    Args:
        classes: The walking classes, indexed by ``walk_class``.
        release: Whether a tree of the classes can slip (:attr:`TreeFactor.walk_slip_capacity`).
    """
    from .kernel_template import THREADS, KernelTemplate  # noqa: PLC0415

    width = max(walk.coordinates for walk in classes)
    if release:
        head = substitute(_LEAVE_RELEASE, width=width)
        read = "[&](const int c) -> float { return slipped ? r[c] : WALK_LOAD(du.data + WALK_INDEX(at + c)); }"
    else:
        head = "    const auto ready = []() {};\n"
        read = "[&](const int c) -> float { return WALK_LOAD(du.data + WALK_INDEX(at + c)); }"
    body = substitute(_LEAVE, release=head, read=read, coordinate=WALK_COORDINATE, link=WALK_LINK)
    tag = hashlib.sha256(repr(classes).encode()).hexdigest()[:8]
    return KernelTemplate(
        f"walk_leave_{len(classes)}_{tag}" + ("_release" if release else ""),
        (
            ("arts", wp.array[int]),
            ("lanes", int),
            ("walk", WalkTrees),
            ("walk_class", wp.array[int]),
            ("du", wp.array[float]),
            ("du_index", wp.array[int]),
            ("v_out", wp.array[float]),
        ),
        body,
        THREADS,
        preludes=((walk_paths_prelude(1),) if release else ()) + (walk_down_prelude(classes),),
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
    the all-``K`` walk, which keeps every row's support (``WALK_RELEASE``, ``WALK_RELEASE_T``, ``WALK_RELEASE_IT``; the
    Warp twin :func:`release_path`). Two slips' paths meet if and only if their roots are equal (their ``R_j`` touch
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
    - ``now``, ``now_link``: each coordinate's column at its body's frame, and each body's link (:data:`WALK_LINK`'s
      layout: the offset to its parent's frame and its frame point), at the step's poses
      (:meth:`TreeFactor.walk_step`, every step): a seed enters the walk as ``J`` there, so a held factor (a step
      between refreshes) whitens the step's ``J``, and a row's incident ``J v_hat`` is summed along its walk;
    - ``scratch``: the leave's and the audit's twists;
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
    now: wp.array2d[float]
    now_link: wp.array2d[float]
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
    view.now = wp.zeros((1, WALK_NOW), dtype=wp.float32, device=device)
    view.now_link = wp.zeros((1, WALK_LINK), dtype=wp.float32, device=device)
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
    float p[6], w[6];
    for (int k = 0; k < 6; ++k) p[k] = w[k] = wrench[k];
    int top = WALK_FIRST_OF(body + 1) - 1;
    const float* now = walk.now.data + (size_t)dof0 * $now;
    const float* now_link = walk.now_link.data + (size_t)joint0 * $link;
    if (coordinate >= 0) {
        const float x[1] = {value};
        WALK_SEED(data, coordinate, x, p, sink);
        top = coordinate - 1;
        now = nullptr;
    }
    WALK_UP(data, link, now, now_link, body, top, p, w, sink);
    return wp::vec2(incident, squares);
"""


@functools.cache
def _walk_into_snippet() -> str:
    from .device_preludes import dual_native  # noqa: PLC0415

    return dual_native(
        substitute(_WALK_INTO, coordinate=WALK_COORDINATE, link=WALK_LINK, now=WALK_NOW),
        preludes=(walk_paths_prelude(1),),
    )


@wp.func_native(_walk_into_snippet())
def walk_into(
    walk: WalkTrees,
    joint0: int,
    dof0: int,
    bodies: int,
    coordinates: int,
    body: int,
    wrench: wp.spatial_vector,
    coordinate: int,
    value: float,
    qd: wp.array[float],
    add: int,
    out: wp.array3d[float],
    line: int,
    slot: int,
    entry: int,
) -> wp.vec2:
    """Walk one row's seed into ``out[line, slot, entry + c]``: a wrench at local ``body``'s frame at the step's poses
    (``coordinate`` -1), or the covector ``value e_coordinate`` (``body`` its local body, ``wrench`` zero), on the tree
    of articulation-local path table ``joint0``. Stores the footprints (adds them with ``add`` 1: a second seed on one
    tree; stores none with ``add`` -1) and returns the seed's incident ``J qd`` (the step's ``J`` entries the walk
    passes, against ``qd``) and the sum of its footprints' squares."""
    ...


RELEASE_SLIPS = 16
"""The most slipping drives a tree's release holds (:class:`WalkRelease`, three floats and a coordinate a slip and path
step): a tree past it keeps the rest's ``K`` and ``constraint_capacity_status()["slip_frames"]`` flags it. It bounds the
release's memory until a data-sized arena holds it (DECISIONS 10-24); nothing else reads it."""


@wp.func
def release_path(release: WalkRelease, art: int, j: int, x: wp.array3d[float], line: int, slot: int, entry: int):
    """``x[line, slot, entry + c] <- (R_j x)_c``: slip ``j``'s release along its path, leaf first (:class:`WalkRelease`;
    the Warp twin of ``WALK_RELEASE``)."""
    s = float(0.0)
    for t in range(release.b.shape[2]):
        c = release.coordinate[art, j, t]
        if c < 0:
            break
        u = x[line, slot, entry + c] - release.b[art, j, t] * s
        s += release.a[art, j, t] * u
        x[line, slot, entry + c] = release.g[art, j, t] * u


# ---------------------------------------------------------------------------------------------------------------------
# The walk's step on the device (FL3.3): the step's geometry every step, the inward pass on the factor's refreshes,
# one launch, a lane group per tree
# ---------------------------------------------------------------------------------------------------------------------

WALK_JOIN_SLOTS = 32
"""The most joins one tree's inward pass holds at once: a parent that waits for the lifted inertia of a child other
than the joint after it (:meth:`TreeFactor.plan_walk` assigns the slots). On CUDA lane ``k`` of the tree's group
holds slot ``k`` in registers, so a group has at least as many lanes as its tree's joins in flight; the CPU's one
thread holds them all."""

WALK_ARMATURE = 14
"""Where a coordinate's record holds its effective armature ``R`` for the inward pass (a pad of
:data:`WALK_COORDINATE`, written by the step's geometry on a refresh)."""

WALK_ADDED = 15
"""Where a coordinate's record holds everything the factor adds to its diagonal, ``R + K``, for the inward pass (a pad
of :data:`WALK_COORDINATE`)."""


def walk_group(bodies: int, joins: int) -> int:
    """The lane group of a tree in :func:`walk_step_kernel` on CUDA, from the class's data: a batch of the inward pass
    builds one body a lane, so a group has no more lanes than its tree's bodies; it holds the tree's joins in flight,
    one a lane; and it stays within a warp.

    Args:
        bodies: The most bodies of one of the class's trees.
        joins: The most joins one of its trees holds at once (:meth:`TreeFactor.plan_walk`).
    """
    lanes = 1
    while lanes < min(max(bodies, joins), WALK_BATCH):
        lanes *= 2
    while lanes < joins:
        lanes *= 2
    return lanes


WALK_BATCH = 8
"""The bodies a batch of the inward pass builds at most, a lane each, before its chain consumes them."""

WALK_BLOCK = 64
"""Threads of a block of :func:`walk_step_kernel` on CUDA: whole groups, so a group never straddles a warp."""


@wp.func
def _walk_frame(
    j: int,
    parent_body: int,
    joint_type: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
) -> wp.transform:
    """Joint ``j``'s frame at the poses ``body_q`` holds: its anchor (its parent body's pose times its joint frame, the
    world's at a root, ``parent_body`` -1), at a free body's centre of mass for a free joint, with the anchor's
    rotation."""
    anchor = joint_X_p[j]
    if parent_body >= 0:
        anchor = wp.transform_multiply(body_q[parent_body], joint_X_p[j])
    kind = joint_type[j]
    if kind == JointType.FREE or kind == JointType.DISTANCE:
        child = joint_child[j]
        return wp.transform(wp.transform_point(body_q[child], body_com[child]), wp.transform_get_rotation(anchor))
    return anchor


@wp.func
def _walk_column(walk_now: wp.array2d[float], dof: int, column: wp.spatial_vector):
    """Store coordinate ``dof``'s motion column ``s`` at its body's frame at the step's poses."""
    for k in range(6):
        walk_now[dof, k] = column[k]


_INWARD = """    // The refresh's inward pass of one tree, float32, leaf first, by the tree's lane group (TreeWalk's, on the factor's
    // diagonal), after walk_step_kernel's geometry left each coordinate's column in now and each body's link and frame
    // point in now_link: on a refresh they become the factor's frames. The lane feed (FEED, leaves to root) hands the chain one body at a time: a
    // lane builds each body of a batch (its spatial inertia at its frame, from the body's pose, centre of mass, mass and
    // inertia; its link offset, its path entry, its join slots and its last coordinate's record), so the chain's state
    // (the articulated inertia h, symmetric: A 6, B 9, D 6) lives in registers. Per coordinate: Is = h s, own = s . Is,
    // 1/ell = (own + R + K)^-1/2, ut = Is / ell, P0 = own + R, h -= ut ut^T. Per body: h lifted to its parent's frame,
    // [A, A X + B; ., D - X (A X + B) + B^T X] with X the skew of d (the parent's frame less the body's), carried when
    // the parent is the next joint inward, else added into the parent's join slot (joins[parent], a lane's registers
    // on CUDA) until the chain reaches it. The outputs leave whole: each body's link row (its lane, in the batch) and
    // each coordinate's record (s, ut, 1/ell, P0, R, R + K: four float4, one lane), each written once.
    SYNC();
    const float* Lk = now_link.data;
    const float* Nw = now.data;
    float* Lo = link.data;
    float* Dt = data.data;
    const float* Ra = armature.data;
    const int* Dr = drive_row_by_dof.data;
    const float* Kr = row_K.data;
    // A coordinate's column at the step's poses and what the factor adds to its diagonal, R and R + K.
    const auto COLUMN = [&](int c, float* x) {
        for (int k = 0; k < 6; ++k) x[k] = Nw[$now * c + k];
        float added = Ra[c];
        x[6] = added;
        const int row = Dr[c];
        if (row >= 0) {
            if (Kr[row] > 0.0f) added += Kr[row];
        }
        x[7] = added;
    };
    const int* Pt = reinterpret_cast<const int*>(path.data);
    const int* Jt = joins.data;
    const int* Ct = joint_child.data;
    const float* Qb = reinterpret_cast<const float*>(body_q.data);
    const float* Cb = reinterpret_cast<const float*>(body_com.data);
    const float* Mb = body_mass.data;
    const float* Ib = reinterpret_cast<const float*>(body_inertia.data);
#if defined(__CUDA_ARCH__)
    const auto PREFETCH = [](const void* p) { asm volatile("prefetch.global.L1 [%0];" ::"l"(p)); };
    constexpr int HELD = 1;
#else
    const auto PREFETCH = [](const void* p) {};
    constexpr int HELD = $slots;
#endif
    for (int c = dof0 + LANE; c < dof1; c += LANES) {
        PREFETCH(Nw + $now * c);
        PREFETCH(Ra + c);
        PREFETCH(Dr + c);
    }
    // The join slots: lane k holds slot k on CUDA (HELD 1), the CPU's one thread every slot.
    float held[HELD][21];
    for (int s = 0; s < HELD; ++s)
        for (int k = 0; k < 21; ++k) held[s][k] = 0.0f;
    const auto OWNS = [&](int slot) -> bool { return HELD > 1 || LANE == slot; };
    const auto AT = [&](int slot) -> int { return HELD > 1 ? slot : 0; };
    // h: A (00 01 02 11 12 22), B (row-major 3x3), D (00 01 02 11 12 22).
    float h[21];
    for (int k = 0; k < 21; ++k) h[k] = 0.0f;
    bool carrying = false;
    const auto SYM = [](int a, int b) -> int { return a <= b ? (a == 0 ? b : (a == 1 ? 2 + b : 5)) : (b == 0 ? a : (b == 1 ? 2 + a : 5)); };
    // A body as the feed hands it to the chain: m, m c (c its centre of mass less its frame), R I R^T - m X X (6), its
    // link offset d (3), its last coordinate's record (s 6, R, R + K), and as bits its parent, coordinates [lo, hi),
    // its own join slot and its parent's.
    struct Body { float w[26]; };
    enum { M = 0, MC = 1, ROT = 4, OFFSET = 10, RECORD = 13, UP = 21, LO = 22, HI = 23, SLOT = 24, INTO = 25 };
    FEED(j0, j1 - 1, -1,
        [&](int j) {
            const int b = Ct[j];
            PREFETCH(Qb + 7 * b);
            PREFETCH(Qb + 7 * b + 6);
            PREFETCH(Cb + 3 * b);
            PREFETCH(Cb + 3 * b + 2);
            PREFETCH(Mb + b);
            PREFETCH(Ib + 9 * b);
            PREFETCH(Ib + 9 * b + 8);
            PREFETCH(Lk + $link * j);
            PREFETCH(Pt + 2 * j);
            PREFETCH(Jt + j);
        },
        [&](int mine, bool valid) -> Body {
            Body r;
            for (int k = 0; k < 26; ++k) r.w[k] = 0.0f;
            int up = -1, lo = 0, hi = 0, slot = -1, into = -1;
            if (valid) {
                up = Pt[2 * mine];
                lo = dof0 + Pt[2 * mine + 1];
                hi = mine + 1 < j1 ? dof0 + Pt[2 * (mine + 1) + 1] : dof1;
                slot = Jt[mine];
                into = up >= 0 ? Jt[j0 + up] : -1;
                const float* Lr = Lk + $link * mine;
                for (int k = 0; k < 3; ++k) r.w[OFFSET + k] = Lr[k];
                // The body's link row becomes the factor's.
#if defined(__CUDA_ARCH__)
                reinterpret_cast<float4*>(Lo + $link * mine)[0] = reinterpret_cast<const float4*>(Lr)[0];
                reinterpret_cast<float4*>(Lo + $link * mine)[1] = reinterpret_cast<const float4*>(Lr)[1];
#else
                for (int k = 0; k < $link; ++k) Lo[$link * mine + k] = Lr[k];
#endif
                const int b = Ct[mine];
                const float* q = Qb + 7 * b;
                const float* com = Cb + 3 * b;
                const float* I = Ib + 9 * b;
                const float m = Mb[b];
                // The body's rotation and its centre of mass less its frame, c.
                const float x = q[3], y = q[4], z = q[5], w = q[6];
                const float R[3][3] = {
                    {1.0f - 2.0f * (y * y + z * z), 2.0f * (x * y - z * w), 2.0f * (x * z + y * w)},
                    {2.0f * (x * y + z * w), 1.0f - 2.0f * (x * x + z * z), 2.0f * (y * z - x * w)},
                    {2.0f * (x * z - y * w), 2.0f * (y * z + x * w), 1.0f - 2.0f * (x * x + y * y)}};
                float c[3], mc[3];
                for (int a = 0; a < 3; ++a)
                    c[a] = (q[a] + (R[a][0] * com[0] + R[a][1] * com[1] + R[a][2] * com[2])) - Lr[$frame + a];
                for (int a = 0; a < 3; ++a) mc[a] = m * c[a];
                // R I R^T - m X X, X the skew of c: -m X X = m (|c|^2 1 - c c^T).
                float RI[3][3];
                for (int a = 0; a < 3; ++a)
                    for (int e = 0; e < 3; ++e) RI[a][e] = R[a][0] * I[e] + R[a][1] * I[3 + e] + R[a][2] * I[6 + e];
                const float cc = c[0] * c[0] + c[1] * c[1] + c[2] * c[2];
                r.w[M] = m;
                for (int a = 0; a < 3; ++a) r.w[MC + a] = mc[a];
                for (int a = 0; a < 3; ++a)
                    for (int e = a; e < 3; ++e)
                        r.w[ROT + SYM(a, e)] = (RI[a][0] * R[e][0] + RI[a][1] * R[e][1] + RI[a][2] * R[e][2]) + mc[a] * -c[e] + (a == e ? m * cc : 0.0f);
                if (hi > lo) COLUMN(hi - 1, r.w + RECORD);
            }
            r.w[UP] = __int_as_float(up);
            r.w[LO] = __int_as_float(lo);
            r.w[HI] = __int_as_float(hi);
            r.w[SLOT] = __int_as_float(slot);
            r.w[INTO] = __int_as_float(into);
            return r;
        },
        [&](int b, const Body& r) {
            // The body's inertia at its frame: [m 1, -m X; m X, rot], X the skew of c.
            const float bm = r.w[M];
            const float* bc = r.w + MC;
            const float* br = r.w + ROT;
            const float* bd = r.w + OFFSET;
            float s[8];
            for (int k = 0; k < 8; ++k) s[k] = r.w[RECORD + k];
            const int b_up = __float_as_int(r.w[UP]), b_lo = __float_as_int(r.w[LO]), b_hi = __float_as_int(r.w[HI]);
            const int b_slot = __float_as_int(r.w[SLOT]), b_into = __float_as_int(r.w[INTO]);
            const float body[21] = {bm, 0.0f, 0.0f, bm, 0.0f, bm,
                                    0.0f, bc[2], -bc[1], -bc[2], 0.0f, bc[0], bc[1], -bc[0], 0.0f,
                                    br[0], br[1], br[2], br[3], br[4], br[5]};
            for (int k = 0; k < 21; ++k) h[k] = carrying ? body[k] + h[k] : body[k];
            if (b_slot >= 0) {
                for (int k = 0; k < 21; ++k) h[k] += READ(held[AT(b_slot)][k], b_slot);
                if (OWNS(b_slot))
                    for (int k = 0; k < 21; ++k) held[AT(b_slot)][k] = 0.0f;
            }
            for (int dof = b_hi - 1; dof >= b_lo; --dof) {
                // The next coordinate's record, loaded before this one stores (a body with several coordinates).
                float next[8];
                if (dof > b_lo) COLUMN(dof - 1, next);
                const float* A = h;
                const float* B = h + 6;
                const float* D = h + 15;
                float Is[6];
                for (int a = 0; a < 3; ++a) {
                    Is[a] = (A[SYM(a, 0)] * s[0] + A[SYM(a, 1)] * s[1] + A[SYM(a, 2)] * s[2])
                            + (B[3 * a] * s[3] + B[3 * a + 1] * s[4] + B[3 * a + 2] * s[5]);
                    Is[3 + a] = (B[a] * s[0] + B[3 + a] * s[1] + B[6 + a] * s[2])
                                + (D[SYM(a, 0)] * s[3] + D[SYM(a, 1)] * s[4] + D[SYM(a, 2)] * s[5]);
                }
                const float own = (Is[0] * s[0] + Is[1] * s[1] + Is[2] * s[2]) + (Is[3] * s[3] + Is[4] * s[4] + Is[5] * s[5]);
#if defined(__CUDA_ARCH__)
                const float inv = rsqrtf(own + s[7]);
#else
                const float inv = 1.0f / sqrtf(own + s[7]);
#endif
                float ut[6];
                for (int k = 0; k < 6; ++k) ut[k] = Is[k] * inv;
                if (LANE == 0) {
                    float* o = Dt + $coordinate * dof;
#if defined(__CUDA_ARCH__)
                    float4* o4 = reinterpret_cast<float4*>(o);
                    o4[0] = make_float4(s[0], s[1], s[2], s[3]);
                    o4[1] = make_float4(s[4], s[5], ut[0], ut[1]);
                    o4[2] = make_float4(ut[2], ut[3], ut[4], ut[5]);
                    o4[3] = make_float4(inv, own + s[6], s[6], s[7]);
#else
                    for (int k = 0; k < 6; ++k) o[k] = s[k];
                    for (int k = 0; k < 6; ++k) o[6 + k] = ut[k];
                    o[12] = inv;
                    o[$pivot] = own + s[6];
                    o[$armature] = s[6];
                    o[$added] = s[7];
#endif
                }
                // h -= ut ut^T
                for (int a = 0; a < 3; ++a)
                    for (int e = a; e < 3; ++e) {
                        h[SYM(a, e)] -= ut[a] * ut[e];
                        h[15 + SYM(a, e)] -= ut[3 + a] * ut[3 + e];
                    }
                for (int a = 0; a < 3; ++a)
                    for (int e = 0; e < 3; ++e) h[6 + 3 * a + e] -= ut[a] * ut[3 + e];
                if (dof > b_lo)
                    for (int k = 0; k < 8; ++k) s[k] = next[k];
            }
            carrying = false;
            if (b_up >= 0) {
                // The lift: B' = B + A X, D' = D - X B' + B^T X (its symmetric part).
                const float X[3][3] = {{0.0f, -bd[2], bd[1]}, {bd[2], 0.0f, -bd[0]}, {-bd[1], bd[0], 0.0f}};
                float Bn[9];
                for (int a = 0; a < 3; ++a)
                    for (int e = 0; e < 3; ++e)
                        Bn[3 * a + e] = h[6 + 3 * a + e] + (h[SYM(a, 0)] * X[0][e] + h[SYM(a, 1)] * X[1][e] + h[SYM(a, 2)] * X[2][e]);
                for (int a = 0; a < 3; ++a)
                    for (int e = a; e < 3; ++e) {
                        const float xb = X[a][0] * Bn[e] + X[a][1] * Bn[3 + e] + X[a][2] * Bn[6 + e];
                        const float bx = h[6 + a] * X[0][e] + h[9 + a] * X[1][e] + h[12 + a] * X[2][e];
                        h[15 + SYM(a, e)] = (h[15 + SYM(a, e)] - xb) + bx;
                    }
                for (int k = 0; k < 9; ++k) h[6 + k] = Bn[k];
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

    return dual_native(
        substitute(
            _INWARD,
            coordinate=WALK_COORDINATE,
            link=WALK_LINK,
            now=WALK_NOW,
            frame=WALK_FRAME,
            pivot=WALK_PIVOT,
            armature=WALK_ARMATURE,
            added=WALK_ADDED,
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
    now: wp.array2d[float],
    now_link: wp.array2d[float],
    armature: wp.array[float],
    drive_row_by_dof: wp.array[int],
    row_K: wp.array[float],
    joint_child: wp.array[int],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    link: wp.array2d[float],
    data: wp.array2d[float],
):
    """The refresh's inward pass of the tree of joints ``[j0, j1)`` and coordinates ``[dof0, dof1)`` by its group of
    ``lanes`` (one thread on the CPU), after :func:`walk_step_kernel`'s geometry (``now``, ``now_link``): the factor's
    frames (each body's ``link``) and each coordinate's whole record in ``data`` (``s``, ``ut``, ``1 / ell``, ``P0``,
    ``R``, ``R + K``), with each body's spatial inertia built in the pass from its pose and mass properties."""
    ...


@functools.cache
def _walk_sync_snippet() -> str:
    from .device_preludes import dual_native, lane_group  # noqa: PLC0415

    return dual_native("    SYNC();\n", preludes=(lane_group("lanes"),))


@wp.func_native(_walk_sync_snippet())
def walk_group_sync(lanes: int):
    """Synchronize this thread's group of ``lanes`` (nothing on the CPU, where one thread runs it): what a lane stored
    is visible to the group after it."""
    ...


@wp.kernel(enable_backward=False)
def walk_step_kernel(
    arts: wp.array[int],
    lanes: int,
    mass_update_mask: wp.array[int],
    articulation_start: wp.array[int],
    articulation_joint_end: wp.array[int],
    walk_path: wp.array[wp.vec2i],
    walk_parent_body: wp.array[int],
    walk_joins: wp.array[int],
    joint_type: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_child: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_q: wp.array[float],
    body_q: wp.array[wp.transform],
    body_mass: wp.array[float],
    body_com: wp.array[wp.vec3],
    body_inertia: wp.array[wp.mat33],
    armature: wp.array[float],
    drive_row_by_dof: wp.array[int],
    row_K: wp.array[float],
    # outputs
    walk_now: wp.array2d[float],
    walk_now_link: wp.array2d[float],
    walk_data: wp.array2d[float],
    walk_link: wp.array2d[float],
):
    """The walk's one launch a step, a group of ``lanes`` threads per held tree (:meth:`TreeFactor.walk_step`),
    float32, at the poses ``body_q`` holds.

    The step's geometry, the group striding over the tree's joints: each body's frame (its joint's anchor, a free
    body's centre of mass), its coordinates' columns there in world axes, as :func:`local_frames` takes them (a free
    joint turns about the articulation's origin, its first body's centre of mass; a D6 joint's turns about the pivot
    its slides moved), and its link (:data:`WALK_LINK`'s layout: its parent's frame less its own, its frame's world
    point). The rows' seeds enter the walk here. On a tree the refresh just factored (``mass_update_mask``), the same geometry becomes the factor's frames, with each coordinate's ``R`` and ``R + K``
    (the diagonal the refresh's ``L`` holds, :meth:`TreeFactor.diagonal`), and the group runs the inward pass
    (:func:`walk_inward`), which builds each body's spatial inertia itself, so the walk and ``L`` are one factor."""
    tree, lane = wp.tid()
    art = arts[tree]
    j0 = articulation_start[art]
    j1 = articulation_joint_end[art]
    refresh = mass_update_mask[art] != 0
    for j in range(j0 + lane, j1, lanes):
        X = _walk_frame(j, walk_parent_body[j], joint_type, joint_child, joint_X_p, body_q, body_com)
        R_a = wp.transform_get_rotation(X)
        frame = wp.transform_get_translation(X)
        kind = joint_type[j]
        free = kind == JointType.FREE or kind == JointType.DISTANCE
        q0 = joint_q_start[j]
        d0 = joint_qd_start[j]
        if kind == JointType.REVOLUTE:
            column = wp.spatial_vector(wp.vec3(0.0), wp.quat_rotate(R_a, joint_axis[d0]))
            _walk_column(walk_now, d0, column)
        elif kind == JointType.PRISMATIC:
            column = wp.spatial_vector(wp.quat_rotate(R_a, joint_axis[d0]), wp.vec3(0.0))
            _walk_column(walk_now, d0, column)
        elif kind == JointType.BALL:
            for k in range(3):
                e = wp.vec3(0.0)
                e[k] = 1.0
                column = wp.spatial_vector(wp.vec3(0.0), wp.quat_rotate(R_a, e))
                _walk_column(walk_now, d0 + k, column)
        elif free:
            first = joint_child[j0]
            origin = wp.transform_point(body_q[first], body_com[first])
            for k in range(3):
                e = wp.vec3(0.0)
                e[k] = 1.0
                w = wp.quat_rotate(R_a, e)
                column = wp.spatial_vector(w, wp.vec3(0.0))
                _walk_column(walk_now, d0 + k, column)
                # The factor turns a free joint about the articulation's origin: the same twist at the body's frame.
                column = wp.spatial_vector(wp.cross(w, frame - origin), w)
                _walk_column(walk_now, d0 + 3 + k, column)
        elif kind == JointType.D6:
            lin = joint_dof_dim[j, 0]
            ang = joint_dof_dim[j, 1]
            slide = wp.vec3(0.0)
            for k in range(lin):
                column = wp.spatial_vector(wp.quat_rotate(R_a, joint_axis[d0 + k]), wp.vec3(0.0))
                _walk_column(walk_now, d0 + k, column)
                slide += joint_axis[d0 + k] * joint_q[q0 + k]
            turned = wp.quat_identity()
            for k in range(ang):
                axis = wp.quat_rotate(turned, joint_axis[d0 + lin + k])
                w = wp.quat_rotate(R_a, axis)
                # A turn about the pivot the slides moved, at the anchor: v = (R_a slide) x w.
                column = wp.spatial_vector(wp.cross(wp.quat_rotate(R_a, slide), w), w)
                _walk_column(walk_now, d0 + lin + k, column)
                turned = wp.quat_from_axis_angle(axis, joint_q[q0 + lin + k]) * turned
        for k in range(3):
            walk_now_link[j, 4 + k] = frame[k]
    # Each body's link, its parent's frame less its own, from the frame points the group just stored.
    walk_group_sync(lanes)
    for j in range(j0 + lane, j1, lanes):
        parent = walk_path[j][0]
        offset = wp.vec3(0.0)
        if parent >= 0:
            jp = j0 + parent
            offset = wp.vec3(
                walk_now_link[jp, 4] - walk_now_link[j, 4],
                walk_now_link[jp, 5] - walk_now_link[j, 5],
                walk_now_link[jp, 6] - walk_now_link[j, 6],
            )
        for k in range(3):
            walk_now_link[j, k] = offset[k]
    if refresh:
        walk_inward(
            lanes,
            j0,
            j1,
            joint_qd_start[j0],
            joint_qd_start[j1],
            walk_path,
            walk_joins,
            walk_now,
            walk_now_link,
            armature,
            drive_row_by_dof,
            row_K,
            joint_child,
            body_q,
            body_com,
            body_mass,
            body_inertia,
            walk_link,
            walk_data,
        )
