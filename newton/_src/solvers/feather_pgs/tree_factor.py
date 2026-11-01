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
:meth:`TreeFactor.leave` (``v = v_hat + P^T L^-T du``, where the sweep stores ``du``) and :meth:`TreeFactor.finish`
(``qdd``, integrate, publish).
"""

from __future__ import annotations

import functools
import warnings

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
    "WALK_COORDINATE",
    "WALK_FRAME",
    "WALK_LINK",
    "WALK_NOW",
    "WALK_PIVOT",
    "TreeFactor",
    "prepare_augmented_drives",
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
        self.T_by_size: dict[int, wp.array] = {}
        self.R_by_size: dict[int, wp.array] = {}
        self.diagonal_inverse_mass = None
        self.diagonal_inverse_trial = None
        self.driven_articulations = np.zeros(0, dtype=bool)
        self.body_factors = None
        self.eliminates = False
        self.elimination = None
        self.block_count = 0
        self.held = None
        self.dynamics_stream = None
        self.walk_arts = None

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
        """The bilateral blocks' tables (:func:`.response_builds_rows.elimination`).

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
        bounds, order = host_index.group_by(block, heads.size)
        capacity = int(np.bincount(block, weights=rows).max())
        self.block_count = n = int(heads.size)
        self.eliminates = True
        self.elimination = response_builds_rows.elimination(capacity)
        art_to_block = np.full(arts, -1, dtype=np.int32)
        held = np.isin(component, heads)
        art_to_block[held] = np.searchsorted(heads, component[held])
        self._preelim_art_to_idx = wp.array(art_to_block, dtype=wp.int32, device=model.device)
        self._arrays.allocate(PREELIMINATION, preelim=n, preelim_row=n * capacity)
        self._arrays.upload(
            PREELIMINATION_PLAN,
            _preelim_world=forest.articulation_world[heads],
            _preelim_start=bounds,
            _preelim_entry=entries[order],
        )
        self._no_slots = self._arrays.placeholder(wp.int32)

    def eliminate(self, *, first, Z, sides, counts, owners, cfm, rhs, incident, diag, impulses) -> None:
        """Make each block's rows orthonormal, then correct every other row, its right-hand side and its diagonal, and
        give the block's rows the projection's impulses, which the sweep seeds
        (:func:`.response_builds_rows.elimination`). Runs once the right-hand sides are final (bias and restitution)."""
        block = field_group.bind(Preelim, factor=self)
        wp.launch(
            self.elimination.setup,
            dim=self.block_count,
            inputs=[
                first,
                block,
                self._no_slots if self.laws.mimic_slot is None else self.laws.mimic_slot,
                self._no_slots if self.laws.link_slot is None else self.laws.link_slot,
                owners,
                cfm,
            ],
            outputs=[Z, rhs, incident, diag],
            device=self.model.device,
        )
        wp.launch(
            self.elimination.correct,
            dim=sides.shape[0] * sides.shape[1],
            inputs=[first, block, sides, counts, owners, cfm],
            outputs=[Z, rhs, diag],
            device=self.model.device,
        )
        wp.launch(
            self.elimination.weights,
            dim=self.block_count,
            inputs=[first, block, rhs],
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
        self.driven_articulations = np.zeros(model.articulation_count, dtype=bool)
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
        self.driven_articulations = row_counts > 0
        self.parallel_drives = True

    def _driven(self, articulations) -> bool:
        """Whether any of ``articulations`` holds a drive row, so its factor needs a trial factor beside it."""
        articulations = np.asarray(articulations, dtype=np.int64)
        return bool(self.driven_articulations.size and np.any(self.driven_articulations[articulations]))

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
        driven_diagonal = any(self._driven(forest.group_to_art_host[size]) for size in forest.diagonal_mass_sizes)
        self.diagonal_inverse_trial = (
            wp.zeros_like(model.joint_qd) if driven_diagonal else arrays.placeholder(wp.float32)
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
        trial = articulations if self.driven_articulations.any() else 1
        self._arrays.allocate(SPARSE_FACTOR, group=articulations, trial_group=trial, nonzero=plan.nonzero_count)

    def allocate_size_groups(self, placeholder_sizes) -> None:
        """Allocate each size group's factor L and armature R; ``placeholder_sizes`` (the compact diagonal and the
        native sparse factor) keep (1, 1, 1) placeholders of L."""
        for size in self.forest.size_groups:
            n_arts = self.forest.n_arts_by_size[size]
            placeholder = size in placeholder_sizes
            factor = 1 if placeholder else 0
            trial = factor or (0 if self._driven(self.forest.group_to_art_host[size]) else 1)
            sizes = {
                "art": n_arts,
                "size": size,
                "factor_art": factor or n_arts,
                "factor_size": factor or size,
                "trial_art": trial or n_arts,
                "trial_size": trial or size,
            }
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
            self.walk_geometry(state_in)
            self.refresh_walk(state_in)
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
        drives' saturation decision and, where it moved a drive, the tree's new step factor and solve
        (:data:`.tree_kinetic_state.SATURATE_DRIVES`); the trial factor keeps every ``K`` while the step factor holds a
        decided set, so no saturation ever asks the refresh for a factor."""
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

    def leave(self, velocity: wp.array, *, size: int, indices, Linv: wp.array, delta: wp.array, base: wp.array) -> None:
        """``v = base + P^T L^-T du`` through the native sparse factor (:func:`.response_builds_rows.factor_publish_kernel`).

        A sweep that keeps ``du`` in shared memory leaves in its kernel, with the same text: this is the publication of
        the one that stores it.
        """
        forest = self.forest
        wp.launch(
            response_builds_rows.factor_publish_kernel,
            dim=(forest.n_arts_by_size[size], size),
            inputs=[
                forest.group_to_art[size],
                forest.art_to_world,
                forest.articulation_world_dof_offset,
                forest.articulation_dof_start,
                indices.permutation,
                indices.lookup,
                Linv,
                delta,
                base,
            ],
            outputs=[velocity],
            device=self.model.device,
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
        :data:`WALK_COORDINATE` floats, each body's :data:`WALK_LINK` floats, inertia scratch and twists, and each
        body's path entry (its parent and first coordinate within its tree, :func:`walk_paths_prelude`). Its refresh
        runs inside the factor's (:meth:`refresh_walk`), so the walk is the factor ``L`` refreshes, to rounding. The
        step's geometry (:meth:`walk_geometry`, :data:`WALK_NOW` floats a coordinate and :data:`WALK_LINK` a body) is
        where the rows' seeds enter, every step.

        Raises:
            ValueError: For a tree whose bodies do not come after their parents.
        """
        model, forest = self.model, self.forest
        device = model.device
        arts = np.asarray(articulations, dtype=np.int32)
        start = model.articulation_start.numpy()
        end = forest.articulation_joint_end_host
        qd_start = model.joint_qd_start.numpy()
        joint_parent, joint_child = model.joint_parent.numpy(), model.joint_child.numpy()
        joint_of = np.full(max(int(model.body_count), 1), -1, dtype=np.int64)
        joint_of[joint_child[joint_child >= 0]] = np.flatnonzero(joint_child >= 0)
        path = np.zeros((max(int(model.joint_count), 1), 2), dtype=np.int32)
        join = np.zeros(max(int(model.joint_count), 1), dtype=np.int32)
        joint_art = np.full(max(int(model.joint_count), 1), -1, dtype=np.int32)
        dof_body = np.full(max(int(model.joint_dof_count), 1), -1, dtype=np.int32)
        for art in arts.tolist():
            j0, j1 = int(start[art]), int(end[art])
            joint_art[j0:j1] = art
            for j in range(j0, j1):
                up = int(joint_of[joint_parent[j]]) if joint_parent[j] >= 0 else -1
                parent = up - j0 if j0 <= up < j1 else -1
                if parent >= j - j0:
                    raise ValueError(f"The walk needs every parent before its children (articulation {art})")
                path[j] = (parent, int(qd_start[j] - qd_start[j0]))
                dof_body[qd_start[j] : qd_start[j + 1]] = j - j0
                if parent >= 0 and parent != j - j0 - 1:
                    join[j0 + parent] = 1
        joints, dofs = max(int(model.joint_count), 1), max(int(model.joint_dof_count), 1)
        self.walk_arts = wp.array(arts, dtype=wp.int32, device=device)
        free_host = forest.is_free_rigid_host if forest.is_free_rigid is not None else np.zeros(0, dtype=np.int32)
        trees = arts[~np.isin(arts, np.flatnonzero(np.asarray(free_host) != 0))] if len(free_host) else arts
        # The trees the twists walk: a free rigid body's incident is its six-coordinate J against v_hat.
        self.walk_trees_arts = wp.array(trees if trees.size else np.zeros(1, np.int32), dtype=wp.int32, device=device)
        self.walk_tree_count = int(trees.size)
        self.walk_path = wp.array(path, dtype=wp.vec2i, device=device)
        self.walk_join = wp.array(join, dtype=wp.int32, device=device)
        self.walk_data = wp.zeros((dofs, WALK_COORDINATE), dtype=wp.float32, device=device)
        self.walk_link = wp.zeros((joints, WALK_LINK), dtype=wp.float32, device=device)
        self.walk_inertia = wp.zeros(joints, dtype=wp.spatial_matrix, device=device)
        self.walk_twist = wp.zeros(joints, dtype=wp.spatial_vector, device=device)
        self.walk_leave_twist = wp.zeros(joints, dtype=wp.spatial_vector, device=device)
        self.walk_armature = wp.zeros(dofs, dtype=wp.float32, device=device)
        if self.armature is not None and self.armature.size:
            self.walk_armature.assign(self.armature.astype(np.float32))
        # The step's geometry (walk_geometry_kernel, every step): a seed enters as J at the step's poses.
        walked = np.flatnonzero(joint_art >= 0).astype(np.int32)
        self.walk_joints = wp.array(walked if walked.size else np.zeros(1, np.int32), dtype=wp.int32, device=device)
        self.walk_joint_count = int(walked.size)
        self.walk_joint_art = wp.array(joint_art, dtype=wp.int32, device=device)
        self.walk_now = wp.zeros((dofs, WALK_NOW), dtype=wp.float32, device=device)
        self.walk_now_link = wp.zeros((joints, WALK_LINK), dtype=wp.float32, device=device)
        view = WalkTrees()
        view.data, view.link, view.path = self.walk_data, self.walk_link, self.walk_path
        view.now, view.now_link = self.walk_now, self.walk_now_link
        view.twist, view.scratch = self.walk_twist, self.walk_leave_twist
        view.joint_start = model.articulation_start
        view.joint_end = forest.articulation_joint_end
        view.dof_start = forest.articulation_dof_start
        view.dofs = forest.articulation_dof_count
        view.dof_body = wp.array(dof_body, dtype=wp.int32, device=device)
        body_joint = np.full(max(int(model.body_count), 1), -1, dtype=np.int32)
        body_joint[: joint_of.size] = joint_of[: body_joint.size]
        view.body_joint = wp.array(body_joint, dtype=wp.int32, device=device)
        free = forest.is_free_rigid_host if forest.is_free_rigid is not None else np.zeros(1, dtype=np.int32)
        view.free = wp.array(np.asarray(free, dtype=np.int32), dtype=wp.int32, device=device)
        view.group = forest.art_group_idx
        six = self.L_by_size.get(6)
        view.L6 = six if six is not None and six.shape[1] == 6 else wp.zeros((1, 6, 6), dtype=wp.float32, device=device)
        # The slips' frame (walk_frame_kernel): as many slips per tree as it has limited, driven coordinates (from the
        # model's gains and effort limits, data-sized), at most FRAME_SLIPS; none without a limited drive.
        slips = 0
        if self.parallel_drives and model.joint_dof_count:
            ke, kd = model.joint_target_ke.numpy(), model.joint_target_kd.numpy()
            damping, limit = self.passive_joint_damping.numpy(), model.joint_effort_limit.numpy()
            driven = self.drive_row_by_dof.numpy() >= 0
            limited = driven & ((ke > 0) | (kd > 0) | (damping > 0)) & (limit > 0) & np.isfinite(limit)
            starts, counts = forest.articulation_dof_start_host, forest.articulation_dof_count_host
            for art in arts.tolist():
                slips = max(slips, int(np.count_nonzero(limited[starts[art] : starts[art] + counts[art]])))
        self.walk_slip_capacity = min(slips, FRAME_SLIPS)
        cap = max(self.walk_slip_capacity, 1)
        arts_total = max(int(model.articulation_count), 1)
        width = max(int(forest.articulation_max_dofs), 1) if self.walk_slip_capacity else 1
        view.slips = wp.zeros(arts_total, dtype=wp.int32, device=device)
        view.basis = wp.zeros(
            (arts_total if self.walk_slip_capacity else 1, cap, width), dtype=wp.float32, device=device
        )
        view.scale = wp.zeros((arts_total if self.walk_slip_capacity else 1, cap, cap), dtype=wp.float32, device=device)
        self.walk_frame_work = wp.zeros((view.scale.shape[0], 2 * cap, cap), dtype=wp.float32, device=device)
        self.walk_slip_rows = wp.zeros((view.scale.shape[0], cap), dtype=wp.int32, device=device)
        self.walk_frame_status = wp.zeros(1, dtype=wp.int32, device=device)
        self.walk_trees = view

    def walk_frames(self) -> None:
        """The slips' frame of every tree whose predictor decided a set this step (:func:`walk_frame_kernel`): the block
        primitive's first phase, before the one writer corrects its rows and the leave reads it."""
        if self.walk_arts is None or not self.walk_slip_capacity:
            return
        view = self.walk_trees
        wp.launch(
            walk_frame_kernel,
            dim=self.walk_arts.shape[0],
            inputs=[
                self.walk_arts,
                view,
                self.aug_decided,
                self.aug_row_counts,
                self.aug_row_dof_index,
                self.aug_row_K,
                self.aug_row_Kd,
                self.forest.articulation_max_dofs,
            ],
            outputs=[
                view.slips,
                self.walk_slip_rows,
                view.basis,
                view.scale,
                self.walk_frame_work,
                self.walk_frame_status,
            ],
            device=self.model.device,
        )

    def walk_view(self) -> WalkTrees:
        """The walk's view (:class:`WalkTrees`) the one writer, the leave and the audit read."""
        return self.walk_trees

    def walk_geometry(self, state) -> None:
        """The step's geometry of every held tree (:func:`walk_geometry_kernel`, a thread per joint, every step): each
        coordinate's column at its body's frame and each body's frame point at the poses ``state.body_q`` holds, where
        the rows' seeds enter the walk as ``J``."""
        if self.walk_arts is None or not self.walk_joint_count:
            return
        model = self.model
        wp.launch(
            walk_geometry_kernel,
            dim=self.walk_joint_count,
            inputs=[
                self.walk_joints,
                self.walk_joint_art,
                model.articulation_start,
                self.walk_path,
                model.joint_type,
                model.joint_dof_dim,
                model.joint_child,
                model.joint_q_start,
                model.joint_qd_start,
                model.joint_X_p,
                model.joint_axis,
                state.joint_q,
                state.body_q,
                model.body_com,
            ],
            outputs=[self.walk_now, self.walk_now_link],
            device=model.device,
        )

    def twists(self, v_hat: wp.array) -> None:
        """Every held tree's predicted body twists at its frames at the step's poses (:meth:`walk_geometry`), from
        ``v_hat`` (:func:`walk_twists_kernel`)."""
        if self.walk_arts is None or not self.walk_tree_count:
            return
        wp.launch(
            walk_twists_kernel(),
            dim=self.walk_tree_count,
            inputs=[self.walk_trees_arts, self.walk_trees, v_hat],
            device=self.model.device,
        )

    def refresh_walk(self, state) -> None:
        """The device inward pass (:func:`walk_refresh_kernel`) of every held tree the refresh just factored
        (``mass_update_mask``), at the poses its forward kinematics published in ``state.body_q``: the step's geometry
        (:meth:`walk_geometry`, launched before it at the same poses) becomes the factor's frames."""
        if self.walk_arts is None:
            return
        model = self.model
        wp.launch(
            walk_refresh_kernel,
            dim=self.walk_arts.shape[0],
            inputs=[
                self.walk_arts,
                self.mass_update_mask,
                model.articulation_start,
                self.forest.articulation_joint_end,
                self.walk_path,
                model.joint_child,
                model.joint_qd_start,
                state.body_q,
                model.body_mass,
                model.body_com,
                model.body_inertia,
                self.walk_armature,
                self.drive_row_by_dof,
                self.aug_row_K if self.parallel_drives else self._arrays.placeholder(wp.float32),
                self.walk_trees.free,
                self.walk_join,
                self.walk_now,
                self.walk_now_link,
            ],
            outputs=[self.walk_data, self.walk_link, self.walk_inertia],
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

WALK_COORDINATE = 16
"""Floats per coordinate of the device walk's data: ``s`` and ``ut`` (``(v, w)``, six each), ``1 / ell``, the pivot
without any drive's ``K`` (``P0 = s^T I s + R``, the split pivot's, SAT_PHYS §1.2) and two pads, so a coordinate's record
is four aligned ``float4``."""

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
    // TreeWalk.device_data and TreeFactor.refresh_walk lay them out.
    //
    // WALK_UP(data, link, now, now_link, body, top, p, w, sink): the states p of $rows rows, p[6 r + k] the wrench
    // (f, t) of row r at body's frame, walked up to the root from coordinate top of body (its last for a wrench seed):
    // each coordinate c takes the footprint z = (s_c . p) / ell_c, given to sink(r, c, z), and passes p - ut_c z; a
    // joint passes t - d x f to its parent. now (null for rows seeded by covectors alone): the step's geometry, each
    // coordinate's column at its body's frame at the step's poses (WALK_NOW floats) and each body's link there
    // (now_link, link's layout), with w[6 r + k] row r's seed wrench at the current body's frame there. A seed enters
    // as the step's J and the refresh's factor whitens it (a held factor between refreshes, L^-1 P J^T as the dense rows
    // do): p is the seed's wrench less what the factor has taken, so the footprint takes (s_now_c - s_c) . w and a joint
    // passes (d - d_now) x f_w more. On a refresh step the two geometries are one and both terms are zero.
    const auto WALK_UP = [&](const float* data, const float* link, const float* now, const float* now_link, int body,
                             int top, float* p, float* w, auto&& sink) {
        for (int b = body; b >= 0; b = WALK_PARENT_OF(b)) {
            for (int c = b == body ? top : WALK_FIRST_OF(b + 1) - 1; c >= WALK_FIRST_OF(b); --c) {
                const float* d = data + $coordinate * c;
                const float* e = now != nullptr ? now + $now * c : nullptr;
                for (int r = 0; r < $rows; ++r) {
                    float* q = p + 6 * r;
                    float x = (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]) + (d[3] * q[3] + d[4] * q[4] + d[5] * q[5]);
                    if (e != nullptr) {
                        const float* v = w + 6 * r;
                        x += ((e[0] - d[0]) * v[0] + (e[1] - d[1]) * v[1] + (e[2] - d[2]) * v[2])
                             + ((e[3] - d[3]) * v[3] + (e[4] - d[4]) * v[4] + (e[5] - d[5]) * v[5]);
                    }
                    const float z = x * d[12];
                    for (int k = 0; k < 6; ++k) q[k] -= d[6 + k] * z;
                    sink(r, c, z);
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
    // velocity limit's), on the states p: z = (x + s_c . p) / ell_c. WALK_UP continues from c - 1 of c's body.
    const auto WALK_SEED = [&](const float* data, int c, const float* x, float* p, auto&& sink) {
        const float* d = data + $coordinate * c;
        for (int r = 0; r < $rows; ++r) {
            float* q = p + 6 * r;
            const float z = (x[r] + ((d[0] * q[0] + d[1] * q[1] + d[2] * q[2]) + (d[3] * q[3] + d[4] * q[4] + d[5] * q[5])))
                            * d[12];
            for (int k = 0; k < 6; ++k) q[k] -= d[6 + k] * z;
            sink(r, c, z);
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
    // WALK_DOWN(data, link, u, qd, twist): the leave P^T L^-T u of the class's trees, root first: each coordinate c
    // takes qd_c = (u_c - ut_c . V) / ell_c, then V += s_c qd_c, V its body's twist; twist receives every body's
    // twist (v, w) at its frame, six floats a body.
    const auto WALK_DOWN = [&](const float* data, const float* link, const float* u, float* qd, float* twist) {
        for (int b = 0; b < WALK_BODIES; ++b) {
            float V[6];
            WALK_PARENT_TWIST(link, twist, b, V);
            for (int c = WALK_FIRST_OF(b); c < WALK_FIRST_OF(b + 1); ++c) {
                const float* d = data + $coordinate * c;
                const float q = (u[c] - ((d[6] * V[0] + d[7] * V[1] + d[8] * V[2])
                                         + (d[9] * V[3] + d[10] * V[4] + d[11] * V[5]))) * d[12];
                qd[c] = q;
                for (int k = 0; k < 6; ++k) V[k] += d[k] * q;
            }
            for (int k = 0; k < 6; ++k) twist[6 * b + k] = V[k];
        }
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
    // WALK_TWISTS(now, now_link, qd, twist): every body's twist at its frame from known coordinates qd, root first, at
    // the step's poses (WALK_UP's now and now_link): V = V_parent + d x w_parent, then V += s_now_c qd_c. The writer's
    // prologue: the predicted twists a row's incident velocity reads against its seed's wrench there.
    const auto WALK_TWISTS = [&](const float* now, const float* now_link, const float* qd, float* twist) {
        for (int b = 0; b < WALK_BODIES; ++b) {
            float V[6];
            WALK_PARENT_TWIST(now_link, twist, b, V);
            for (int c = WALK_FIRST_OF(b); c < WALK_FIRST_OF(b + 1); ++c) {
                const float* s = now + $now * c;
                for (int k = 0; k < 6; ++k) V[k] += s[k] * qd[c];
            }
            for (int k = 0; k < 6; ++k) twist[6 * b + k] = V[k];
        }
    };"""


@functools.cache
def walk_prelude(parent: tuple[int, ...], first: tuple[int, ...], rows: int) -> DevicePrelude:
    """The device walk of a class of trees (one tree, or a world's forest), for both devices: a prelude declaring

    - the class's path table in constant memory (``WALK_PARENT``, ``WALK_FIRST``): topology, one table per class, so a
      lane walks a root path without a global load in front of its steps (GPU_WALK §8, variant C);
    - ``WALK_UP`` and ``WALK_SEED``, the rows: a lane per seed walks its ``rows`` rows up the seed's root path, every
      load shared (Z_UNION Theorem B), and gives each footprint to the caller's sink, the placement's store;
    - ``WALK_DOWN``, the leave, and ``WALK_TWISTS``, the twists of known coordinates: a lane per tree, root first.

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
def walk_paths_prelude(rows: int) -> DevicePrelude:
    """The device walk of any tree, for both devices: :func:`walk_prelude`'s functions over a path table in global
    memory, one tree at a time (``WALK_TREE(path, bodies, coordinates)`` before its walks), the table
    :attr:`TreeFactor.walk_path` holds for every articulation. The general placement's worlds hold any forest, so its
    one writer and leave read each tree's table where a class's kernel bakes it into constant memory.

    Args:
        rows: The rows a lane walks together (:func:`walk_prelude`).
    """
    if rows < 1:
        raise ValueError(f"a lane walks at least one row, got {rows}")
    walk = _WALK_PATHS + "\n" + substitute(_WALK, rows=rows, coordinate=WALK_COORDINATE, link=WALK_LINK, now=WALK_NOW)
    return DevicePrelude(f"walk_paths_{rows}", walk, cpu=walk)


# ---------------------------------------------------------------------------------------------------------------------
# The walk's view (FL3.3): what the one writer, the leave and the audit read of the converted class's trees
# ---------------------------------------------------------------------------------------------------------------------


@wp.struct
class WalkTrees:
    """The walk of every held tree of the converted class (:meth:`TreeFactor.plan_walk`), by global coordinate, body
    (its joint) and articulation:

    - ``data``, ``link``, ``path``: :data:`WALK_COORDINATE` floats a coordinate, :data:`WALK_LINK` floats a body and
      each body's parent and first coordinate within its tree (:func:`walk_paths_prelude`'s table);
    - ``now``, ``now_link``: each coordinate's column at its body's frame, and each body's link (:data:`WALK_LINK`'s
      layout: the offset to its parent's frame and its frame point), at the step's poses
      (:meth:`TreeFactor.walk_geometry`, every step): a seed enters the walk as ``J`` there, so a held factor (a step
      between refreshes) whitens the step's ``J``;
    - ``twist``: each body's predicted twist at its frame at the step's poses (:meth:`TreeFactor.twists`), which a
      row's incident reads; ``scratch``: the leave's twists;
    - each articulation's first joint, joint end, first coordinate and coordinates, and whether it is a free rigid
      body (``free``), whose rows and leave keep its own 6x6 factor (``L6`` at ``group``) until the free bodies
      convert;
    - ``dof_body``: each coordinate's body within its tree; ``body_joint``: each body's joint;
    - the slips' frame of each tree (the block primitive's first phase, :func:`walk_frame_kernel`): its slipping drives
      this step (``slips``, 0 for none), the orthonormal basis ``Bh`` of their columns (``basis``, a row per slip over
      the tree's coordinates) and the lower triangle ``Ln`` (``scale``), so that the rows take ``F = I + Bh (Ln^T - I)
      Bh^T`` and the leave ``F^T``.
    """

    data: wp.array2d[float]
    link: wp.array2d[float]
    path: wp.array[wp.vec2i]
    now: wp.array2d[float]
    now_link: wp.array2d[float]
    twist: wp.array[wp.spatial_vector]
    scratch: wp.array[wp.spatial_vector]
    joint_start: wp.array[int]
    joint_end: wp.array[int]
    dof_start: wp.array[int]
    dofs: wp.array[int]
    dof_body: wp.array[int]
    body_joint: wp.array[int]
    free: wp.array[int]
    group: wp.array[int]
    L6: wp.array3d[float]
    slips: wp.array[int]
    basis: wp.array3d[float]
    scale: wp.array3d[float]


_TWISTS = """    if (tid >= arts.shape[0]) return;
    const int art = arts.data[tid];
    const int j0 = walk.joint_start.data[art], d0 = walk.dof_start.data[art];
    WALK_TREE(reinterpret_cast<const int*>(walk.path.data + j0), walk.joint_end.data[art] - j0, walk.dofs.data[art]);
    WALK_TWISTS(walk.now.data + (size_t)d0 * $now, walk.now_link.data + (size_t)j0 * $link, qd.data + d0,
                reinterpret_cast<float*>(walk.twist.data + j0));
"""


_ENTER = """    if (tid >= arts.shape[0]) return;
    const int art = arts.data[tid];
    if (walk.free.data[art] != 0) return;
    const int world = art_to_world.data[art], offset = world_offset.data[art];
    if (world < 0 || offset < 0) return;
    const int j0 = walk.joint_start.data[art], d0 = walk.dof_start.data[art];
    WALK_TREE(reinterpret_cast<const int*>(walk.path.data + j0), walk.joint_end.data[art] - j0, walk.dofs.data[art]);
    float* u = coordinates.data + (size_t)world * coordinates.shape[1] + offset;
    WALK_ENTER(walk.data.data + (size_t)d0 * $coordinate, walk.link.data + (size_t)j0 * $link, velocity.data + d0, u,
               reinterpret_cast<float*>(walk.scratch.data + j0));
    const int slips = walk.slips.data[art], n = walk.dofs.data[art];
    if (slips > 0) {
        // The rows' frame: u <- F^-T u = u + Bh (Ln^-1 - I) Bh^T u (tree_factor.walk_frame_kernel).
        const int width = walk.basis.shape[2], cap = walk.scale.shape[2];
        const float* basis = walk.basis.data + (size_t)art * walk.basis.shape[1] * width;
        const float* scale = walk.scale.data + (size_t)art * walk.scale.shape[1] * cap;
        float y[$slips], x[$slips];
        for (int j = 0; j < slips; ++j) {
            float acc = 0.0f;
            for (int c = 0; c < n; ++c) acc += basis[j * width + c] * u[c];
            y[j] = acc;
        }
        for (int j = 0; j < slips; ++j) {
            float value = y[j];
            for (int k = 0; k < j; ++k) value -= scale[j * cap + k] * x[k];
            x[j] = value / scale[j * cap + j];
        }
        for (int j = 0; j < slips; ++j)
            for (int c = 0; c < n; ++c) u[c] += (x[j] - y[j]) * basis[j * width + c];
    }
"""


@functools.cache
def walk_enter_kernel() -> wp.Kernel:
    """Each held tree's coordinates of a velocity in the walk's frame, ``u = L^T P v`` (``WALK_ENTER``), a thread per
    tree, at its world coordinates: what the audit reads the walked rows against (:mod:`.audit`). A free rigid body
    keeps its 6x6 factor's."""
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
        substitute(_ENTER, coordinate=WALK_COORDINATE, link=WALK_LINK, slips=FRAME_SLIPS),
        THREADS,
        preludes=(walk_paths_prelude(1),),
        dual=True,
        native_name="walk_enter_native",
    ).kernel()


@functools.cache
def walk_twists_kernel() -> wp.Kernel:
    """Each held tree's body twists from known coordinates ``qd`` at its frames at the step's poses, a thread per tree
    (``WALK_TWISTS``): the predicted twists every row's incident velocity reads (:meth:`TreeFactor.twists`)."""
    from .kernel_template import THREADS, KernelTemplate  # noqa: PLC0415

    body = substitute(_TWISTS, now=WALK_NOW, link=WALK_LINK)
    return KernelTemplate(
        "walk_twists",
        (("arts", wp.array[int]), ("walk", WalkTrees), ("qd", wp.array[float])),
        body,
        THREADS,
        preludes=(walk_paths_prelude(1),),
        dual=True,
        native_name="walk_twists_native",
    ).kernel()


_WALK_INTO = """    WALK_TREE(reinterpret_cast<const int*>(walk.path.data + joint0), bodies, coordinates);
    const float* data = walk.data.data + (size_t)dof0 * $coordinate;
    const float* link = walk.link.data + (size_t)joint0 * $link;
    float* row = out.data + ((size_t)line * out.shape[1] + slot) * out.shape[2] + entry;
    const auto sink = [&](int, int c, float z) { row[c] += z; };
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
    out: wp.array3d[float],
    line: int,
    slot: int,
    entry: int,
): ...


FRAME_SLIPS = 16
"""The most slipping drives a tree's frame holds (the one writer's local vector): a tree past it keeps its first
``FRAME_SLIPS`` and the frame capacity status flags it."""

_FrameVector = wp.types.vector(length=FRAME_SLIPS, dtype=wp.float32)


@wp.kernel(enable_backward=False)
def walk_frame_kernel(
    arts: wp.array[int],
    walk: WalkTrees,
    decided: wp.array[int],
    row_counts: wp.array[int],
    row_dof: wp.array[int],
    row_K: wp.array[float],
    row_Kd: wp.array[float],
    max_dofs: int,
    # outputs
    slips: wp.array[int],
    slip_row: wp.array2d[int],
    basis: wp.array3d[float],
    scale: wp.array3d[float],
    work: wp.array3d[float],
    status: wp.array[int],
):
    """The slips' frame of each tree whose predictor decided a set (``decided``, H7's release), a thread per tree
    (FL3_design §1.4, SAT_PHYS §1.2). A slipping drive is a drive row whose decided ``K`` (``row_Kd``) is below the
    trial's: its released stiffness is ``K_i = row_K - row_Kd``. With ``B`` the walk's columns of the slipping
    coordinates (``walk_up(e_i)``), ``B = Bh T`` by modified Gram-Schmidt, and the capacitance ``C = K_S^-1 - B^T B`` with
    the split pivot on its diagonal, ``C_ii = P_i / (K_i D_i) - sum over the ancestors of B_ki^2`` (``P_i`` the pivot
    with the decided ``K`` alone, ``D_i`` with all of it), the frame's scale is ``Ln = chol(I + T C^-1 T^T)``: then
    ``F = I + Bh (Ln^T - I) Bh^T`` and ``F^T F = I + B C^-1 B^T``, the decided operator on the all-``K`` walk."""
    art = arts[wp.tid()]
    slips[art] = 0
    if decided[art] == 0:
        return
    cap = basis.shape[1]
    n = walk.dofs[art]
    d0 = walk.dof_start[art]
    j0 = walk.joint_start[art]
    s = int(0)
    for r in range(row_counts[art]):
        row = art * max_dofs + r
        released = row_K[row] - row_Kd[row]
        if row_Kd[row] != row_K[row] and released > 0.0:
            if s < cap:
                slip_row[art, s] = row
                s += 1
            else:
                status[0] = 1
    if s == 0:
        return
    # The columns B_j = walk_up(e_i), and the capacitance from them (work rows [0, s)).
    for j in range(s):
        for c in range(n):
            basis[art, j, c] = 0.0
        dof = row_dof[slip_row[art, j]]
        walk_into(
            walk,
            j0,
            d0,
            walk.joint_end[art] - j0,
            n,
            walk.dof_body[dof],
            wp.spatial_vector(),
            dof - d0,
            1.0,
            basis,
            art,
            j,
            0,
        )
    for i in range(s):
        row_i = slip_row[art, i]
        own_c = row_dof[row_i] - d0
        for j in range(i):
            dot = float(0.0)
            for c in range(n):
                dot += basis[art, i, c] * basis[art, j, c]
            work[art, i, j] = -dot
            work[art, j, i] = -dot
        ancestors = float(0.0)
        for c in range(n):
            if c != own_c:
                ancestors += basis[art, i, c] * basis[art, i, c]
        K = row_K[row_i] - row_Kd[row_i]
        pivot = walk.data[d0 + own_c, 13] + row_Kd[row_i]
        work[art, i, i] = pivot / (K * (pivot + K)) - ancestors
    # B = Bh T by modified Gram-Schmidt, in place; T in work rows [cap, cap + s).
    for j in range(s):
        for i in range(j):
            t = float(0.0)
            for c in range(n):
                t += basis[art, i, c] * basis[art, j, c]
            work[art, cap + i, j] = t
            for c in range(n):
                basis[art, j, c] = basis[art, j, c] - t * basis[art, i, c]
        norm2 = float(0.0)
        for c in range(n):
            norm2 += basis[art, j, c] * basis[art, j, c]
        norm = wp.sqrt(norm2)
        work[art, cap + j, j] = norm
        for i in range(j + 1, s):
            work[art, cap + i, j] = 0.0
        for c in range(n):
            basis[art, j, c] = basis[art, j, c] / norm
    # C = Lc Lc^T in place (its lower triangle).
    for j in range(s):
        diag = work[art, j, j]
        for k in range(j):
            diag -= work[art, j, k] * work[art, j, k]
        diag = wp.sqrt(diag)
        work[art, j, j] = diag
        for i in range(j + 1, s):
            value = work[art, i, j]
            for k in range(j):
                value -= work[art, i, k] * work[art, j, k]
            work[art, i, j] = value / diag
    # N = T C^-1 T^T, a column at a time: x = C^-1 (T^T e_m) through Lc, then N[:, m] = T x; M = I + N in scale.
    for m in range(s):
        x = _FrameVector()
        for i in range(s):
            value = work[art, cap + m, i]  # (T^T)[i, m] = T[m, i]
            for k in range(i):
                value -= work[art, i, k] * x[k]
            x[i] = value / work[art, i, i]
        for ii in range(s):
            i = s - 1 - ii
            value = x[i]
            for k in range(i + 1, s):
                value -= work[art, k, i] * x[k]
            x[i] = value / work[art, i, i]
        for i in range(s):
            value = float(0.0)
            for k in range(i, s):
                value += work[art, cap + i, k] * x[k]
            if i == m:
                value += 1.0
            scale[art, i, m] = value
    # Ln = chol(I + N) in place.
    for j in range(s):
        diag = scale[art, j, j]
        for k in range(j):
            diag -= scale[art, j, k] * scale[art, j, k]
        diag = wp.sqrt(diag)
        scale[art, j, j] = diag
        for i in range(j + 1, s):
            value = scale[art, i, j]
            for k in range(j):
                value -= scale[art, i, k] * scale[art, j, k]
            scale[art, i, j] = value / diag
        for i in range(j):
            scale[art, i, j] = 0.0
    slips[art] = s


# ---------------------------------------------------------------------------------------------------------------------
# The walk's refresh on the device (FL3.3): one inward pass per tree, in local frames, the factor's own cadence
# ---------------------------------------------------------------------------------------------------------------------


@wp.func
def _walk_column(data: wp.array2d[float], dof: int, column: wp.spatial_vector):
    """Store coordinate ``dof``'s motion column ``s`` at its body's frame (:data:`WALK_COORDINATE`'s first six)."""
    for k in range(6):
        data[dof, k] = column[k]


@wp.func
def _walk_lift(inertia: wp.spatial_matrix, d: wp.vec3) -> wp.spatial_matrix:
    """``inertia`` at a body's frame taken at its parent's, ``d`` the parent's frame less the body's
    (:meth:`TreeWalk._lift`): ``W I T``, ``T`` taking a twist at the parent's frame to the body's (``v + d x w``) and
    ``W`` a wrench at the body's frame to the parent's (``t - d x f``)."""
    D = wp.skew(d)
    to = wp.identity(6, dtype=float)
    wrench = wp.identity(6, dtype=float)
    for r in range(3):
        for c in range(3):
            to[r, 3 + c] = D[r, c]
            wrench[3 + r, c] = -D[r, c]
    return wp.mul(wp.mul(wrench, inertia), to)


@wp.func
def _walk_frame(
    j: int,
    j0: int,
    walk_path: wp.array[wp.vec2i],
    joint_type: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
) -> wp.transform:
    """Joint ``j``'s frame at the poses ``body_q`` holds: its anchor (its parent body's pose times its joint frame, the
    world's at a root), at a free body's centre of mass for a free joint, with the anchor's rotation."""
    parent = walk_path[j][0]
    anchor = joint_X_p[j]
    if parent >= 0:
        anchor = wp.transform_multiply(body_q[joint_child[j0 + parent]], joint_X_p[j])
    kind = joint_type[j]
    if kind == JointType.FREE or kind == JointType.DISTANCE:
        child = joint_child[j]
        return wp.transform(wp.transform_point(body_q[child], body_com[child]), wp.transform_get_rotation(anchor))
    return anchor


@wp.kernel(enable_backward=False)
def walk_geometry_kernel(
    joints: wp.array[int],
    joint_art: wp.array[int],
    articulation_start: wp.array[int],
    walk_path: wp.array[wp.vec2i],
    joint_type: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_child: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_q: wp.array[float],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    # outputs
    walk_now: wp.array2d[float],
    walk_now_link: wp.array2d[float],
):
    """The step's geometry of every held tree, a thread per joint (:meth:`TreeFactor.walk_geometry`), float32, at the
    poses ``body_q`` holds: the body's frame (its joint's anchor, a free body's centre of mass), its coordinates'
    columns there in world axes, as :func:`local_frames` takes them (a free joint turns about the articulation's origin,
    its first body's centre of mass; a D6 joint's turns about the pivot its slides moved), and its link (:data:`WALK_LINK`'s
    layout: its parent's frame less its own, its frame's world point). The rows' seeds enter the walk here, and the
    refresh takes its frames from it."""
    j = joints[wp.tid()]
    j0 = articulation_start[joint_art[j]]
    parent = walk_path[j][0]
    X = _walk_frame(j, j0, walk_path, joint_type, joint_child, joint_X_p, body_q, body_com)
    R_a = wp.transform_get_rotation(X)
    frame = wp.transform_get_translation(X)
    kind = joint_type[j]
    free = kind == JointType.FREE or kind == JointType.DISTANCE
    q0 = joint_q_start[j]
    d0 = joint_qd_start[j]
    if kind == JointType.REVOLUTE:
        _walk_column(walk_now, d0, wp.spatial_vector(wp.vec3(0.0), wp.quat_rotate(R_a, joint_axis[d0])))
    elif kind == JointType.PRISMATIC:
        _walk_column(walk_now, d0, wp.spatial_vector(wp.quat_rotate(R_a, joint_axis[d0]), wp.vec3(0.0)))
    elif kind == JointType.BALL:
        for k in range(3):
            e = wp.vec3(0.0)
            e[k] = 1.0
            _walk_column(walk_now, d0 + k, wp.spatial_vector(wp.vec3(0.0), wp.quat_rotate(R_a, e)))
    elif free:
        first = joint_child[j0]
        origin = wp.transform_point(body_q[first], body_com[first])
        for k in range(3):
            e = wp.vec3(0.0)
            e[k] = 1.0
            w = wp.quat_rotate(R_a, e)
            _walk_column(walk_now, d0 + k, wp.spatial_vector(w, wp.vec3(0.0)))
            # The factor turns a free joint about the articulation's origin: the same twist at the body's frame.
            _walk_column(walk_now, d0 + 3 + k, wp.spatial_vector(wp.cross(w, frame - origin), w))
    elif kind == JointType.D6:
        lin = joint_dof_dim[j, 0]
        ang = joint_dof_dim[j, 1]
        slide = wp.vec3(0.0)
        for k in range(lin):
            _walk_column(walk_now, d0 + k, wp.spatial_vector(wp.quat_rotate(R_a, joint_axis[d0 + k]), wp.vec3(0.0)))
            slide += joint_axis[d0 + k] * joint_q[q0 + k]
        turned = wp.quat_identity()
        for k in range(ang):
            axis = wp.quat_rotate(turned, joint_axis[d0 + lin + k])
            w = wp.quat_rotate(R_a, axis)
            # A turn about the pivot the slides moved, at the anchor: v = (R_a slide) x w.
            _walk_column(walk_now, d0 + lin + k, wp.spatial_vector(wp.cross(wp.quat_rotate(R_a, slide), w), w))
            turned = wp.quat_from_axis_angle(axis, joint_q[q0 + lin + k]) * turned
    offset = wp.vec3(0.0)
    if parent >= 0:
        up = _walk_frame(j0 + parent, j0, walk_path, joint_type, joint_child, joint_X_p, body_q, body_com)
        offset = wp.transform_get_translation(up) - frame
    for k in range(3):
        walk_now_link[j, k] = offset[k]
        walk_now_link[j, 4 + k] = frame[k]


@wp.kernel(enable_backward=False)
def walk_refresh_kernel(
    arts: wp.array[int],
    mass_update_mask: wp.array[int],
    articulation_start: wp.array[int],
    articulation_joint_end: wp.array[int],
    walk_path: wp.array[wp.vec2i],
    joint_child: wp.array[int],
    joint_qd_start: wp.array[int],
    body_q: wp.array[wp.transform],
    body_mass: wp.array[float],
    body_com: wp.array[wp.vec3],
    body_inertia: wp.array[wp.mat33],
    armature: wp.array[float],
    drive_row_by_dof: wp.array[int],
    row_K: wp.array[float],
    free_rigid: wp.array[int],
    walk_join: wp.array[int],
    walk_now: wp.array2d[float],
    walk_now_link: wp.array2d[float],
    # outputs
    walk_data: wp.array2d[float],
    walk_link: wp.array2d[float],
    walk_inertia: wp.array[wp.spatial_matrix],
):
    """The walk's data of each refreshed tree (``mass_update_mask``), a thread per articulation, float32: the factor of
    ``P (H + R + K) P^T`` as :class:`TreeWalk` forms it from :func:`local_frames`, at the refresh's poses (``body_q``,
    which the held tree's forward kinematics publishes before any factor read).

    Outward, every parent before its children: the step's geometry (:func:`walk_geometry_kernel`, at the same poses)
    becomes the factor's, each coordinate's column and each body's frame and link to its parent's frame. Inward, leaf
    first: a body's spatial inertia at its frame plus its children's lifts, then each coordinate takes ``ell = (s^T I s
    + R + K)^1/2``, ``ut = I s / ell`` and ``P0 = s^T I s + R``, ``I -= ut ut^T``, and the body lifts what is left to
    its parent: in registers when the parent is the next joint inward, through ``walk_inertia`` at a join
    (``walk_join``: a joint with a child other than the next joint). ``R`` is the factor's effective armature and ``K``
    each drive's, the diagonal the refresh's ``L`` holds (:meth:`TreeFactor.diagonal`), so the walk and ``L`` are one
    factor. A free rigid body (``free_rigid``) keeps its 6x6 factor and takes nothing."""
    art = arts[wp.tid()]
    if mass_update_mask[art] == 0 or free_rigid[art] != 0:
        return
    j0 = articulation_start[art]
    j1 = articulation_joint_end[art]
    for j in range(j0, j1):
        for dof in range(joint_qd_start[j], joint_qd_start[j + 1]):
            for k in range(6):
                walk_data[dof, k] = walk_now[dof, k]
        for k in range(3):
            walk_link[j, k] = walk_now_link[j, k]
            walk_link[j, 4 + k] = walk_now_link[j, 4 + k]
        if walk_join[j] != 0:
            # A join gathers the lifted inertias of its children other than the next joint.
            walk_inertia[j] = wp.spatial_matrix()
    carry = wp.spatial_matrix()
    for jj in range(j1 - j0):
        jb = j1 - 1 - jj
        child = joint_child[jb]
        pose = body_q[child]
        frame = wp.vec3(walk_now_link[jb, 4], walk_now_link[jb, 5], walk_now_link[jb, 6])
        X = wp.skew(wp.transform_point(pose, body_com[child]) - frame)
        m = body_mass[child]
        R = wp.quat_to_matrix(wp.transform_get_rotation(pose))
        rotational = R * body_inertia[child] * wp.transpose(R) - m * X * X
        held = carry
        for a in range(3):
            held[a, a] = held[a, a] + m
            for c in range(3):
                held[a, 3 + c] = held[a, 3 + c] - m * X[a, c]
                held[3 + a, c] = held[3 + a, c] + m * X[a, c]
                held[3 + a, 3 + c] = held[3 + a, 3 + c] + rotational[a, c]
        if walk_join[jb] != 0:
            held = held + walk_inertia[jb]
        for cc in range(joint_qd_start[jb + 1] - joint_qd_start[jb]):
            dof = joint_qd_start[jb + 1] - 1 - cc
            s = wp.spatial_vector(
                walk_now[dof, 0],
                walk_now[dof, 1],
                walk_now[dof, 2],
                walk_now[dof, 3],
                walk_now[dof, 4],
                walk_now[dof, 5],
            )
            Is = held * s
            own = wp.dot(s, Is)
            added = armature[dof]
            pivot = own + added
            row = drive_row_by_dof[dof]
            if row >= 0:
                if row_K[row] > 0.0:
                    added += row_K[row]
            ell = wp.sqrt(own + added)
            ut = Is / ell
            for k in range(6):
                walk_data[dof, 6 + k] = ut[k]
            walk_data[dof, 12] = 1.0 / ell
            walk_data[dof, 13] = pivot  # WALK_PIVOT
            held = held - wp.outer(ut, ut)
        up = walk_path[jb][0]
        carry = wp.spatial_matrix()
        if up >= 0:
            lifted = _walk_lift(held, wp.vec3(walk_link[jb, 0], walk_link[jb, 1], walk_link[jb, 2]))
            if j0 + up == jb - 1:
                carry = lifted
            else:
                walk_inertia[j0 + up] = walk_inertia[j0 + up] + lifted
