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

Its evaluator is the held tree (:mod:`.held_tree`) where that admits the solver, else the ordinary stages
(:mod:`.tree_ordinary`) behind the same calls, until D. A step calls :meth:`TreeFactor.factor` (FK/ID, the drives'
K and u0, the mass-update mask, the factor refresh), :meth:`TreeFactor.predict` (``v_hat``),
:meth:`TreeFactor.leave` (``v = v_hat + P^T L^-T du``, where the sweep stores ``du``) and
:meth:`TreeFactor.finish` (``qdd``, integrate, publish).
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager

import numpy as np
import warp as wp

from ...sim import BodyFlags, JointType, ModelFlags
from . import field_group, host_index, response_builds_rows
from .conditioning import ArmatureFloor, conditioning_armature
from .field_group import SolverArrays
from .joint_table import JointTable
from .kernels import (
    BodyFactors,
    apply_free_root_transport_to_predictor,
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
    JOINT_FOREST,
    PREELIMINATION,
    PREELIMINATION_PLAN,
    SIZE_FACTORS,
    SIZE_GROUPED_SOLVE,
    SPARSE_FACTOR,
    STATE_AUX,
    Preelim,
)
from .sparse_mass_matrix import factor_indices
from .spatial_laws import drive_force, drive_stiffness
from .tree_plan import TreePlan

__all__ = ["TreeFactor", "prepare_augmented_drives"]


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
    dt: float,
    row_K: wp.array[float],
    drive_tau: wp.array[float],
):
    """Each coordinate's augmented drive, ``K`` (:func:`.spatial_laws.drive_stiffness`, its passive damper's with it)
    and ``u0`` (:func:`.spatial_laws.drive_force`), from the immutable DOF-to-row topology."""
    dof = wp.tid()
    drive_tau[dof] = 0.0
    row = drive_row_by_dof[dof]
    if row < 0:
        return
    ke = joint_target_ke[dof]
    kd = joint_target_kd[dof]
    K = drive_stiffness(ke, kd, passive_damping[dof], dt)
    if K <= 0.0:
        row_K[row] = 0.0
        return
    u0 = drive_force(
        ke,
        kd,
        joint_q[drive_q_index_by_dof[dof]],
        joint_qd[dof],
        joint_target_pos[drive_target_index_by_dof[dof]],
        joint_target_vel[dof],
        joint_effort_limit[dof],
        dt,
    )
    row_K[row] = K
    drive_tau[dof] = u0


class TreeFactor:
    """The tree factor of one solver (see the module docstring); the solver holds it as ``factor``.

    Built in the solver's construction order: the cadence, the cache and the augmented state here, then
    :meth:`kinematic_state` and :meth:`apply_kinematic_state`, :meth:`plan` once the forest plan and the row laws
    exist, :meth:`allocate_common`, :meth:`allocate_size_groups`, :meth:`build_body_factors`,
    :meth:`scatter_armature`, :meth:`install` and :meth:`init_size_streams`.

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
        self.fk_id_cache_available = bool(model.articulation_count and model.joint_count)
        # The held tree's cache of next-step dynamics (enabled by install); ordinary dynamics derive them from the
        # state at every step.
        self.fk_id_cache_enabled = False
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
        self.tau_by_size: dict[int, wp.array] = {}
        self.qdd_by_size: dict[int, wp.array] = {}
        self.diagonal_inverse_mass = None
        self.body_factors = None
        self.eliminates = False
        self.elimination = None
        self.block_count = 0
        self.held = None
        self.held_reason = None
        self.ordinary = None
        self.dynamics_stream = None
        self.size_streams: dict[int, wp.Stream | None] = {}
        self.size_events: dict[int, wp.Event | None] = {}
        self._init_event = None

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
        hold wherever each coordinate has one owning drive row. The held tree's admission, which the
        representations only it evaluates (the compact diagonal, the native sparse factor) read before they are
        chosen, needs them; ordinary dynamics drop them once the held tree declines (:meth:`install`) and evaluate
        drives with the joint forces.
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

    def eliminate(self, *, Z, sides, counts, owners, cfm, rhs, incident, diag, impulses) -> None:
        """Make each block's rows orthonormal, then correct every other row, its right-hand side and its diagonal, and
        give the block's rows the projection's impulses, which the sweep seeds
        (:func:`.response_builds_rows.elimination`). Runs once the right-hand sides are final (bias and restitution)."""
        block = field_group.bind(Preelim, factor=self)
        wp.launch(
            self.elimination.setup,
            dim=self.block_count,
            inputs=[
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
            inputs=[block, sides, counts, owners, cfm],
            outputs=[Z, rhs, diag],
            device=self.model.device,
        )
        wp.launch(
            self.elimination.weights,
            dim=self.block_count,
            inputs=[block, rhs],
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
            )
        self.parallel_drives = False
        self.drive_row_by_dof = wp.full((max(1, int(model.joint_dof_count)),), -1, dtype=wp.int32, device=model.device)
        self.drive_q_index_by_dof = None
        self.drive_target_index_by_dof = None
        if not eligible or forest.has_loop_joints or not model.joint_dof_count:
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
        frames and the joint forest's upload) and the articulation origins."""
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
            self._upload_joint_forest()
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

    def _upload_joint_forest(self) -> None:
        """Upload the joint forest's depth-first order, positions and subtree ends (``tree_passes.crba_fill``'s
        ancestor test) and its up flags (the up scans', :attr:`.tree_plan.TreePlan.up_flags`)."""
        if not self.model.joint_count:
            self._arrays.release(JOINT_FOREST)
            return
        forest = self.forest.joint_forest
        self._arrays.upload(
            JOINT_FOREST,
            _joint_dfs_order=forest.dfs_order,
            _joint_dfs_position=forest.dfs_position,
            _joint_dfs_end=forest.dfs_end,
            _joint_up_flags=forest.up_flags,
        )

    def allocate_sparse(self, plan, articulations: int) -> None:
        """The native sparse factor of ``articulations`` of one size: its pattern ``plan``, its device indices, its
        inverse entries and its status."""
        self.sparse_mass_matrix_plan = plan
        self.sparse_mass_matrix_indices = factor_indices(plan, self.model.device)
        self._arrays.allocate(SPARSE_FACTOR, group=articulations, nonzero=plan.nonzero_count)

    def allocate_size_groups(self, placeholder_sizes) -> None:
        """Allocate each size group's factor L, armature R and dense-solve scratch; ``placeholder_sizes`` (the compact
        diagonal and the native sparse factor) keep (1, 1, 1) placeholders of L and no scratch."""
        for size in self.forest.size_groups:
            n_arts = self.forest.n_arts_by_size[size]
            placeholder = size in placeholder_sizes
            factor = 1 if placeholder else 0
            sizes = {"art": n_arts, "size": size, "factor_art": factor or n_arts, "factor_size": factor or size}
            buffers = self._arrays.allocate(SIZE_FACTORS, owner={}, **sizes)
            if not placeholder:
                # Grouped scratch belongs to the dense triangular solve, not the compact paths.
                buffers.update(self._arrays.allocate(SIZE_GROUPED_SOLVE, owner={}, **sizes))
            for name, array in buffers.items():
                getattr(self, f"{name}_by_size")[size] = array

    def scatter_armature(self) -> None:
        """Copy the effective armature (DOF-ordered) to the size groups' R."""
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
        """Choose the evaluator: the held tree where :func:`.held_tree.install` admits it, else the ordinary stages.

        Ordinary dynamics take the joint forces and the drives in one pass (K added after the assembly); the
        per-coordinate drive rows are the held tree's, and their preparation overlaps FK/ID on a stream of its own.
        """
        from . import held_tree  # noqa: PLC0415

        self.held, self.held_reason = held_tree.install(solver)
        if self.held is not None:
            self.fk_id_cache_enabled = True
        else:
            self.parallel_drives = False
            from .tree_ordinary import OrdinaryTree  # noqa: PLC0415

            self.ordinary = OrdinaryTree(solver)
        model = self.model
        self.dynamics_stream = wp.Stream(model.device) if self.parallel_drives and model.device.is_cuda else None

    def init_size_streams(self) -> None:
        """One CUDA stream and event per size group where several size groups launch in parallel (none on the CPU or
        for a single size group)."""
        model = self.model
        self.size_streams = {}
        self.size_events = {}
        if model.device.is_cuda and len(self.forest.size_groups) > 1:
            for size in self.forest.size_groups:
                self.size_streams[size] = wp.Stream(model.device)
                self.size_events[size] = wp.Event(model.device)
        else:
            for size in self.forest.size_groups:
                self.size_streams[size] = None
                self.size_events[size] = None

    # ---------------------------------------------------------------------------------------------------------------
    # Notifications
    # ---------------------------------------------------------------------------------------------------------------

    def forget_repair_continuity(self) -> None:
        """Launch the held tree's next repair after a host call that changes its validity or inputs."""
        if self.held is not None:
            self.held.forget_continuity()

    def track_step_dt(self, dt: float) -> None:
        """Record the step duration; a change refreshes mass and invalidates the FK/ID cache of every articulation."""
        if self.last_step_dt is not None and abs(self.last_step_dt - dt) > 1.0e-8:
            self.force_mass_update = True
            if self.fk_id_cache_enabled:
                self.fk_id_cache_valid.zero_()
            self.forget_repair_continuity()
        self.last_step_dt = dt

    def notify_state_changed(self, articulations: wp.array | None = None) -> None:
        """Invalidate the cached kinematics of ``articulations`` (``None``: every articulation)."""
        if not self.fk_id_cache_enabled:
            return
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
        frames_proven = held is not None and held.validate_model(flags)
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
        elif held is not None and not frames_proven:
            from .tree_kinetic_owner import proof_runs  # noqa: PLC0415

            if proof_runs(flags):
                # The proof ran and found a moved joint frame (a changed axis raises in every group's proof):
                # invalidate every cache, whichever flag announced the move.
                cache_inputs |= int(ModelFlags.JOINT_PROPERTIES)
        if held is not None and cache_inputs:
            held.invalidate()
        if self.fk_id_cache_enabled and cache_inputs:
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
        if self.fk_id_cache_enabled:
            wp.launch(
                invalidate_articulation_fk_id_cache,
                dim=self.model.articulation_count,
                inputs=[world_mask, self.forest.art_to_world],
                outputs=[self.fk_id_cache_valid],
                device=self.model.device,
            )
        if (
            self.fk_id_cache_enabled
            and self.fk_id_cache_source_state is not None
            and state is not self.fk_id_cache_source_state
            and not getattr(self.held, "bank_compare", False)
        ):
            self.fk_id_cache_valid.zero_()
            self.fk_id_cache_source_state = state

    # ---------------------------------------------------------------------------------------------------------------
    # The step
    # ---------------------------------------------------------------------------------------------------------------

    @contextmanager
    def _parallel_size_region(self):
        """Context for parallel dispatch across size groups."""
        if not self.model.device.is_cuda or len(self.forest.size_groups) <= 1:
            yield
            return

        main_stream = wp.get_stream(self.model.device)
        self._init_event = main_stream.record_event()
        try:
            yield
        finally:
            for size in self.forest.size_groups:
                stream = self.size_streams.get(size)
                if stream is not None:
                    main_stream.wait_event(stream.record_event())
            self._init_event = None

    @contextmanager
    def _on_size_stream(self, size: int):
        """Execute block on this size's CUDA stream."""
        stream = self.size_streams.get(size)
        if stream is not None and self._init_event is not None:
            stream.wait_event(self._init_event)
            with wp.ScopedStream(stream):
                yield
        else:
            yield

    def for_sizes(self):
        """Each size group with the context that runs its launches on the size's stream."""
        with self._parallel_size_region():
            for size in self.forest.size_groups:
                yield size, self._on_size_stream(size)

    def factor(self, state_in, control, state_out, dt: float, step: int, qd_work: wp.array):
        """Stages 1 and 2: FK/ID, the drives' K and u0, the joint forces, the mass-update mask and the factor refresh
        (the held tree's in stage 1, the ordinary factor per size in stage 2).

        Returns:
            ``(qd, ready)``: the predictor's velocity (``qd_work`` prescaled when velocities are limited) and the event
            the predictor waits for, if any.
        """
        model = self.model
        with wp.ScopedTimer("S1_FK_ID_CRBA", print=False, use_nvtx=self.nvtx, synchronize=False):
            drive_rows_ready = self.prepare_drives(state_in, control, dt)
            qd = self.fk_id(state_in, state_out, step, qd_work)
            if model.articulation_count:
                # The held tree's joint_tau holds only u0: the predictor adds the current external, control and bias
                # contributions after waiting for the drives. Ordinary dynamics take the joint forces and the drives
                # in one pass.
                if self.held is not None:
                    ready = drive_rows_ready
                else:
                    self.ordinary.joint_tau(state_in, control, dt)
                    ready = None
            else:
                ready = None
            self.refresh(drive_rows_ready, step)
        with wp.ScopedTimer("S2_Cholesky", print=False, use_nvtx=self.nvtx, synchronize=False):
            for size, context in self.for_sizes():
                with context:
                    if self.ordinary is not None:
                        self.ordinary.factor(size)
        return qd, ready

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
        if self.fk_id_cache_enabled:
            if not getattr(self.held, "bank_compare", False):
                # Held trees that compare input banks validate a changed bank on the device.
                if self.fk_id_cache_source_state is not None and state_in is not self.fk_id_cache_source_state:
                    self.fk_id_cache_valid.zero_()
                    self.forget_repair_continuity()
            self.fk_id_cache_source_state = state_in
        else:
            self.body_f_s.zero_()
        if self.options.enable_joint_velocity_limits:
            self.prescale_velocity(state_in.joint_qd, qd_work)
            qd = qd_work
        else:
            qd = state_in.joint_qd

        refresh_composite = (step % self.options.update_mass_matrix_interval) == 0 or self.force_mass_update
        if self.held is not None:
            self.held.begin(state_in, self, qd, self.last_step_dt, refresh_composite, state_out=state_out)
            return qd
        self.ordinary.fk_id(state_in, state_out, qd, refresh_composite)
        return qd

    def prepare_drives(self, state_in, control, dt: float) -> wp.Event | None:
        """Launch the per-coordinate augmented-drive preparation independently of FK/ID."""
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
                        dt,
                    ],
                    outputs=[self.aug_row_K, self.joint_tau],
                    block_dim=256,
                    device=model.device,
                )
        return None if dynamics_stream is None else dynamics_stream.record_event()

    def refresh(self, drive_rows_ready: wp.Event | None, step: int) -> None:
        """The mass-update mask, then the held tree's factor refresh or the ordinary CRBA; consumes the requests."""
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

        wp.launch(
            build_mass_update_mask,
            dim=model.articulation_count,
            inputs=[
                global_flag,
                self.mass_update_requested,
            ],
            outputs=[self.mass_update_mask],
            device=model.device,
        )

        if self.held is not None:
            if drive_rows_ready is not None:
                wp.get_stream(model.device).wait_event(drive_rows_ready)
            self.held.refresh(self)
        else:
            self.ordinary.crba(global_flag)
        # Consume the requests: leaving either set changes the factor cadence on subsequent captured steps.
        self.mass_update_requested.zero_()
        self.force_mass_update = False

    def predict(self, state_in, control, qd: wp.array, ready: wp.Event | None, dt: float, v_hat: wp.array) -> None:
        """Stage 3: ``v_hat = qd + h H^-1 (tau - c)``, the free-root correction included."""
        with wp.ScopedTimer("S3_Trisolve_Vhat", print=False, use_nvtx=self.nvtx, synchronize=False):
            if ready is not None:
                wp.get_stream(self.model.device).wait_event(ready)
            if self.held is not None:
                self.held.predict(state_in, self, control, qd, dt)
                self.correct_free_root_velocity(state_in, qd, dt, v_hat, apply_transport=False)
            else:
                self.ordinary.predict(state_in, qd, dt)

    def correct_free_root_velocity(
        self, state_in, qd: wp.array, dt: float, v_hat: wp.array, *, apply_transport: bool = True
    ) -> None:
        """Apply canonical gyro physics and the free roots' spin damping after either ordinary or held prediction."""
        model = self.model
        forest = self.forest
        # Keep v_hat on jcalc_integrate's free-root convention. The compact
        # launch excludes every non-root joint; a live device mask handles
        # runtime kinematic changes without changing graph topology.
        if not forest.free_root_joint_count:
            return
        if forest.free_rigid_body_count:
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
                    apply_transport,
                ],
                outputs=[v_hat],
                device=model.device,
            )
        elif apply_transport:
            wp.launch(
                apply_free_root_transport_to_predictor,
                dim=forest.free_root_joint_count,
                inputs=[
                    forest.free_root_joint_indices,
                    model.joint_qd_start,
                    self.kinematic_joint_mask,
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
            if self.held is not None:
                next_refresh = ((step + 1) % self.options.update_mass_matrix_interval) == 0
                self.held.finish(state_in, self, state_out, dt, next_refresh)
                self.fk_id_cache_source_state = state_out
            else:
                self.ordinary.finish(state_in, state_out, dt)
