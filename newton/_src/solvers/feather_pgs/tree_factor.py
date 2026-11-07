# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The tree factor: each articulation's ``P (H + R + K) P^T = L L^T``, the predictor, and the factor's maps of
velocities (BACKBONE §2).

:class:`TreeFactor` owns what the factor reads and writes across a step:

- the augmented state, the step's ``state_aug`` (joint forces, motion axes, body twists and wrenches);
- the kinematic masks and the effective armature R (a kinematic coordinate holds 1e10 armature);
- the drive topology and its rows K, prepared per coordinate apart from inverse dynamics;
- the walk of every held tree (:class:`.factor.walk.WalkTrees`, its refresh the factor, where each body's tree and coordinates
  live among it) and the compact diagonal's inverse mass;
- the body inertia records, the articulation origins and the passive joint forces;
- the mass-update cadence, its requests and the FK/ID cache.

Its evaluator is the held tree (:mod:`.held_tree`). A step calls :meth:`TreeFactor.factor` (FK/ID, the drives' K and
u0, the mass-update mask, the factor refresh), :meth:`TreeFactor.predict` (``v_hat`` and the drives' saturation),
and :meth:`TreeFactor.finish` (``qdd``, integrate, publish); the walk's rows and leaves read its walk
(:meth:`TreeFactor.plan_walk`).
"""

from __future__ import annotations

import warnings

import numpy as np
import warp as wp

from ...core import execution, memory
from ...sim import BodyFlags, JointType, ModelFlags
from . import field_group, host_index
from .factor.closure import seed_blocks_kernel
from .factor.fields import (
    ARTICULATION_ORIGIN,
    ARTICULATION_WORK,
    AUGMENTED_DRIVE_ROWS,
    BODY_INERTIA,
    PREELIMINATION,
    PREELIMINATION_PLAN,
    STATE_AUX,
    DriveRows,
    Preelim,
    drive_terms,
    hold_drive_row,
)
from .factor.kernels import (
    Clock,
    advance_clock,
    apply_free_root_velocity_corrections,
    build_mass_update_mask,
    compute_com_transforms,
    damp_free_root_spin,
    invalidate_articulation_fk_id_cache,
    invalidate_fk_id_cache_for_articulations,
    prescale_joint_velocity_limits,
)
from .factor.walk import (
    RELEASE_SLIPS,
    WALK_COORDINATE,
    WALK_JOIN_SLOTS,
    WALK_LINK,
    TreeWalk,
    WalkTrees,
    leave_work,
    local_frames,
    release_view,
    walk_leave_kernel,
    walk_step_kernel,
    walk_work,
)
from .field_group import SolverArrays
from .joint_table import JointTable
from .path_selection import json_value
from .row_laws import LINK_ROWS
from .spatial_laws import spring_damper_stiffness
from .tree_scan import (
    WALK_ANGULAR,
    WALK_BOTH,
    WALK_LINEAR,
    WalkClass,
)

__all__ = [
    "TreeFactor",
    "prepare_augmented_drives",
]


@wp.kernel
def prepare_augmented_drives(
    dofs: wp.array[int],
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
    """The augmented drive of each coordinate of ``dofs`` (those no one-coordinate tree's predict decides; without a
    list, the thread's own), from the immutable DOF-to-row topology: ``u0`` and ``K`` (:func:`.factor.fields.drive_terms`, its
    passive damper's with it) held on its row (:func:`.factor.fields.hold_drive_row`), which requests the
    articulation's factor refresh wherever ``K`` changed, a drive whose ``K`` fell to zero included."""
    dof = wp.tid()
    if dofs.shape[0] > 0:
        dof = dofs[dof]
    drive_tau[dof] = 0.0
    row = drive_row_by_dof[dof]
    if row < 0:
        return
    u0, K = drive_terms(
        row,
        dof,
        drive_q_index_by_dof[dof],
        joint_q,
        joint_qd,
        dt,
        drive_target_index_by_dof,
        joint_target_ke,
        joint_target_kd,
        joint_target_pos,
        joint_target_vel,
        passive_damping,
    )
    passive = spring_damper_stiffness(0.0, passive_damping[dof], 0.0, dt)
    drive_tau[dof] = hold_drive_row(rows, row, row // max_dofs, dof, u0, K, passive, joint_effort_limit[dof])


class TreeFactor:
    """The tree factor of one solver (see the module docstring); the solver holds it as ``factor``.

    Built in the solver's construction order: the cadence, the cache and the augmented state here, then
    :meth:`kinematic_state` and :meth:`apply_kinematic_state`, :meth:`plan` once the forest plan and the row laws
    exist, :meth:`allocate_common`, :meth:`scatter_armature`, :meth:`install` and :meth:`plan_walk`.

    Args:
        model: The solver's model.
        options: The solver's physics options (:class:`.configuration.Options`).
    """

    def __init__(self, model, options):
        self.model = model
        self.options = options
        self.forest = None
        self.laws = None
        self._arrays = SolverArrays(self, model.device)
        self.mass_update_requested = wp.zeros(model.articulation_count, dtype=wp.int32, device=model.device)
        # The step's clock on the device (factor.kernels.Clock): the refresh cadence and the duration's change replay
        # with the device's own step index in a captured graph. The first step refreshes.
        clock = Clock()
        clock.step, clock.interval, clock.cadence = 0, int(options.update_mass_matrix_interval), 1
        self.clock = wp.array([clock], dtype=Clock, device=model.device)
        # The last step's duration on the host, which the contact forces' export reads (sweep_plan.forces).
        self.last_step_dt = None
        # The held tree's cache of next-step dynamics.
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
        # The effective armature by coordinate (scatter_armature): the walk's diagonal and the drives' K-free mass.
        self.dof_armature = wp.zeros(max(1, int(model.joint_dof_count)), dtype=wp.float32, device=model.device)
        self.diagonal_inverse_mass = None
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

    def kinematic_state(self, *, sparse_velocity_limits: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Derive the kinematic masks and effective joint armature from the body flags, changing nothing.

        The coordinate mask is the rows' prescribed coordinates (:data:`.joint_table.JOINTS`): none takes a row on
        it alone.

        Args:
            sparse_velocity_limits: Whether the native sparse factor carries velocity-limit rows.

        Returns:
            ``(armature, joint_mask, dof_mask)`` for :meth:`apply_kinematic_state`.

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
        return armature, joint_mask, dof_mask

    def apply_kinematic_state(self, kinematic: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        """Publish derived kinematic masks and armature in place, where captured graphs read them."""
        armature, joint_mask, dof_mask = kinematic
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
        self.walk_trees = self._walk_without_trees()

    def _walk_without_trees(self) -> WalkTrees:
        """The walk's view until :meth:`plan_walk` holds a tree, and where it holds none: the predictor's border reads
        it where no tree could slip at construction, and clears its slip counts. Its coordinate ranges are the forest's
        and its data has no row, so a decided step there (a limit set after construction) finds no tree, keeps its trial
        and flags the border's status (``constraint_capacity_status()["slip_frames"]``), never reading past an array."""
        model, forest, device = self.model, self.forest, self.model.device
        view = WalkTrees()
        view.data = wp.zeros((0, WALK_COORDINATE), dtype=wp.float32, device=device)
        view.link = wp.zeros((1, WALK_LINK), dtype=wp.float32, device=device)
        view.now = wp.zeros(1, dtype=wp.spatial_vector, device=device)
        view.now_link = wp.zeros((1, WALK_LINK), dtype=wp.float32, device=device)
        view.origin = view.now_origin = wp.zeros(1, dtype=wp.vec3, device=device)
        view.joint_art = wp.zeros(1, dtype=wp.int32, device=device)
        view.joint_start = model.articulation_start
        view.joint_end = forest.articulation_joint_end
        view.dof_start = forest.articulation_dof_start
        view.dofs = forest.articulation_dof_count
        view.dof_body = view.body_joint = wp.zeros(1, dtype=wp.int32, device=device)
        view.du = wp.zeros(1, dtype=wp.float32, device=device)
        view.release = release_view(max(int(model.articulation_count), 1), 0, 1, device)
        view.first_art = wp.full(1, -1, dtype=wp.int32, device=device)
        view.cls = wp.zeros(1, dtype=wp.int32, device=device)
        return view

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

        def _zeros(count=n_dofs):
            return self._arrays.placeholder(wp.float32, (count,))

        # Upstream's passive springs: stiffness per DOF, rest configuration per coordinate.
        spring_k = getattr(model, "joint_stiffness", None)
        spring_ref = getattr(model, "joint_rest_q", None)
        if spring_k is None or spring_ref is None:
            self.passive_spring_stiffness = _zeros()
            self.passive_spring_ref = _zeros(int(model.joint_coord_count))
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
        """The bilateral blocks' tables (the closure, :func:`.factor.closure.close_block`).

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
        """The bilateral blocks (:data:`.factor.fields.Preelim`) the one writer's block phase closes."""
        return field_group.bind(Preelim, factor=self)

    def seed_blocks(self, *, first, rhs, impulses) -> None:
        """The closure's seed, once the warm start has gathered the rows' impulses: each block's kept rows take
        ``-r_Q`` (:func:`.factor.closure.seed_blocks_kernel`); the one writer closed the blocks and corrected the
        other rows."""
        wp.launch(
            seed_blocks_kernel,
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

    def install(self, solver) -> None:
        """Hold the dynamics (:func:`.held_tree.install`); the drive rows' preparation overlaps FK/ID on a stream of
        its own."""
        from . import held_tree  # noqa: PLC0415

        self.held = held_tree.install(solver)
        model = self.model
        self.dynamics_stream = wp.Stream(model.device) if self.parallel_drives and model.device.is_cuda else None

    def record(self, classes, placement) -> dict:
        """The dynamics' route record (``routes.dynamics``), the packed rows ``placement``'s."""
        forest, laws = self.forest, self.laws
        return {
            "dynamics": {
                "compact_diagonal_mass_dofs": json_value(classes.diagonal_size),
                "diagonal_mass_dofs": sorted(int(size) for size in forest.diagonal_mass_sizes),
                # The rows' graph's vertices: how many coordinate trees hold each number of coordinates.
                "tree_sizes": {str(size): int(n) for size, n in enumerate(np.bincount(forest.tree_size)) if n},
                # The matrix-free sweeps' dense rows packed by support (packed_rows), and whether they serve the step
                # alone.
                "packed_rows": {
                    "support": int(placement.packed_support) if classes.mf_sweeps else None,
                    "dense": bool(placement.packed_dense),
                    "response_only": bool(placement.packed_response),
                    "reason": placement.packed_response_reason,
                },
                "parallel_augmented_drive_topology": bool(self.parallel_drives),
                "preelimination": bool(self.eliminates),
                "mimic_rows": int(laws.mimic_count or 0),
                "link_rows": int(laws.link_count or 0),
                "prescribed_response": bool(forest.has_prescribed_response),
            }
        }

    # ---------------------------------------------------------------------------------------------------------------
    # Notifications
    # ---------------------------------------------------------------------------------------------------------------

    def notify_state_changed(self, articulations: wp.array | None = None) -> None:
        """Invalidate the cached kinematics of ``articulations`` (``None``: every articulation)."""
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

    def reset_cache(self, world_mask: wp.array | None) -> None:
        """Invalidate the FK/ID cache of the worlds ``world_mask`` selects. A step from another state than the one the
        cache was derived from needs nothing here: the repair's stamps name the bank each tree's cache holds."""
        wp.launch(
            invalidate_articulation_fk_id_cache,
            dim=self.model.articulation_count,
            inputs=[world_mask, self.forest.art_to_world],
            outputs=[self.fk_id_cache_valid],
            device=self.model.device,
        )

    # ---------------------------------------------------------------------------------------------------------------
    # The step
    # ---------------------------------------------------------------------------------------------------------------

    # Stages 1-3 (:func:`.step_plan._stages` chooses which run): the drives' K and u0 (on the dynamics stream where
    # they overlap FK/ID), the prescaled velocity, FK/ID, the mass-update mask and the factor refresh, the walk's
    # refresh, which advances the step's clock, then the predictor and the free roots' corrections. joint_tau holds
    # only u0: the predictor adds the current external, control and bias contributions.

    def prescale_velocity(self, joint_qd: wp.array, prescaled_joint_qd: wp.array) -> None:
        """Write every DOF's PhysX-prescaled joint velocity in one launch, without a separate copy."""
        model = self.model
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

    def fk_id(self, state_in, state_out, qd: wp.array, dt: float) -> None:
        """Forward kinematics and inverse-dynamics terms, at the predictor's velocity ``qd`` (prescaled when
        limited)."""
        self.held.begin(state_in, self, qd, dt, state_out=state_out)

    def prepare_drives(
        self, state_in, control, dt: float, v_hat: wp.array, v_out: wp.array, dofs: wp.array | None = None
    ) -> None:
        """Launch the augmented drives of ``dofs`` (:func:`prepare_augmented_drives`; ``None``, every coordinate's
        without a list), apart from FK/ID; ``v_hat`` and ``v_out`` are last step's. Beside the one-coordinate class, whose
        predict decides its own, ``dofs`` are the others (:attr:`.held_tree.HeldPlan.drive_dofs`)."""
        model = self.model
        wp.launch(
            prepare_augmented_drives,
            dim=model.joint_dof_count if dofs is None else dofs.shape[0],
            inputs=[
                dofs,
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

    def drive_rows(self, v_hat: wp.array, v_out: wp.array) -> DriveRows:
        """What a step's drives write (:func:`.factor.fields.hold_drive_row`); ``v_hat`` and ``v_out`` are last step's."""
        return field_group.bind(DriveRows, factor=self, v_hat=v_hat, v_out=v_out)

    def refresh(self, dt: float) -> None:
        """The mass-update mask, the clock's cadence with the requests (which it consumes; the drives' have joined),
        then the held tree's factor refresh."""
        model = self.model
        wp.launch(
            build_mass_update_mask,
            dim=model.articulation_count,
            inputs=[self.clock, dt],
            outputs=[self.mass_update_requested, self.mass_update_mask],
            device=model.device,
        )
        self.held.refresh(self)

    def predict(self, state_in, control, qd: wp.array, dt: float) -> None:
        """Stage 3: ``v_hat = qd + h H^-1 (tau - c)``. Each tree's launch holds the drives' saturation decision and,
        where it moved a drive, the decided step by the border on the walk (:data:`.tree_kinetic_state.SATURATE_DRIVES`,
        :func:`.tree_kinetic_state.border_solve`); the factor keeps every ``K``, so no saturation ever asks the refresh
        for a factor."""
        self.held.predict(state_in, self, control, qd, dt)

    def correct_free_roots(self, state_in, qd: wp.array, dt: float, v_hat: wp.array) -> None:
        """Apply canonical gyro physics to the free roots after the held prediction, where a free root and a free
        rigid body exist."""
        model, forest = self.model, self.forest
        # Keep v_hat on jcalc_integrate's free-root convention. The compact launch excludes every non-root joint; a
        # live device mask handles runtime kinematic changes without changing graph topology.
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

    def damp_spin(self, dt: float, v_hat: wp.array) -> None:
        """Damp the free roots' spin, where the options damp it."""
        model, forest = self.model, self.forest
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

    def finish(self, state_in, state_out, dt: float) -> None:
        """Stage 7: ``qdd`` from the solved velocity, integrate the coordinates and publish the output state (with the
        next step's geometry where the clock refreshes it)."""
        self.held.finish(state_in, self, state_out, dt)

    # ---------------------------------------------------------------------------------------------------------------
    # The walk on the device (FL3.3): the converted class's factor as the one writer and the leave read it
    # ---------------------------------------------------------------------------------------------------------------

    def plan_walk(self, articulations: np.ndarray) -> None:
        """Hold the walk's data for ``articulations`` (the converted class's, every responding tree): each coordinate's
        :data:`.factor.walk.WALK_COORDINATE` floats, each body's :data:`.factor.walk.WALK_LINK` floats, each tree's
        walking class (:attr:`walk_classes`, :attr:`walk_class`: its path and join slots, :class:`.tree_scan.WalkClass`,
        which every walk reads from its classes' tables in constant memory), the trees' lane group
        (:func:`.factor.walk.walk_work`), and the step's geometry the finish publishes (:attr:`WalkTrees.now`,
        :attr:`WalkTrees.now_link`): the rows' seeds enter there. One launch on the factor's own refreshes
        (:meth:`walk_step`, the classes' refresh) takes that geometry as the factor's frames and runs the inward pass, so
        the walk is the factor ``L`` refreshes, to rounding.

        Raises:
            ValueError: For a tree whose bodies do not come after their parents, or that holds more than
                :data:`.factor.walk.WALK_JOIN_SLOTS` joins at once.
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
        # The drives' K the refresh adds where a coordinate has a drive row.
        self.walk_drive_K = self.aug_row_K if self.parallel_drives else self._arrays.placeholder(wp.float32)
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
        self.walk_class_host = walk_class
        # The refresh's join slots are its class's (WalkClass.slots): a parent waits in one for the lifted inertia of
        # each child other than the joint after it.
        held = max((walk.slots[1] for walk in classes), default=0)
        if held > WALK_JOIN_SLOTS:
            raise ValueError(f"The walk holds at most {WALK_JOIN_SLOTS} joins of a tree at once, a tree needs {held}")
        # The refresh's launch (walk_work): on CUDA a lane a body of a batch and a join slot, the trees in flight
        # those with a chain to walk (a free rigid body leaves its group after the batch's first body).
        bodies = max([1, *(len(walk.parent) for walk in classes)])
        sizes = np.array([int(end[a]) - int(start[a]) for a in arts.tolist()], dtype=np.int64)
        walked = int(np.count_nonzero(sizes > 1)) or int(arts.size)
        self.walk_shape = execution.shape(walk_work(walked, bodies, held), memory.device(device))
        self.walk_refresh = walk_step_kernel(self.walk_classes)
        # The one flag every walk reader's text is chosen by: whether a record (and a step column) can have a side.
        self.walk_side = both
        self.walk_data = wp.zeros(((2 if both else 1) * dofs, WALK_COORDINATE), dtype=wp.float32, device=device)
        self.walk_link = wp.zeros((joints, WALK_LINK), dtype=wp.float32, device=device)
        # The step's geometry, which the held tree's finish publishes with its motion columns (every step, no launch
        # of the walk's): each coordinate's column at its body's frame and each body's link there. A seed enters as J
        # there, and a refresh takes it as the factor's frames.
        # A model with a column of two halves keeps the finish's columns at each body's frame; else the step's columns
        # are S's own (WALK_NOW).
        self.walk_now = wp.zeros(dofs if both else 0, dtype=wp.spatial_vector, device=device)
        self.walk_now_link = wp.zeros((joints, WALK_LINK), dtype=wp.float32, device=device)
        # Each articulation's world origin at the factor's frames (the refresh's copy of the step's): the frame points
        # are the articulation's own, relative to its origin, so a world point meets them through it.
        self.walk_origin = wp.zeros(max(int(model.articulation_count), 1), dtype=wp.vec3, device=device)
        view = WalkTrees()
        view.data, view.link = self.walk_data, self.walk_link
        view.now = self.walk_now if both else self.joint_S_s
        view.now_link = self.walk_now_link
        view.origin, view.now_origin = self.walk_origin, self.articulation_origin
        view.joint_art = model.joint_articulation
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
        # The responding articulation whose coordinates start at each global coordinate (-1 elsewhere), by which a
        # matrix-free sweep finds a free body's tree.
        first_art = np.full(dofs, -1, dtype=np.int32)
        starts = forest.articulation_dof_start.numpy()
        responding = np.flatnonzero(forest.articulation_response_dof_count.numpy() > 0)
        first_art[starts[responding]] = responding
        view.first_art = wp.array(first_art, dtype=wp.int32, device=device)
        view.cls = self.walk_class
        self.walk_trees = view
        self.plan_leave()

    def plan_leave(self, nodes: tuple[tuple[int, ...], ...] | None = None) -> None:
        """The step's one leave of every walking tree (:func:`.factor.walk.walk_leave_kernel`), the release compiled where a tree
        can slip, reading each class's coordinate ``c`` at ``node(c)`` of its tree's changes: ``nodes`` (each class's,
        a sweep's own layout), or the walk's own coordinates."""
        classes = getattr(self, "walk_classes", ())
        self.walk_leave = walk_leave_kernel(classes, self.walk_slip_capacity > 0, nodes) if classes else None
        if classes:
            self.leave_shape = execution.shape(leave_work(self.walk_arts.shape[0]), memory.device(self.model.device))

    def leave(self, v_out: wp.array, du: wp.array | None = None, du_base: wp.array | None = None) -> None:
        """The step's one leave (:func:`.factor.walk.walk_leave_kernel`): every walking tree's velocity change walked down onto
        ``v_out`` (``v += P^T L^-T R^T du``): coordinate ``c``'s change is ``du[du_base[art] + node(c)]``, by default
        the walk's own :attr:`WalkTrees.du` at the tree's coordinates, which the sweeps add into from zero each
        step."""
        if self.walk_leave is None:
            return
        lanes = self.leave_shape.lanes
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
            block_dim=self.leave_shape.block,
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
        :data:`.factor.walk.RELEASE_SLIPS`. A tree's border holds that many; 0 where none can slip, so a consumer compiles no
        release."""
        slips = self.slips_by_tree(articulations)
        return min(int(slips.max(initial=0)), RELEASE_SLIPS)

    def walk_view(self) -> WalkTrees:
        """The walk's view (:class:`.factor.walk.WalkTrees`) the one writer, the leave and the audit read."""
        return self.walk_trees

    def walk_step(self, state, dt: float) -> None:
        """The walk's refresh (:func:`.factor.walk.walk_step_kernel`, a lane group per held tree), at the poses the held tree
        published in ``state.body_q``, on the trees the refresh just factored (``mass_update_mask``, which the device
        may set on any step): the step's geometry the finish published (:attr:`WalkTrees.now`) becomes the factor's
        frames and the group runs the inward pass. A free rigid body is a one-body tree: its walk is its six
        coordinates' factor. A tree that does not refresh takes nothing here. Its first thread advances the step's clock
        past this step (``dt``); a model where no tree walks advances it alone (:meth:`advance_clock`)."""
        model = self.model
        lanes = self.walk_shape.lanes
        wp.launch(
            self.walk_refresh,
            dim=(self.walk_arts.shape[0], lanes),
            inputs=[
                self.walk_arts,
                lanes,
                self.mass_update_mask,
                model.articulation_start,
                self.forest.articulation_joint_end,
                self.walk_class,
                model.joint_child,
                model.joint_qd_start,
                state.body_q,
                model.body_mass,
                model.body_inertia,
                self.dof_armature,
                self.drive_row_by_dof,
                self.walk_drive_K,
                self.walk_trees.now,
                self.walk_now_link,
                self.joint_com_offset,
                self.articulation_origin,
                self.clock,
                dt,
            ],
            outputs=[self.walk_data, self.walk_link, self.walk_origin],
            block_dim=self.walk_shape.block,
            device=model.device,
        )

    def advance_clock(self, dt: float) -> None:
        """Advance the step's clock past this step (``dt``) where no tree walks, so no refresh carries it."""
        wp.launch(advance_clock, dim=1, inputs=[self.clock, dt], device=self.model.device)

    def two_halves(self, coordinates) -> bool:
        """Whether a coordinate of ``coordinates`` has a step column of two halves at its frame (:data:`.factor.walk.WALK_BOTH`, from
        its joint's kind): the one rule a walk reader's text is chosen by. :attr:`walk_side` is it over the walked
        coordinates; a held group's finish reads it over its own (:class:`.tree_kinetic_owner.TreeKineticState`)."""
        if self.walk_half_host is None:
            return False
        return bool(np.any(self.walk_half_host[np.asarray(coordinates, dtype=np.int64)] == WALK_BOTH))

    def diagonal(self, articulation: int) -> np.ndarray:
        """What this factor adds to ``articulation``'s mass matrix: each coordinate's effective armature ``R``
        (kinematic coordinates included) and the drive stiffness ``K`` it last refreshed with."""
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
        (:class:`.factor.walk.TreeWalk`): its local frames from the model in float64 (:func:`.factor.walk.local_frames`), its diagonal this
        factor's (:meth:`diagonal`), its inward pass in ``dtype``. No step calls it: the tests hold it to the factor
        (Z_UNION §6, Z0)."""
        frames = local_frames(self.model, joint_q, articulation)
        return TreeWalk(*frames, self.diagonal(articulation), dtype=dtype)
