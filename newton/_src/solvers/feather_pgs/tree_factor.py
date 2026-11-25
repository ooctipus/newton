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
from ...core.host_unique import row_ids
from ...sim import BodyFlags, JointType, ModelBuilder, ModelFlags
from ...sim.articulation import eval_articulation_ik
from . import field_group, host_index, walk_rows
from .factor.closure import seed_blocks_kernel
from .factor.fields import (
    ARTICULATION_ORIGIN,
    ARTICULATION_WORK,
    AUGMENTED_DRIVE_ROWS,
    PREELIMINATION,
    PREELIMINATION_PLAN,
    STATE_AUX,
    DriveRows,
    Preelim,
    drive_terms,
    hold_drive_row,
)
from .factor.kernels import (
    TURN_BLOCK,
    Clock,
    advance_clock,
    build_mass_update_mask,
    damp_free_root_spin,
    follow_prescribed_bodies,
    forget_prescribed_bodies,
    invalidate_articulation_fk_id_cache,
    invalidate_fk_id_cache_for_articulations,
    invalidate_fk_id_cache_for_gravity,
    publish_prescribed_bodies,
    take_gravity,
    turn_forces,
    turn_moments,
)
from .factor.walk import (
    RELEASE_SLIPS,
    WALK_COORDINATE,
    WALK_JOIN_SLOTS,
    WALK_LINK,
    TreeLanes,
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
from .path_selection import json_value
from .row_laws import LINK_ROWS
from .spatial_laws import motor_line, spring_damper_stiffness
from .tree_scan import (
    WALK_ANGULAR,
    WALK_BOTH,
    WALK_LINEAR,
    WalkClass,
)

# FeatherPGS takes no gradients: its kernels compile no adjoints.
wp.set_module_options({"enable_backward": False})

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
    joint_velocity_limit: wp.array[float],
    unauthored_velocity_limit: float,
    passive_damping: wp.array[float],
    max_dofs: int,
    dt: float,
    rows: DriveRows,
    drive_tau: wp.array[float],
):
    """The augmented drive of each coordinate of ``dofs`` (those no one-coordinate tree's predict decides; without a
    list, the thread's own), from the immutable DOF-to-row topology: ``u0`` and ``K`` (:func:`.factor.fields.drive_terms`, its
    passive damper's with it) and its motor's line (:func:`.spatial_laws.motor_line`, from its effort and velocity
    limits) held on its row (:func:`.factor.fields.hold_drive_row`), which requests the articulation's factor refresh
    wherever ``K`` changed, a drive whose ``K`` fell to zero included."""
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
    limit = joint_effort_limit[dof]
    line = motor_line(limit, joint_velocity_limit[dof], unauthored_velocity_limit)
    drive_tau[dof] = hold_drive_row(rows, row, row // max_dofs, dof, u0, K, passive, limit, line, joint_qd[dof])


class TreeFactor:
    """The tree factor of one solver (see the module docstring); the solver holds it as ``factor``.

    Built in the solver's construction order: the cadence, the cache and the augmented state here, then
    :meth:`kinematic_state` and :meth:`apply_kinematic_state`, :meth:`plan` once the forest plan and the row laws
    exist, :meth:`allocate_common`, :meth:`scatter_armature`, :meth:`install` and :meth:`plan_walk`.

    Args:
        model: The solver's model.
        options: The solver's physics options (:func:`.configuration.check_options`).
    """

    def __init__(self, model, options):
        self.model = model
        self.options = options
        # The joint velocity limit the builder writes where none is authored: no motor line (spatial_laws.motor_line).
        self.unauthored_velocity_limit = float(ModelBuilder.JointDofConfig().velocity_limit)
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
        self._plan_prescribed_roots(model)
        # The augmented state is allocated here rather than on the first step: a wp.zeros() issued inside a CUDA
        # graph capture becomes a memset node that replays every launch.
        if model.body_count:
            self._arrays.allocate(STATE_AUX, dof=model.joint_dof_count, body=model.body_count)
        # Each body's gravity compensation fraction: the model's attribute (the kernels read its array), or none.
        compensation = getattr(model, "body_gravity_compensation", None)
        if compensation is None:
            compensation = wp.zeros(max(1, int(model.body_count)), dtype=wp.float32, device=model.device)
        self.body_gravity_compensation = compensation
        # Each world's gravity as the cached dynamics hold it (the bias wrench of each body is its own world's): a
        # model notification compares the model's with it and repairs the worlds whose vector changed.
        self.held_gravity = wp.clone(model.gravity)
        self.gravity_changed = wp.zeros(model.gravity.shape[0], dtype=wp.int32, device=model.device)
        self.kinematic_joint_mask = wp.zeros(model.joint_count, dtype=wp.int32, device=model.device)
        # The articulated joints of two or more turns (free, ball, universal, a D6's) by tree: each one's gyroscopic
        # moment on its body as coordinate forces (turn_forces, J^T of the moment) on the turn coordinates of the joint
        # and of its ancestors. The predictor adds turn_tau to each coordinate's force; it stays zero on every other.
        self._turn_trees(model)
        self.kinematic_dof_mask = wp.zeros(model.joint_dof_count, dtype=wp.int32, device=model.device)
        self.kinematic_dof_mask_host = None
        self.armature = None
        # The effective armature by coordinate (scatter_armature): the walk's diagonal and the drives' K-free mass.
        self.dof_armature = wp.zeros(max(1, int(model.joint_dof_count)), dtype=wp.float32, device=model.device)
        self.diagonal_inverse_mass = None
        self.eliminates = False
        self.block_capacity = 0
        self.block_entries = 0
        self.block_count = 0
        # Where the closure keeps a block's staged rows, and their lines where that is global memory (plan_blocks).
        self.closure_place = memory.Place.SHARED
        self.closure_lines = self._arrays.placeholder(wp.float32, (1, 1))
        self.held = None
        self.dynamics = None
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

    def kinematic_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Derive the kinematic masks and effective joint armature from the body flags, changing nothing.

        The coordinate mask is the rows' prescribed coordinates (:data:`.joint_table.JOINTS`): none takes a row on
        it alone.

        Returns:
            ``(armature, joint_mask, dof_mask)`` for :meth:`apply_kinematic_state`.

        Raises:
            RuntimeError: If the flags make a DOF of a prescribed-response articulation dynamic, which needs a new
                solver.
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

    def _plan_prescribed_roots(self, model) -> None:
        """The roots whose prescribed state the body state gives (:meth:`follow_prescribed`): every root joint with
        coordinates (its parent the world) in an articulation, where the body flags make one of them kinematic at
        construction; each one's body state as published, the model's initial one until a step publishes its own, and
        each articulation's root. A model without a kinematic root follows none (``prescribes``): a root its flags make
        kinematic later keeps its joint state."""
        self.prescribes = False
        if not (model.joint_count and model.body_count and model.articulation_count):
            return
        parent, child = model.joint_parent.numpy(), model.joint_child.numpy()
        qd_start = model.joint_qd_start.numpy()
        articulation = model.joint_articulation.numpy()
        roots = np.flatnonzero((parent < 0) & (qd_start[1:] > qd_start[:-1]) & (articulation >= 0))
        if not np.any(model.body_flags.numpy()[child[roots]] & int(BodyFlags.KINEMATIC)):
            return
        device = model.device
        self.prescribes = True
        root_articulation = articulation[roots]
        root_of_articulation = np.full(model.articulation_count, -1, dtype=np.int32)
        root_of_articulation[root_articulation] = np.arange(len(roots), dtype=np.int32)
        starts = model.articulation_start.numpy()
        self.prescribed_joints = int(np.max(starts[root_articulation + 1] - starts[root_articulation]))
        self.prescribed_roots = wp.array(roots.astype(np.int32), dtype=wp.int32, device=device)
        self.prescribed_root_articulation = wp.array(root_articulation.astype(np.int32), dtype=wp.int32, device=device)
        self.prescribed_root_of_articulation = wp.array(root_of_articulation, dtype=wp.int32, device=device)
        self.prescribed_q = wp.array(model.body_q.numpy()[child[roots]], dtype=wp.transform, device=device)
        self.prescribed_qd = wp.array(model.body_qd.numpy()[child[roots]], dtype=wp.spatial_vector, device=device)
        self.prescribed_valid = wp.ones(len(roots), dtype=wp.int32, device=device)
        self.prescribed_written = wp.zeros(model.articulation_count, dtype=wp.bool, device=device)

    def follow_prescribed(self, state_in) -> None:
        """The step's entry: a kinematic root whose body state (``body_q``, ``body_qd``) was written since the solver
        published it takes its joint coordinates and speeds from it, by Newton's inverse kinematics of its joint, and
        its articulation's cached kinematics are invalidated; any other root keeps its joint state. So a kinematic
        body is prescribed by whichever state its user writes: its body state (maximal coordinates, as the collision
        reads it) or its joint state (with :meth:`notify_state_changed` or a reset)."""
        model, count = self.model, self.prescribed_roots.shape[0]
        wp.launch(
            follow_prescribed_bodies,
            dim=count,
            inputs=[
                self.prescribed_roots,
                self.prescribed_root_articulation,
                model.joint_child,
                self.kinematic_joint_mask,
                state_in.body_q,
                state_in.body_qd,
                self.prescribed_q,
                self.prescribed_qd,
                self.prescribed_valid,
            ],
            outputs=[self.prescribed_written, self.fk_id_cache_valid],
            device=model.device,
        )
        wp.launch(
            eval_articulation_ik,
            dim=(count, self.prescribed_joints),
            inputs=[
                model.articulation_start,
                model.articulation_end,
                model.articulation_count,
                self.prescribed_written,
                self.prescribed_root_articulation,
                state_in.body_q,
                state_in.body_qd,
                model.body_com,
                model.joint_type,
                model.joint_parent,
                model.joint_child,
                model.joint_X_p,
                model.joint_X_c,
                model.joint_axis,
                model.joint_dof_dim,
                model.joint_q_start,
                model.joint_qd_start,
                model.body_flags,
                int(BodyFlags.KINEMATIC),
            ],
            outputs=[state_in.joint_q, state_in.joint_qd],
            device=model.device,
        )

    def publish_prescribed(self, state_out) -> None:
        """The step's exit: record each root body's published state, which :meth:`follow_prescribed` tells a write
        from."""
        wp.launch(
            publish_prescribed_bodies,
            dim=self.prescribed_roots.shape[0],
            inputs=[self.prescribed_roots, self.model.joint_child, state_out.body_q, state_out.body_qd],
            outputs=[self.prescribed_q, self.prescribed_qd, self.prescribed_valid],
            device=self.model.device,
        )

    def forget_prescribed(self, world_mask=None, articulations=None) -> None:
        """Let the joint state of the roots of ``articulations`` (a joint-state write), or else of the worlds
        ``world_mask`` selects (a reset; ``None``: every world), stand at the next step."""
        if not self.prescribes:
            return
        wp.launch(
            forget_prescribed_bodies,
            dim=self.prescribed_roots.shape[0] if articulations is None else articulations.shape[0],
            inputs=[
                world_mask,
                articulations,
                self.prescribed_root_of_articulation,
                self.prescribed_root_articulation,
                self.forest.art_to_world,
            ],
            outputs=[self.prescribed_valid],
            device=self.model.device,
        )

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

    def plan_blocks(self, forest, laws, pins) -> None:
        """The bilateral blocks' tables (the closure, :func:`.factor.closure.close_world`).

        A block holds a linked component's constraints: its trees' mimics and its links' six rows each. Its buffers
        are sized once from the model's largest block (CUDA-graph-safe); its slots and its factor are rebuilt every
        step on the device, since the rows' slots and ``J_B`` depend on the state. Its rows' entries lie in its
        trees' world coordinates, its span (the closure stages the widest span, :attr:`block_entries`), in the one
        writer's block stage, or where that declines them (:func:`.walk_rows.closure_place`; the test pin
        ``closure='global'``) in a line of global memory a world (:attr:`closure_lines`).
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
        # Each block's span: the world coordinates of its trees that respond, first to end.
        offset = forest.articulation_world_dof_offset.numpy()
        responds = held & (forest.response_dof_count > 0)
        first = np.full(n, np.iinfo(np.int32).max, dtype=np.int64)
        end = np.zeros(n, dtype=np.int64)
        np.minimum.at(first, art_to_block[responds], offset[responds])
        np.maximum.at(end, art_to_block[responds], offset[responds] + forest.response_dof_count[responds])
        first = np.minimum(first, end)
        self.block_entries = max(int((end - first).max()), 1)
        self.closure_place = walk_rows.closure_place(capacity, self.block_entries, model.device.arch)
        if pins.closure == "global":
            self.closure_place = memory.Place.GLOBAL
        if self.closure_place is not memory.Place.SHARED:
            lines = (forest.world_count, capacity * self.block_entries)
            self.closure_lines = wp.zeros(lines, dtype=wp.float32, device=model.device)
        self._arrays.upload(
            PREELIMINATION_PLAN,
            _preelim_world=world_of[by_world],
            _preelim_start=bounds,
            _preelim_entry=entries[order],
            _preelim_world_start=host_index.offsets(world_blocks, dtype=np.int32),
            _preelim_span=np.stack((first, end), axis=1).astype(np.int32),
        )

    def blocks(self):
        """The bilateral blocks (:data:`.factor.fields.Preelim`) the one writer's block phase closes."""
        return field_group.bind(Preelim, factor=self)

    def seed_blocks(self, first, rhs, impulses) -> None:
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
        """Allocate the mass-update mask, the compact diagonal's inverse mass and the articulation origins."""
        model, forest, arrays = self.model, self.forest, self._arrays
        if model.joint_count:
            arrays.allocate(ARTICULATION_WORK, articulation=model.articulation_count)
        else:
            arrays.release(ARTICULATION_WORK)
        self.diagonal_inverse_mass = (
            wp.zeros_like(model.joint_qd) if forest.diagonal_mass_sizes else arrays.placeholder(wp.float32)
        )
        if not model.articulation_count or not model.joint_count:
            self.articulation_origin = None
            return

        arrays.allocate(ARTICULATION_ORIGIN, articulation=model.articulation_count)

    def scatter_armature(self) -> None:
        """Copy the effective armature (DOF-ordered) to the coordinates' (:attr:`dof_armature`)."""
        if self.armature is not None and self.armature.size:
            self.dof_armature.assign(self.armature.astype(np.float32))

    def install(self, solver) -> None:
        """Hold the dynamics (:func:`.held_tree.install`); the drive rows' preparation overlaps FK/ID beside the step's
        stream (:class:`.core.execution.Beside`)."""
        from . import held_tree  # noqa: PLC0415

        self.held = held_tree.install(solver)
        model = self.model
        self.dynamics = execution.Beside(model.device) if self.parallel_drives and model.device.is_cuda else None

    def record(self, classes, placement) -> dict:
        """The dynamics' route record (``routes.dynamics``), the packed rows ``placement``'s."""
        forest, laws = self.forest, self.laws
        return {
            "dynamics": {
                "compact_diagonal_mass_dofs": json_value(classes.diagonal_size),
                "diagonal_mass_dofs": sorted(int(size) for size in forest.diagonal_mass_sizes),
                # The rows' graph's vertices: how many coordinate trees hold each number of coordinates.
                "tree_sizes": {str(size): int(n) for size, n in enumerate(np.bincount(forest.tree_size)) if n},
                # The dense rows packed by support (packed_rows), and whether they serve the step alone.
                "packed_rows": {
                    "support": int(placement.packed_support) if placement.packs_rows else None,
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
        """Invalidate the cached kinematics of ``articulations`` (``None``: every articulation); their joint state was
        written, so a kinematic root's stands (:meth:`forget_prescribed`)."""
        self.forget_prescribed(articulations=articulations)
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
        the caches the changed inputs feed, publish ``kinematic`` (:meth:`kinematic_state`, or ``None``) and ask every
        tree's factor again on an inertial change (the kernels read mass, inertia and centre of mass live). A change
        of the model's properties (``MODEL_PROPERTIES``) reaches the cached dynamics through gravity alone, a vector a
        world: the worlds whose vector changed are found on the device and only their trees' caches are invalidated
        (:meth:`_take_gravity`); the factors hold no gravity."""
        # The held tree's proof stores a moved joint frame as its new reference, so it runs after every check that
        # can still refuse the notification.
        held = self.held
        frames_proven = held.validate_model(flags)
        cache_inputs = int(flags) & int(
            ModelFlags.JOINT_PROPERTIES
            | ModelFlags.JOINT_DOF_PROPERTIES
            | ModelFlags.BODY_PROPERTIES
            | ModelFlags.BODY_INERTIAL_PROPERTIES
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
        if flags & ModelFlags.MODEL_PROPERTIES:
            self._take_gravity()
        if kinematic is not None:
            self.apply_kinematic_state(kinematic)
            self.scatter_armature()
            self.mass_update_requested.fill_(1)
        if flags & ModelFlags.BODY_INERTIAL_PROPERTIES and self.model.body_count:
            self.mass_update_requested.fill_(1)

    def _take_gravity(self) -> None:
        """Hold the model's gravity and invalidate the cached dynamics of the trees it changed: each articulation
        with a body in a world whose vector differs from the one held. Two launches, no readback: a captured graph
        replays them."""
        model = self.model
        wp.launch(
            take_gravity,
            dim=model.gravity.shape[0],
            inputs=[model.gravity],
            outputs=[self.held_gravity, self.gravity_changed],
            device=model.device,
        )
        if model.body_count:
            wp.launch(
                invalidate_fk_id_cache_for_gravity,
                dim=model.body_count,
                inputs=[self.gravity_changed, model.body_world, self.forest.body_to_articulation],
                outputs=[self.fk_id_cache_valid],
                device=model.device,
            )

    def reset_cache(self, world_mask: wp.array | None) -> None:
        """Invalidate the FK/ID cache of the worlds ``world_mask`` selects, where a kinematic root's joint state then
        stands (:meth:`forget_prescribed`). A step from another state than the one the cache was derived from needs
        nothing here: the repair's stamps name the bank each tree's cache holds."""
        self.forget_prescribed(world_mask=world_mask)
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
    # they overlap FK/ID), FK/ID, the mass-update mask and the factor refresh, the walk's
    # refresh, which advances the step's clock, then the predictor and the free roots' corrections. joint_tau holds
    # only u0: the predictor adds the current external, control and bias contributions.

    def fk_id(self, state_in, state_out, dt: float) -> None:
        """Forward kinematics and inverse-dynamics terms, at the input state's velocity."""
        self.held.begin(state_in, self, dt, state_out=state_out)

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
                model.joint_velocity_limit,
                self.unauthored_velocity_limit,
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

    def predict(self, state_in, control, dt: float) -> None:
        """Stage 3: ``v_hat = qd + h H^-1 (tau - c)`` at the input state's ``qd``. Each tree's launch holds the drives' saturation decision and,
        where it moved a drive, the decided step by the border on the walk (:data:`.tree_kinetic_state.SATURATE_DRIVES`,
        :func:`.tree_kinetic_state.border_solve`); the factor keeps every ``K``, so no saturation ever asks the refresh
        for a factor."""
        self.held.predict(state_in, self, control, dt)

    def _turn_trees(self, model) -> None:
        """The trees holding a joint of two or more turns, each one's turning joints in joint order with their bodies,
        their turns' first coordinate and count and their linear axes (:func:`.factor.kernels.turn_forces`), each tree
        joint's parent joint, and the stream the turns' forces run on beside the factor refresh."""
        turning = np.zeros(0, dtype=np.int32)
        starts = np.zeros(1, dtype=np.int32)
        parent_joint = np.full(max(1, int(model.joint_count)), -1, dtype=np.int32)
        if model.joint_count:
            dof_dim = model.joint_dof_dim.numpy()
            parent = model.joint_parent.numpy()
            tree = model.joint_articulation.numpy()
            articulated = tree >= 0
            child_joint = np.full(max(1, int(model.body_count)), -1, dtype=np.int32)
            child_joint[model.joint_child.numpy()[articulated]] = np.flatnonzero(articulated)
            parent_joint[: model.joint_count] = np.where(
                articulated & (parent >= 0), child_joint[np.maximum(parent, 0)], -1
            )
            turning = np.flatnonzero(articulated & (dof_dim[:, 1] > 1)).astype(np.int32)
            turning = turning[np.argsort(tree[turning], kind="stable")]
            _trees, first = np.unique(tree[turning], return_index=True)
            starts = np.append(first, turning.size).astype(np.int32)
        lin, ang = (dof_dim[turning, 0], dof_dim[turning, 1]) if turning.size else (turning, turning)
        qd_start = model.joint_qd_start.numpy() if model.joint_count else np.zeros(0, dtype=np.int32)
        dofs = np.stack([qd_start[turning] + lin, ang, lin], axis=1) if turning.size else turning
        # The coordinates the turns' moments reach, each with its joint and tree: a turning joint's turn coordinates and
        # its ancestors', each once a tree (turn_forces).
        reached = []
        for t in range(starts.size - 1):
            joints = set()
            for joint in turning[starts[t] : starts[t + 1]]:
                up = int(joint)
                while up >= 0:
                    joints.add(up)
                    up = int(parent_joint[up])
            for joint in sorted(joints):
                angular = qd_start[joint] + dof_dim[joint, 0]
                reached += [(dof, joint, t) for dof in range(angular, angular + dof_dim[joint, 1])]
        device = model.device
        self.turning_joints = wp.array(turning, dtype=wp.int32, device=device)
        self.turn_body = wp.array(
            model.joint_child.numpy()[turning] if turning.size else turning, dtype=wp.int32, device=device
        )
        self.turn_dofs = wp.array(np.asarray(dofs, dtype=np.int32).reshape(-1, 3), dtype=wp.vec3i, device=device)
        self.turn_tree_start = wp.array(starts, dtype=wp.int32, device=device)
        self.turn_parent_joint = wp.array(parent_joint, dtype=wp.int32, device=device)
        self.turn_coordinates = wp.array(
            np.asarray(reached, dtype=np.int32).reshape(-1, 3), dtype=wp.vec3i, device=device
        )
        self.turn_moments = wp.zeros(turning.size, dtype=wp.vec3, device=device)
        self.turn_tau = wp.zeros(max(1, int(model.joint_dof_count)), dtype=wp.float32, device=device)
        # The prediction waits for the turns' forces: their stream goes first beside the refreshes.
        self.turn_beside = execution.Beside(device, priority=-1)

    def turn_forces(self, state_in, dt: float) -> None:
        """The turning joints' gyroscopic moments as coordinate forces (from FK/ID's columns, for the prediction): each
        joint's moment at once, then each coordinate's sum of those it takes."""
        model = self.model
        wp.launch(
            turn_moments,
            dim=self.turning_joints.shape[0],
            inputs=[
                self.turning_joints,
                self.turn_body,
                self.turn_dofs,
                model.joint_X_c,
                self.kinematic_joint_mask,
                state_in.body_q,
                state_in.body_qd,
                model.body_com,
                model.body_mass,
                model.body_inertia,
                self.joint_S_s,
                state_in.joint_qd,
                self.dof_armature,
                dt,
            ],
            outputs=[self.turn_moments],
            block_dim=TURN_BLOCK,
            device=model.device,
        )
        wp.launch(
            turn_forces,
            dim=self.turn_coordinates.shape[0],
            inputs=[
                self.turn_tree_start,
                self.turning_joints,
                self.turn_parent_joint,
                self.kinematic_joint_mask,
                self.turn_coordinates,
                self.joint_S_s,
                self.turn_moments,
            ],
            outputs=[self.turn_tau],
            block_dim=TURN_BLOCK,
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
        arts = np.asarray(articulations, dtype=np.int32)
        start = model.articulation_start.numpy()
        end = forest.articulation_joint_end_host
        # The trees of more than one body first, then the one-body trees (free rigid bodies among them): the
        # refresh gives each part its own lane groups (TreeLanes), whole blocks apart; the leave's groups are one.
        single = end[arts] - start[arts] <= 1
        arts = np.concatenate([arts[~single], arts[single]]).astype(np.int32)
        qd_start = model.joint_qd_start.numpy()
        joint_parent, joint_child = model.joint_parent.numpy(), model.joint_child.numpy()
        joint_of = np.full(max(int(model.body_count), 1), -1, dtype=np.int64)
        joint_of[joint_child[joint_child >= 0]] = np.flatnonzero(joint_child >= 0)
        # Each joint's parent within its walked tree (-1: a root, and every joint outside the walk) and first coordinate.
        path = np.zeros((max(int(model.joint_count), 1), 2), dtype=np.int32)
        path[:, 0] = -1
        dof_body = np.full(max(int(model.joint_dof_count), 1), -1, dtype=np.int32)
        # Every joint of the walked trees, tree by tree: its tree's place in arts, its rank in the tree.
        tree, rank = host_index.expand(np.maximum(end[arts] - start[arts], 0))
        j0, j1 = start[arts][tree], end[arts][tree]
        j = j0 + rank
        up = np.where(joint_parent[j] >= 0, joint_of[np.maximum(joint_parent[j], 0)], -1)
        parent = np.where((j0 <= up) & (up < j1), up - j0, -1)
        unordered = np.flatnonzero(parent >= rank)
        if unordered.size:
            raise ValueError(
                f"The walk needs every parent before its children (articulation {arts[tree[unordered[0]]]})"
            )
        path[j, 0] = parent
        path[j, 1] = qd_start[j] - qd_start[j0]
        joint, dof = host_index.expand(qd_start[j + 1] - qd_start[j])
        dof_body[qd_start[j][joint] + dof] = rank[joint]
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
        walked[j] = True
        lin, ang = dims[:, 0].astype(np.int64), dims[:, 1].astype(np.int64)
        two = (ang > 0) & (
            ((kinds == int(JointType.D6)) & (lin > 0))
            | (np.isin(kinds, (int(JointType.FREE), int(JointType.DISTANCE))) & (path[: kinds.size, 0] >= 0))
        )
        # Each joint's linear coordinates, then its angular ones, joint by joint (a later write wins, as in order).
        joint, k = host_index.expand(lin + ang)
        half = np.where(k < lin[joint], WALK_LINEAR, np.where(two[joint], WALK_BOTH, WALK_ANGULAR))
        place = qd_start[: kinds.size][joint] + k
        inside = place < dofs  # a slice stops at the array's end
        self.walk_half_host[place[inside]] = half[inside]
        both = self.two_halves(np.flatnonzero(np.repeat(walked[: kinds.size], np.diff(qd_start[: kinds.size + 1]))))
        # The walking classes: each held tree's topology, which the step's one leave (walk_leave_kernel) and the
        # predictor's solve emit as its walk, and each articulation's class among them.
        classes, walk_class = self._walk_tables(arts, start, end, qd_start, path)
        self.walk_classes = classes
        self.walk_class = wp.array(walk_class, dtype=wp.int32, device=device)
        self.walk_class_host = walk_class
        # The refresh's join slots are its class's (WalkClass.slots): a parent waits in one for the lifted inertia of
        # each child other than the joint after it.
        held = max((walk.slots[1] for walk in classes), default=0)
        if held > WALK_JOIN_SLOTS:
            raise ValueError(f"The walk holds at most {WALK_JOIN_SLOTS} joins of a tree at once, a tree needs {held}")
        # The refresh's launch (walk_work): on CUDA a lane a body of a batch and a join slot, the articulated trees'
        # groups by their own count; a one-body tree's chain is its one body, a group of its own shape (TreeLanes).
        bodies = max([1, *(len(walk.parent) for walk in classes)])
        split = int(np.count_nonzero(~single))
        first = execution.shape(walk_work(split or 1, bodies, held), memory.device(device))
        second = execution.shape(walk_work(max(arts.size - split, 1), 1, 0), memory.device(device))
        self.walk_shape = TreeLanes(int(arts.size), split, first.lanes, second.lanes, first.block)
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
        depth = max(
            [1, *(sum(s.c >= 0 for s in walk.steps(False, c)) for walk in classes for c in range(walk.coordinates))]
        )
        view.du = wp.zeros(dofs, dtype=wp.float32, device=device)
        view.release = release_view(
            max(int(model.articulation_count), 1),
            self.walk_slip_capacity,
            depth if self.walk_slip_capacity else 1,
            device,
        )
        view.cls = self.walk_class
        self.walk_trees = view
        self.plan_leave()

    def plan_leave(self) -> None:
        """The step's one leave of every walking tree (:func:`.factor.walk.walk_leave_kernel`), the release compiled where a tree
        can slip, reading its tree's changes in the walk's coordinates."""
        classes = getattr(self, "walk_classes", ())
        self.walk_leave = walk_leave_kernel(classes, self.walk_slip_capacity > 0) if classes else None
        if classes:
            self.leave_shape = execution.shape(leave_work(self.walk_arts.shape[0]), memory.device(self.model.device))

    def leave(self, v_out: wp.array) -> None:
        """The step's one leave (:func:`.factor.walk.walk_leave_kernel`): every walking tree's velocity change walked down onto
        ``v_out`` (``v += P^T L^-T R^T du``): the walk's own :attr:`WalkTrees.du` at the tree's coordinates, which the
        sweeps add into from zero each step."""
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
                v_out,
            ],
            block_dim=self.leave_shape.block,
            device=self.model.device,
        )

    def launch_variants(self):
        """Each kernel the walk and the held tree launch, with its threads per block
        (:func:`~newton._src.core.execution.compile_ahead`)."""
        if self.walk_arts is not None:
            yield self.walk_refresh, self.walk_shape.block
            if self.walk_leave is not None:
                yield self.walk_leave, self.leave_shape.block
        for group in self.held.groups if self.held is not None else ():
            yield from group.launch_variants()

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
        # Each tree's count of limited coordinates, as running totals over its coordinate range (clipped as a slice).
        total = np.concatenate(([0], np.cumsum(limited, dtype=np.int64)))
        first = np.clip(starts[arts].astype(np.int64), 0, limited.size)
        last = np.clip(first + counts[arts].astype(np.int64), first, limited.size)
        return total[last] - total[first]

    def _walk_tables(self, arts, start, end, qd_start, path) -> tuple[tuple[WalkClass, ...], np.ndarray]:
        """The walking classes of the trees ``arts`` in the order they first occur, and each articulation's class (-1
        for one not walked): a tree's class is its bodies' parents, their first coordinates with the count after the
        last, and its coordinates' halves (:attr:`walk_half_host`), so trees of equal tables share one."""
        model = self.model
        walk_class = np.full(max(int(model.articulation_count), 1), -1, dtype=np.int32)
        if arts.size == 0:
            return (), walk_class
        j0, j1 = start[arts].astype(np.int64), end[arts].astype(np.int64)
        d0, d1 = qd_start[j0].astype(np.int64), qd_start[j1].astype(np.int64)
        # Equal tables have equal joint and coordinate counts: rank the tables of each (joints, coordinates) group.
        counts = np.stack((j1 - j0, d1 - d0), axis=1)
        group = row_ids(counts)
        key = np.full(arts.size, -1, dtype=np.int64)
        found = 0
        for g in range(int(group.max()) + 1):
            members = np.flatnonzero(group == g)
            joints, coordinates = (int(n) for n in counts[members[0]])
            body = j0[members, None] + np.arange(joints)
            rows = np.concatenate(
                (path[body, 0], path[body, 1], self.walk_half_host[d0[members, None] + np.arange(coordinates)]), axis=1
            )
            ids = row_ids(rows)
            key[members] = found + ids
            found += int(ids.max()) + 1
        # The classes in the order their first tree occurs in arts (a dict's insertion order).
        first = np.full(found, arts.size, dtype=np.int64)
        np.minimum.at(first, key, np.arange(arts.size))
        order = np.argsort(first, kind="stable")
        number = np.empty(found, dtype=np.int64)
        number[order] = np.arange(found)
        walk_class[arts] = number[key]
        classes = []
        for t in first[order].tolist():
            a, b, c, d = int(j0[t]), int(j1[t]), int(d0[t]), int(d1[t])
            classes.append(
                WalkClass(
                    tuple(path[a:b, 0].tolist()),
                    (*path[a:b, 1].tolist(), d - c),
                    tuple(self.walk_half_host[c:d].tolist()),
                )
            )
        return tuple(classes), walk_class

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
        wp.launch(
            self.walk_refresh,
            dim=self.walk_shape.threads,
            inputs=[
                self.walk_arts,
                *self.walk_shape.inputs(),
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
