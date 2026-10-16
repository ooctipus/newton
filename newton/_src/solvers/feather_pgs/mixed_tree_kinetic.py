# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compose complete tree caches over retained heterogeneous factor groups."""

import os

import numpy as np
import warp as wp

from ...sim import JointType
from .device_proof import DeviceProof
from .franka_kinetic_factor import get_kernel as geometric_factor
from .sparse_factor import bind_operator_plan, operator_metadata
from .sparse_mass_matrix import _SparseMassMatrixPlan
from .tree_kinetic_owner import _NUMERIC_FLAGS, TreeKineticState
from .tree_kinetic_plan import make_plan
from .tree_kinetic_state import HeldResponse


@wp.kernel(enable_backward=False)
def _diagonal_geometry(
    group_to_art: wp.array[int],
    dof_start: wp.array[int],
    mask: wp.array[int],
    geometry_index: wp.array[int],
    geometric: wp.array2d[float],
    armature: wp.array2d[float],
    drive_row: wp.array[int],
    drive_k: wp.array[float],
    n: int,
    inverse: wp.array[float],
):
    element = wp.tid()
    group, local = element // n, element % n
    art = group_to_art[group]
    if mask[art] == 0:
        return
    dof = dof_start[art] + local
    mass = geometric[group, geometry_index[local]] + armature[group, local]
    row = drive_row[dof]
    if row >= 0 and drive_k[row] > 0.0:
        mass += drive_k[row]
    inverse[dof] = 1.0 / mass


def supported(solver):
    """Require original immediate rows and complete scalar-tree factor families."""
    return bool(
        solver.model.device.is_cuda
        and not solver.model.requires_grad
        and not solver.model.particle_count
        and len(solver.size_groups) >= 1
        and solver._sparse_mass_matrix_size is None
        and solver._point_contact_paths
        # Split mode composes its rows from the same held factors this owner
        # refreshes; the sparse tree owner has served split sweeps since it
        # existed, and the dense held owner does the same.
        and solver.pgs_mode in ("matrix_free", "split")
        # The factor stage returns after this owner's refresh, before the
        # paired-response inverse would be materialized.
        and getattr(solver, "_paired_response_primary_size", None) is None
        and solver.articulated_contact_response == "immediate"
        and solver.drive_mode == "augmented"
        and solver.pgs_velocity_iterations == 0
        and not solver.enable_joint_velocity_limits
        # Prescribed (kinematic) free bodies are owned as zero-response roots;
        # see PrescribedRootOperator. Mimic rows are ordinary matrix-free rows
        # over the held factors this owner refreshes, as they are for the
        # Franka held owner; they need nothing from the predictor itself.
        and not solver._connect_count
        and not solver._has_loop_joints
        and not solver.lazy_kinematics
        and not solver._fused_k1
        and solver._fk_id_cache_available
        and solver._async_augmented_drives
        and solver._parallel_augmented_drive_topology
        and solver._articulation_dynamics_stream is not None
        and not solver._debug_buffers_enabled
        and all(
            getattr(solver, name, None) is None
            for name in (
                "_sparse_factor",
                "_kinetic_world",
                "_franka_kinetic_state",
                "_joint_world",
                "_world_scan_publication",
                "_prismatic_publication",
            )
        )
        and all(
            size == solver._compact_diagonal_mass_size
            or (1 <= size <= 12 and solver._crba_cholesky_warp_kernels_by_size.get(size) is not None)
            for size in solver.size_groups
        )
    )


def _prescribed_articulations(solver):
    """Articulations whose response the solver prescribes (kinematic free bodies)."""
    return np.flatnonzero(solver._prescribed_articulation.numpy() != 0).astype(np.int32)


def _coverage(solver):
    """Prove complete disjoint articulation and physical coordinate ownership."""
    model = solver.model
    prescribed = _prescribed_articulations(solver)
    if not solver.size_groups or model.joint_count != model.body_count:
        raise ValueError("Mixed tree cache requires complete factor groups")
    arts = np.concatenate([solver.group_to_art[size].numpy() for size in solver.size_groups] + [prescribed])
    if not np.array_equal(np.sort(arts), np.arange(model.articulation_count)):
        raise ValueError("Mixed tree factors must cover every articulation exactly once")
    starts = model.articulation_start.numpy()
    if starts[0] != 0 or starts[-1] != model.joint_count or np.any(np.diff(starts) <= 0):
        raise ValueError("Mixed tree cache requires contiguous nonempty articulations")
    qd = model.joint_qd_start.numpy()
    ds = solver.articulation_dof_start.numpy()
    if not np.array_equal(ds, qd[starts[:-1]]):
        raise ValueError("Mixed tree factor starts must preserve physical coordinate ownership")
    for size in solver.size_groups:
        group = solver.group_to_art[size].numpy()
        if np.any(qd[starts[group + 1]] - qd[starts[group]] != size):
            raise ValueError("Mixed tree factor size disagrees with its physical coordinates")
        if solver.n_arts_by_size[size] != len(group):
            raise ValueError("Mixed tree factor count disagrees with its group map")
    if prescribed.size and np.any(qd[starts[prescribed + 1]] - qd[starts[prescribed]] != 6):
        raise ValueError("Prescribed cached roots must be six-coordinate free bodies")
    if not np.array_equal(np.sort(model.joint_child.numpy()), np.arange(model.body_count)):
        raise ValueError("Mixed tree cache must publish every body exactly once")
    return (
        tuple(solver.size_groups),
        tuple(sorted(solver.n_arts_by_size.items())),
        solver._compact_diagonal_mass_size,
        int(prescribed.size),
    )


class TreeHeldOperator:
    """Bind packed physical geometry to an unchanged Cholesky or diagonal action."""

    def __init__(self, solver, size):
        self.solver, self.size = solver, size
        self.requires_body_velocity = True
        model, device = solver.model, solver.model.device
        group = solver.group_to_art[size].numpy()
        starts = model.articulation_start.numpy()
        first, end = starts[group[0]], starts[group[0] + 1]
        ancestors = model.joint_ancestor.numpy()[first:end]
        parents = np.where(ancestors < 0, -1, ancestors - first)
        counts = np.diff(model.joint_qd_start.numpy())[first:end]
        root_type = model.joint_type.numpy()[first]
        fixed_root = counts[0] == 0 and root_type in (int(JointType.FIXED), int(JointType.D6))
        # A free rigid body is a one-body tree on a free root: its six coordinates
        # share the held Cholesky action and the generic free-root transport.
        free_root = counts[0] == 6 and root_type == int(JointType.FREE)
        if not (fixed_root or free_root):
            raise ValueError("Mixed held tree groups require fixed or free roots")
        self.free_root = bool(free_root)
        mass = self.mass = _SparseMassMatrixPlan.build(parents, counts)
        if mass.dof_count != size:
            raise ValueError("Mixed tree physical and factor dimensions differ")
        self.host = operator_metadata(mass)
        self.plan = bind_operator_plan(solver, size, self.host)
        self.layout, _, _ = make_plan(model, self, complete=False)
        physical_index = np.full((size, size), -1, np.int32)
        for index, (row, col) in enumerate(zip(mass.entry_rows, mass.columns, strict=True)):
            a, b = mass.permutation[row], mass.permutation[col]
            physical_index[a, b] = physical_index[b, a] = index
        self.diagonal = size == solver._compact_diagonal_mass_size
        if self.diagonal and mass.nonzero_count != size:
            raise ValueError("Compact diagonal cache requires independent physical coordinates")
        self.data = HeldResponse()
        self.data.valid = wp.ones(len(group), dtype=int, device=device)
        self.data.status = wp.zeros(len(group), dtype=int, device=device)
        if self.diagonal:
            self.data.inverse = solver._diagonal_inverse_mass
            self.geometry_index = wp.array(np.diag(physical_index).copy(), dtype=int, device=device)
            self.refresh_kernel = _diagonal_geometry
        else:
            self.data.lower = solver.L_by_size[size]
            self.geometry_index = wp.array(physical_index.ravel(), dtype=int, device=device)
            self.refresh_kernel = geometric_factor(
                str(device.arch), size=size, geometric_index=tuple(map(int, physical_index.ravel()))
            )
            self.factor_offsets = wp.zeros(size, dtype=int, device=device)
            self.factor_schedule = wp.array(
                [row * size + col for row in range(size) for col in range(row + 1)], dtype=int, device=device
            )
        self.kinetic_state = None

    def refresh(self, state_aug):
        """Refresh only original requested factor groups from their current geometry."""
        solver, size = self.solver, self.size
        if self.diagonal:
            wp.launch(
                self.refresh_kernel,
                dim=solver.n_arts_by_size[size] * size,
                inputs=[
                    self.plan.group_to_art,
                    solver.articulation_dof_start,
                    solver.mass_update_mask,
                    self.geometry_index,
                    self.kinetic_state.geometric,
                    solver.R_by_size[size],
                    solver._augmented_drive_row_by_dof,
                    solver.aug_row_K,
                    size,
                ],
                outputs=[solver._diagonal_inverse_mass],
                device=solver.model.device,
            )
        else:
            wp.launch(
                self.refresh_kernel,
                dim=solver.n_arts_by_size[size] * 32,
                inputs=[
                    solver.n_arts_by_size[size],
                    solver.model.articulation_start,
                    solver.articulation_dof_start,
                    solver.mass_update_mask,
                    solver.model.joint_child,
                    state_aug.joint_S_s,
                    state_aug.body_I_s,
                    self.plan.group_to_art,
                    solver.R_by_size[size],
                    self.factor_offsets,
                    self.factor_schedule,
                    1,
                    solver._augmented_drive_row_by_dof,
                    solver.aug_row_K,
                    self.kinetic_state.geometric,
                    self.geometry_index,
                ],
                outputs=[solver.L_by_size[size]],
                block_dim=128,
                device=solver.model.device,
            )

    def bind_response(self):
        """Bind current held-array descriptors before a new launch or capture."""
        if self.diagonal:
            self.data.inverse = self.solver._diagonal_inverse_mass
        else:
            self.data.lower = self.solver.L_by_size[self.size]


class PrescribedRootOperator:
    """Own prescribed free bodies as zero-response roots of the same tree law.

    A kinematic free body has six prescribed coordinates and no factor. With a
    diagonal response whose inverse is zero, the shared predictor yields zero
    acceleration and the prescribed velocity, the free-root transport is
    skipped on kinematic roots, and the shared finish integrates the pose and
    publishes the body exactly as it does for a dynamic free body. This is
    the generic form of the free-root slots the Franka and Kuka owners keep.
    """

    prescribed = True

    def __init__(self, solver):
        self.solver, self.size = solver, 6
        self.requires_body_velocity = True
        device = solver.model.device
        arts = _prescribed_articulations(solver)
        if not arts.size:
            raise ValueError("No prescribed articulations to own")
        self.free_root = True
        self.diagonal = True
        self.mass = mass = _SparseMassMatrixPlan.build(np.array([-1], np.int32), np.array([6], np.int32))
        self.host = operator_metadata(mass)
        self.group_to_art = wp.array(arts, dtype=int, device=device)
        self.plan = bind_operator_plan(solver, 6, self.host, group_to_art=self.group_to_art)
        self.layout, _, _ = make_plan(solver.model, self, complete=False)
        self.data = HeldResponse()
        self.data.valid = wp.ones(len(arts), dtype=int, device=device)
        self.data.status = wp.zeros(len(arts), dtype=int, device=device)
        # Zero inverse mass: the prescribed coordinates never respond.
        self.data.inverse = wp.zeros(solver.model.joint_dof_count, dtype=float, device=device)
        self.kinetic_state = None

    def refresh(self, state_aug):
        """Prescribed roots hold no factor; nothing to refresh."""

    def bind_response(self):
        """The zero inverse is owned here and never rebinds."""


def forest_supported(solver):
    """Whether every world is one forest of consecutive articulations with held or zero families."""
    if not supported(solver):
        return False
    model = solver.model
    worlds = int(solver.world_count)
    art_world = solver.art_to_world.numpy()
    prescribed = solver._prescribed_articulation.numpy()
    counts = np.diff([*solver.articulation_dof_start.numpy().tolist(), model.joint_dof_count])
    per_world = [np.flatnonzero(art_world == w) for w in range(worlds)]
    if not per_world or any(len(a) != len(per_world[0]) or len(a) == 0 for a in per_world):
        return False
    if len(per_world[0]) > 4 or len(per_world[0]) < 2:
        return False
    signature = [(int(counts[a]), int(prescribed[a])) for a in per_world[0]]
    for arts in per_world:
        if np.any(np.diff(arts) != 1) or [(int(counts[a]), int(prescribed[a])) for a in arts] != signature:
            return False
    families = [0 if pres else size for size, pres in signature]
    for size in sorted({f for f in families if f}):
        if size == solver._compact_diagonal_mass_size or solver._crba_cholesky_warp_kernels_by_size.get(size) is None:
            return False
        group = solver.group_to_art[size].numpy()
        if len(group) != worlds or not np.array_equal(art_world[group], np.arange(worlds)):
            return False
    return True


class ForestHeldOperator:
    """Own every articulation of a world as one group: shared scratch, one launch per phase.

    Each root keeps its own held factor family (the solver's dense factor
    for its size, refreshed from the forest's geometry) or responds with zero
    when prescribed. This is the generic form of the per-world forest the
    Franka owner was written around: roots, families and offsets come from
    the topology.
    """

    def __init__(self, solver):
        self.solver = solver
        self.requires_body_velocity = True
        self.free_root = True
        self.diagonal = False
        model, device = solver.model, solver.model.device
        worlds = int(solver.world_count)
        art_world = solver.art_to_world.numpy()
        prescribed = solver._prescribed_articulation.numpy()
        counts = np.diff([*solver.articulation_dof_start.numpy().tolist(), model.joint_dof_count])
        arts = np.stack([np.flatnonzero(art_world == w) for w in range(worlds)]).astype(np.int32)
        self.forest_arts = arts
        self.root_families = tuple(0 if prescribed[a] else int(counts[a]) for a in arts[0])
        starts = model.articulation_start.numpy()
        first, end = int(starts[arts[0, 0]]), int(starts[arts[0, -1] + 1])
        ancestors = model.joint_ancestor.numpy()[first:end]
        parents = np.where(ancestors < 0, -1, ancestors - first)
        joint_counts = np.diff(model.joint_qd_start.numpy())[first:end]
        self.mass = mass = _SparseMassMatrixPlan.build(parents, joint_counts)
        self.host = operator_metadata(mass)
        self.group_to_art = wp.array(arts[:, 0].copy(), dtype=int, device=device)
        self.plan = bind_operator_plan(solver, 0, self.host, group_to_art=self.group_to_art)
        self.size = mass.dof_count
        self.layout, _, _ = make_plan(model, self, complete=False)
        # Physical (row, col) -> forest entry, for each root family's refresh.
        physical_index = np.full((self.size, self.size), -1, np.int32)
        for index, (row, col) in enumerate(zip(mass.entry_rows, mass.columns, strict=True)):
            a, b = mass.permutation[row], mass.permutation[col]
            physical_index[a, b] = physical_index[b, a] = index
        self.data = HeldResponse()
        self.data.valid = wp.ones(worlds, dtype=int, device=device)
        self.data.status = wp.zeros(worlds, dtype=int, device=device)
        self.data.inverse = wp.zeros(model.joint_dof_count, dtype=float, device=device)
        factor_index = np.zeros((worlds, len(self.root_families)), np.int32)
        self.families = []
        for r, family in enumerate(self.root_families):
            if family == 0:
                continue
            group = solver.group_to_art[family].numpy()
            position = {int(a): g for g, a in enumerate(group)}
            factor_index[:, r] = [position[int(a)] for a in arts[:, r]]
            offset = self.layout.root_offsets[r]
            block = physical_index[offset : offset + family, offset : offset + family]
            # The family's factor groups read this root's block of their
            # world's geometry record (one record per forest group).
            self.families.append(
                (
                    r,
                    family,
                    geometric_factor(
                        str(device.arch),
                        size=family,
                        geometric_index=tuple(map(int, block.ravel())),
                        geometric_stride=self.layout.entries,
                        geometric_group=True,
                    ),
                    wp.zeros(family, dtype=int, device=device),
                    wp.array(
                        [row * family + col for row in range(family) for col in range(row + 1)],
                        dtype=int,
                        device=device,
                    ),
                    wp.array(block.ravel().astype(np.int32), dtype=int, device=device),
                    wp.array(art_world[group].astype(np.int32), dtype=int, device=device),
                )
            )
            setattr(self.data, "lower" if r == 0 else f"lower{r}", solver.L_by_size[family])
        for r, family in enumerate(self.root_families):
            if family == 0 and r > 0:
                setattr(self.data, f"lower{r}", self.data.lower)
        self.data.factor_index = wp.array(factor_index, dtype=int, device=device)
        self.kinetic_state = None

    def refresh(self, state_aug):
        """Refresh every root family's dense factor from the forest geometry."""
        solver = self.solver
        for _r, family, kernel, offsets, schedule, geometry_index, geometry_group in self.families:
            wp.launch(
                kernel,
                dim=solver.n_arts_by_size[family] * 32,
                inputs=[
                    solver.n_arts_by_size[family],
                    solver.model.articulation_start,
                    solver.articulation_dof_start,
                    solver.mass_update_mask,
                    solver.model.joint_child,
                    state_aug.joint_S_s,
                    state_aug.body_I_s,
                    solver.group_to_art[family],
                    solver.R_by_size[family],
                    offsets,
                    schedule,
                    1,
                    solver._augmented_drive_row_by_dof,
                    solver.aug_row_K,
                    self.kinetic_state.geometric,
                    geometry_index,
                    geometry_group,
                ],
                outputs=[solver.L_by_size[family]],
                block_dim=128,
                device=solver.model.device,
            )

    def bind_response(self):
        """Rebind the solver's held factors before a launch or capture."""
        for r, family in enumerate(self.root_families):
            if family:
                setattr(self.data, "lower" if r == 0 else f"lower{r}", self.solver.L_by_size[family])


class MixedTreeKineticState:
    """Compose disjoint factor-group caches with complete public-state coverage."""

    def __init__(self, solver, *, admission=None):
        self.solver = solver
        self.coverage = _coverage(solver)
        self._dimensions = (solver.model.joint_count, solver.model.body_count, solver.model.articulation_count)
        # The coverage proof's inputs, compared on the device on notification.
        self._proof = DeviceProof(list(self._coverage_arrays()), solver.model.device)
        if os.environ.get("FEATHER_PGS_TREE_FOREST", "0") == "1" and forest_supported(solver):
            operators = [ForestHeldOperator(solver)]
        else:
            operators = [TreeHeldOperator(solver, size) for size in solver.size_groups]
            if self.coverage[-1]:
                operators.append(PrescribedRootOperator(solver))
        self.groups = []
        for operator in operators:
            group = TreeKineticState(
                operator,
                admission=admission,
                chain_scan_default=False,
                complete=False,
                response="diagonal" if operator.diagonal else "cholesky",
                # Copied-state reuse requires fixed scalar coordinates.
                copy_reuse=not operator.free_root,
            )
            operator.kinetic_state = group
            self.groups.append(group)
        # The existing solver lifecycle delegates factor refresh to this binding.
        self.sparse = self
        self.copy_reuse = all(group.copy_reuse for group in self.groups)

    def _coverage_arrays(self):
        solver, model = self.solver, self.solver.model
        for size in solver.size_groups:
            yield f"group_to_art[{size}]", solver.group_to_art[size]
        yield "articulation_start", model.articulation_start
        yield "joint_qd_start", model.joint_qd_start
        yield "joint_child", model.joint_child
        yield "articulation_dof_start", solver.articulation_dof_start
        yield "prescribed_articulation", solver._prescribed_articulation

    def validate_model(self, flags=None):
        """Reject ownership changes before any cache or notification mutation."""
        if flags is not None and int(flags) != 0 and int(flags) & ~_NUMERIC_FLAGS == 0:
            # Numeric notifications cannot change ownership; see TreeKineticState.
            return
        solver = self.solver
        sizes = (
            tuple(solver.size_groups),
            tuple(sorted(solver.n_arts_by_size.items())),
            solver._compact_diagonal_mass_size,
        )
        if (
            solver.model.joint_count,
            solver.model.body_count,
            solver.model.articulation_count,
        ) != self._dimensions or sizes != self.coverage[:3]:
            raise RuntimeError("Mixed tree factor groups changed; reconstruct and recapture")
        current = dict(self._coverage_arrays())
        self._proof.validate(current.__getitem__, "Mixed tree factor")
        for group in self.groups:
            # Each group's plan binds the same group map it was built from.
            group.validate_model(flags)

    def invalidate(self, world_mask=None):
        """Invalidate all component caches for selected worlds."""
        for group in self.groups:
            group.invalidate(world_mask)

    def begin(self, state, augmented, velocity, dt, refresh):
        """Repair every current component before any factor or force consumer."""
        for group in self.groups:
            group.begin(state, augmented, velocity, dt, refresh)

    def refresh(self, augmented):
        """Preserve independent factor masks and numerical representations."""
        for group in self.groups:
            group.sparse.refresh(augmented)

    def predict(self, state, augmented, control, velocity, dt):
        """Project current forces and apply each original held response."""
        for group in self.groups:
            group.sparse.bind_response()
            group.predict(state, augmented, control, velocity, dt)

    def finish(self, state, augmented, output, dt, refresh):
        """Integrate and publish each physical coordinate and body once."""
        for group in self.groups:
            group.finish(state, augmented, output, dt, refresh)


def create_owner(solver):
    """Keep unsupported structures on their complete ordinary solver route."""
    if not supported(solver):
        return None
    try:
        owner = MixedTreeKineticState(solver, admission=supported)
    except ValueError:
        return None
    solver._fk_id_cache_enabled = True
    solver._fk_id_cache_uses_snapshot = False
    solver._fk_id_cache = None
    return owner
