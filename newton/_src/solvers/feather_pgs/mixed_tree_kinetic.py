# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compose complete tree caches over retained heterogeneous factor groups."""

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
        and len(solver.size_groups) > 1
        and solver._sparse_mass_matrix_size is None
        and solver._point_contact_paths
        and solver.pgs_mode == "matrix_free"
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
    if len(solver.size_groups) + (1 if prescribed.size else 0) < 2 or model.joint_count != model.body_count:
        raise ValueError("Mixed tree cache requires multiple complete factor groups")
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


class MixedTreeKineticState:
    """Compose disjoint factor-group caches with complete public-state coverage."""

    def __init__(self, solver, *, admission=None):
        self.solver = solver
        self.coverage = _coverage(solver)
        self._dimensions = (solver.model.joint_count, solver.model.body_count, solver.model.articulation_count)
        # The coverage proof's inputs, compared on the device on notification.
        self._proof = DeviceProof(list(self._coverage_arrays()), solver.model.device)
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
