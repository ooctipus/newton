# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The held-tree owner: the CUDA evaluator of articulated dynamics, its groups chosen from the topology.

One owner (:class:`HeldTree`) composes the groups that evaluate every articulation
(:class:`~.tree_kinetic_owner.TreeKineticState`, plus the scalar branches' group). The solver's data chooses them:
the native sparse factor, when the solver admitted one, makes one complete group (:class:`TreeSparseOperator`);
otherwise each size group of the dense factors is a group with its held leaf-first Cholesky or diagonal response
(:class:`TreeHeldOperator`), kinematic free bodies a zero-response group (:class:`PrescribedRootOperator`), or one
group per world spans its articulations (:class:`ForestHeldOperator`). :func:`decline` names why a solver's
dynamics stay on the serial passes; :func:`install` installs the owner.
"""

from types import SimpleNamespace

import numpy as np
import warp as wp

from ...sim import JointType
from . import host_index
from .device_proof import DeviceProof
from .response_builds_dense import factor_warps_kernel, leaf_index
from .response_builds_rows import sparse_factor_kernel
from .route_table import capable
from .sparse_factor import SparseData, bind_operator_plan, operator_metadata
from .tree_kinetic_owner import TreeKineticState, kernel_switches, launch_switches, proof_runs
from .tree_kinetic_plan import make_plan
from .tree_kinetic_state import HeldResponse
from .tree_plan import TreePlan, local_parents


@wp.kernel(enable_backward=False)
def _diagonal_geometry(
    group_to_art: wp.array[int],
    dof_start: wp.array[int],
    mask: wp.array[int],
    geometry_index: wp.array[int],
    geometry_group: wp.array[int],
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
    mass = geometric[geometry_group[group], geometry_index[local]] + armature[group, local]
    row = drive_row[dof]
    if row >= 0 and drive_k[row] > 0.0:
        mass += drive_k[row]
    inverse[dof] = 1.0 / mass


def decline(solver, *, sparse: tuple[bool, bool] | None = None, compact: int | None = None) -> str | None:
    """Why the held tree does not evaluate the solver's dynamics, or ``None`` when it does.

    The terms every group needs, then those of the representation: the native sparse factor's group (``sparse``:
    its ``(velocity_limits, split)``), else the dense size groups with ``compact`` as the compact diagonal size.
    Both default to the representation the solver chose; the native sparse factor and the compact diagonal, which
    only the held tree evaluates, pass their candidates (:func:`sparse_decline`, :func:`compact_decline`). A term
    is read only once the earlier ones hold.
    """
    if sparse is None and getattr(solver, "_sparse_mass_matrix_size", None) is not None:
        sparse = (solver._sparse_velocity_limits, solver._sparse_split)
    if compact is None:
        compact = getattr(solver, "_compact_diagonal_mass_size", None)
    model = solver.model
    reason = capable("held", solver._laws)
    if reason is not None:
        return reason
    if model.particle_count:
        return "Particles are outside the held tree."
    if sparse is not None:
        velocity_limits, split = sparse
        if solver._has_free_rigid_bodies:
            return "The sparse factor's group holds no free rigid bodies."
        if solver._mimic_count and not (
            velocity_limits and solver._mimic_valid_np is not None and np.all(solver._mimic_valid_np)
        ):
            return "Mimic rows on the sparse factor need its velocity-limit rows and valid mimic pairs."
    else:
        # The owner holds no rows, impulses or patch frames: the patch build, link, seed and finish stay on the
        # solver pipeline around it, so it serves every contact law. Torsion rows read the joint motions and body
        # twists every group publishes (joint_S_s, body_v_s); compliance reads no dynamics.
        # Split mode composes its rows from the same held factors the owner refreshes.
        if not solver.size_groups:
            return "No articulation has response coordinates."
    if sparse is not None and solver._has_prescribed_response:
        return "The sparse factor's group holds no prescribed articulations."
    if not solver._fk_id_cache_available:
        return "The FK/ID cache is unavailable."
    if sparse is not None and not (solver.pgs_mode == "matrix_free" or split):
        return "The sparse factor's group serves the matrix-free and sparse split sweeps."
    if not solver._parallel_augmented_drive_topology:
        return "The held tree needs the per-coordinate drive preparation (augmented drives, owned drive rows)."
    if sparse is not None:
        return None
    # Every family refreshes through the geometric small Cholesky (one lane per factor row, up to 32
    # coordinates) into the solver's dense factor of the same shape, in place of the size's stage-2 factor; the
    # compact diagonal size keeps its compact inverse.
    for size in solver.size_groups:
        if size != compact and size > 32:
            return f"Size {size} has no held Cholesky family (1 to 32 coordinates)."
    return None


def admitted(solver) -> bool:
    """Whether the held tree evaluates the solver's dynamics (:func:`decline`); its groups re-check it on every
    call."""
    return decline(solver) is None


def _prescribed_articulations(solver):
    """Articulations whose response the solver prescribes (kinematic free bodies)."""
    return np.flatnonzero(solver._prescribed_articulation.numpy() != 0).astype(np.int32)


def _coverage(solver, compact):
    """Prove complete disjoint articulation and physical coordinate ownership; ``compact`` is the compact diagonal
    size the groups carry."""
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
        compact,
        int(prescribed.size),
    )


def _plan_view(group_to_art, art_to_world, host, *, prescribed=False):
    """What :func:`.make_plan` reads of a group's operator: its articulations, their worlds and its metadata."""
    return SimpleNamespace(
        plan=SimpleNamespace(group_to_art=group_to_art, art_to_world=art_to_world), host=host, prescribed=prescribed
    )


def _size_group(solver, size, *, diagonal):
    """Size group ``size`` as a held group: whether its trees stand on free roots, their mass pattern, its metadata
    and its layout (:func:`.make_plan`). Raises ValueError where the trees do not fit a held group."""
    model = solver.model
    first_art = int(solver.group_to_art[size].numpy()[0])
    starts = model.articulation_start.numpy()
    first, end = starts[first_art], starts[first_art + 1]
    parents = local_parents(model.joint_ancestor.numpy(), [first], end - first)[0]
    counts = np.diff(model.joint_qd_start.numpy())[first:end]
    root_type = model.joint_type.numpy()[first]
    fixed_root = counts[0] == 0 and root_type in (int(JointType.FIXED), int(JointType.D6))
    # A free rigid body is a one-body tree on a free root: its six coordinates
    # share the held Cholesky action and the generic free-root transport.
    free_root = counts[0] == 6 and root_type == int(JointType.FREE)
    if not (fixed_root or free_root):
        raise ValueError("Mixed held tree groups require fixed or free roots")
    mass = TreePlan.build(parents, counts).factor
    if mass.dof_count != size:
        raise ValueError("Mixed tree physical and factor dimensions differ")
    if diagonal and mass.nonzero_count != size:
        raise ValueError("Compact diagonal cache requires independent physical coordinates")
    host = operator_metadata(mass)
    layout, _ = make_plan(model, _plan_view(solver.group_to_art[size], solver.art_to_world, host), complete=False)
    return bool(free_root), mass, host, layout


def _prescribed_group(solver, group_to_art):
    """The prescribed free bodies as a held group: their mass pattern, metadata and layout (:func:`.make_plan`)."""
    mass = TreePlan.build([-1], [6]).factor
    host = operator_metadata(mass)
    view = _plan_view(group_to_art, solver.art_to_world, host, prescribed=True)
    layout, _ = make_plan(solver.model, view, complete=False)
    return mass, host, layout


def sparse_decline(solver, size, plan, *, velocity_limits, split) -> str | None:
    """Why the held tree would not take a native sparse factor of ``size`` coordinates and pattern ``plan``, or
    ``None``: :func:`options_decline`, :func:`decline` of the candidate, then its one complete group's fit.

    Only the held tree evaluates native sparse factors, so their admission reads this before choosing one.
    """
    reason = options_decline(solver) or decline(solver, sparse=(velocity_limits, split))
    if reason is not None:
        return reason
    view = _plan_view(solver.group_to_art[size], solver.art_to_world, operator_metadata(plan))
    try:
        make_plan(solver.model, view, complete=True)
    except ValueError as error:
        return f"The topology does not fit the held groups: {error}"
    return None


def compact_decline(solver, size, *, scalar) -> str | None:
    """Why the held tree would not take the dense size groups with a compact diagonal of ``size`` coordinates, or
    ``None``: :func:`options_decline`, :func:`decline` of the candidate, then the groups' fit (the coverage, the
    diagonal group unless ``scalar`` branches take it, every other size group, the prescribed roots).

    Only the held tree evaluates a compact diagonal, so the sparse-diagonal response reads this before choosing one.
    """
    reason = options_decline(solver) or decline(solver, compact=size)
    if reason is not None:
        return reason
    try:
        _coverage(solver, size)
        for group_size in solver.size_groups:
            if group_size != size or not scalar:
                _size_group(solver, group_size, diagonal=group_size == size)
        prescribed = _prescribed_articulations(solver)
        if prescribed.size:
            _prescribed_group(solver, wp.array(prescribed, dtype=int, device=solver.model.device))
    except ValueError as error:
        return f"The topology does not fit the held groups: {error}"
    return None


class TreeSparseOperator:
    """Respond through the solver's native sparse factor, refreshed from the held geometry."""

    def __init__(self, solver):
        self.solver = solver
        self.requires_body_velocity = True
        mass = solver._sparse_mass_matrix_plan
        n, device = mass.dof_count, solver.model.device
        self.host = operator_metadata(mass)
        self.plan = bind_operator_plan(solver, n, self.host)
        self.data = SparseData()
        self.data.W = solver._sparse_Linv
        # The native factor publishes status instead of a second validity bit.
        # Initial and reset mass requests are consumed before any prediction.
        self.data.valid = wp.ones(self.plan.group_to_art.size, dtype=int, device=device)
        self.data.status = solver._sparse_mass_matrix_status
        self.refresh_kernel = sparse_factor_kernel(
            n, mass.nonzero_count, warps_per_block=solver._sparse_factor_warps_per_block
        )
        self.kinetic_state = None

    def refresh(self, state_aug):
        """Read cached physical mass terms through the unchanged sparse factor."""
        s, n = self.solver, self.solver._sparse_mass_matrix_size
        wp.launch(
            self.refresh_kernel,
            dim=s.n_arts_by_size[n] * 32,
            inputs=[
                s.group_to_art[n],
                s.mass_update_mask,
                s.articulation_dof_start,
                self.kinetic_state.geometric,
                s.R_by_size[n],
                s._augmented_drive_row_by_dof,
                s.aug_row_K,
                s._sparse_mass_matrix_indices,
            ],
            outputs=[s._sparse_Linv, s._sparse_mass_matrix_status],
            block_dim=32 * s._sparse_factor_warps_per_block,
            device=s.model.device,
        )

    def bind_response(self):
        """The native factor's descriptors are bound once."""


class TreeHeldOperator:
    """Bind packed physical geometry to an unchanged Cholesky or diagonal action."""

    def __init__(self, solver, size):
        self.solver, self.size = solver, size
        self.requires_body_velocity = True
        device = solver.model.device
        group = solver.group_to_art[size].numpy()
        self.diagonal = size == solver._compact_diagonal_mass_size
        self.free_root, mass, self.host, self.layout = _size_group(solver, size, diagonal=self.diagonal)
        self.mass = mass
        self.plan = bind_operator_plan(solver, size, self.host)
        physical_index = mass.physical_lookup
        self.data = HeldResponse()
        self.data.valid = wp.ones(len(group), dtype=int, device=device)
        self.data.status = wp.zeros(len(group), dtype=int, device=device)
        if self.diagonal:
            self.data.inverse = solver._diagonal_inverse_mass
            self.geometry_index = wp.array(np.diag(physical_index).copy(), dtype=int, device=device)
            self.geometry_group = wp.array(np.arange(len(group), dtype=np.int32), dtype=int, device=device)
            self.refresh_kernel = _diagonal_geometry
        else:
            self.data.lower = solver.L_by_size[size]
            index = leaf_index(physical_index)
            self.geometry_index = wp.array(index, dtype=int, device=device)
            self.refresh_kernel = factor_warps_kernel(size, index)
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
                    self.geometry_group,
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
                    solver.articulation_dof_start,
                    solver.mass_update_mask,
                    self.plan.group_to_art,
                    solver.R_by_size[size],
                    self.factor_schedule,
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
    the generic form of a per-world free-root slot.
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
        self.group_to_art = wp.array(arts, dtype=int, device=device)
        self.mass, self.host, self.layout = _prescribed_group(solver, self.group_to_art)
        self.plan = bind_operator_plan(solver, 6, self.host, group_to_art=self.group_to_art)
        self.data = HeldResponse()
        self.data.valid = wp.ones(len(arts), dtype=int, device=device)
        self.data.status = wp.zeros(len(arts), dtype=int, device=device)
        # Zero inverse mass: the prescribed coordinates never respond.
        self.data.inverse = solver._arrays.placeholder(wp.float32, (solver.model.joint_dof_count,))
        self.kinetic_state = None

    def refresh(self, state_aug):
        """Prescribed roots hold no factor; nothing to refresh."""

    def bind_response(self):
        """The zero inverse is owned here and never rebinds."""


def forest_supported(solver):
    """Whether every world is one forest of consecutive articulations with held or zero families."""
    return forest_families(solver) is not None


def forest_default(solver):
    """Whether the forest is the default group: a prescribed root and no diagonal-mass family.

    The forest's gain is one group and one launch set per world where the
    per-size groups would run a third, zero-response group for a prescribed
    root: measured -9% on a 9+6+prescribed world and -1% on 23+6+prescribed.
    Without a prescribed root the per-size groups' lane packing wins (+3% for
    the forest on a 16+6 world), and a diagonal-mass family, which the per-size
    owner applies in a two-lane block, costs the forest 5% (9+4 with a drawer).
    """
    families = forest_families(solver)
    return families is not None and 0 in families and solver._compact_diagonal_mass_size not in families


def forest_families(solver):
    """The per-root family sizes of the forest (0 for a prescribed root), or ``None``."""
    if not admitted(solver):
        return None
    if getattr(solver, "_scalar_branches", None) is not None:
        # Scalar branches own their size group outside any tree plan.
        return None
    model = solver.model
    worlds = int(solver.world_count)
    art_world = solver.art_to_world.numpy()
    prescribed = solver._prescribed_articulation.numpy()
    counts = np.diff([*solver.articulation_dof_start.numpy().tolist(), model.joint_dof_count])
    world_bounds, order = host_index.group_by(art_world, worlds)
    world_arts = np.diff(world_bounds)[:worlds]
    if not worlds or np.any(world_arts != world_arts[0]) or world_arts[0] == 0:
        return None
    if world_arts[0] > 4 or world_arts[0] < 2:
        return None
    # Each world's articulations in ascending order, one row per world.
    per_world = order.reshape(worlds, world_arts[0])
    if np.any(np.diff(per_world, axis=1) != 1):
        return None
    if np.any(counts[per_world] != counts[per_world[0]]) or np.any(prescribed[per_world] != prescribed[per_world[0]]):
        return None
    # One layout holds a world's trees: at most 64 bodies and coordinates (make_plan).
    bodies = np.diff(model.articulation_start.numpy())
    if bodies[per_world[0]].sum() > 64 or counts[per_world[0]].sum() > 64:
        return None
    signature = [(int(counts[a]), int(prescribed[a])) for a in per_world[0]]
    families = [0 if pres else size for size, pres in signature]
    for size in sorted({f for f in families if f}):
        if size == solver._compact_diagonal_mass_size:
            if solver._diagonal_inverse_mass is None:
                return None
        elif not 1 <= size <= 32 or solver._execution_plan.use_diagonal_mass(size):
            return None
        group = solver.group_to_art[size].numpy()
        if len(group) != worlds or not np.array_equal(art_world[group], np.arange(worlds)):
            return None
    return families


class ForestHeldOperator:
    """Own every articulation of a world as one group: shared scratch, one launch per phase.

    Each root keeps its own held factor family (the solver's dense factor
    for its size, refreshed from the forest's geometry) or responds with zero
    when prescribed. Roots, families and offsets come from the topology.
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
        # Each world's articulations in ascending order, one row per world (forest_families admits equal counts).
        world_bounds, order = host_index.group_by(art_world, worlds)
        arts = order.reshape(worlds, int(world_bounds[1] - world_bounds[0])).astype(np.int32)
        self.forest_arts = arts
        # Family per root: its coordinate count for a held Cholesky factor,
        # negated for a diagonal-mass size (compact inverse), 0 when prescribed.
        diagonal = solver._compact_diagonal_mass_size
        self.root_families = tuple(
            0 if prescribed[a] else (-int(counts[a]) if counts[a] == diagonal else int(counts[a])) for a in arts[0]
        )
        starts = model.articulation_start.numpy()
        first, end = int(starts[arts[0, 0]]), int(starts[arts[0, -1] + 1])
        parents = local_parents(model.joint_ancestor.numpy(), [first], end - first)[0]
        joint_counts = np.diff(model.joint_qd_start.numpy())[first:end]
        self.mass = mass = TreePlan.build(parents, joint_counts).factor
        self.host = operator_metadata(mass)
        self.group_to_art = wp.array(arts[:, 0].copy(), dtype=int, device=device)
        self.plan = bind_operator_plan(solver, 0, self.host, group_to_art=self.group_to_art)
        self.size = mass.dof_count
        self.layout, _ = make_plan(model, self, complete=False)
        # Physical (row, col) -> forest entry, for each root family's refresh.
        physical_index = mass.physical_lookup
        self.data = HeldResponse()
        self.data.valid = wp.ones(worlds, dtype=int, device=device)
        self.data.status = wp.zeros(worlds, dtype=int, device=device)
        self.data.inverse = (
            solver._diagonal_inverse_mass
            if any(f < 0 for f in self.root_families)
            else wp.zeros(model.joint_dof_count, dtype=float, device=device)
        )
        factor_index = np.zeros((worlds, len(self.root_families)), np.int32)
        self.families = []
        for r, family in enumerate(self.root_families):
            if family == 0:
                continue
            offset = self.layout.root_offsets[r]
            if family < 0:
                size = -family
                group = solver.group_to_art[size].numpy()
                block = physical_index[offset : offset + size, offset : offset + size]
                self.families.append(
                    (
                        r,
                        family,
                        _diagonal_geometry,
                        None,
                        wp.array(np.diag(block).astype(np.int32), dtype=int, device=device),
                        wp.array(art_world[group].astype(np.int32), dtype=int, device=device),
                    )
                )
                continue
            group = solver.group_to_art[family].numpy()
            position = {int(a): g for g, a in enumerate(group)}
            factor_index[:, r] = [position[int(a)] for a in arts[:, r]]
            index = leaf_index(physical_index[offset : offset + family, offset : offset + family])
            # The family's factor groups read this root's block of their
            # world's geometry record (one record per forest group).
            self.families.append(
                (
                    r,
                    family,
                    factor_warps_kernel(family, index, geometric_stride=self.layout.entries, geometric_group=True),
                    wp.array(
                        [row * family + col for row in range(family) for col in range(row + 1)],
                        dtype=int,
                        device=device,
                    ),
                    wp.array(index, dtype=int, device=device),
                    wp.array(art_world[group].astype(np.int32), dtype=int, device=device),
                )
            )
            setattr(self.data, "lower" if r == 0 else f"lower{r}", solver.L_by_size[family])
        if not any(f > 0 for f in self.root_families):
            self.data.lower = wp.zeros((1, 1, 1), dtype=float, device=device)
        for r, family in enumerate(self.root_families):
            if family <= 0 and r > 0:
                setattr(self.data, f"lower{r}", self.data.lower)
        self.data.factor_index = wp.array(factor_index, dtype=int, device=device)
        self.kinetic_state = None

    def refresh(self, state_aug):
        """Refresh every root family's dense factor from the forest geometry."""
        solver = self.solver
        for _r, family, kernel, schedule, geometry_index, geometry_group in self.families:
            if family < 0:
                size = -family
                wp.launch(
                    kernel,
                    dim=solver.n_arts_by_size[size] * size,
                    inputs=[
                        solver.group_to_art[size],
                        solver.articulation_dof_start,
                        solver.mass_update_mask,
                        geometry_index,
                        geometry_group,
                        self.kinetic_state.geometric,
                        solver.R_by_size[size],
                        solver._augmented_drive_row_by_dof,
                        solver.aug_row_K,
                        size,
                    ],
                    outputs=[solver._diagonal_inverse_mass],
                    device=solver.model.device,
                )
                continue
            wp.launch(
                kernel,
                dim=solver.n_arts_by_size[family] * 32,
                inputs=[
                    solver.n_arts_by_size[family],
                    solver.articulation_dof_start,
                    solver.mass_update_mask,
                    solver.group_to_art[family],
                    solver.R_by_size[family],
                    schedule,
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
            if family > 0:
                setattr(self.data, "lower" if r == 0 else f"lower{r}", self.solver.L_by_size[family])


class HeldTree:
    """Evaluate every articulation's dynamics in held groups chosen from the topology (see the module docstring).

    ``admission`` (:func:`admitted` when installed) is re-checked by every group on every call.
    """

    def __init__(self, solver, *, admission=None):
        self.solver = solver
        # One read of the kernel and launch switches for every group this owner composes.
        self.reference, self.clocked = kernel_switches()
        skip, classify = launch_switches(solver._route_pins)
        self.scalar = None
        if solver._sparse_mass_matrix_size is not None:
            # The native sparse factor: one complete group, which proves its own coverage (make_plan).
            self.coverage = self.forest_reason = None
            operator = TreeSparseOperator(solver)
            group = TreeKineticState(
                operator,
                admission=admission,
                reference=self.reference,
                clocked=self.clocked,
                skip=skip,
                classify=classify,
            )
            operator.kinetic_state = group
            self.groups = [group]
            self.bank_compare = group.bank_compare
            self.skip_captured_repairs = group.skip_captured_repairs
            return
        self.coverage = _coverage(solver, solver._compact_diagonal_mass_size)
        self._dimensions = (solver.model.joint_count, solver.model.body_count, solver.model.articulation_count)
        # The coverage proof's inputs, compared on the device on notification.
        self._proof = DeviceProof(list(self._coverage_arrays()), solver.model.device)
        # A size group of scalar branches (scalar_branches.py) is owned in place
        # of a tree group: one thread per coordinate, no tree plan.
        scalar = getattr(solver, "_scalar_branches", None)
        override = solver._route_pins.get("grouping")
        forest = override == "forest" if override is not None else forest_default(solver)
        if forest and forest_supported(solver):
            operators = [ForestHeldOperator(solver)]
            self.forest_reason = (
                "A test pin selects one group per world (grouping='forest')."
                if override is not None
                else "Default: a prescribed root and no diagonal-mass family make one group per world cheaper."
            )
        else:
            operators = [
                TreeHeldOperator(solver, size) for size in solver.size_groups if scalar is None or size != scalar.size
            ]
            if self.coverage[-1]:
                operators.append(PrescribedRootOperator(solver))
            if forest and scalar is not None:
                self.forest_reason = "Scalar branches own the diagonal-mass group, so no group spans a world."
            elif forest:
                self.forest_reason = "The world layout does not admit one forest group."
            elif override is not None:
                self.forest_reason = "A test pin keeps per-size groups (grouping='per_size')."
            else:
                self.forest_reason = (
                    "Default: per-size groups (the layout admits no forest, has no prescribed root, or has a "
                    "diagonal-mass family the forest applies at a loss)."
                )
        self.groups = []
        for operator in operators:
            group = TreeKineticState(
                operator,
                admission=admission,
                chains=False,
                complete=False,
                response="diagonal" if operator.diagonal else "cholesky",
                reference=self.reference,
                clocked=self.clocked,
                skip=skip,
                classify=classify,
                # Scalar branches hold no bank stamps, so beside them a changed input bank
                # invalidates every cache of the owner (bank_compare below), and a tree group
                # that compared would compare words whose cache was already discarded.
                compare_banks=scalar is None,
            )
            operator.kinetic_state = group
            self.groups.append(group)
        if scalar is not None:
            from .scalar_branches import ScalarBranchState  # noqa: PLC0415

            # A tree group's proof holds the model topology, the joint axes and
            # art_to_world and watches the joint frames, and the coverage proof every
            # size's group map, so the scalar group needs no proof of its own: the
            # sparse-diagonal pair always leaves a tree group beside it.
            self.scalar = ScalarBranchState(solver, scalar, admission=admission)
            self.groups.append(self.scalar)
        # Every tree group compares input banks (fixed and free roots alike) and skips
        # captured repairs unless the solver prescales velocities; see TreeKineticState.
        # Beside scalar branches, which do neither, every group validates a changed
        # input bank by identity, and the solver invalidates the whole owner.
        self.bank_compare = all(group.bank_compare for group in self.groups)
        self.skip_captured_repairs = all(group.skip_captured_repairs for group in self.groups)

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
        """Reject ownership changes before any cache or notification mutation.

        Returns:
            Whether every group's proof ran and found its inputs unchanged (see
            :meth:`TreeKineticState.validate_model`). A scalar group without a
            proof of its own counts as passed: its tree neighbours prove the
            same model arrays.
        """
        if not proof_runs(flags):
            # Numeric notifications cannot change ownership; see TreeKineticState.
            return False
        solver = self.solver
        if self.coverage is not None:
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
        # Each group's plan binds the same group map it was built from.
        proven = [group.validate_model(flags) for group in self.groups]
        return bool(proven) and all(proven)

    def invalidate(self, world_mask=None):
        """Invalidate all component caches for selected worlds."""
        for group in self.groups:
            group.invalidate(world_mask)

    def forget_continuity(self):
        """Launch every component's next repair (see :meth:`TreeKineticState.forget_continuity`)."""
        for group in self.groups:
            group.forget_continuity()

    def begin(self, state, augmented, velocity, dt, refresh, *, state_out=None):
        """Repair every current component before any factor or force consumer (see :meth:`TreeKineticState.begin`)."""
        for group in self.groups:
            group.begin(state, augmented, velocity, dt, refresh, state_out=state_out)

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


def options_decline(solver) -> str | None:
    """Why the device or a test pin keeps ordinary dynamics, or ``None``.

    The CPU keeps the serial passes, and so does the test pin ``evaluator='ordinary'``. Only the held tree evaluates
    the native sparse factor and the compact diagonal, so their admissions read this before :func:`install` does.
    """
    if solver.model.device.is_cpu:
        return "CPU uses ordinary dynamics; cached prediction requires CUDA."
    if solver._route_pins.get("evaluator") == "ordinary":
        return "A test pin keeps ordinary dynamics (evaluator='ordinary')."
    return None


def install(solver) -> None:
    """Install the held tree as ``solver._tree_kinetic_state`` where :func:`options_decline` and :func:`decline` admit
    it, with the reason.

    The native sparse factor and the compact diagonal are the held tree's representations: they are chosen only
    where :func:`sparse_decline` and :func:`compact_decline` admit them, so the held tree takes every solver that
    holds one.
    """
    solver._tree_kinetic_state = None
    reason = options_decline(solver)
    if reason is None:
        reason = decline(solver)
        if reason is None:
            try:
                solver._tree_kinetic_state = HeldTree(solver, admission=admitted)
            except ValueError as error:
                reason = f"The topology does not fit the held groups: {error}"
        if reason is None:
            solver._fk_id_cache_enabled = True
            reason = (
                "Automatic cache selection: tree cache admitted for this topology, representation and state lifecycle."
            )
        else:
            reason = f"Ordinary dynamics retained: {reason}"
    solver._cached_dynamics_selection_reason = reason
