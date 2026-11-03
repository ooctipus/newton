# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The held-tree owner: the evaluator of articulated dynamics, its groups chosen from the topology.

One owner (:class:`HeldTree`) composes the groups that evaluate every articulation
(:class:`~.tree_kinetic_owner.TreeKineticState`, plus the scalar branches' group). The solver's data chooses them:
the native sparse factor, when the solver admitted one, makes one complete group (:class:`TreeSparseOperator`);
otherwise each size group is a group that responds by its walk or by the compact diagonal
(:class:`TreeHeldOperator`), kinematic free bodies a zero-response group (:class:`PrescribedRootOperator`), or one
group per world spans its articulations (:class:`ForestHeldOperator`). Every group's collection writes its drives' K-free
mass (``row_H``) where it forms the mass diagonal; the compact diagonal's refresh forms its inverse mass.
:func:`decline` names why a representation does not fit; :func:`install` returns the owner.
"""

import itertools
from types import SimpleNamespace

import numpy as np
import warp as wp

from . import host_index
from .device_proof import DeviceProof
from .sparse_factor import SparseData, bind_operator_plan, operator_metadata
from .tree_kinetic_owner import TreeKineticState, launch_switches, proof_runs, stage_clocks_on
from .tree_kinetic_plan import FREE_JOINTS, make_plan, tree_joints
from .tree_kinetic_state import HeldResponse
from .tree_plan import TreePlan, local_parents


@wp.kernel(enable_backward=False)
def _diagonal_geometry(
    group_to_art: wp.array[int],
    dof_start: wp.array[int],
    mask: wp.array[int],
    offset: int,
    geometry_group: wp.array[int],
    diagonal: wp.array2d[float],
    armature: wp.array[float],
    drive_row: wp.array[int],
    drive_k: wp.array[float],
    n: int,
    inverse: wp.array[float],
):
    """The compact diagonal's inverse mass ``1 / (H_ii + R + K)`` from ``H_ii`` as the collection keeps it (its held
    group's coordinate ``offset + local``), with the ``K`` the drives hold now."""
    element = wp.tid()
    group, local = element // n, element % n
    art = group_to_art[group]
    if mask[art] == 0:
        return
    dof = dof_start[art] + local
    mass = diagonal[geometry_group[group], offset + local] + armature[dof]
    row = drive_row[dof]
    if row >= 0:
        if drive_k[row] > 0.0:
            mass += drive_k[row]
    inverse[dof] = 1.0 / mass


def decline(solver, *, sparse: bool | None = None) -> str | None:
    """Why the held tree does not evaluate a representation of the solver's dynamics, or ``None`` when it does.

    The terms of the representation: the native sparse factor's group (``sparse``: whether it carries velocity-limit
    rows; by default the representation the solver chose, and its admission passes its candidate through
    :func:`sparse_decline`), then the drive rows every dense group reads. A term is read only once the earlier ones
    hold.
    """
    if sparse is None and getattr(solver.placement, "sparse_mass_matrix_size", None) is not None:
        sparse = solver.sweep.sparse_velocity_limits
    # The owner holds no rows, impulses or patch frames: the patch build, link, seed and finish stay on the solver
    # pipeline around it, so it serves every contact law. Torsion rows read the joint motions and body twists every
    # group publishes (joint_S_s, body_v_s); compliance reads no dynamics.
    if sparse is not None:
        velocity_limits = sparse
        if solver.forest.has_free_rigid_bodies:
            return "The sparse factor's group holds no free rigid bodies."
        if solver.laws.mimic_count and not (
            velocity_limits and solver.laws.mimic_valid_np is not None and np.all(solver.laws.mimic_valid_np)
        ):
            return "Mimic rows on the sparse factor need its velocity-limit rows and valid mimic pairs."
        if solver.forest.has_prescribed_response:
            return "The sparse factor's group holds no prescribed articulations."
    if not solver.factor.parallel_drives and solver.forest.size_groups:
        return "The held tree needs the per-coordinate drive preparation (augmented drives, owned drive rows)."
    return None


def admitted(solver) -> bool:
    """Whether the held tree evaluates the solver's dynamics (:func:`decline`); its groups re-check it on every
    call."""
    return decline(solver) is None


def _prescribed_articulations(solver):
    """Articulations whose response the solver prescribes (kinematic free bodies)."""
    return np.flatnonzero(solver.forest.prescribed_articulation.numpy() != 0).astype(np.int32)


def _coverage(solver, compact):
    """Prove disjoint articulation and physical coordinate ownership; ``compact`` is the compact diagonal size the
    groups carry. An articulation no size group holds has no coordinates (:class:`RestOperator`), and a body outside
    the articulations stays outside the solver."""
    model = solver.model
    prescribed = _prescribed_articulations(solver)
    owned, _ = tree_joints(model)
    arts = np.concatenate(
        [solver.forest.group_to_art[size].numpy() for size in solver.forest.size_groups] + [prescribed]
    )
    if np.unique(arts).size != arts.size:
        raise ValueError("Mixed tree factors must hold an articulation at most once")
    starts = model.articulation_start.numpy()
    if starts[0] != 0 or starts[-1] != model.joint_count or np.any(np.diff(starts) <= 0):
        raise ValueError("Mixed tree cache requires contiguous nonempty articulations")
    qd = model.joint_qd_start.numpy()
    ends = solver.forest.articulation_joint_end_host
    ds = solver.forest.articulation_dof_start.numpy()
    if not np.array_equal(ds, qd[starts[:-1]]):
        raise ValueError("Mixed tree factor starts must preserve physical coordinate ownership")
    for size in solver.forest.size_groups:
        group = solver.forest.group_to_art[size].numpy()
        if np.any(qd[ends[group]] - qd[starts[group]] != size):
            raise ValueError("Mixed tree factor size disagrees with its physical coordinates")
        if solver.forest.n_arts_by_size[size] != len(group):
            raise ValueError("Mixed tree factor count disagrees with its group map")
    if prescribed.size and np.any(qd[ends[prescribed]] - qd[starts[prescribed]] != 6):
        raise ValueError("Prescribed cached roots must be six-coordinate free bodies")
    if np.unique(model.joint_child.numpy()[owned]).size != np.count_nonzero(owned):
        raise ValueError("Mixed tree cache must publish a body at most once")
    return (
        tuple(solver.forest.size_groups),
        tuple(sorted(solver.forest.n_arts_by_size.items())),
        compact,
        int(prescribed.size),
    )


def _plan_view(group_to_art, art_to_world, host, *, prescribed=False):
    """What :func:`.make_plan` reads of a group's operator: its articulations, their worlds and its metadata."""
    return SimpleNamespace(
        plan=SimpleNamespace(group_to_art=group_to_art, art_to_world=art_to_world), host=host, prescribed=prescribed
    )


def _topologies(solver, size) -> list:
    """The ranges of size group ``size`` that hold one topology each (:attr:`.forest_plan.ForestPlan.size_topologies`)."""
    bounds = solver.forest.size_topologies[size]
    return [(int(first), int(last)) for first, last in itertools.pairwise(bounds)]


def _size_group(solver, size, span, *, diagonal):
    """The range ``span`` of size group ``size`` (:func:`_topologies`) as a held group: whether its trees stand on
    free roots, their mass pattern, its metadata and its layout (:func:`.make_plan`). Raises ValueError where the
    trees do not fit a held group."""
    model = solver.model
    group_to_art = solver.forest.group_to_art[size][span[0] : span[1]]
    first_art = int(group_to_art.numpy()[0])
    first, end = model.articulation_start.numpy()[first_art], solver.forest.articulation_joint_end_host[first_art]
    parents = local_parents(model.joint_ancestor.numpy(), [first], end - first)[0]
    counts = np.diff(model.joint_qd_start.numpy())[first:end]
    root_type = model.joint_type.numpy()[first]
    # A free rigid body is a one-body tree on a free root: its six coordinates
    # share the held Cholesky action and the generic free-root transport. Any
    # other root's coordinates (none, or a joint's) are a descendant's.
    free_root = counts[0] == 6 and root_type in FREE_JOINTS
    mass = TreePlan.build(parents, counts).factor
    if mass.dof_count != size:
        raise ValueError("Mixed tree physical and factor dimensions differ")
    if diagonal and mass.nonzero_count != size:
        raise ValueError("Compact diagonal cache requires independent physical coordinates")
    host = operator_metadata(mass)
    layout, _ = make_plan(model, _plan_view(group_to_art, solver.forest.art_to_world, host), complete=False)
    return bool(free_root), mass, host, layout


def _prescribed_group(solver, group_to_art):
    """The prescribed free bodies as a held group: their mass pattern, metadata and layout (:func:`.make_plan`)."""
    mass = TreePlan.build([-1], [6]).factor
    host = operator_metadata(mass)
    view = _plan_view(group_to_art, solver.forest.art_to_world, host, prescribed=True)
    layout, _ = make_plan(solver.model, view, complete=False)
    return mass, host, layout


def sparse_decline(solver, size, plan, *, velocity_limits) -> str | None:
    """Why the held tree would not take a native sparse factor of ``size`` coordinates and pattern ``plan``, or
    ``None``: :func:`decline` of the candidate, then its one complete group's fit.

    Only the held tree evaluates native sparse factors, so their admission reads this before choosing one.
    """
    reason = decline(solver, sparse=velocity_limits)
    if reason is not None:
        return reason
    view = _plan_view(solver.forest.group_to_art[size], solver.forest.art_to_world, operator_metadata(plan))
    try:
        make_plan(solver.model, view, complete=True)
    except ValueError as error:
        return f"The topology does not fit the held groups: {error}"
    return None


def compact_decline(solver, size, *, scalar) -> str | None:
    """Why the held tree would not take the dense size groups with a compact diagonal of ``size`` coordinates, or
    ``None``: :func:`decline`, then the groups' fit (the coverage, the
    diagonal group unless ``scalar`` branches take it, every other size group, the prescribed roots).

    Only the held tree evaluates a compact diagonal, so the sparse-diagonal response reads this before choosing one.
    """
    reason = decline(solver)
    if reason is not None:
        return reason
    try:
        _coverage(solver, size)
        for group_size in solver.forest.size_groups:
            if group_size != size or not scalar:
                for span in _topologies(solver, group_size):
                    _size_group(solver, group_size, span, diagonal=group_size == size)
        prescribed = _prescribed_articulations(solver)
        if prescribed.size:
            _prescribed_group(solver, wp.array(prescribed, dtype=int, device=solver.model.device))
    except ValueError as error:
        return f"The topology does not fit the held groups: {error}"
    return None


class TreeSparseOperator:
    """Respond through the walk over the sparse class's pattern (its plan): the walk is the factor (Z4.3), refreshed by
    :meth:`.tree_factor.TreeFactor.walk_step`, so the group holds no factor array and refreshes nothing of its own."""

    def __init__(self, solver):
        self.solver = solver
        self.requires_body_velocity = True
        mass = solver.factor.sparse_mass_matrix_plan
        device = solver.model.device
        self.host = operator_metadata(mass)
        self.plan = bind_operator_plan(solver, mass.dof_count, self.host)
        self.data = SparseData()
        # The walk has no factor status: a pivot that is not positive reaches the predictor as a non-finite response.
        self.data.valid = wp.ones(self.plan.group_to_art.size, dtype=int, device=device)
        self.data.status = wp.zeros(self.plan.group_to_art.size, dtype=int, device=device)
        self.diagonal_mass = False
        self.kinetic_state = None

    def refresh(self, state_aug):
        """Nothing to refresh: the walk's refresh is the factor's, and the drives' K-free mass (``row_H``) is the
        finish's collection's, where it forms ``H_ii``."""

    def bind_response(self):
        """Nothing to bind: the predictor reads the walk."""


class TreeHeldOperator:
    """The range ``span`` of a size group as a held group: the trees of one topology. A walking group refreshes nothing of its own (the walk's refresh is its factor, its collection
    writes ``row_H``); the compact diagonal refreshes its inverse mass from the held geometry."""

    def __init__(self, solver, size, span):
        self.solver, self.size, self.span = solver, size, span
        self.requires_body_velocity = True
        device = solver.model.device
        count = span[1] - span[0]
        self.diagonal = size == solver._compact_diagonal_mass_size
        # The compact diagonal's refresh reads H's diagonal, which the collection keeps.
        self.diagonal_mass = self.diagonal
        self.free_root, mass, self.host, self.layout = _size_group(solver, size, span, diagonal=self.diagonal)
        self.mass = mass
        self.plan = bind_operator_plan(
            solver, size, self.host, group_to_art=solver.forest.group_to_art[size][span[0] : span[1]]
        )
        self.data = HeldResponse()
        self.data.valid = wp.ones(count, dtype=int, device=device)
        self.data.status = wp.zeros(count, dtype=int, device=device)
        if self.diagonal:
            self.data.inverse = solver.factor.diagonal_inverse_mass
            self.geometry_group = wp.array(np.arange(count, dtype=np.int32), dtype=int, device=device)
        self.kinetic_state = None

    def refresh(self, state_aug):
        """The compact diagonal's inverse mass from the requested groups' current geometry; a walking group has nothing
        to refresh."""
        if not self.diagonal:
            return
        solver, size = self.solver, self.size
        count = self.span[1] - self.span[0]
        wp.launch(
            _diagonal_geometry,
            dim=count * size,
            inputs=[
                self.plan.group_to_art,
                solver.forest.articulation_dof_start,
                solver.factor.mass_update_mask,
                0,
                self.geometry_group,
                self.kinetic_state.mass_diagonal,
                solver.factor.dof_armature,
                solver.factor.drive_row_by_dof,
                solver.factor.aug_row_K,
                size,
            ],
            outputs=[solver.factor.diagonal_inverse_mass],
            device=solver.model.device,
        )

    def bind_response(self):
        """Bind the compact diagonal's inverse before a new launch or capture; a walking group binds nothing."""
        if self.diagonal:
            self.data.inverse = self.solver.factor.diagonal_inverse_mass


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
        self.diagonal_mass = False
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


class RestOperator:
    """Own the trees of the range ``span`` of the articulations without coordinates: a held group of none.

    Nothing responds, so there is no factor and every stage's loop over coordinates runs no time: the group holds
    the poses its joint frames give, as the tree law writes them.
    """

    def __init__(self, solver, span):
        self.solver, self.size = solver, 0
        self.requires_body_velocity = True
        model, device = solver.model, solver.model.device
        arts = solver.forest.rest_articulations[span[0] : span[1]]
        self.free_root = False
        self.diagonal = True
        self.diagonal_mass = False
        self.group_to_art = wp.array(arts, dtype=int, device=device)
        first, end = model.articulation_start.numpy()[arts[0]], solver.forest.articulation_joint_end_host[arts[0]]
        parents = local_parents(model.joint_ancestor.numpy(), [first], end - first)[0]
        self.mass = TreePlan.build(parents, np.zeros(end - first, np.int32)).factor
        self.host = operator_metadata(self.mass)
        self.layout, _ = make_plan(
            model, _plan_view(self.group_to_art, solver.forest.art_to_world, self.host), complete=False
        )
        self.plan = bind_operator_plan(solver, 0, self.host, group_to_art=self.group_to_art)
        self.data = HeldResponse()
        self.data.valid = wp.ones(len(arts), dtype=int, device=device)
        self.data.status = wp.zeros(len(arts), dtype=int, device=device)
        self.data.inverse = solver._arrays.placeholder(wp.float32, (model.joint_dof_count,))
        self.kinetic_state = None

    def refresh(self, state_aug):
        """No coordinate, no factor."""

    def bind_response(self):
        """Nothing rebinds."""


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
    worlds = int(solver.forest.world_count)
    art_world = solver.forest.art_to_world.numpy()
    prescribed = solver.forest.prescribed_articulation.numpy()
    counts = solver.forest.articulation_dof_count_host
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
    # One group holds a world's trees: at most 64 bodies and coordinates, each family a warp's lanes per row.
    bodies = solver.forest.articulation_joint_end_host - model.articulation_start.numpy()[:-1]
    if bodies[per_world[0]].sum() > 64 or counts[per_world[0]].sum() > 64:
        return None
    signature = [(int(counts[a]), int(prescribed[a])) for a in per_world[0]]
    families = [0 if pres else size for size, pres in signature]
    for size in sorted({f for f in families if f}):
        if len(_topologies(solver, size)) > 1:
            return None
        if size == solver._compact_diagonal_mass_size:
            if solver.factor.diagonal_inverse_mass is None:
                return None
        elif not 1 <= size <= 32 or solver._execution_plan.use_diagonal_mass(size):
            return None
        group = solver.forest.group_to_art[size].numpy()
        if len(group) != worlds or not np.array_equal(art_world[group], np.arange(worlds)):
            return None
    return families


class ForestHeldOperator:
    """Own every articulation of a world as one group: shared scratch, one launch per phase.

    Each root responds by its walk, by the compact diagonal (refreshed from
    the forest's geometry) or with zero when prescribed. Roots, families and
    offsets come from the topology.
    """

    def __init__(self, solver):
        self.solver = solver
        self.requires_body_velocity = True
        self.free_root = True
        self.diagonal = False
        model, device = solver.model, solver.model.device
        worlds = int(solver.forest.world_count)
        art_world = solver.forest.art_to_world.numpy()
        prescribed = solver.forest.prescribed_articulation.numpy()
        counts = solver.forest.articulation_dof_count_host
        # Each world's articulations in ascending order, one row per world (forest_families admits equal counts).
        world_bounds, order = host_index.group_by(art_world, worlds)
        arts = order.reshape(worlds, int(world_bounds[1] - world_bounds[0])).astype(np.int32)
        self.forest_arts = arts
        # Family per root: its coordinate count for a walking root, negated
        # for a diagonal-mass size (compact inverse), 0 when prescribed.
        diagonal = solver._compact_diagonal_mass_size
        self.root_families = tuple(
            0 if prescribed[a] else (-int(counts[a]) if counts[a] == diagonal else int(counts[a])) for a in arts[0]
        )
        first = int(model.articulation_start.numpy()[arts[0, 0]])
        end = int(solver.forest.articulation_joint_end_host[arts[0, -1]])
        parents = local_parents(model.joint_ancestor.numpy(), [first], end - first)[0]
        joint_counts = np.diff(model.joint_qd_start.numpy())[first:end]
        self.mass = mass = TreePlan.build(parents, joint_counts).factor
        self.host = operator_metadata(mass)
        self.group_to_art = wp.array(arts[:, 0].copy(), dtype=int, device=device)
        self.plan = bind_operator_plan(solver, 0, self.host, group_to_art=self.group_to_art)
        self.size = mass.dof_count
        self.layout, _ = make_plan(model, self, complete=False)
        self.data = HeldResponse()
        self.data.valid = wp.ones(worlds, dtype=int, device=device)
        self.data.status = wp.zeros(worlds, dtype=int, device=device)
        self.data.inverse = (
            solver.factor.diagonal_inverse_mass
            if any(f < 0 for f in self.root_families)
            else wp.zeros(model.joint_dof_count, dtype=float, device=device)
        )
        # Each diagonal-mass root's refresh reads its coordinates' H_ii from the forest's collection (from the root's
        # offset); a walking root refreshes nothing of its own.
        self.diagonal_mass = any(f < 0 for f in self.root_families)
        self.families = []
        for r, family in enumerate(self.root_families):
            if family >= 0:
                continue
            size = -family
            group = solver.forest.group_to_art[size].numpy()
            self.families.append(
                (
                    size,
                    int(self.layout.root_offsets[r]),
                    wp.array(art_world[group].astype(np.int32), dtype=int, device=device),
                )
            )
        self.kinetic_state = None

    def refresh(self, state_aug):
        """Refresh every diagonal-mass root's inverse mass from the forest geometry."""
        solver = self.solver
        for size, offset, geometry_group in self.families:
            wp.launch(
                _diagonal_geometry,
                dim=solver.forest.n_arts_by_size[size] * size,
                inputs=[
                    solver.forest.group_to_art[size],
                    solver.forest.articulation_dof_start,
                    solver.factor.mass_update_mask,
                    offset,
                    geometry_group,
                    self.kinetic_state.mass_diagonal,
                    solver.factor.dof_armature,
                    solver.factor.drive_row_by_dof,
                    solver.factor.aug_row_K,
                    size,
                ],
                outputs=[solver.factor.diagonal_inverse_mass],
                device=solver.model.device,
            )

    def bind_response(self):
        """Nothing rebinds: the walking roots read the walk, the diagonal roots the solver's inverse."""


class HeldTree:
    """Evaluate every articulation's dynamics in held groups chosen from the topology (see the module docstring).

    ``admission`` (:func:`admitted` when installed) is re-checked by every group on every call.
    """

    def __init__(self, solver, *, admission=None):
        self.solver = solver
        # One read of the kernel and launch switches for every group this owner composes.
        self.clocked = stage_clocks_on()
        skip, classify = launch_switches(solver._route_pins)
        self.scalar = None
        if not (solver.model.joint_count and solver.model.articulation_count):
            # Nothing to evaluate: the step reads no group.
            self.coverage = self.forest_reason = None
            self.groups = []
            self.bank_compare = self.skip_captured_repairs = True
            return
        if solver.placement.sparse_mass_matrix_size is not None:
            # The native sparse factor: one complete group, which proves its own coverage (make_plan).
            self.coverage = self.forest_reason = None
            operator = TreeSparseOperator(solver)
            group = TreeKineticState(
                operator,
                admission=admission,
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
                TreeHeldOperator(solver, size, span)
                for size in solver.forest.size_groups
                if scalar is None or size != scalar.size
                for span in _topologies(solver, size)
            ]
            if self.coverage[-1]:
                operators.append(PrescribedRootOperator(solver))
            operators += [RestOperator(solver, span) for span in itertools.pairwise(solver.forest.rest_topologies)]
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
        for size in solver.forest.size_groups:
            yield f"group_to_art[{size}]", solver.forest.group_to_art[size]
        yield "articulation_start", model.articulation_start
        yield "joint_qd_start", model.joint_qd_start
        yield "joint_child", model.joint_child
        yield "articulation_dof_start", solver.forest.articulation_dof_start
        yield "prescribed_articulation", solver.forest.prescribed_articulation

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
                tuple(solver.forest.size_groups),
                tuple(sorted(solver.forest.n_arts_by_size.items())),
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


def install(solver) -> HeldTree:
    """The owner of the solver's dynamics. A model it does not fit raises ``ValueError``, naming why.

    The native sparse factor and the compact diagonal are the held tree's representations: they are chosen only
    where :func:`sparse_decline` and :func:`compact_decline` admit them.
    """
    reason = decline(solver)
    if reason is not None:
        raise ValueError(f"FeatherPGS cannot hold this model: {reason}")
    try:
        return HeldTree(solver, admission=admitted)
    except ValueError as error:
        raise ValueError(f"FeatherPGS cannot hold this model: {error}") from error
