# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The held-tree owner: the evaluator of articulated dynamics, its groups chosen from the topology.

One owner (:class:`HeldTree`) composes the groups that evaluate every articulation
(:class:`~.tree_kinetic_owner.TreeKineticState`, plus the scalar branches' group), each over one :class:`HeldGroup`
whose kind the solver's data chooses: the sparse class, when the solver admitted one, makes one complete group that
walks its pattern (``"sparse"``); otherwise each size group is a group that responds by its walk or by the compact
diagonal (``"tree"``), kinematic free bodies a zero-response group (``"prescribed"``), articulations without
coordinates a group of none (``"rest"``), or one group per world spans its articulations (``"forest"``). Every group's
collection writes its drives' K-free mass (``row_H``) where it forms the mass diagonal; the compact diagonal's refresh
forms its inverse mass. :func:`decline` names why a representation does not fit; :func:`install` returns the owner.
"""

import itertools
from types import SimpleNamespace

import numpy as np
import warp as wp

from ...sim.contacts import capture_scope
from . import field_group, host_index
from .device_proof import DeviceProof
from .tree_kinetic_owner import TreeKineticState, proof_runs
from .tree_kinetic_plan import make_plan, tree_joints
from .tree_kinetic_state import GroupTrees, HeldResponse
from .tree_plan import TreePlan, local_parents


def _continuation(state, dt: float) -> tuple | None:
    """What a finish publishing into ``state`` leaves the next step's repair: the bank (its coordinates and velocities),
    the capture scope that records it (``None`` eagerly) and the step's duration; ``None`` inside a capture Warp does
    not know, whose steps nothing can tell apart from another capture's."""
    device = state.joint_q.device
    scope = capture_scope(device)
    if scope is None and device.is_capturing:
        return None
    return scope, state.joint_q.ptr, state.joint_qd.ptr, dt


def operator_metadata(mass) -> dict:
    """What the held groups read on the host of a tree's mass pattern ``mass`` (:class:`.tree_plan.FactorPattern`):
    its coordinate lookup (``index``), each coordinate's joint (``dof_joint``) and its inverse columns' rows
    (``inverse_nodes``, whose width the layout records)."""
    counts = np.diff(mass.column_offsets).astype(np.int32)
    nodes = np.full((mass.dof_count, int(counts.max(initial=0))), -1, np.int32)
    col, rank = host_index.expand(counts)
    nodes[col, rank] = mass.entry_rows[mass.column_entries]
    return {"index": mass.lookup, "dof_joint": mass.dof_joint, "inverse_nodes": nodes}


def bind_group_trees(solver, host, group_to_art) -> GroupTrees:
    """A held group's trees (:data:`.tree_kinetic_state.GROUP_TREES`): the articulations ``group_to_art`` and the
    template tree's joint of each coordinate from ``host`` (:func:`operator_metadata`)."""
    model = solver.model
    return field_group.upload(
        GroupTrees,
        model.device,
        dof_joint=host["dof_joint"],
        group_to_art=group_to_art,
        art_to_world=solver.forest.art_to_world,
        art_dof_start=solver.forest.articulation_dof_start,
        art_joint_start=model.articulation_start,
        joint_child=model.joint_child,
    )


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


def _prescribed_articulations(forest):
    """Articulations whose response the solver prescribes (kinematic free bodies)."""
    return np.flatnonzero(forest.prescribed_articulation.numpy() != 0).astype(np.int32)


def _coverage(solver, compact):
    """Prove disjoint articulation and physical coordinate ownership; ``compact`` is the compact diagonal size the
    groups carry. An articulation no size group holds has no coordinates (a ``"rest"`` group), and a body outside
    the articulations stays outside the solver."""
    model = solver.model
    prescribed = _prescribed_articulations(solver.forest)
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


def _template(solver, first_art, last_art=None, *, coordinates=True):
    """The mass pattern of the trees of articulations ``first_art`` through ``last_art`` (one tree by default; a
    world's forest in local order), from their local parents and each joint's coordinates (none without
    ``coordinates``)."""
    model = solver.model
    first = int(model.articulation_start.numpy()[first_art])
    end = int(solver.forest.articulation_joint_end_host[first_art if last_art is None else last_art])
    parents = local_parents(model.joint_ancestor.numpy(), [first], end - first)[0]
    counts = np.diff(model.joint_qd_start.numpy())[first:end] if coordinates else np.zeros(end - first, np.int32)
    return TreePlan.build(parents, counts).factor


def _fit(solver, group_to_art, mass, *, prescribed=False, complete=False):
    """The metadata (:func:`operator_metadata`) of a held group of the trees ``group_to_art`` on the pattern
    ``mass``, once its layout fits (:func:`.make_plan`; raises ValueError where it does not)."""
    host = operator_metadata(mass)
    plan = SimpleNamespace(group_to_art=group_to_art, art_to_world=solver.forest.art_to_world)
    make_plan(solver.model, SimpleNamespace(plan=plan, host=host, prescribed=prescribed), complete=complete)
    return host


def _topologies(forest, size) -> list:
    """The ranges of size group ``size`` that hold one topology each (:attr:`.forest_plan.ForestPlan.size_topologies`)."""
    bounds = forest.size_topologies[size]
    return [(int(first), int(last)) for first, last in itertools.pairwise(bounds)]


def _size_trees(solver, size, span, *, diagonal):
    """The range ``span`` of size group ``size`` (:func:`_topologies`): its articulations, mass pattern and metadata.
    Raises ValueError where the trees do not fit a held group."""
    group_to_art = solver.forest.group_to_art[size][span[0] : span[1]]
    mass = _template(solver, int(group_to_art.numpy()[0]))
    if mass.dof_count != size:
        raise ValueError("Mixed tree physical and factor dimensions differ")
    if diagonal and mass.nonzero_count != size:
        raise ValueError("Compact diagonal cache requires independent physical coordinates")
    return group_to_art, _fit(solver, group_to_art, mass)


def _prescribed_trees(solver):
    """The prescribed free bodies: their articulations and the metadata of their one-body six-coordinate pattern."""
    group_to_art = wp.array(_prescribed_articulations(solver.forest), dtype=int, device=solver.model.device)
    return group_to_art, _fit(solver, group_to_art, TreePlan.build([-1], [6]).factor, prescribed=True)


def sparse_decline(solver, size, plan, *, velocity_limits) -> str | None:
    """Why the held tree would not take a native sparse factor of ``size`` coordinates and pattern ``plan``, or
    ``None``: :func:`decline` of the candidate, then its one complete group's fit.

    Only the held tree evaluates native sparse factors, so their admission reads this before choosing one.
    """
    reason = decline(solver, sparse=velocity_limits)
    if reason is not None:
        return reason
    try:
        _fit(solver, solver.forest.group_to_art[size], plan, complete=True)
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
                for span in _topologies(solver.forest, group_size):
                    _size_trees(solver, group_size, span, diagonal=group_size == size)
        if _prescribed_articulations(solver.forest).size:
            _prescribed_trees(solver)
    except ValueError as error:
        return f"The topology does not fit the held groups: {error}"
    return None


class HeldGroup:
    """One held group: the trees ``group_to_art`` (for a forest, each world's first; ``forest_arts`` holds them all)
    on one pattern, whose metadata is ``host`` (:func:`operator_metadata`), with their response's status.

    ``kind`` names what the solver's data chose (the builders below): ``"sparse"``, ``"tree"``, ``"prescribed"``,
    ``"rest"`` or ``"forest"``. Every group responds through the walk, whose refresh is the factor's, or with zero; its
    collection writes the drives' K-free mass (``row_H``) where it forms ``H_ii``. Its own refresh forms only the
    inverse mass of its compact diagonal ``families``, each ``(size, offset, group_to_art, geometry_group)``: the
    articulations of one diagonal size and, per articulation, the group whose collection keeps its ``H_ii`` from
    coordinate ``offset``.
    """

    requires_body_velocity = True

    def __init__(
        self,
        solver,
        kind,
        group_to_art,
        host,
        *,
        size,
        response="cholesky",
        families=(),
        forest_arts=None,
        root_families=(),
    ):
        self.solver, self.kind, self.size, self.host, self.response = solver, kind, size, host, response
        self.prescribed = kind == "prescribed"
        self.families = families
        # The compact diagonal's refresh reads H's diagonal, which the collection keeps.
        self.diagonal_mass = bool(families)
        self.forest_arts, self.root_families = forest_arts, root_families
        self.plan = bind_group_trees(solver, host, group_to_art)
        count = self.plan.group_to_art.size
        self.data = HeldResponse()
        # The walk has no factor status: a pivot that is not positive reaches the predictor as a non-finite response.
        self.data.valid = wp.ones(count, dtype=int, device=solver.model.device)
        self.data.status = wp.zeros(count, dtype=int, device=solver.model.device)
        self.kinetic_state = None

    def refresh(self, state_aug):
        """Each compact diagonal family's inverse mass from its groups' current geometry."""
        solver = self.solver
        for size, offset, group_to_art, geometry_group in self.families:
            wp.launch(
                _diagonal_geometry,
                dim=group_to_art.size * size,
                inputs=[
                    group_to_art,
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


def sparse_group(solver) -> HeldGroup:
    """The sparse class's trees as one complete group: it responds through the walk over the class's pattern
    (:attr:`.tree_factor.TreeFactor.sparse_pattern`), refreshed by :meth:`.tree_factor.TreeFactor.walk_step`."""
    mass = solver.factor.sparse_pattern
    trees = solver.forest.group_to_art[mass.dof_count]
    return HeldGroup(solver, "sparse", trees, operator_metadata(mass), size=None, response="sparse")


def _tree_group(solver, size, span) -> HeldGroup:
    """The range ``span`` of a size group as a held group: the trees of one topology. A walking group refreshes
    nothing of its own; the compact diagonal refreshes its inverse mass from the held geometry."""
    diagonal = size == solver._compact_diagonal_mass_size
    group_to_art, host = _size_trees(solver, size, span, diagonal=diagonal)
    if not diagonal:
        return HeldGroup(solver, "tree", group_to_art, host, size=size)
    count = span[1] - span[0]
    geometry = wp.array(np.arange(count, dtype=np.int32), dtype=int, device=solver.model.device)
    return HeldGroup(
        solver,
        "tree",
        group_to_art,
        host,
        size=size,
        response="diagonal",
        families=((size, 0, group_to_art, geometry),),
    )


def _prescribed_group(solver) -> HeldGroup:
    """Own prescribed free bodies as zero-response roots of the same tree law.

    A kinematic free body has six prescribed coordinates and no factor. With no
    walking class (its root's class is ``None``), the shared predictor yields zero
    acceleration and the prescribed velocity, the free-root transport is
    skipped on kinematic roots, and the shared finish integrates the pose and
    publishes the body exactly as it does for a dynamic free body. This is
    the generic form of a per-world free-root slot.
    """
    group_to_art, host = _prescribed_trees(solver)
    return HeldGroup(solver, "prescribed", group_to_art, host, size=6, response="diagonal")


def _rest_group(solver, span) -> HeldGroup:
    """Own the trees of the range ``span`` of the articulations without coordinates: a held group of none.

    Nothing responds, so there is no factor and every stage's loop over coordinates runs no time: the group holds
    the poses its joint frames give, as the tree law writes them.
    """
    arts = solver.forest.rest_articulations[span[0] : span[1]]
    group_to_art = wp.array(arts, dtype=int, device=solver.model.device)
    host = _fit(solver, group_to_art, _template(solver, arts[0], coordinates=False))
    return HeldGroup(solver, "rest", group_to_art, host, size=0, response="diagonal")


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
    if decline(solver) is not None:
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
        if len(_topologies(solver.forest, size)) > 1:
            return None
        if size == solver._compact_diagonal_mass_size:
            if solver.factor.diagonal_inverse_mass is None:
                return None
        elif not 1 <= size <= 32 or (solver.model.device.is_cuda and size in solver.forest.diagonal_mass_sizes):
            # A star's group other than the compact diagonal's: on CUDA the retired dense response plan kept it out of
            # the held family (its diagonal generalized inertia); the CPU's held family takes it.
            return None
        group = solver.forest.group_to_art[size].numpy()
        if len(group) != worlds or not np.array_equal(art_world[group], np.arange(worlds)):
            return None
    return families


def _forest_group(solver) -> HeldGroup:
    """Own every articulation of a world as one group: shared scratch, one launch per phase.

    Each root responds by its walk, by the compact diagonal (refreshed from
    the forest's geometry) or with zero when prescribed. Roots, families and
    offsets come from the topology.
    """
    device = solver.model.device
    worlds = int(solver.forest.world_count)
    art_world = solver.forest.art_to_world.numpy()
    prescribed = solver.forest.prescribed_articulation.numpy()
    counts = solver.forest.articulation_dof_count_host
    # Each world's articulations in ascending order, one row per world (forest_families admits equal counts).
    world_bounds, order = host_index.group_by(art_world, worlds)
    arts = order.reshape(worlds, int(world_bounds[1] - world_bounds[0])).astype(np.int32)
    # Family per root: its coordinate count for a walking root, negated
    # for a diagonal-mass size (compact inverse), 0 when prescribed.
    diagonal = solver._compact_diagonal_mass_size
    roots = tuple(
        0 if prescribed[a] else (-int(counts[a]) if counts[a] == diagonal else int(counts[a])) for a in arts[0]
    )
    mass = _template(solver, arts[0, 0], arts[0, -1])
    # Each diagonal-mass root's refresh reads its coordinates' H_ii from the forest's collection, from the root's
    # offset (the coordinates of the world's trees before it).
    offsets = np.concatenate(([0], np.cumsum(counts[arts[0]])))
    families = tuple(
        (
            -family,
            int(offsets[r]),
            solver.forest.group_to_art[-family],
            wp.array(art_world[solver.forest.group_to_art[-family].numpy()].astype(np.int32), dtype=int, device=device),
        )
        for r, family in enumerate(roots)
        if family < 0
    )
    group_to_art = wp.array(arts[:, 0].copy(), dtype=int, device=device)
    return HeldGroup(
        solver,
        "forest",
        group_to_art,
        operator_metadata(mass),
        size=mass.dof_count,
        families=families,
        forest_arts=arts,
        root_families=roots,
    )


class HeldTree:
    """Evaluate every articulation's dynamics in held groups chosen from the topology (see the module docstring)."""

    def __init__(self, solver):
        self.solver = solver
        # One read of the kernel switches for every group this owner composes: the stage clocks, and the test pin
        # repair_classify=False (SolverFeatherPGS._route_pins), which keeps the one-pass repair over every block.
        self.clocked = solver.options.diagnostics.stage_clocks
        classify = solver._route_pins.get("repair_classify", True)
        # What the last finish left the next step (_continuation); an invalidation or a notification forgets it.
        self.continuation = None
        self.scalar = None
        if not (solver.model.joint_count and solver.model.articulation_count):
            # Nothing to evaluate: the step reads no group.
            self.coverage = self.forest_reason = None
            self.groups = []
            self.bank_compare = True
            return
        if solver.placement.sparse_mass_matrix_size is not None:
            # The sparse class: one complete group, which proves its own coverage (make_plan).
            self.coverage = self.forest_reason = scalar = None
            held = [sparse_group(solver)]
        else:
            self.coverage = _coverage(solver, solver._compact_diagonal_mass_size)
            self._dimensions = (solver.model.joint_count, solver.model.body_count, solver.model.articulation_count)
            # The coverage proof's inputs, compared on the device on notification.
            self._proof = DeviceProof(list(self._coverage_arrays()), solver.model.device)
            # A size group of scalar branches (scalar_branches.py) is owned in place
            # of a tree group: one thread per coordinate, no tree plan.
            scalar = getattr(solver, "_scalar_branches", None)
            held = self._dense_groups(solver, scalar)
        self.groups = []
        for operator in held:
            complete = operator.kind == "sparse"
            group = TreeKineticState(
                operator,
                chains=complete,
                complete=complete,
                response=operator.response,
                clocked=self.clocked,
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
            self.scalar = ScalarBranchState(solver, scalar)
            self.groups.append(self.scalar)
        # Every tree group compares input banks (fixed and free roots alike) unless the
        # solver prescales velocities; see TreeKineticState. Beside scalar branches, which
        # do not, every group validates a changed input bank by identity, and the solver
        # invalidates the whole owner.
        self.bank_compare = all(group.bank_compare for group in self.groups)

    def _dense_groups(self, solver, scalar) -> list:
        """The held groups of dense or compact coordinates: one per world (a forest) where the default or a test pin
        chooses it and the layout admits it, else one per size group's topology beside the prescribed and rest
        groups; ``forest_reason`` names the choice."""
        override = solver._route_pins.get("grouping")
        forest = override == "forest" if override is not None else forest_default(solver)
        if forest and forest_supported(solver):
            self.forest_reason = (
                "A test pin selects one group per world (grouping='forest')."
                if override is not None
                else "Default: a prescribed root and no diagonal-mass family make one group per world cheaper."
            )
            return [_forest_group(solver)]
        groups = [
            _tree_group(solver, size, span)
            for size in solver.forest.size_groups
            if scalar is None or size != scalar.size
            for span in _topologies(solver.forest, size)
        ]
        if self.coverage[-1]:
            groups.append(_prescribed_group(solver))
        groups += [_rest_group(solver, span) for span in itertools.pairwise(solver.forest.rest_topologies)]
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
        return groups

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
        self.continuation = None
        for group in self.groups:
            group.invalidate(world_mask)

    def begin(self, state, augmented, velocity, dt, *, state_out=None):
        """Repair every current component before any factor or force consumer (see :meth:`TreeKineticState.begin`).

        A step that continues the last finish (:func:`_continuation`: its input the bank that finish published into,
        in the same capture scope, at the same duration, no invalidation or notification between) finds every cache
        that finish published current, so the groups whose repair would only confirm it launch none: in a graph, the
        second step of a captured pair.
        """
        key = _continuation(state, dt)
        continues = key is not None and key == self.continuation
        self.continuation = None
        for group in self.groups:
            group.begin(state, augmented, velocity, dt, state_out=state_out, continues=continues)

    def refresh(self, augmented):
        """Preserve independent factor masks and numerical representations."""
        for group in self.groups:
            group.sparse.refresh(augmented)

    def predict(self, state, augmented, control, velocity, dt):
        """Project current forces and apply each original held response."""
        for group in self.groups:
            group.predict(state, augmented, control, velocity, dt)

    def finish(self, state, augmented, output, dt):
        """Integrate and publish each physical coordinate and body once."""
        for group in self.groups:
            group.finish(state, augmented, output, dt)
        self.continuation = _continuation(output, dt)


def install(solver) -> HeldTree:
    """The owner of the solver's dynamics. A model it does not fit raises ``ValueError``, naming why.

    The native sparse factor and the compact diagonal are the held tree's representations: they are chosen only
    where :func:`sparse_decline` and :func:`compact_decline` admit them.
    """
    reason = decline(solver)
    if reason is not None:
        raise ValueError(f"FeatherPGS cannot hold this model: {reason}")
    try:
        return HeldTree(solver)
    except ValueError as error:
        raise ValueError(f"FeatherPGS cannot hold this model: {error}") from error
