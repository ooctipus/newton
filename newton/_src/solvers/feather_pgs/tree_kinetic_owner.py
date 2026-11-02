# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Admission and live bindings for complete tree current/next kinetic state."""

import dataclasses
import math
import os

import numpy as np
import warp as wp

from ...sim import ModelFlags
from . import cached_state_binding, capture_joins, field_group
from . import tree_scan as ts
from .device_proof import DeviceProof, fingerprint
from .host_index import expand
from .publication_data import PLAN_FIELDS
from .tree_kinetic_plan import make_plan
from .tree_plan import TreePlan

_NUMERIC_FLAGS = int(
    ModelFlags.JOINT_DOF_PROPERTIES
    | ModelFlags.BODY_INERTIAL_PROPERTIES
    | ModelFlags.SHAPE_PROPERTIES
    | ModelFlags.MODEL_PROPERTIES
)

_PLAN_FIELDS = (
    *PLAN_FIELDS,
    "joint_ancestor",
    "joint_axis",
)
# Joint frames are watched, not frozen: the kernels read joint_X_p and joint_X_c
# live, so a JOINT_PROPERTIES notification that moves a frame (Isaac Lab re-poses
# fixed roots this way on resets) invalidates the caches instead of raising.
_FRAME_FIELDS = ("joint_X_p", "joint_X_c")
_MAPPING_FIELDS = ("group_to_art", "art_to_world", "art_dof_start", "free_root_joint_indices")

_fingerprint = fingerprint


def proof_runs(flags) -> bool:
    """Whether a held-tree group's ``validate_model`` runs its proof for a notification's flags.

    Numeric notifications alone (joint DOF, inertial, shape and model
    properties) cannot change what the proof covers and skip it; every other
    notification, and one without flags, runs it.
    """
    return flags is None or int(flags) == 0 or int(flags) & ~_NUMERIC_FLAGS != 0


def _bank_pointers(state):
    """The pointers a cache stamp records for a state bank: its coordinates and velocities."""
    return state.joint_q.ptr, state.joint_qd.ptr


def stage_clocks_on() -> bool:
    """Read the tree kernels' instrumentation switch, once per owner.

    ``FEATHER_PGS_STAGE_CLOCKS=1`` adds per-stage cycle counters. It changes the emitted code, so an owner reads it
    at construction and passes it to every kernel factory as an argument, which keeps it in the factories' cache keys.
    """
    return os.environ.get("FEATHER_PGS_STAGE_CLOCKS") == "1"


def launch_switches(pins=None):
    """Return the held-tree repair launch mapping, once per owner: ``(skip, classify)``.

    A captured step's repair is skipped where every block would take the early exit, and the repair lists the blocks
    to repair before repairing them. Test pins (``SolverFeatherPGS._route_pins``) ``repair_skip=False`` launch the
    repair of every captured step and ``repair_classify=False`` keep the one-pass repair launch over every block. Both
    choose launches, not emitted code.
    """
    pins = pins or {}
    return pins.get("repair_skip", True), pins.get("repair_classify", True)


# Blocks per SM of a listed repair's grid: the most a CUDA SM keeps resident, so a
# long list keeps the one-pass launch's concurrency and a short one exits in about a
# wave. A grid-stride loop covers every block of the list.
_LISTED_BLOCKS_PER_SM = 32


def listed_repair_blocks(device):
    """Grid of a listed repair: :data:`_LISTED_BLOCKS_PER_SM` per SM, or 0 (every block) without grid-stride kernels."""
    if not device.is_cuda or not getattr(wp.config, "default_grid_stride", True):
        return 0
    return _LISTED_BLOCKS_PER_SM * int(device.sm_count)


# Threads per block of a classification launch, one per repair block it lists.
# tools/fpgs_bench/kernel_sass.py compiles the classification at this block.
_CLASSIFY_BLOCK_DIM = 64

# Listed grids of blocks a one-pass repair must exceed to be classified. Up to two
# grids the one-pass launch's early exits cost about what the classified repair's
# three or four launches do (head: 8.4-9.6 us for a current 8192-block packed repair,
# 11-31 us at 16384 blocks), and a copied bank is compared inside the one-pass repair
# without the passes; a packed repair keeps its registers with the comparison inside
# (sm_120: 72 against 72 to 80), a chain-scan or unpacked one does not (78 to 96,
# 80 to 117).
_CLASSIFIED_GRIDS = 2


WALK_SOLVE_THREADS = 32
"""Threads a block of the walk's solve, a thread a tree root (WALK_MAPPING E5): one warp, so g1's 16,384 trees spread
over every SM (512 blocks)."""


class TreeKineticState:
    """Own one current cache with device state-bank tags and a separate held W.

    A repair proves each group's cache by the input bank's identity (the bank
    the last finish or repair stamped) or, outside motion-velocity layouts
    (:attr:`bank_compare`), by its content: when a group is stamped with the
    step's output bank instead, as after a caller copies the output bank back
    into the input bank, the two banks are compared and a group whose banks
    hold the same words is restamped with the input bank and keeps its cache.
    A classified repair compares in a pass of its own over the blocks its
    classification lists for comparison; a one-pass repair compares inside
    its blocks (a second repair kernel, launched only by repairs that compare).
    An owner that composes a group validated by identity alone passes
    ``compare_banks=False``: the solver then invalidates every cache of the
    owner on a changed input bank, so no comparison could keep one, and none
    is bound or launched.

    The host chooses the launch from the bank it expects the stamps to name.
    It knows that bank eagerly once a repair or finish ran, and inside a graph
    capture once a repair or finish of the same capture ran. The first repair
    of a capture runs on every replay after whatever the host does not see
    (eager steps, resets, copies, other graphs), so it lets the device decide:
    it compares whenever the output bank is another bank and is classified
    whenever the layout classifies.

    Within one graph capture, a repair that follows this owner's finish and
    reads the bank that finish published launches nothing outside
    motion-velocity layouts (:attr:`skip_captured_repairs`): every block
    would take the early exit. The solver calls :meth:`forget_continuity`
    from every host call that changes validity, requests or the step.

    On CUDA (:attr:`classify_first`), a repair of more blocks than two capped
    grids of :func:`listed_repair_blocks` lists the blocks that need it, one
    thread per block, and repairs over that list on the capped grid, instead
    of launching a block per group that mostly confirms its cache: inside a
    capture, and eagerly when the groups are expected to be stamped with the
    input bank or compared with the output bank.

    Motion-velocity layouts keep the one-pass repair, a measured trade-off.
    Their repair launches on every step, and a group exits early only when,
    besides the conditions above, each of its velocities compares equal to
    the step's prescaled velocity (computed after the last finish): as when
    the articulation's prescale ratio stayed 1 and no velocity is NaN (a NaN
    velocity never compares equal, so its group repairs on every step). A
    classification adds a list clear and a launch to every step, and it can
    save no more than the one-pass launch's early exits. On a 16384-group
    velocity-limited chain the classification and the listed repair took as
    long as the one-pass repair, and their two extra graph nodes per step
    cost 36 us per eight steps; where nearly every group exits, the
    classification measured no gain.
    """

    def __init__(
        self,
        sparse_owner,
        *,
        admission=None,
        chains=True,
        complete=True,
        response="sparse",
        clocked=None,
        skip=None,
        classify=None,
        compare_banks=True,
    ):
        from . import tree_kinetic_state as native  # noqa: PLC0415

        self.sparse = sparse_owner
        self.solver = solver = sparse_owner.solver
        self.motion_velocity = bool(solver.options.enable_joint_velocity_limits)
        self._motion_binding = None
        model, device = solver.model, solver.model.device
        # Direct construction is also used by CPU physical representation tests.
        # Production admission remains CUDA-only through supported().
        self.admission = admission
        self.complete = complete
        host = sparse_owner.host
        self.layout, parent = make_plan(model, sparse_owner, complete=complete)
        # Motion-velocity layouts keep identity validation: their repair reads the
        # prescaled velocity, and the solver keeps its host identity check for them.
        self.bank_compare = bool(compare_banks) and not self.motion_velocity
        switches = launch_switches(solver._route_pins)
        skip = switches[0] if skip is None else skip
        classify = switches[1] if classify is None else classify
        # A motion-velocity repair also compares the prescaled velocity, which the
        # step recomputes after the finish; its launches always run.
        self.skip_captured_repairs = bool(skip) and not self.motion_velocity
        # The CPU twin keeps the one-pass repair; _bind_kernels sets classify_first
        # from this, the motion contract and the launch size of the bound layout.
        self._classify_switch = bool(classify) and device.is_cuda
        # (capture id, whether the finish computed the next geometry) of the last
        # finish inside a capture; the next begin consumes it.
        self._continuity = None
        self.group_count = sparse_owner.plan.group_to_art.size
        # A composite owner passes what it read for all of its groups; a direct construction reads it here.
        self.clocked = stage_clocks_on() if clocked is None else bool(clocked)
        if self.clocked:
            solver._stage_clocks = wp.zeros((self.group_count, 28), dtype=wp.int64, device=solver.model.device)
        self._cache_to_art = sparse_owner.plan.group_to_art.reshape((self.group_count, 1))
        bodies, dofs = self.layout.bodies, self.layout.dofs
        # Each body's coordinate range: the plan's dof_joint is contiguous per body, in body order.
        dof_joint = np.asarray(host["dof_joint"], dtype=np.int64)
        body_dof_count = np.bincount(dof_joint, minlength=bodies).astype(np.int32)
        if not np.array_equal(dof_joint, expand(body_dof_count)[0]):
            raise ValueError("Tree kinetic state requires each body's coordinates to be contiguous")
        # The group's template tree: every traversal structure the scans read (TreePlan).
        self.tree = tree = TreePlan.build(parent, body_dof_count)
        if tree.children.size != bodies - len(self.layout.roots):
            raise ValueError("Tree kinetic state requires every true parent edge")

        group_to_art = sparse_owner.plan.group_to_art.numpy()
        starts = model.articulation_start.numpy()
        q_starts = model.joint_q_start.numpy()
        free_roots = solver.forest.free_root_joint_indices
        free_roots = free_roots.numpy() if free_roots is not None else []
        root_lookup = {int(joint): slot for slot, joint in enumerate(free_roots)}
        root_slot = np.empty(self.group_count, np.int32)
        q_index = np.full((self.group_count, dofs), -1, np.int32)
        # Coordinate of each DOF: the joint's first coordinate plus the DOF's
        # offset within its joint (one coordinate per DOF; a ball joint's
        # quaternion is never a spring coordinate, its stiffness is zero).
        dof_offset = np.arange(dofs) - tree.dof_start[dof_joint]
        for group, art in enumerate(group_to_art):
            start = int(starts[art])
            if self.layout.free_root and start not in root_lookup:
                raise ValueError("Tree kinetic state is missing a free-root transport slot")
            root_slot[group] = root_lookup.get(start, -1)
            for dof in range(self.layout.root_dofs if self.layout.free_root else 0, dofs):
                q_index[group, dof] = q_starts[start + dof_joint[dof]] + dof_offset[dof]
        # Each group's coordinates are one contiguous range, from its first joint's.
        first_joint = starts[group_to_art]
        coordinates = q_starts[first_joint + bodies] - q_starts[first_joint]
        if np.any(coordinates != coordinates[0]):
            raise ValueError("Tree kinetic state requires one coordinate count per group")
        self.bank_coordinates = int(coordinates[0])
        # The launch: lanes per tree and the scans' schedules, from the tree's data (forests keep their trees in
        # lane groups: no chain schedules).
        self.chains = bool(chains) and not self.layout.forest
        lanes = native.held_lanes(tree, self.group_count, cuda=device.is_cuda)
        self.layout = dataclasses.replace(self.layout, lanes=lanes)
        self.schedules = native.held_schedules(tree, lanes, chains=self.chains)
        if self.group_count % self.layout.pack:
            raise ValueError("Tree kinetic lane packing requires whole blocks of trees")
        # Forest metadata (a single tree is the one-root case of the same arrays).
        root_locals = np.asarray(self.layout.roots, np.int32)
        if not np.array_equal(root_locals, tree.roots):
            raise ValueError("Tree kinetic state requires its forest's roots in local order")
        root_of = np.searchsorted(tree.roots, tree.root_of).astype(np.int32)
        arts = getattr(sparse_owner, "forest_arts", None)
        arts = np.asarray(arts, np.int32) if arts is not None else group_to_art.reshape(-1, 1).astype(np.int32)
        self._root_arts = arts
        # Predicted body twists for the resolved-world classifier: the predictor holds their publication for
        # warp-sized trees whose scans keep to their lane groups (a CUDA register scan), and publishes them unless
        # the test pin predicted_twists=False leaves the classifier to its own kernel.
        self._predicts_twists = native.publishes_twists(self.layout, self.schedules, cuda=device.is_cuda)
        self.publishes_twists = self._predicts_twists and solver._route_pins.get("predicted_twists", True)
        self._no_twists = solver._arrays.placeholder(wp.spatial_vector)
        if self.publishes_twists:
            if getattr(solver, "_predicted_twist", None) is None:
                solver._predicted_twist = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=device)
                solver._predicted_twist_arts_host = np.zeros(model.articulation_count, np.int32)
            solver._predicted_twist_arts_host[np.asarray(arts, np.int64).reshape(-1)] = 1
        free = [r for r, is_free in enumerate(self.layout.free_roots) if is_free]
        free_root_local = root_locals[free] if free else np.zeros(1, np.int32)
        free_root_dof = (
            np.asarray([self.layout.root_offsets[r] for r in free], np.int32) if free else np.zeros(1, np.int32)
        )
        dof_passive = np.ones(dofs, np.int32)
        for r in free:
            dof_passive[self.layout.root_offsets[r] : self.layout.root_offsets[r] + 6] = 0
        if self.layout.forest:
            root_slot = np.full((self.group_count, max(1, len(free))), -1, np.int32)
            for group in range(self.group_count):
                for k, r in enumerate(free):
                    root_slot[group, k] = root_lookup.get(int(starts[arts[group, r]]), -1)
                    if root_slot[group, k] < 0:
                        raise ValueError("Tree kinetic state is missing a free-root transport slot")
            root_slot = root_slot.reshape(-1)
        self.plan = field_group.upload(
            native.HeldPlan,
            device,
            **ts.scan_arrays(tree),
            q_index=q_index.reshape(-1),
            root_slot=root_slot,
            body_dof_start=tree.dof_start[:-1],
            body_dof_count=body_dof_count,
            arts=arts,
            root_of=root_of,
            root_local=root_locals,
            free_root_local=free_root_local,
            free_root_dof=free_root_dof,
            dof_passive=dof_passive,
            scratch=np.zeros(self.group_count * self.layout.total if self.layout.global_block else 1, np.float32),
        )
        groups = self.group_count
        self.data = native.KineticData()
        field_group.SolverArrays(self.data, device).allocate(
            native.KINETIC_CACHE.owned("cache"), group=groups, dof=dofs, body=bodies, entry=self.layout.entries
        )
        self.data.current_valid = solver.factor.fk_id_cache_valid
        self.data.stage_clocks = cached_state_binding.stage_clocks(solver)
        # A classified repair's block lists (at most one entry per block each) and their
        # counts (repair, comparison), allocated before any capture.
        self._repair_blocks = wp.empty(groups, dtype=int, device=device)
        self._compare_blocks = wp.empty(groups, dtype=int, device=device)
        self._list_counts = wp.zeros(2, dtype=int, device=device)
        # The bank argument of the shared launch ABI where no kernel reads it: no
        # arrays, so a read would fault rather than compare stale words.
        self._no_bank = native.StateBank()
        self._repair_grid = listed_repair_blocks(device)
        # Blocks a repair launch must exceed to be classified.
        self._classify_above = _CLASSIFIED_GRIDS * self._repair_grid
        self.data.current_valid.zero_()
        self.bias = self.data.bias
        self.geometric = self.data.geometric
        self.geometry_valid = self.data.geometry_valid
        self.status = self.data.status
        self._body_velocity = getattr(sparse_owner, "requires_body_velocity", False)
        self._response = response
        # Pointers of the bank every current group is stamped with after the last
        # finish (its output bank) or repair (its input bank), as the host knows it,
        # and the capture that launch was recorded in (0 eagerly). It only chooses
        # launches; the kernels check each group's own stamp.
        self._published = None
        self._published_capture = 0
        # (capture id, launch form of each repair recorded in it) of the latest capture,
        # for the route record.
        self._captured_repairs = None
        self._bind_kernels()
        self._plan_dimensions = self._dimensions()
        # Everything make_plan and the kernels bake in, proven on the device, and the
        # joint frames, which the kernels read live, watched.
        self._proof = DeviceProof(
            [(name, getattr(model, name)) for name in (*_PLAN_FIELDS, *_FRAME_FIELDS)]
            + [(f"{name} mapping", self._mapping(name)) for name in _MAPPING_FIELDS],
            device,
            watched=_FRAME_FIELDS,
        )

    def _bind_kernels(self):
        from . import tree_kinetic_state as native  # noqa: PLC0415

        layout = self.layout
        self.block_dim = layout.block
        self.launch_count = self.group_count // layout.pack
        # Only a launch of more blocks than _CLASSIFIED_GRIDS listed grids is classified, and
        # never a motion-velocity layout's, which repairs on every step (see the class docstring).
        self.classify_first = (
            self._classify_switch
            and not self.motion_velocity
            and 0 < self._repair_grid
            and self._classify_above < self.launch_count
        )
        self.repair_kernel = native.get_state_kernel(
            False,
            self.schedules,
            layout=layout,
            body_velocity=self._body_velocity,
            motion_velocity=self.motion_velocity,
            clocked=self.clocked,
        )
        # A classified repair compares in a pass of its own over the blocks its classification
        # lists for comparison. Otherwise a second one-pass repair kernel compares inside its
        # blocks, and only repairs that compare launch it: no extra launch, and every other
        # repair keeps its kernel (comparing inside raises a chain-scan repair's registers, 80
        # to 151 on a 13-body chain on sm_120, but not a packed one's, 72 either way).
        self.bank_kernel = self.bank_repair_kernel = None
        if self.bank_compare and self.classify_first:
            self.bank_kernel = native.get_bank_kernel(layout=layout, coordinates=self.bank_coordinates)
        elif self.bank_compare:
            self.bank_repair_kernel = native.get_state_kernel(
                False,
                self.schedules,
                layout=layout,
                body_velocity=self._body_velocity,
                bank_coordinates=self.bank_coordinates,
                clocked=self.clocked,
            )
        self.classify_kernel = self.listed_repair_kernel = None
        if self.classify_first:
            # Ordinary motion only: classify_first excludes motion-velocity layouts.
            self.classify_kernel = native.get_classify_kernel(layout=layout)
            self.listed_repair_kernel = native.get_state_kernel(
                False,
                self.schedules,
                layout=layout,
                body_velocity=self._body_velocity,
                listed=True,
                clocked=self.clocked,
            )
        self.finish_kernel = native.get_state_kernel(
            True,
            self.schedules,
            layout=layout,
            body_velocity=self._body_velocity,
            clocked=self.clocked,
        )
        classes = self._walk_classes()
        self.force_kernel = native.get_force_kernel(
            self.schedules, layout=layout, response=self._response, classes=classes
        )
        self.walk_solve_kernel = native.get_walk_solve_kernel(layout=layout, classes=classes)
        self.predictor_kernel = native.get_predictor_kernel(
            self.schedules,
            layout=layout,
            response=self._response,
            clocked=self.clocked,
            twists=self._predicts_twists,
            classes=classes,
            slips=self.solver.factor.walk_slip_capacity,
        )
        # The trial's words between the predictor's halves and the walk's solve, at their global coordinates.
        if getattr(self, "trial", None) is None:
            dofs = max(int(self.solver.model.joint_dof_count), 1)
            self.trial = native.TrialWords()
            self.trial.force = wp.zeros(dofs, dtype=float, device=self.solver.model.device)
            self.trial.response = wp.zeros(dofs, dtype=float, device=self.solver.model.device)

    def _walk_classes(self):
        """Each root's walking class (:class:`.tree_factor.WalkClass`, its path table in the walk the predictor solves
        by), or ``None`` for a prescribed root (no coordinates to solve); every group's tree of a root has the same
        table."""
        from .tree_factor import WALK_BOTH, WalkClass  # noqa: PLC0415

        factor, layout = self.solver.factor, self.layout
        path = getattr(factor, "walk_path_host", None)
        starts, ends = self.solver.model.articulation_start.numpy(), self.solver.forest.articulation_joint_end_host
        classes = []
        for r, family in enumerate(layout.root_families if layout.forest else (layout.dofs,)):
            if family == 0:
                classes.append(None)
                continue
            if path is None:
                raise RuntimeError("The predictor solves by the walk: the solver holds it for every responding tree")
            arts = self._root_arts[:, r]
            first = arts[0]
            table = path[starts[first] : ends[first]]
            for art in arts[1:].tolist():
                if not np.array_equal(path[starts[art] : ends[art]], table):
                    raise ValueError("A predictor group's trees of one root must share their walk's path table")
            count = int(self.solver.forest.articulation_dof_count.numpy()[first])
            # Each coordinate's column half where every tree of the root agrees, else read from its record.
            d0 = self.solver.forest.articulation_dof_start.numpy()
            halves = np.stack([factor.walk_half_host[int(d0[a]) : int(d0[a]) + count] for a in arts.tolist()])
            half = tuple(int(h[0]) if np.all(h == h[0]) else WALK_BOTH for h in halves.T)
            classes.append(WalkClass(tuple(int(x) for x in table[:, 0]), (*(int(x) for x in table[:, 1]), count), half))
        return tuple(classes)

    def _walk(self):
        """The walk the predictor's border reads (:func:`.tree_kinetic_state.border_solve`): the factor's, or, where no
        tree can slip and the walk is not held, a view with no tree whose ``slips`` the predictor clears."""
        factor = self.solver.factor
        if factor.walk_arts is not None:
            return factor.walk_view()
        if getattr(self, "_no_walk", None) is None:
            from .tree_factor import empty_walk_view  # noqa: PLC0415

            self._no_walk = empty_walk_view(self.solver.model, self.solver.forest)
        return self._no_walk

    def _force_input(self, state_in, state_aug, control, predictor_qd, dt):
        """Bind the predictor's inputs (tree_kinetic_state.FORCE_INPUT) for one step."""
        from .tree_kinetic_state import ForceInput  # noqa: PLC0415

        solver = self.solver
        twists = getattr(solver, "_predicted_twist", None)
        # Publish only when the classifier, built after this owner, consumes them.
        consumer = solver.sweep.resolved_worlds_owner
        return field_group.bind(
            ForceInput,
            solver=solver,
            factor=solver.factor,
            model=solver.model,
            state_in=state_in,
            state_aug=state_aug,
            control=control,
            predictor_qd=predictor_qd,
            dt=dt,
            stage_clocks=cached_state_binding.stage_clocks(solver),
            predicted_twist=self._no_twists if twists is None else twists,
            publish_twists=int(self.publishes_twists and consumer is not None and consumer.published),
        )

    def _dimensions(self):
        model = self.solver.model
        return (
            self.solver.forest.world_count,
            model.world_count,
            model.articulation_count,
            model.body_count,
            model.joint_count,
            model.joint_coord_count,
            model.joint_dof_count,
        )

    def _mapping(self, name):
        return (
            self.solver.forest.free_root_joint_indices
            if name == "free_root_joint_indices"
            else getattr(self.sparse.plan, name)
        )

    def validate_model(self, flags=None):
        """Reject changed baked ownership before notification invalidates any state.

        Numeric notifications (joint DOF, inertial, shape and model properties)
        change values the caches refresh from, never parents, axes or body
        membership, so they are accepted outright. Every other notification
        compares the baked plan arrays on the device (one small readback)
        instead of fingerprinting them on the host; Isaac Lab issues a
        notification on every reset. Joint frames are not baked but watched:
        the kernels read them live, so a ``JOINT_PROPERTIES`` notification may
        move them, as Isaac Lab does to re-pose a fixed root; the proof then
        reports them moved and stores their new checksums, and the solver
        invalidates every cache. Body flags are not baked either: the kernels
        read the kinematic masks and armature the solver re-derives in place,
        after rejecting any flag change it cannot absorb.

        Returns:
            Whether the proof ran and found every input unchanged: the device
            checksums of every baked plan array, the joint axes included, and
            of the joint frames ``joint_X_p`` and ``joint_X_c`` match the
            values stored at construction or at the last notification that
            moved a frame. ``False`` when a frame moved. A device checksum is
            two position-sensitive 32-bit sums, not a word-by-word compare: one
            changed word always shows, and a random multi-word change collides
            with a probability of about 2**-64. Arrays the proof cannot view as
            device words (every array on CPU) compare by an exact host
            fingerprint. Numeric notifications skip the proof and return
            ``False``.
        """
        if not proof_runs(flags):
            return False
        if self._dimensions() != self._plan_dimensions:
            raise RuntimeError("Tree kinetic dimensions changed; reconstruct and recapture")

        def current(label):
            if label.endswith(" mapping"):
                return self._mapping(label[: -len(" mapping")])
            return getattr(self.solver.model, label)

        # A moved frame invalidates instead of raising; any other change raises.
        return not self._proof.validate(current, "Tree kinetic static")

    def forget_continuity(self):
        """Launch the next repair: a host call changed what the last finish left on the device."""
        self._continuity = None

    def invalidate(self, world_mask=None):
        """Invalidate current state for every owned articulation in selected worlds."""
        self.forget_continuity()
        cached_state_binding.invalidate(
            self.solver.forest.world_count,
            world_mask,
            self._cache_to_art,
            self.sparse.plan.art_to_world,
            self.data.current_valid,
            self.data.geometry_valid,
        )

    def _validate_call(self, state, state_aug, dt):
        if self.admission is not None and not self.admission(self.solver):
            raise RuntimeError("Tree kinetic configuration changed; reconstruct and recapture")
        if bool(self.solver.options.enable_joint_velocity_limits) != self.motion_velocity:
            raise RuntimeError("Tree kinetic motion-velocity contract changed; reconstruct and recapture")
        if dt is None or not math.isfinite(dt) or dt <= 0:
            raise ValueError("Tree kinetic state requires a positive finite step duration")
        for name in ("joint_q", "joint_qd", "body_q", "body_qd", "body_f"):
            value = getattr(state, name)
            if value.device != self.solver.model.device or not value.is_contiguous:
                raise RuntimeError("Tree kinetic state requires contiguous same-device state arrays")

    def _publication(self, state_in, state_aug, state_out, dt):
        """Bind live current and next-state publication descriptors."""
        return cached_state_binding.publication(self.solver, state_in, state_aug, state_out, dt)

    def _bank(self, state, state_in):
        """Bind the bank a repair compares ``state_in`` with: same device, contiguous, same shapes."""
        from .tree_kinetic_state import StateBank  # noqa: PLC0415

        bank = StateBank()
        for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
            value, reference = getattr(state, name), getattr(state_in, name)
            if value.device != reference.device or not value.is_contiguous or value.shape != reference.shape:
                raise RuntimeError("Tree kinetic bank comparison requires a contiguous state of the same model")
            setattr(bank, name, value)
        return bank

    def begin(self, state_in, state_aug, stage3_qd, dt, global_refresh, *, state_out=None):
        """Repair current caches or late-request geometry before any factor read.

        Args:
            state_in: The step's input bank, which the repair publishes into.
            state_aug: The solver's augmented state.
            stage3_qd: The predictor velocity (the prescaled velocity of a motion-velocity layout).
            dt: The step duration [s].
            global_refresh: Whether this step refreshes every group's geometry.
            state_out: The step's output bank. A group whose cache the last finish
                published into it keeps that cache when ``state_in`` holds the same
                words (see :func:`~.tree_kinetic_state.get_bank_kernel`). ``None``
                validates by identity alone.
        """
        self._validate_call(state_in, state_aug, dt)
        binding = self._validate_motion(state_in, state_aug, stage3_qd)
        device = self.solver.model.device
        source = _bank_pointers(state_in)
        capture = capture_joins.capture_id(device)
        continuity, self._continuity = self._continuity, None
        if (
            continuity is not None
            and source == self._published
            and continuity[0] == capture
            and (not global_refresh or continuity[1])
        ):
            # This capture's last finish published into this bank and no host call
            # has touched validity or requests since. Every group with a finite cache
            # is valid, stamped with this bank, unrequested and, unless this step
            # needs geometry the finish did not compute, geometrically current: its
            # block would take the early exit. (A non-finite cache stays invalid, and
            # the predictor then makes its articulations' next state NaN with or
            # without a repair.)
            self._record_capture(capture, "skipped")
            self._motion_binding = binding
            return
        data = self._publication(state_in, state_aug, state_in, dt)
        compare, classify = self._repair_form(source, state_out, capture, device)
        # Every repair takes the launch ABI of the finish (get_state_kernel): only an
        # alternate-motion repair reads stage3_qd, otherwise the state's own velocity
        # (_validate_motion); only a listed repair reads the block list, and only a
        # comparing repair a bank.
        arguments = [
            self.sparse.plan,
            self.plan,
            data,
            self.data,
            self.solver.factor.mass_update_requested,
            int(global_refresh),
            stage3_qd,
        ]
        lists = [self._repair_blocks, self._list_counts]
        if classify:
            # List the blocks that repair (and, when comparing, those that compare
            # first) and run the repair over the list on the capped grid. Passing the
            # input bank as the comparison bank lists nothing for comparison.
            bank = self._bank(state_out if compare else state_in, state_in)
            self._list_counts.zero_()
            # The classification is a contract of ordinary motion: it takes the six common
            # arguments, not the motion input of the repair ABI.
            wp.launch(
                self.classify_kernel,
                dim=self.launch_count,
                inputs=[*arguments[:6], bank, self._repair_blocks, self._compare_blocks, self._list_counts],
                block_dim=_CLASSIFY_BLOCK_DIM,
                device=device,
            )
            if compare:
                wp.launch_tiled(
                    self.bank_kernel,
                    dim=[self.launch_count],
                    inputs=[*arguments[:6], bank, self._compare_blocks, self._repair_blocks, self._list_counts],
                    block_dim=self.block_dim,
                    max_blocks=self._repair_grid,
                    device=device,
                )
            wp.launch_tiled(
                self.listed_repair_kernel,
                dim=[self.launch_count],
                inputs=[*arguments, *lists, self._no_bank],
                block_dim=self.block_dim,
                max_blocks=self._repair_grid,
                device=device,
            )
        elif compare:
            wp.launch_tiled(
                self.bank_repair_kernel,
                dim=[self.launch_count],
                inputs=[*arguments, *lists, self._bank(state_out, state_in)],
                block_dim=self.block_dim,
                device=device,
            )
        else:
            wp.launch_tiled(
                self.repair_kernel,
                dim=[self.launch_count],
                inputs=[*arguments, *lists, self._no_bank],
                block_dim=self.block_dim,
                device=device,
            )
        self._published, self._published_capture = source, capture
        form = ("classified" if classify else "one_pass") + ("_compared" if compare else "")
        self._record_capture(capture, form)
        self._motion_binding = binding

    def _repair_form(self, source, state_out, capture, device):
        """Choose whether a repair compares input banks and whether it is classified.

        Returns:
            ``(compare, classify)``.
        """
        output = None if state_out is None else _bank_pointers(state_out)
        distinct = output is not None and output != source
        if (capture != 0 or device.is_capturing) and (capture == 0 or self._published_capture != capture):
            # The first repair of a capture (or of one whose id is unknown) runs on every
            # replay after whatever the host does not see, such as a copy of the output
            # bank back into the input bank or an eager reset: the device decides.
            return self.bank_compare and distinct, self.classify_first
        if self._published is None:
            # The first eager step: every cache is invalid, and every group repairs in
            # the one-pass launch.
            return False, False
        # The last repair or finish ran right before this one: eagerly, or earlier in this
        # capture. The stamps name its bank. The output bank, as after an odd substep
        # count copied it back into the input bank, is compared; any other bank but the
        # input bank repairs every group, which the one-pass launch does more cheaply.
        compare = self.bank_compare and distinct and self._published == output
        return compare, self.classify_first and (compare or source == self._published)

    def _record_capture(self, capture, form):
        """Remember how each repair of the latest capture was recorded, for the route record."""
        if capture == 0:
            return
        if self._captured_repairs is None or self._captured_repairs[0] != capture:
            self._captured_repairs = (capture, [])
        self._captured_repairs[1].append(form)

    def _validate_motion(self, state, state_aug, velocity):
        """Reject invalid working descriptors before changing current-cache validity."""
        raw = state.joint_qd
        if (
            not isinstance(velocity, wp.array)
            or velocity.device != raw.device
            or velocity.dtype != raw.dtype
            or velocity.shape != raw.shape
            or not velocity.is_contiguous
        ):
            raise RuntimeError("Tree kinetic motion velocity requires a matching contiguous state-vector descriptor")
        if not self.motion_velocity and velocity.ptr != raw.ptr:
            raise RuntimeError("Tree kinetic state excludes alternate prescaled predictor velocity")

        if (velocity.ptr != raw.ptr and cached_state_binding.overlap(velocity, raw)) or any(
            cached_state_binding.overlap(velocity, array)
            for array in (state.joint_q, state_aug.joint_qdd, self.solver.v_hat, self.solver.v_out)
        ):
            raise RuntimeError("Tree kinetic motion velocity overlaps state coordinates or response outputs")
        return state.joint_q.ptr, raw.ptr, velocity.ptr

    def predict(self, state_in, state_aug, control, stage3_qd, dt):
        """Apply live force buckets through held W with original root transport."""

        self._validate_call(state_in, state_aug, dt)
        binding = self._validate_motion(state_in, state_aug, stage3_qd)
        if self.motion_velocity and binding != self._motion_binding:
            raise RuntimeError("Tree kinetic predictor motion velocity differs from the current-cache binding")
        solver = self.solver
        force = self._force_input(state_in, state_aug, control, stage3_qd, dt)
        walk = self._walk()
        inputs = [self.sparse.plan, self.sparse.data, self.plan, self.data, force, walk, self.trial]
        device = solver.model.device
        # The force half, the walk's solve (a thread a tree root), the response half (Z4.2).
        wp.launch_tiled(
            self.force_kernel, dim=[self.launch_count], inputs=inputs, block_dim=self.block_dim, device=device
        )
        roots = len(self.layout.roots)
        wp.launch(
            self.walk_solve_kernel,
            dim=self.group_count * roots,
            inputs=[self.group_count, self.sparse.plan, self.plan, walk, self.trial],
            block_dim=WALK_SOLVE_THREADS,
            device=device,
        )
        wp.launch_tiled(
            self.predictor_kernel, dim=[self.launch_count], inputs=inputs, block_dim=self.block_dim, device=device
        )

    def finish(self, state_in, state_aug, state_out, dt, next_refresh):
        """Integrate once, publish complete public state and construct next bias."""
        self._validate_call(state_in, state_aug, dt)
        self._validate_call(state_out, state_aug, dt)
        if not cached_state_binding.disjoint_states(state_in, state_out):
            raise RuntimeError("Tree kinetic integration requires non-overlapping generalized state banks")
        data = self._publication(state_in, state_aug, state_out, dt)
        wp.launch_tiled(
            self.finish_kernel,
            dim=[self.launch_count],
            inputs=[
                self.sparse.plan,
                self.plan,
                data,
                self.data,
                self.solver.factor.mass_update_requested,
                int(next_refresh),
                # The repair inputs of the shared launch ABI, unread by a finish.
                state_in.joint_qd,
                self._repair_blocks,
                self._list_counts,
                self._no_bank,
            ],
            block_dim=self.block_dim,
            device=self.solver.model.device,
        )
        capture = capture_joins.capture_id(self.solver.model.device)
        self._published, self._published_capture = _bank_pointers(state_out), capture
        self._continuity = (capture, bool(next_refresh)) if capture and self.skip_captured_repairs else None
