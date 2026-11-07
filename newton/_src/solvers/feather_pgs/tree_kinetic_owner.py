# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Admission and live bindings for complete tree current/next kinetic state."""

import dataclasses

import numpy as np
import warp as wp

from ...core import execution, memory
from ...core.execution import Work
from ...sim import ModelFlags
from . import cached_state_binding, field_group
from . import tree_scan as ts
from .device_proof import DeviceProof
from .host_index import expand
from .path_selection import json_value, kernel_key
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


def proof_runs(flags) -> bool:
    """Whether a held-tree group's ``validate_model`` runs its proof for a notification's flags.

    Numeric notifications alone (joint DOF, inertial, shape and model
    properties) cannot change what the proof covers and skip it; every other
    notification, and one without flags, runs it.
    """
    return flags is None or int(flags) == 0 or int(flags) & ~_NUMERIC_FLAGS != 0


def walk_solve_work(roots: int) -> Work:
    """The walk's solve (:func:`.tree_kinetic_state.get_walk_solve_kernel`): a thread a tree root in blocks of one warp
    (WALK_MAPPING E5), so g1's 16,384 trees spread over every SM (512 blocks)."""
    return Work(roots, lanes=1, block=32)


class TreeKineticState:
    """Own one current cache with device state-bank tags and a separate held W.

    Every step launches one repair, a block per group, and the device decides it group by group: a group whose cache
    stands for the step's input exits at once, every other group recomputes its state. The cache stands when it is
    valid, current for the step's geometry, and stamped with the input bank (the bank the last finish or repair
    published from), or, outside motion-velocity layouts (:attr:`bank_compare`), stamped with the step's output bank
    while the input bank holds the same words, as after a caller copies the output bank back into the input bank: the
    group is then restamped with the input bank and keeps its cache. The host keeps no record of which bank the stamps
    name.

    A motion-velocity layout's group also exits only where each of its velocities compares equal to the step's
    prescaled velocity (computed after the last finish): as when the articulation's prescale ratio stayed 1 and no
    velocity is NaN (a NaN velocity never compares equal, so its group repairs on every step).

    The cache is worth its stamps (UNB's PRO hold, 2026-11-05, 16,384 worlds): invalidated every step, g1 ran 11-13%
    slower, franka-reach 4-5%, allegro 2-3%. The classification pass and its listed repair, which this one launch
    replaced, saved g1 0.4-0.8% (a 16,384-block early-exit launch of the comparing repair is about 20 us).
    """

    def __init__(
        self,
        sparse_owner,
        *,
        complete=True,
        response="sparse",
        clocked=None,
    ):
        from . import tree_kinetic_state as native  # noqa: PLC0415

        self.sparse = sparse_owner
        self.solver = solver = sparse_owner.solver
        self.motion_velocity = bool(solver.options.enable_joint_velocity_limits)
        model, device = solver.model, solver.model.device
        self.complete = complete
        host = sparse_owner.host
        self.layout, parent = make_plan(model, sparse_owner, complete=complete)
        # Motion-velocity layouts validate by the stamps alone: their repair reads the prescaled velocity.
        self.bank_compare = not self.motion_velocity
        self.group_count = sparse_owner.plan.group_to_art.size
        # A composite owner passes what it read for all of its groups; a direct construction reads it here.
        self.clocked = solver.options.diagnostics.stage_clocks if clocked is None else bool(clocked)
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
        # Lanes and schedules from the tree's data: chain scans for a tree past a warp (its own block), not in forests;
        # within a warp a tree group keeps the warp scans and the predictor's twist publication.
        lanes = native.held_lanes(tree, self.group_count, memory.device(device))
        self.chains = not self.layout.forest and (complete or lanes > 32)
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
        self.publishes_twists = self._predicts_twists and solver.spec.pins.predicted_twists
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
        # Whether the group's refresh reads H's diagonal (a compact diagonal's inverse): its collection keeps H_ii.
        self.diagonal_mass = bool(getattr(sparse_owner, "diagonal_mass", False))
        # Whether a coordinate of the group's trees has a step column of two halves at its frame (the walk's one rule,
        # tree_factor.TreeFactor.two_halves; walk_side is it over every walked tree): only then does its finish form
        # the walk's columns a second time.
        owned = self._root_arts.reshape(-1)
        owner, local = expand(solver.forest.articulation_dof_count_host[owned])
        self.two_half = solver.factor.two_halves(solver.forest.articulation_dof_start_host[owned][owner] + local)
        self.data = native.KineticData()
        field_group.SolverArrays(self.data, device).allocate(
            native.KINETIC_CACHE.owned("cache"),
            group=groups,
            dof=dofs,
            body=bodies,
            diagonal=dofs if self.diagonal_mass else 0,
        )
        self.data.current_valid = solver.factor.fk_id_cache_valid
        self.data.com_offset = solver.factor.joint_com_offset
        self.data.stage_clocks = cached_state_binding.stage_clocks(solver)
        # The bank argument of the shared launch ABI where no kernel reads it: no
        # arrays, so a read would fault rather than compare stale words.
        self._no_bank = native.StateBank()
        self._walk_solve = execution.shape(
            walk_solve_work(self.group_count * len(self.layout.roots)), memory.device(device)
        )
        self.data.current_valid.zero_()
        self.bias = self.data.bias
        self.mass_diagonal = self.data.diagonal
        self.geometry_valid = self.data.geometry_valid
        self.status = self.data.status
        self._body_velocity = getattr(sparse_owner, "requires_body_velocity", False)
        self._response = response
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

    def record(self) -> dict:
        """The group's record in the held tree's (``routes.held_tree.groups``)."""
        layout, operator = self.layout, self.sparse
        return {
            "operator": operator.kind,
            "size": json_value(getattr(operator, "size", None)),
            "group_count": int(self.group_count),
            "bodies": int(layout.bodies),
            "dofs": int(layout.dofs),
            "lanes": int(layout.lanes),
            "trees_per_block": int(layout.pack),
            "block_dim": int(self.block_dim),
            "roots": json_value(layout.roots),
            "root_coordinates": json_value(layout.root_coordinates),
            # Per root, whether it stands on a free joint (a joint root's coordinates are a descendant's).
            "free_roots": json_value(layout.free_roots),
            # Factor family per root: its coordinate count, negated for a diagonal-mass family, 0 when prescribed.
            "root_families": json_value(layout.root_families),
            # The scans' schedules, chosen from the tree's data with its lanes (tree_kinetic_state.held_schedules).
            "schedules": {name: getattr(self.schedules, name).value for name in ("pose", "motion", "reduce")},
            "response": self._response,
            # "content": a repair whose input bank is not the bank the cache was published into compares the two banks
            # word for word; "identity": the bank's pointers alone.
            "input_validation": "content" if self.bank_compare else "identity",
            "motion_velocity": bool(self.motion_velocity),
            "publishes_twists": bool(self.publishes_twists),
            "kernels": {
                "repair": kernel_key(self.repair_kernel),
                "finish": kernel_key(self.finish_kernel),
                "predict": kernel_key(self.predictor_kernel),
            },
        }

    def _bind_kernels(self):
        from . import tree_kinetic_state as native  # noqa: PLC0415

        layout = self.layout
        self.block_dim = layout.block
        self.launch_count = self.group_count // layout.pack
        # The one repair a step launches; a comparing owner's compares the banks inside its blocks.
        self.repair_kernel = native.get_state_kernel(
            False,
            self.schedules,
            layout=layout,
            body_velocity=self._body_velocity,
            motion_velocity=self.motion_velocity,
            bank_coordinates=self.bank_coordinates if self.bank_compare else 0,
            clocked=self.clocked,
            diagonal=self.diagonal_mass,
            two_half=self.two_half,
        )
        self.finish_kernel = native.get_state_kernel(
            True,
            self.schedules,
            layout=layout,
            body_velocity=self._body_velocity,
            clocked=self.clocked,
            diagonal=self.diagonal_mass,
            two_half=self.two_half,
        )
        # A rest tree (no coordinates) predicts nothing: no force, solve or response launch.
        self.force_kernel = self.walk_solve_kernel = self.predictor_kernel = None
        if not layout.dofs:
            return
        classes = self._walk_classes()
        self.force_kernel = native.get_force_kernel(
            self.schedules, layout=layout, response=self._response, classes=classes
        )
        self.walk_solve_kernel = (
            native.get_walk_solve_kernel(layout=layout, classes=classes) if native.walk_solves(layout) else None
        )
        self.predictor_kernel = native.get_predictor_kernel(
            self.schedules,
            layout=layout,
            response=self._response,
            clocked=self.clocked,
            twists=self._predicts_twists,
            classes=classes,
            slips=self.solver.factor.walk_slip_capacity,
            side=self.solver.factor.walk_side,
        )
        # The trial's words between the predictor's halves and the walk's solve, at their global coordinates.
        if getattr(self, "trial", None) is None:
            dofs = max(int(self.solver.model.joint_dof_count), 1)
            self.trial = native.TrialWords()
            self.trial.force = wp.zeros(dofs, dtype=float, device=self.solver.model.device)
            self.trial.response = wp.zeros(dofs, dtype=float, device=self.solver.model.device)

    def _walk_classes(self):
        """Each root's walking class (:class:`.tree_scan.WalkClass`, its path table in the walk the predictor solves
        by), or ``None`` for a prescribed root (no coordinates to solve) and a diagonal root (a negative family: its walk
        is the compact inverse, no table); every group's tree of a root has the same table."""
        from .tree_scan import WALK_BOTH, WalkClass  # noqa: PLC0415

        factor, layout = self.solver.factor, self.layout
        walking = getattr(factor, "walk_class_host", None)
        classes = []
        for r, family in enumerate(layout.root_families):
            if family <= 0:
                classes.append(None)
                continue
            arts = self._root_arts[:, r]
            if walking is None or np.any(walking[arts] < 0):
                raise RuntimeError("The predictor solves by the walk: the solver holds it for every responding tree")
            table = factor.walk_classes[walking[arts[0]]]
            for art in arts[1:].tolist():
                other = factor.walk_classes[walking[art]]
                if (other.parent, other.first) != (table.parent, table.first):
                    raise ValueError("A predictor group's trees of one root must share their walk's path table")
            count = table.coordinates
            # Each coordinate's column half where every tree of the root agrees, else read from its record.
            d0 = self.solver.forest.articulation_dof_start.numpy()
            halves = np.stack([factor.walk_half_host[int(d0[a]) : int(d0[a]) + count] for a in arts.tolist()])
            half = tuple(int(h[0]) if np.all(h == h[0]) else WALK_BOTH for h in halves.T)
            classes.append(WalkClass(table.parent, table.first, half))
        return tuple(classes)

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

    def invalidate(self, world_mask=None):
        """Invalidate current state for every owned articulation in selected worlds."""
        cached_state_binding.invalidate(
            self.solver.forest.world_count,
            world_mask,
            self._cache_to_art,
            self.sparse.plan.art_to_world,
            self.data.current_valid,
            self.data.geometry_valid,
        )

    def _publication(self, state_in, state_aug, state_out, dt):
        """Bind live current and next-state publication descriptors."""
        return cached_state_binding.publication(self.solver, state_in, state_aug, state_out, dt)

    @staticmethod
    def _bank(state):
        """Bind the bank a repair compares the input bank with."""
        from .tree_kinetic_state import StateBank  # noqa: PLC0415

        bank = StateBank()
        bank.joint_q, bank.joint_qd, bank.body_q, bank.body_qd = (
            state.joint_q,
            state.joint_qd,
            state.body_q,
            state.body_qd,
        )
        return bank

    def begin(self, state_in, state_aug, stage3_qd, dt, *, state_out=None):
        """Repair current caches or late-request geometry before any factor read.

        Args:
            state_in: The step's input bank, which the repair publishes into.
            state_aug: The solver's augmented state.
            stage3_qd: The predictor velocity (the prescaled velocity of a motion-velocity layout).
            dt: The step duration [s].
            state_out: The step's output bank. A group whose cache the last finish
                published into it keeps that cache when ``state_in`` holds the same
                words (see :func:`~.tree_kinetic_state._bank_binding`). ``None``
                validates by the input bank's stamps alone.
        """
        data = self._publication(state_in, state_aug, state_in, dt)
        # The repair takes the launch ABI of the finish (get_state_kernel): only an alternate-motion repair reads
        # stage3_qd, otherwise the state's own velocity; only a comparing repair reads a bank: the output bank, or the
        # input bank itself where there is none to compare (every group stamped with it exits before comparing).
        bank = self._bank(state_out if self.bank_compare and state_out is not None else state_in)
        wp.launch_tiled(
            self.repair_kernel,
            dim=[self.launch_count],
            inputs=[
                self.sparse.plan,
                self.plan,
                data,
                self.data,
                self.solver.factor.mass_update_requested,
                self.solver.factor.clock,
                stage3_qd,
                bank,
            ],
            block_dim=self.block_dim,
            device=self.solver.model.device,
        )

    def predict(self, state_in, state_aug, control, stage3_qd, dt):
        """Apply live force buckets through held W with original root transport (a rest tree has none)."""
        if self.predictor_kernel is None:
            return
        solver = self.solver
        force = self._force_input(state_in, state_aug, control, stage3_qd, dt)
        walk = solver.factor.walk_view()
        inputs = [self.sparse.plan, self.sparse.data, self.plan, self.data, force, walk, self.trial]
        device = solver.model.device
        # The force half, the walk's solve (a thread a tree root; none where every root is diagonal), the response
        # half (Z4.2).
        wp.launch_tiled(
            self.force_kernel, dim=[self.launch_count], inputs=inputs, block_dim=self.block_dim, device=device
        )
        if self.walk_solve_kernel is not None:
            wp.launch(
                self.walk_solve_kernel,
                dim=self.group_count * len(self.layout.roots),
                inputs=[self.group_count, self.sparse.plan, self.plan, walk, self.trial],
                block_dim=self._walk_solve.block,
                device=device,
            )
        wp.launch_tiled(
            self.predictor_kernel, dim=[self.launch_count], inputs=inputs, block_dim=self.block_dim, device=device
        )

    def finish(self, state_in, state_aug, state_out, dt):
        """Integrate once, publish complete public state and construct next bias."""
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
                self.solver.factor.clock,
                # The repair inputs of the shared launch ABI, unread by a finish.
                state_in.joint_qd,
                self._no_bank,
            ],
            block_dim=self.block_dim,
            device=self.solver.model.device,
        )
