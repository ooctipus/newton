# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Admission and live bindings for complete tree current/next kinetic state."""

import dataclasses
import math
import os

import numpy as np
import warp as wp

from ...sim import ModelFlags
from . import cached_state_binding, capture_joins
from .device_proof import DeviceProof, fingerprint
from .publication_data import PLAN_FIELDS
from .tree_kinetic_plan import cooperative_block_dim, lane_candidates, make_plan

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
    "joint_X_p",
    "joint_X_c",
)
_MAPPING_FIELDS = ("group_to_art", "art_to_world", "art_dof_start", "free_root_joint_indices")

_fingerprint = fingerprint


def dfs_intervals(parent):
    """Depth-first order of a parent-first forest and, per position, the end of its subtree.

    Children are visited in increasing local index; roots (parent -1) in
    increasing local index. Returns ``order`` (position to body) and ``end``
    (position to one past the subtree's last position), so the subtree of the
    body at position ``p`` is exactly the interval ``[p, end[p])``.
    """
    parent = np.asarray(parent, dtype=np.int64)
    count = int(parent.size)
    children = [[] for _ in range(count)]
    for body in range(count):
        if parent[body] >= 0:
            children[int(parent[body])].append(body)
    order, end = [], np.zeros(count, np.int32)
    stack = [(root, False) for root in range(count - 1, -1, -1) if parent[root] < 0]
    starts = {}
    while stack:
        body, done = stack.pop()
        if done:
            end[starts[body]] = len(order)
            continue
        starts[body] = len(order)
        order.append(body)
        stack.append((body, True))
        stack.extend((child, False) for child in reversed(children[body]))
    if len(order) != count:
        raise ValueError("Tree kinetic state requires a parent-first forest")
    return np.asarray(order, np.int32), end


def _bank_pointers(state):
    """The pointers a cache stamp records for a state bank: its coordinates and velocities."""
    return state.joint_q.ptr, state.joint_qd.ptr


def kernel_switches():
    """Read the tree kernels' comparison and instrumentation switches, once per owner.

    ``FEATHER_PGS_TREE_REFERENCE=1`` selects the reference forms the A/B tests
    compare against (the serial one-lane held solve, the depth-level subtree
    reduction and the shared-memory scans); ``FEATHER_PGS_STAGE_CLOCKS=1`` adds
    per-stage cycle counters. Both change the emitted code, so an owner reads
    them at construction and passes them to every kernel factory as arguments,
    which keeps them in the factories' cache keys.

    Returns:
        ``(reference, clocked)``.
    """
    return (
        os.environ.get("FEATHER_PGS_TREE_REFERENCE") == "1",
        os.environ.get("FEATHER_PGS_STAGE_CLOCKS") == "1",
    )


def launch_switches():
    """Read the held-tree repair launch switches, once per owner.

    ``FEATHER_PGS_TREE_REPAIR_SKIP=0`` launches the repair of every captured
    step, also where every block would take the early exit.
    ``FEATHER_PGS_TREE_REPAIR_CLASSIFY=0`` keeps the one-pass repair launch
    over every block. The switches choose launches, not emitted code.

    Returns:
        ``(skip, classify)``.
    """
    return (
        os.environ.get("FEATHER_PGS_TREE_REPAIR_SKIP", "1") != "0",
        os.environ.get("FEATHER_PGS_TREE_REPAIR_CLASSIFY", "1") != "0",
    )


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

    Motion-velocity layouts keep the one-pass repair. Their repair launches on
    every step, and a group exits early only when, besides the conditions
    above, the step's velocity prescale (computed after the last finish) left
    each of its velocities unchanged, as it does for an articulation within
    every joint velocity limit. A classification would add a list clear and a
    launch to every step to skip at most the one-pass launch's early exits:
    on a 16384-group velocity-limited chain, the classification and the listed
    repair took as long as the one-pass repair, and the two extra graph nodes
    per step cost 36 us per eight steps.
    """

    def __init__(
        self,
        sparse_owner,
        *,
        admission=None,
        chain_scan_default=None,
        complete=True,
        response="sparse",
        reference=None,
        clocked=None,
        skip=None,
        classify=None,
    ):
        from . import tree_kinetic_state as native  # noqa: PLC0415

        self.sparse = sparse_owner
        self.solver = solver = sparse_owner.solver
        self.motion_velocity = bool(solver.enable_joint_velocity_limits)
        self._motion_binding = None
        model, device = solver.model, solver.model.device
        # Direct construction is also used by CPU physical representation tests.
        # Production admission remains CUDA-only through supported().
        self.admission = admission
        self.complete = complete
        host = sparse_owner.host
        self.layout, parent, depth = make_plan(model, sparse_owner, complete=complete)
        # Motion-velocity layouts keep identity validation: their repair reads the
        # prescaled velocity, and the solver keeps its host identity check for them.
        self.bank_compare = not self.motion_velocity
        switches = launch_switches()
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
        # Kernel-selecting switches: a composite owner passes the values it read
        # for all of its groups; a direct construction reads them here.
        if reference is None or clocked is None:
            read_reference, read_clocked = kernel_switches()
            reference = read_reference if reference is None else reference
            clocked = read_clocked if clocked is None else clocked
        self.reference, self.clocked = bool(reference), bool(clocked)
        if self.clocked:
            solver._stage_clocks = wp.zeros((self.group_count, 28), dtype=wp.int64, device=solver.model.device)
        self._cache_to_art = sparse_owner.plan.group_to_art.reshape((self.group_count, 1))
        bodies, dofs = self.layout.bodies, self.layout.dofs
        jump = np.empty((self.layout.rounds, bodies), np.int32)
        jump[0] = parent
        for level in range(1, self.layout.rounds):
            previous = jump[level - 1]
            jump[level] = np.where(previous >= 0, previous[np.maximum(previous, 0)], -1)
        children, offsets = [], [0]
        for body in range(bodies):
            children.extend(np.flatnonzero(parent == body)[::-1].tolist())
            offsets.append(len(children))
        dfs_order, dfs_end = dfs_intervals(parent)
        # Each body's DOF range: the plan's dof_joint is contiguous per body.
        dof_joint_host = np.asarray(host["dof_joint"], dtype=np.int64)
        body_dof_start = np.zeros(bodies, np.int32)
        body_dof_count = np.zeros(bodies, np.int32)
        for dof, body in enumerate(dof_joint_host):
            if body_dof_count[body] == 0:
                body_dof_start[body] = dof
            elif body_dof_start[body] + body_dof_count[body] != dof:
                raise ValueError("Tree kinetic state requires each body's coordinates to be contiguous")
            body_dof_count[body] += 1
        if len(children) != bodies - len(self.layout.roots):
            raise ValueError("Tree kinetic state requires every true parent edge")

        group_to_art = sparse_owner.plan.group_to_art.numpy()
        starts = model.articulation_start.numpy()
        q_starts = model.joint_q_start.numpy()
        free_roots = solver._free_root_joint_indices
        free_roots = free_roots.numpy() if free_roots is not None else []
        root_lookup = {int(joint): slot for slot, joint in enumerate(free_roots)}
        root_slot = np.empty(self.group_count, np.int32)
        q_index = np.full((self.group_count, dofs), -1, np.int32)
        # Coordinate of each DOF: the joint's first coordinate plus the DOF's
        # offset within its joint (one coordinate per DOF; a ball joint's
        # quaternion is never a spring coordinate, its stiffness is zero).
        dof_joint = np.asarray(host["dof_joint"], dtype=np.int64)
        dof_offset = np.arange(dofs) - np.array([np.flatnonzero(dof_joint == dof_joint[d])[0] for d in range(dofs)])
        for group, art in enumerate(group_to_art):
            start = int(starts[art])
            if self.layout.root_dofs and start not in root_lookup:
                raise ValueError("Tree kinetic state is missing a free-root transport slot")
            root_slot[group] = root_lookup.get(start, -1)
            for dof in range(self.layout.root_dofs, dofs):
                q_index[group, dof] = q_starts[start + dof_joint[dof]] + dof_offset[dof]
        # Each group's coordinates are one contiguous range, from its first joint's.
        first_joint = starts[group_to_art]
        coordinates = q_starts[first_joint + bodies] - q_starts[first_joint]
        if np.any(coordinates != coordinates[0]):
            raise ValueError("Tree kinetic state requires one coordinate count per group")
        self.bank_coordinates = int(coordinates[0])
        if chain_scan_default is None:
            chain_scan_default = True
        self.chain_scan_requested = os.environ.get("FEATHER_PGS_TREE_CHAIN_SCAN", str(int(chain_scan_default))) == "1"
        self.chain_scan = False
        chain_host = None
        if self.chain_scan_requested:
            from . import tree_chain_scan  # noqa: PLC0415

            try:
                chain_host = tree_chain_scan.make_plan(parent)
            except ValueError:
                # Unsupported packing preserves the complete original owner.
                pass
        if chain_host is not None and self.layout.pack > 1:
            # Packed chain scans address threads directly; keep one tree per block.
            self.layout = dataclasses.replace(self.layout, lanes=0)
        if self.group_count % self.layout.pack:
            raise ValueError("Tree kinetic lane packing requires whole blocks of trees")
        self._chain_block_dim = cooperative_block_dim(self.layout, chain_host)
        if chain_host is None:
            self.plan = native.KineticPlan()
        else:
            self.plan = tree_chain_scan.ChainPlan()
            self.plan.chain_meta = wp.array(chain_host["meta"].reshape(-1), dtype=int, device=device)
            for name in ("light_offsets", "light_children", "stage_width"):
                setattr(self.plan, name, wp.array(chain_host[name], dtype=int, device=device))
            self.plan.chain_levels = chain_host["levels"]
            self.plan.chain_width = int(chain_host["widths"].max())
            self.chain_scan = True
        # Forest metadata (a single tree is the one-root case of the same arrays).
        root_locals = np.asarray(self.layout.roots, np.int32)
        root_of = np.zeros(bodies, np.int32)
        for local in range(1, bodies):
            root_of[local] = (
                root_of[parent[local]] if parent[local] >= 0 else int(np.flatnonzero(root_locals == local)[0])
            )
        arts = getattr(sparse_owner, "forest_arts", None)
        arts = np.asarray(arts, np.int32) if arts is not None else group_to_art.reshape(-1, 1).astype(np.int32)
        # Predicted body twists for the resolved-world classifier: the predictor
        # publishes them for warp-sized plans (the chain plan keeps its own scans);
        # FEATHER_PGS_PREDICTED_TWISTS=0 leaves the classifier to its own kernel.
        self.publishes_twists = (
            chain_host is None
            and self.layout.lanes <= 32
            and os.environ.get("FEATHER_PGS_PREDICTED_TWISTS", "1") == "1"
        )
        self._no_twists = wp.zeros(1, dtype=wp.spatial_vector, device=device)
        if self.publishes_twists:
            if getattr(solver, "_predicted_twist", None) is None:
                solver._predicted_twist = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=device)
                solver._predicted_twist_arts_host = np.zeros(model.articulation_count, np.int32)
            solver._predicted_twist_arts_host[np.asarray(arts, np.int64).reshape(-1)] = 1
        free = [r for r, c in enumerate(self.layout.root_coordinates) if c == 6]
        free_root_local = root_locals[free] if free else np.zeros(1, np.int32)
        free_root_dof = (
            np.asarray([self.layout.root_offsets[r] for r in free], np.int32) if free else np.zeros(1, np.int32)
        )
        dof_passive = np.ones(dofs, np.int32)
        for r, c in enumerate(self.layout.root_coordinates):
            dof_passive[self.layout.root_offsets[r] : self.layout.root_offsets[r] + c] = 0
        if self.layout.forest:
            root_slot = np.full((self.group_count, max(1, len(free))), -1, np.int32)
            for group in range(self.group_count):
                for k, r in enumerate(free):
                    root_slot[group, k] = root_lookup.get(int(starts[arts[group, r]]), -1)
                    if root_slot[group, k] < 0:
                        raise ValueError("Tree kinetic state is missing a free-root transport slot")
            root_slot = root_slot.reshape(-1)
        for name, values in (
            ("parent", parent),
            ("depth", depth),
            ("jump", jump),
            ("children_offsets", np.asarray(offsets, np.int32)),
            ("children", np.asarray(children, np.int32)),
            ("q_index", q_index.reshape(-1)),
            ("root_slot", root_slot),
        ):
            setattr(self.plan, name, wp.array(values, dtype=int, device=device))
        if chain_host is None:
            # Chain-scan plans keep their own struct; forests are kinetic plans only.
            for name, values in (
                ("dfs_order", dfs_order),
                ("dfs_end", dfs_end),
                ("body_dof_start", body_dof_start),
                ("body_dof_count", body_dof_count),
                ("arts", arts),
                ("root_of", root_of),
                ("root_local", root_locals),
                ("free_root_local", free_root_local),
                ("free_root_dof", free_root_dof),
                ("dof_passive", dof_passive),
            ):
                setattr(self.plan, name, wp.array(values, dtype=int, device=device))
        elif self.layout.forest:
            raise ValueError("Forest groups do not use the packed chain scan")
        self.data = native.KineticData()
        self.data.stage_clocks = cached_state_binding.stage_clocks(solver)
        groups = self.group_count
        self.data.bias = wp.empty((groups, dofs), dtype=float, device=device)
        self.data.com_offset = wp.empty((groups, bodies), dtype=wp.vec3, device=device)
        self.data.geometric = wp.empty((groups, self.layout.entries), dtype=float, device=device)
        self.data.current_valid = solver._fk_id_cache_valid
        self.data.geometry_valid = wp.zeros(groups, dtype=int, device=device)
        self.data.generation = wp.zeros(groups, dtype=wp.int64, device=device)
        self.data.geometry_generation = wp.full(groups, -1, dtype=wp.int64, device=device)
        self.data.source_q = wp.zeros(groups, dtype=wp.uint64, device=device)
        self.data.source_qd = wp.zeros(groups, dtype=wp.uint64, device=device)
        self.data.status = wp.zeros(groups, dtype=int, device=device)
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
        self.lane_timing = {}
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
        if not self.chain_scan:
            self._choose_lanes()
        self._plan_dimensions = self._dimensions()
        # Everything make_plan and the kernels bake in, proven on the device.
        self._proof = DeviceProof(
            [(name, getattr(model, name)) for name in _PLAN_FIELDS]
            + [(f"{name} mapping", self._mapping(name)) for name in _MAPPING_FIELDS],
            device,
        )

    def _bind_kernels(self):
        from . import tree_kinetic_state as native  # noqa: PLC0415

        layout = self.layout
        on_cpu = self.solver.model.device.is_cpu
        self.block_dim = max(layout.block, self._chain_block_dim)
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
            self.chain_scan,
            layout=layout,
            body_velocity=self._body_velocity,
            motion_velocity=self.motion_velocity,
            reference=self.reference,
            clocked=self.clocked,
            host=on_cpu,
        )
        # A classified repair compares in a pass of its own over the blocks its classification
        # lists for comparison. Otherwise a second one-pass repair kernel compares inside its
        # blocks, and only repairs that compare launch it: no extra launch, and every other
        # repair keeps its kernel (comparing inside raises a chain-scan repair's registers, 80
        # to 151 on a 13-body chain on sm_120, but not a packed one's, 72 either way).
        self.bank_kernel = self.bank_repair_kernel = None
        if self.bank_compare and self.classify_first:
            self.bank_kernel = native.get_bank_kernel(self.chain_scan, layout=layout, coordinates=self.bank_coordinates)
        elif self.bank_compare:
            self.bank_repair_kernel = native.get_state_kernel(
                False,
                self.chain_scan,
                layout=layout,
                body_velocity=self._body_velocity,
                bank_coordinates=self.bank_coordinates,
                reference=self.reference,
                clocked=self.clocked,
                host=on_cpu,
            )
        self.classify_kernel = self.listed_repair_kernel = None
        if self.classify_first:
            # Ordinary motion only: classify_first excludes motion-velocity layouts.
            self.classify_kernel = native.get_classify_kernel(self.chain_scan, layout=layout)
            self.listed_repair_kernel = native.get_state_kernel(
                False,
                self.chain_scan,
                layout=layout,
                body_velocity=self._body_velocity,
                listed=True,
                reference=self.reference,
                clocked=self.clocked,
                host=on_cpu,
            )
        self.finish_kernel = native.get_state_kernel(
            True,
            self.chain_scan,
            layout=layout,
            body_velocity=self._body_velocity,
            reference=self.reference,
            clocked=self.clocked,
            host=on_cpu,
        )
        self.predictor_kernel = native.get_predictor_kernel(
            self.chain_scan, layout=layout, response=self._response, reference=self.reference, clocked=self.clocked
        )

    def _choose_lanes(self):
        """Time the candidate lane widths on this model and keep the fastest.

        The finish kernel (scan, body terms, reduction, geometry, integration,
        publication) is the owner's dominant launch and the one whose cost
        depends on lanes per tree versus trees per block. No static rule fit
        the measurements, so the width is chosen by running each candidate a
        few times at construction, before any graph capture, on throwaway
        states; the caches are then reset so the first real step repairs.
        """
        candidates = lane_candidates(self.layout.lanes, self.group_count)
        if len(candidates) < 2:
            return
        import time  # noqa: PLC0415

        from .tree_kinetic_state import ForceInput  # noqa: PLC0415

        model, device, solver = self.solver.model, self.solver.model.device, self.solver
        probe_in, probe_out = model.state(), model.state()
        data = self._publication(probe_in, solver, probe_out, 1.0 / 240.0)
        # Steady state: finish without a geometry refresh, then predict (repair
        # exits early on an unchanged bank and refresh is periodic).
        finish_arguments = [
            self.sparse.plan,
            self.plan,
            data,
            self.data,
            solver._mass_update_requested,
            0,
            probe_in.joint_qd,
            self._repair_blocks,
            self._list_counts,
            self._no_bank,
        ]
        force = ForceInput()
        force.joint_q, force.joint_qd, force.predictor_qd = probe_in.joint_q, probe_in.joint_qd, probe_in.joint_qd
        force.joint_f, force.body_f, force.body_flags = model.control().joint_f, probe_in.body_f, model.body_flags
        force.stiffness, force.reference = solver._passive_spring_stiffness, solver._passive_spring_ref
        force.damping, force.u0, force.S = solver._passive_joint_damping, solver.joint_tau, solver.joint_S_s
        force.kinematic_joint, force.qdd, force.vhat = solver._kinematic_joint_mask, solver.joint_qdd, solver.v_hat
        force.dt = 1.0 / 240.0
        force.stage_clocks = cached_state_binding.stage_clocks(solver)
        twists = getattr(solver, "_predicted_twist", None)
        force.predicted_twist = self._no_twists if twists is None else twists
        # Publish only when the classifier, built after this owner, consumes them.
        consumer = getattr(solver, "_resolved_worlds_owner", None)
        force.publish_twists = int(self.publishes_twists and consumer is not None and consumer.published)
        predict_arguments = [self.sparse.plan, self.sparse.data, self.plan, self.data, force]
        base = self.layout
        timings = {}
        for lanes in candidates:
            self.layout = dataclasses.replace(base, lanes=lanes)
            self._bind_kernels()
            launches = (
                (self.finish_kernel, finish_arguments),
                (self.predictor_kernel, predict_arguments),
            )
            # One warm pass populates the caches the timed launches read.
            for kernel, arguments in launches:
                wp.launch_tiled(
                    kernel, dim=[self.launch_count], inputs=arguments, block_dim=self.block_dim, device=device
                )
            best = math.inf
            for _ in range(6):
                wp.synchronize_device(device)
                started = time.perf_counter()
                for kernel, arguments in launches:
                    wp.launch_tiled(
                        kernel, dim=[self.launch_count], inputs=arguments, block_dim=self.block_dim, device=device
                    )
                wp.synchronize_device(device)
                best = min(best, time.perf_counter() - started)
            timings[lanes] = best
        self.lane_timing = timings
        self.layout = dataclasses.replace(base, lanes=min(timings, key=timings.get))
        self._bind_kernels()
        # The probes wrote the caches and the solver's working arrays; make the
        # first real step repair everything from its own state.
        for name in ("current_valid", "geometry_valid", "status", "generation", "source_q", "source_qd"):
            getattr(self.data, name).zero_()
        self.data.geometry_generation.fill_(-1)

    def _dimensions(self):
        model = self.solver.model
        return (
            self.solver.world_count,
            model.world_count,
            model.articulation_count,
            model.body_count,
            model.joint_count,
            model.joint_coord_count,
            model.joint_dof_count,
        )

    def _mapping(self, name):
        return (
            self.solver._free_root_joint_indices
            if name == "free_root_joint_indices"
            else getattr(self.sparse.plan, name)
        )

    def validate_model(self, flags=None):
        """Reject changed baked ownership before notification invalidates any state.

        Numeric notifications (joint DOF, inertial, shape and model properties)
        change values the caches refresh from, never parents, axes, anchors or
        body membership, so they are accepted outright. Every other
        notification compares the baked plan arrays on the device (one small
        readback) instead of fingerprinting them on the host; Isaac Lab issues
        a notification on every reset. Body flags are not baked: the kernels
        read the kinematic masks and armature the solver re-derives in place,
        after rejecting any flag change it cannot absorb.

        Returns:
            Whether the proof ran and passed: the device checksums of every
            baked plan array, the joint frames ``joint_X_p`` and ``joint_X_c``
            and the joint axes included, match their construction values. A
            device checksum is two position-sensitive 32-bit sums, not a
            word-by-word compare: one changed word always shows, and a random
            multi-word change collides with a probability of about 2**-64.
            Arrays the proof cannot view as device words (every array on CPU)
            compare by an exact host fingerprint. Numeric notifications skip
            the proof and return ``False``.
        """
        if flags is not None and int(flags) != 0 and int(flags) & ~_NUMERIC_FLAGS == 0:
            return False
        if self._dimensions() != self._plan_dimensions:
            raise RuntimeError("Tree kinetic dimensions changed; reconstruct and recapture")

        def current(label):
            if label.endswith(" mapping"):
                return self._mapping(label[: -len(" mapping")])
            return getattr(self.solver.model, label)

        self._proof.validate(current, "Tree kinetic static")
        return True

    def forget_continuity(self):
        """Launch the next repair: a host call changed what the last finish left on the device."""
        self._continuity = None

    def invalidate(self, world_mask=None):
        """Invalidate current state for every owned articulation in selected worlds."""
        self.forget_continuity()
        cached_state_binding.invalidate(
            self.solver.world_count,
            world_mask,
            self._cache_to_art,
            self.sparse.plan.art_to_world,
            self.data.current_valid,
            self.data.geometry_valid,
        )

    def _validate_call(self, state, state_aug, dt):
        if state.requires_grad or state_aug is not self.solver:
            raise RuntimeError("Tree kinetic state excludes gradient/alternate augmented ownership")
        if self.admission is not None and not self.admission(self.solver):
            raise RuntimeError("Tree kinetic configuration changed; reconstruct and recapture")
        if bool(self.solver.enable_joint_velocity_limits) != self.motion_velocity:
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
            self.solver._mass_update_requested,
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

        def overlaps(array):
            return velocity.ptr < array.ptr + array.capacity and array.ptr < velocity.ptr + velocity.capacity

        if (velocity.ptr != raw.ptr and overlaps(raw)) or any(
            overlaps(array) for array in (state.joint_q, state_aug.joint_qdd, self.solver.v_hat, self.solver.v_out)
        ):
            raise RuntimeError("Tree kinetic motion velocity overlaps state coordinates or response outputs")
        return state.joint_q.ptr, raw.ptr, velocity.ptr

    def predict(self, state_in, state_aug, control, stage3_qd, dt):
        """Apply live force buckets through held W with original root transport."""
        from .tree_kinetic_state import ForceInput  # noqa: PLC0415

        self._validate_call(state_in, state_aug, dt)
        binding = self._validate_motion(state_in, state_aug, stage3_qd)
        if self.motion_velocity and binding != self._motion_binding:
            raise RuntimeError("Tree kinetic predictor motion velocity differs from the current-cache binding")
        solver = self.solver
        force = ForceInput()
        force.joint_q, force.joint_qd = state_in.joint_q, state_in.joint_qd
        force.predictor_qd = stage3_qd
        force.joint_f, force.body_f = control.joint_f, state_in.body_f
        force.body_flags = solver.model.body_flags
        force.stiffness = solver._passive_spring_stiffness
        force.reference = solver._passive_spring_ref
        force.damping = solver._passive_joint_damping
        force.u0, force.S = state_aug.joint_tau, state_aug.joint_S_s
        force.kinematic_joint = solver._kinematic_joint_mask
        force.qdd, force.vhat = state_aug.joint_qdd, solver.v_hat
        force.dt = dt
        force.stage_clocks = cached_state_binding.stage_clocks(solver)
        twists = getattr(solver, "_predicted_twist", None)
        force.predicted_twist = self._no_twists if twists is None else twists
        # Publish only when the classifier, built after this owner, consumes them.
        consumer = getattr(solver, "_resolved_worlds_owner", None)
        force.publish_twists = int(self.publishes_twists and consumer is not None and consumer.published)
        wp.launch_tiled(
            self.predictor_kernel,
            dim=[self.launch_count],
            inputs=[self.sparse.plan, self.sparse.data, self.plan, self.data, force],
            block_dim=self.block_dim,
            device=solver.model.device,
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
                self.solver._mass_update_requested,
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
