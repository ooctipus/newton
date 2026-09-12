# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental complete local prefix and current-contact packet ownership."""

import ast
import hashlib
import inspect
import linecache
import textwrap
from functools import cache

import numpy as np
import warp as wp


@wp.struct
class PrefixData:
    group_to_art: wp.array[int]
    art_to_world: wp.array[int]
    dof_start: wp.array[int]
    mimic_start: wp.array[int]
    mimic_list: wp.array[int]
    mimic_valid: wp.array[int]
    mimic_enabled: wp.array[wp.bool]
    mimic_dof0: wp.array[int]
    mimic_dof1: wp.array[int]
    mimic_q0: wp.array[int]
    mimic_q1: wp.array[int]
    mimic_coef0: wp.array[float]
    mimic_coef1: wp.array[float]
    limit_q_index: wp.array[int]
    lower: wp.array[float]
    upper: wp.array[float]
    q: wp.array[float]
    activation_gap: float
    beta: float
    cfm: float
    dt: float
    bias_scale: float
    limit_speculative_scale: float
    # Sparse private prefix coefficients, never a global dense J allocation.
    dofs: wp.array3d[int]
    weights: wp.array3d[float]
    row_contact: wp.array2d[int]
    mimic_slot: wp.array[int]
    slot_counter: wp.array[int]
    bounds: wp.array2d[int]
    kind: wp.array2d[int]
    parent: wp.array2d[int]
    mu: wp.array2d[float]
    row_beta: wp.array2d[float]
    row_cfm: wp.array2d[float]
    phi: wp.array2d[float]
    target: wp.array2d[float]
    restitution: wp.array2d[float]
    rhs: wp.array2d[float]
    diag: wp.array2d[float]
    impulses: wp.array2d[float]


@cache
def get_prefix_kernel():
    """Build two-or-fewer mimics and 18 candidate limits, four warps per CTA.

    Launch tiled dim=[ceil(primary_groups/4)], block_dim=128. Admission must
    guarantee one nine-DOF primary per world, at most two total mimic-list entries,
    augmented drives, no dense velocity-limit/connect rows and global cap>=20.
    Current q, enabled/coefficients and limits are read on every invocation.
    """
    source = r"""
#if defined(__CUDA_ARCH__)
    const int lane = threadIdx.x & 31;
    const int group = block * 4 + (threadIdx.x >> 5);
    if (group >= data.group_to_art.shape[0]) return;
#else
    if (logical_lane != 0) return;
    for (int group = block * 4; group < (block + 1) * 4 && group < data.group_to_art.shape[0]; ++group) {
    const int lane = 0;
#endif
    const int art = data.group_to_art.data[group];
    const int world = data.art_to_world.data[art];
    const int first_dof = data.dof_start.data[art];
    const int mimic_first = data.mimic_start.data[art];
    const int mimic_end = data.mimic_start.data[art + 1];
    const int M = data.kind.shape[1];
#if defined(__CUDA_ARCH__)
    for (int row = lane; row < 40; row += 32) data.row_contact.data[world * 40 + row] = -1;
    const int candidate = lane;
#else
    for (int row = 0; row < 40; ++row) data.row_contact.data[world * 40 + row] = -1;
    int count = 0;
    for (int candidate = 0; candidate < 20; ++candidate) {
#endif
    int active = 0, d0 = -1, d1 = -1, type = 3, mimic = -1;
    float w0 = 0.0f, w1 = 0.0f, phi = 0.0f;
    if (candidate < 2 && mimic_first + candidate < mimic_end) {
        mimic = data.mimic_list.data[mimic_first + candidate];
        data.mimic_slot.data[mimic] = -1;
        active = data.mimic_valid.data[mimic] != 0 && data.mimic_enabled.data[mimic];
        if (active) {
            type = 5;
            d0 = data.mimic_dof0.data[mimic] - first_dof;
            d1 = data.mimic_dof1.data[mimic] - first_dof;
            w0 = 1.0f;
            w1 = -data.mimic_coef1.data[mimic];
            phi = data.q.data[data.mimic_q0.data[mimic]]
                - data.mimic_coef1.data[mimic] * data.q.data[data.mimic_q1.data[mimic]]
                - data.mimic_coef0.data[mimic];
        }
    } else if (candidate >= 2 && candidate < 20) {
        const int local = (candidate - 2) / 2, side = (candidate - 2) & 1;
        const int dof = first_dof + local, qi = data.limit_q_index.data[dof];
        if (qi >= 0) {
            const float q = data.q.data[qi];
            const float bound = side == 0 ? data.lower.data[dof] : data.upper.data[dof];
            phi = side == 0 ? q - bound : bound - q;
            active = wp::isfinite(bound) && (side == 0 ? q <= bound + data.activation_gap
                                                                       : q >= bound - data.activation_gap);
            d0 = local;
            w0 = side == 0 ? 1.0f : -1.0f;
        }
    }
#if defined(__CUDA_ARCH__)
    const unsigned mask = __ballot_sync(0xffffffffu, active != 0);
    const int count = __popc(mask);
    const unsigned earlier = lane == 0 ? 0u : ((1u << lane) - 1u);
    const int slot = __popc(mask & earlier);
#else
    const int slot = count;
    count += active;
#endif
    if (active) {
        const int sparse = (world * 20 + slot) * 2;
        data.dofs.data[sparse] = d0;
        data.dofs.data[sparse + 1] = d1;
        data.weights.data[sparse] = w0;
        data.weights.data[sparse + 1] = w1;
        const int row = world * M + slot;
        data.kind.data[row] = type;
        data.parent.data[row] = -1;
        data.mu.data[row] = 0.0f;
        data.row_beta.data[row] = data.beta;
        data.row_cfm.data[row] = data.cfm;
        data.phi.data[row] = phi;
        data.target.data[row] = 0.0f;
        data.restitution.data[row] = 0.0f;
        float rhs = 0.0f;
        if (type == 5 || phi < 0.0f) rhs = data.bias_scale * data.beta * phi * (1.0f / data.dt);
        else rhs = data.limit_speculative_scale * phi * (1.0f / data.dt);
        data.rhs.data[row] = rhs;
        data.diag.data[row] = data.cfm;
        data.impulses.data[row] = 0.0f;
        if (mimic >= 0) data.mimic_slot.data[mimic] = slot;
    }
#if !defined(__CUDA_ARCH__)
    }
#endif
    if (lane == 0) {
        data.slot_counter.data[world] = count;
        data.bounds.data[world * 2] = count;
        data.bounds.data[world * 2 + 1] = count;
    }
#if !defined(__CUDA_ARCH__)
    }
#endif
"""

    @wp.func_native(source)
    def prefix_native(block: int, logical_lane: int, data: PrefixData): ...

    @wp.kernel(module="unique", enable_backward=False)
    def prefix(data: PrefixData):
        block, lane = wp.tid()
        prefix_native(block, lane, data)

    return prefix


def _adapt_local_snippet(snippet, dofs, paired_dofs):
    """Replace only current row reads; retain the original held response and GS."""
    if dofs != 9 or paired_dofs not in (0, 6):
        raise ValueError("Private row packets support the existing 9 and 9+6 local owners")
    primary = f"J_group.data[group_j_base + row * {dofs} + i]"
    replacement = """(packets.row_contact.data[world * 40 + row] >= 0
                ? packets.jacobian.data[(world * 40 + row) * 15 + i]
                : ((prefix.dofs.data[(world * 20 + row) * 2] == i
                        ? prefix.weights.data[(world * 20 + row) * 2] : 0.0f)
                   + (prefix.dofs.data[(world * 20 + row) * 2 + 1] == i
                        ? prefix.weights.data[(world * 20 + row) * 2 + 1] : 0.0f)))"""
    if snippet.count(primary) != 1:
        raise RuntimeError("Original local primary row-read seam changed")
    snippet = snippet.replace(primary, replacement)
    if paired_dofs:
        secondary = f"secondary_J_group.data[\n                secondary_group_j_base + row * {paired_dofs} + i]"
        if snippet.count(secondary) != 1:
            raise RuntimeError("Original local secondary row-read seam changed")
        snippet = snippet.replace(
            secondary,
            "(packets.row_contact.data[world * 40 + row] >= 0 "
            "? packets.jacobian.data[(world * 40 + row) * 15 + 9 + i] : 0.0f)",
        )
    return snippet


@cache
def get_local_factory():
    """Append the two private row operands to the existing native local ABI."""
    from . import solver_feather_pgs as solver  # noqa: PLC0415 - avoid constructor import cycles
    from .franka_contact_packet import ContactPacketData  # noqa: PLC0415

    original = inspect.getsource(solver._get_pgs_solve_local_owned_kernel)
    tree = ast.parse(textwrap.dedent(original))
    factory = tree.body[0]
    factory.name = "packet_local_factory"
    factory.decorator_list = []
    native_calls = 0
    signatures = 0
    for node in ast.walk(factory):
        if isinstance(node, ast.FunctionDef) and node is not factory:
            if any(arg.arg == "v_out" for arg in node.args.args):
                node.args.args += [
                    ast.arg(arg="prefix", annotation=ast.Name(id="PrefixData", ctx=ast.Load())),
                    ast.arg(arg="packets", annotation=ast.Name(id="ContactPacketData", ctx=ast.Load())),
                ]
                signatures += 1
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "pgs_solve_local_internal_native"
        ):
            node.args += [ast.Name(id="prefix", ctx=ast.Load()), ast.Name(id="packets", ctx=ast.Load())]
            native_calls += 1
    if (signatures, native_calls) != (3, 2):
        raise RuntimeError("Original local native/template signature count changed")
    insertion = next(
        i
        for i, node in enumerate(factory.body)
        if isinstance(node, ast.FunctionDef) and node.name == "pgs_solve_local_internal_native"
    )
    factory.body[insertion:insertion] = ast.parse("snippet = _adapt_local_snippet(snippet, dofs, paired_dofs)").body
    # Separate cache/module names; no change to the original callable/globals.
    for node in ast.walk(factory):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = node.value.replace("pgs_solve_local_internal_", "pgs_solve_local_packet_")
    namespace = dict(solver.__dict__)
    namespace.update(
        PrefixData=PrefixData, ContactPacketData=ContactPacketData, _adapt_local_snippet=_adapt_local_snippet
    )
    ast.fix_missing_locations(tree)
    generated = ast.unparse(tree) + "\n"
    filename = f"<franka-row-packet-{hashlib.sha256(generated.encode()).hexdigest()}>"
    linecache.cache[filename] = (len(generated), None, generated.splitlines(keepends=True), filename)
    exec(compile(generated, filename, "exec"), namespace)
    return namespace[factory.name]


def validate_prefix_topology(groups, worlds, starts, mimic_start, mimic_list, mimic_dof0, mimic_dof1, mimic_world=None):
    """Reject omitted list entries, including disabled entries before valid ones."""
    if len(groups) != len(set(worlds[groups])) or len(groups) != len(set(worlds)):
        return False
    for art in groups:
        if starts[art + 1] - starts[art] != 9 or mimic_start[art + 1] - mimic_start[art] > 2:
            return False
        for entry in mimic_list[mimic_start[art] : mimic_start[art + 1]]:
            if mimic_world is not None and mimic_world[entry] != worlds[art]:
                return False
            # Even disabled entries must be safe if subsequently enabled.
            if not (starts[art] <= mimic_dof0[entry] < starts[art + 1]):
                return False
            if not (starts[art] <= mimic_dof1[entry] < starts[art + 1]):
                return False
    return True


@wp.struct
class QueueData:
    primary: wp.array[int]
    secondary: wp.array[int]
    residual_secondary: wp.array[int]
    mf_count: wp.array[int]
    mf_body0: wp.array2d[int]
    mf_body1: wp.array2d[int]
    body_art: wp.array[int]
    count: wp.array[int]
    owner: wp.array[int]
    status: wp.array[int]
    general_count: wp.array[int]
    general: wp.array[int]
    pair_count: wp.array[int]
    pair: wp.array[int]
    pair_secondary: wp.array[int]
    residual_count: wp.array[int]
    residual: wp.array[int]
    residual_pair: wp.array[int]


@wp.func_native("""
#if defined(__CUDA_ARCH__)
return 32;
#else
return 1;
#endif
""")
def _row_stride() -> int: ...


@wp.kernel
def finalize_and_queue(prefix: PrefixData, queue: QueueData):
    """Fuse the original current dense-count clamp, classification and two queues."""
    world = wp.tid()
    rows = prefix.slot_counter[world]
    if rows > prefix.kind.shape[1]:
        wp.atomic_max(queue.status, 0, 1)
        rows = prefix.kind.shape[1]
    queue.count[world] = rows
    mf_count = queue.mf_count[world]
    primary = queue.primary[world]
    secondary = queue.secondary[world]
    residual_secondary = queue.residual_secondary[world]
    single = prefix.bounds[world, 1] == rows
    local_mf = mf_count > 0 and mf_count <= 12 and residual_secondary >= 0
    mf_row = int(0)
    while mf_row < mf_count and local_mf:
        body0, body1 = queue.mf_body0[world, mf_row], queue.mf_body1[world, mf_row]
        if body0 >= 0 and queue.body_art[body0] != residual_secondary:
            local_mf = False
        if body1 >= 0 and queue.body_art[body1] != residual_secondary:
            local_mf = False
        mf_row += 1
    owner = int(0)
    if rows > 0 and mf_count == 0:
        if single and rows <= 9 and primary >= 0:
            owner = 1
        elif not single and rows <= 20 and secondary >= 0:
            owner = 2
    if owner == 0 and rows > 0 and rows <= 40 and residual_secondary >= 0:
        if (mf_count == 0 and not single) or local_mf:
            owner = 3
    queue.owner[world] = owner
    if owner == 0 and (rows > 0 or mf_count > 0):
        slot = wp.atomic_add(queue.general_count, 0, 1)
        queue.general[slot] = world
    elif owner == 2:
        slot = wp.atomic_add(queue.pair_count, 0, 1)
        queue.pair[slot] = primary
        queue.pair_secondary[slot] = secondary
    elif owner == 3:
        slot = wp.atomic_add(queue.residual_count, 0, 1)
        queue.residual[slot] = primary
        queue.residual_pair[slot] = residual_secondary


@wp.kernel
def materialize_general_prefix(
    workers: int,
    prefix: PrefixData,
    queue: QueueData,
    group_start: wp.array[int],
    group_art: wp.array[int],
    art_group: wp.array[int],
    size: int,
    J: wp.array3d[float],
):
    """Clear and populate only current general-world group rows, including zeros."""
    worker, lane = wp.tid()
    for entry in range(worker, queue.general_count[0], workers):
        world = queue.general[entry]
        for item in range(group_start[world], group_start[world + 1]):
            art = group_art[item]
            group = art_group[art]
            for coefficient in range(lane, queue.count[world] * size, _row_stride()):
                row, dof = coefficient // size, coefficient % size
                value = float(0.0)
                if art == queue.primary[world] and row < prefix.bounds[world, 0]:
                    for component in range(2):
                        if prefix.dofs[world, row, component] == dof:
                            value += prefix.weights[world, row, component]
                J[group, row, dof] = value


@wp.kernel
def general_response(
    workers: int,
    queue: QueueData,
    group_start: wp.array[int],
    group_art: wp.array[int],
    art_group: wp.array[int],
    world_offset: wp.array[int],
    size: int,
    L: wp.array3d[float],
    J: wp.array3d[float],
    Y: wp.array3d[float],
    J_world: wp.array3d[float],
    Y_world: wp.array3d[float],
):
    """Retain the original per-row triangular response for the complete fallback."""
    worker, lane = wp.tid()
    for entry in range(worker, queue.general_count[0], workers):
        world = queue.general[entry]
        for item in range(group_start[world], group_start[world + 1]):
            art = group_art[item]
            group = art_group[art]
            for row in range(lane, queue.count[world], _row_stride()):
                for i in range(size):
                    value = J[group, row, i]
                    for k in range(i):
                        value -= L[group, i, k] * Y[group, row, k]
                    diagonal = L[group, i, i]
                    Y[group, row, i] = value / diagonal if diagonal != 0.0 else 0.0
                for reverse in range(size):
                    i = size - 1 - reverse
                    value = Y[group, row, i]
                    for k in range(i + 1, size):
                        value -= L[group, k, i] * Y[group, row, k]
                    diagonal = L[group, i, i]
                    Y[group, row, i] = value / diagonal if diagonal != 0.0 else 0.0
                for i in range(size):
                    J_world[world, row, world_offset[art] + i] = J[group, row, i]
                    Y_world[world, row, world_offset[art] + i] = Y[group, row, i]


@wp.kernel
def general_diag(
    workers: int,
    prefix: PrefixData,
    queue: QueueData,
    world_dofs: wp.array[int],
    J: wp.array3d[float],
    Y: wp.array3d[float],
):
    """Publish the general response diagonal plus CFM; private diagonals stay seeded."""
    worker, lane = wp.tid()
    for entry in range(worker, queue.general_count[0], workers):
        world = queue.general[entry]
        for row in range(lane, queue.count[world], _row_stride()):
            value = float(0.0)
            for dof in range(world_dofs[world]):
                value += J[world, row, dof] * Y[world, row, dof]
            prefix.diag[world, row] = value + prefix.row_cfm[world, row]


class LocalRowPackets:
    """Own one bounded prefix/current-row boundary; all general rows stay complete."""

    def __init__(self, solver):
        from .franka_contact_packet import allocate_packet  # noqa: PLC0415

        self.solver = solver
        device = solver.model.device
        self.workers = max(1, min(solver.world_count, int(device.sm_count) * 4))
        self.prefix = PrefixData()
        self.prefix.dofs = wp.empty((solver.world_count, 20, 2), dtype=int, device=device)
        self.prefix.weights = wp.empty((solver.world_count, 20, 2), dtype=float, device=device)
        self.packet = allocate_packet(solver.world_count, device)
        self.prefix.row_contact = self.packet.row_contact
        self.queue = QueueData()
        for name, source in (
            ("primary", "_local_primary_articulation"),
            ("secondary", "_local_pair_articulation"),
            ("residual_secondary", "_local_residual_pair_articulation"),
            ("mf_count", "mf_constraint_count"),
            ("mf_body0", "mf_body_a"),
            ("mf_body1", "mf_body_b"),
            ("body_art", "body_to_articulation"),
            ("count", "constraint_count"),
            ("owner", "_local_solve_owner"),
            ("general_count", "_local_general_world_count"),
            ("general", "_local_general_worlds"),
        ):
            setattr(self.queue, name, getattr(solver, source))
        for name, source in (
            ("pair_count", "_local_pair_active_counts"),
            ("pair", "_local_pair_active_candidates"),
            ("pair_secondary", "_local_pair_active_secondaries"),
            ("residual_count", "_local_residual_active_counts"),
            ("residual", "_local_residual_active_candidates"),
            ("residual_pair", "_local_residual_active_secondaries"),
        ):
            setattr(self.queue, name, getattr(solver, source)[9])
        self.contact_input = None
        self._topology_signature = topology_signature(solver)

    def validate_notification(self):
        """Allow live numeric changes, but require reconstruction for structural edits."""
        if topology_signature(self.solver) != self._topology_signature or not topology_supported(self.solver):
            raise RuntimeError(
                "Private row-packet topology changed; reconstruct the solver to select a complete fallback"
            )

    def prepare_prefix(self, state_in, dt):
        """Bind current coefficients/state and the current double-buffered row owners."""
        solver, data = self.solver, self.prefix
        # Sticky status is allocated after tiled-kernel construction.
        self.queue.status = solver._constraint_capacity_status
        for name, source in (
            ("art_to_world", "art_to_world"),
            ("dof_start", "articulation_dof_start"),
            ("mimic_start", "_mimic_art_start"),
            ("mimic_list", "_mimic_art_list"),
            ("mimic_valid", "_mimic_valid"),
            ("mimic_dof0", "_mimic_dof0"),
            ("mimic_dof1", "_mimic_dof1"),
            ("mimic_q0", "_mimic_q0"),
            ("mimic_q1", "_mimic_q1"),
            ("limit_q_index", "_joint_limit_q_index"),
            ("mimic_slot", "mimic_slot"),
            ("slot_counter", "slot_counter"),
            ("bounds", "dense_phase_bounds"),
            ("kind", "row_type"),
            ("parent", "row_parent"),
            ("mu", "row_mu"),
            ("row_beta", "row_beta"),
            ("row_cfm", "row_cfm"),
            ("phi", "phi"),
            ("target", "target_velocity"),
            ("restitution", "row_restitution"),
            ("rhs", "rhs"),
            ("diag", "diag"),
            ("impulses", "impulses"),
        ):
            setattr(data, name, getattr(solver, source))
        data.group_to_art = solver.group_to_art[9]
        for name, source in (
            ("mimic_enabled", "constraint_mimic_enabled"),
            ("mimic_coef0", "constraint_mimic_coef0"),
            ("mimic_coef1", "constraint_mimic_coef1"),
            ("lower", "joint_limit_lower"),
            ("upper", "joint_limit_upper"),
        ):
            setattr(data, name, getattr(solver.model, source))
        data.q, data.dt = state_in.joint_q, dt
        data.activation_gap, data.beta, data.cfm = solver.joint_limit_activation_gap, solver.pgs_beta, solver.pgs_cfm
        data.bias_scale, data.limit_speculative_scale = 1.0, 1.0
        self.contact_input = None
        wp.launch_tiled(
            get_prefix_kernel(),
            dim=[(solver.n_arts_by_size[9] + 3) // 4],
            inputs=[data],
            block_dim=128,
            device=solver.model.device,
        )

    def produce_contacts(self, state_in, state_aug, contacts, dt):
        """Publish private current J and complete canonical contact metadata/RHS."""
        from .franka_contact_packet import bind, produce_contacts  # noqa: PLC0415

        self.contact_input = bind(self.solver, state_in, state_aug, contacts, dt, 1.0)
        wp.launch_tiled(
            produce_contacts,
            dim=[self.workers],
            inputs=[self.workers, self.contact_input, self.packet],
            block_dim=32,
            device=self.solver.model.device,
        )

    def finish_rows(self):
        """Finalize and route current counts, then reconstruct only the general path."""
        from .franka_contact_packet import materialize_fallback  # noqa: PLC0415

        solver, queue = self.solver, self.queue
        queue.general_count.zero_()
        queue.pair_count.zero_()
        queue.residual_count.zero_()
        wp.launch(finalize_and_queue, dim=solver.world_count, inputs=[self.prefix, queue], device=solver.model.device)
        for size in solver.size_groups:
            wp.launch_tiled(
                materialize_general_prefix,
                dim=[self.workers],
                inputs=[
                    self.workers,
                    self.prefix,
                    queue,
                    solver.world_response_group_art_start[size],
                    solver.world_response_group_to_art[size],
                    solver.art_group_idx,
                    size,
                    solver.J_by_size[size],
                ],
                block_dim=32,
                device=solver.model.device,
            )
            if self.contact_input is not None:
                wp.launch_tiled(
                    materialize_fallback,
                    dim=[self.workers],
                    inputs=[
                        self.workers,
                        self.contact_input,
                        queue.owner,
                        size,
                        solver.art_group_idx,
                        solver.J_by_size[size],
                    ],
                    block_dim=32,
                    device=solver.model.device,
                )

    def response(self, size):
        """Run the original held-L response on the compact general-world queue."""
        solver = self.solver
        wp.launch_tiled(
            general_response,
            dim=[self.workers],
            inputs=[
                self.workers,
                self.queue,
                solver.world_response_group_art_start[size],
                solver.world_response_group_to_art[size],
                solver.art_group_idx,
                solver.articulation_world_dof_offset,
                size,
                solver.L_by_size[size],
                solver.J_by_size[size],
                solver.Y_by_size[size],
                solver.J_world,
                solver.Y_world,
            ],
            block_dim=32,
            device=solver.model.device,
        )

    def diagonal(self):
        """Keep private CFM seeds and finalize only complete general response diagonals."""
        solver = self.solver
        wp.launch_tiled(
            general_diag,
            dim=[self.workers],
            inputs=[self.workers, self.prefix, self.queue, solver.world_dof_count, solver.J_world, solver.Y_world],
            block_dim=32,
            device=solver.model.device,
        )


def topology_supported(solver):
    """Check all prefix entries, not only currently enabled or valid mimic rows."""
    if sorted(solver.size_groups) != [6, 9] or solver.n_arts_by_size[9] != solver.world_count:
        return False
    if solver._mimic_art_start is None or solver._mimic_art_list is None:
        return False
    if solver._mimic_sizes - {9} or solver._joint_limit_sizes - {9} or solver._connect_count:
        return False
    if any(9 not in getattr(solver, name) for name in ("_local_pair_active_counts", "_local_residual_active_counts")):
        return False
    primary = solver._local_primary_articulation.numpy()
    if np.any(primary < 0):
        return False
    return validate_prefix_topology(
        solver.group_to_art[9].numpy(),
        solver.art_to_world.numpy(),
        solver.articulation_dof_start.numpy(),
        solver._mimic_art_start.numpy(),
        solver._mimic_art_list.numpy(),
        solver._mimic_dof0.numpy(),
        solver._mimic_dof1.numpy(),
        solver._mimic_world.numpy(),
    )


def topology_signature(solver):
    """Pin only structural indices; enabled flags, coefficients, q and limits stay live."""
    names = (
        "constraint_mimic_joint0",
        "constraint_mimic_joint1",
        "constraint_mimic_world",
        "joint_type",
        "joint_parent",
        "joint_child",
        "joint_q_start",
        "joint_qd_start",
        "body_flags",
    )
    return tuple((name, getattr(solver.model, name).numpy().tobytes()) for name in names)


def create_owner(solver):
    """Admit only the original local law; unsupported configurations stay original."""
    from . import solver_feather_pgs as source  # noqa: PLC0415

    if not (
        solver._local_internal_fast_path
        and solver._hinv_jt_writes_world
        and solver.model.device.is_cuda
        and not solver.model.requires_grad
        and solver._local_solve_max_rows == 20
        and solver._local_residual_max_rows == 40
        and solver._local_residual_mf_max_rows == 12
        and solver.dense_max_constraints >= 40
        and solver.pgs_mode == "matrix_free"
        and solver.drive_mode == "augmented"
        and solver.pgs_schedule == "interleaved"
        and solver.friction_mode == "current"
        and not solver.enable_joint_velocity_limits
        and solver.enable_joint_limits
        and solver.pgs_velocity_iterations == 0
        and not solver.pgs_warmstart
        and not solver._mf_warmstart_enabled
        and not solver._regularization_enabled
        and not solver._preelim_active
        and not solver.grouped_dynamics
        and not solver._debug_buffers_enabled
        and not solver._sparse_diagonal_contact_solve
        and not source._COMPACT_CONTACT_BOUNDARY
        and not solver._paired_factor_coordinates
        and not solver._fused_diagonal_joint_limits
        and all(
            not solver._execution_plan.use_tiled_hinv_jt(size) and not solver._execution_plan.use_diagonal_mass(size)
            for size in solver.size_groups
        )
    ):
        return None
    if any(
        (
            source._FPGS_CAPTURE,
            source._INK_ON,
            source._WR_ON,
            source._DEBUG_CACHE,
            source._DEBUG_CACHE_MODE,
            source._DEBUG_CACHE_CMP,
            source._GROUPED_CHECK,
            source._CHECK_ROWS,
            source._CHECK_ROWS_FUSED,
        )
    ):
        return None
    return LocalRowPackets(solver) if topology_supported(solver) else None
