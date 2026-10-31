# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build a world's limit prefix and contact packets in place for the local solve owners.

A world made of one primary articulation (the one carrying joint limits and
mimic rows) and at most one secondary articulation (a free body, say) solves
its rows in the fused local kernels, which read the rows' physical Jacobians
directly. This producer writes those rows where the local kernels read them:
the limit and mimic prefix as sparse coefficients, the contact rows as dense
packets in the world's physical coordinates, both with the complete row
metadata and right-hand sides of the ordinary producers. Only worlds the
local owners decline (too many rows, matrix-free rows of another body,
unsupported row kinds) have their rows materialized into the ordinary
per-size buffers and run the ordinary response. Every size here comes from
the solver's topology plan; nothing names a robot.

Persistent patch friction writes its tangent rows with the patch row law.
The patch link, ``_FrictionPatchState.link``, runs after this producer and
alone links the patch rings and divides the friction coefficient; the
local owners pool each ring's normal load from the linked parents.
"""

from functools import cache, partial
from types import SimpleNamespace

import numpy as np
import warp as wp

from ...sim import ModelFlags
from . import field_group, sweep_routes_local
from .contact_input import ContactInput
from .contact_laws import (
    CONTACT_SPIN_ROW,
    contact_bodies,
    contact_friction_points,
    contact_points,
    contact_row_frame,
    contact_tangent_basis,
)
from .dense_rows import RowSupport, jacobian_row
from .device_preludes import (
    lane_group,
    reserved_count,
    warp_broadcast,
    warp_lane,
    warp_stride,
    warp_sync,
)
from .field_group import Field, FieldGroup, SolverArrays
from .friction_patches import PatchRowLaw, patch_friction_location, patch_row_law
from .kernel_template import KernelTemplate, substitute
from .kernels import PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT
from .prefix_rows import LAW, limits, prefix_rows
from .row_build import _contact_in_sizes, endpoint_projection, populate_contact_J, write_contact_row
from .row_inputs import CompactLocalRows, RowArgument
from .row_laws import ROW_LAW
from .step_plan import JOINT_TOPOLOGY, TopologyPin, first_failed

WARPS_PER_BLOCK = 4
PREFIX_WIDTH = 2  # a limit row has one coefficient, a mimic row two


PREFIX = (
    Field("group_to_art", wp.int32, ("art",), owner="step"),
    Field("art_to_world", wp.int32, ("articulation",), owner="forest"),
    Field("dof_start", wp.int32, ("articulation",), owner="forest", attr="articulation_dof_start"),
    Field("mimic_start", wp.int32, ("articulation_bound",), owner="mimics", attr="_mimic_art_start"),
    Field("mimic_list", wp.int32, ("mimic",), owner="mimics", attr="_mimic_art_list"),
    Field("mimic_valid", wp.int32, ("mimic",), owner="mimics", attr="_mimic_valid"),
    Field("mimic_enabled", wp.bool, ("mimic",), owner="mimic_model", attr="constraint_mimic_enabled"),
    Field("mimic_dof0", wp.int32, ("mimic",), owner="mimics", attr="_mimic_dof0"),
    Field("mimic_dof1", wp.int32, ("mimic",), owner="mimics", attr="_mimic_dof1"),
    Field("mimic_q0", wp.int32, ("mimic",), owner="mimics", attr="_mimic_q0"),
    Field("mimic_q1", wp.int32, ("mimic",), owner="mimics", attr="_mimic_q1"),
    Field("mimic_coef0", wp.float32, ("mimic",), owner="mimic_model", attr="constraint_mimic_coef0"),
    Field("mimic_coef1", wp.float32, ("mimic",), owner="mimic_model", attr="constraint_mimic_coef1"),
    *LAW,
    # Sparse private prefix coefficients, never a global dense J allocation, and each packet row's contact.
    Field("dofs", wp.int32, ("world", "packet", PREFIX_WIDTH), fill=None, owner="packets"),
    Field("weights", wp.float32, ("world", "packet", PREFIX_WIDTH), fill=None, owner="packets"),
    Field("row_contact", wp.int32, ("world", "packet"), fill=-1, owner="packets"),
    Field("mimic_slot", wp.int32, ("mimic",), owner="mimics"),
    Field("first_rejected_slot", wp.int32, ("world",), attr="dense_first_rejected_slot", owner="placement"),
    Field("impulses", wp.float32, ("world", "row"), owner="placement"),
)
"""The prefix kernel's inputs and outputs: the size group, the mimic tables (``mimics``: the solver's, or the
placeholders of a model without mimics), the prefix law and the dense rows (:data:`.prefix_rows.LAW`), and the
private prefix packets."""

PrefixData = field_group.struct("PrefixData", PREFIX, __name__)


PACKETS = FieldGroup(*(field for field in PREFIX if field.owner == "packets"))
"""The prefix's private packets: its coefficients and the contact of each packet row (-1 for none)."""

CONTACT_PACKETS = (Field("jacobian", wp.float32, ("world", "packet", "coordinate")), PACKETS["row_contact"])
"""The contact packets: each packet row's Jacobian over the pair's coordinates, and its contact."""

ContactPacketData = field_group.struct("ContactPacketData", CONTACT_PACKETS, __name__)


QUEUE = (
    Field("primary", wp.int32, ("articulation",), attr="_local_primary_articulation"),
    Field("secondary", wp.int32, ("articulation",), attr="_local_pair_articulation"),
    Field("residual_secondary", wp.int32, ("articulation",), attr="_local_residual_pair_articulation"),
    Field("mf_count", wp.int32, ("world",), attr="mf_constraint_count", owner="placement"),
    Field("mf_body0", wp.int32, ("world", "mf_row"), attr="mf_body_a", owner="placement"),
    Field("mf_body1", wp.int32, ("world", "mf_row"), attr="mf_body_b", owner="placement"),
    Field("mf_row_type", wp.int32, ("world", "mf_row"), owner="placement"),
    Field("body_art", wp.int32, ("body",), owner="forest", attr="body_to_articulation"),
    Field("count", wp.int32, ("world",), attr="constraint_count", owner="placement"),
    Field("owner", wp.int32, ("world",), attr="local_solve_owner", owner="placement"),
    Field("status", wp.int32, (2,), attr="constraint_capacity_status", owner="placement"),
    Field("general_count", wp.int32, (1,), attr="_local_general_world_count"),
    Field("general", wp.int32, ("world",), attr="_local_general_worlds"),
    Field("pair_count", wp.int32, (1,), owner="size"),
    Field("pair", wp.int32, ("candidate",), owner="size"),
    Field("pair_secondary", wp.int32, ("candidate",), owner="size"),
    Field("residual_count", wp.int32, (1,), owner="size"),
    Field("residual", wp.int32, ("candidate",), owner="size"),
    Field("residual_pair", wp.int32, ("candidate",), owner="size"),
)
"""The local owners' world queues: the solver's, and this size group's active pair and residual candidates."""

QueueData = field_group.struct("QueueData", QUEUE, __name__)


_PREFIX_SOURCE = r"""
    // A warp runs each group on CUDA; on the CPU one thread runs the block's groups in turn.
    for (int group = block * $WARPS + SUB; group < (block + 1) * $WARPS && group < data.group_to_art.shape[0]; group += GROUPS) {
    const int art = data.group_to_art.data[group];
    const int world = data.art_to_world.data[art];
    const int first_dof = data.dof_start.data[art];
    const int mimic_first = data.mimic_start.data[art];
    const int mimic_end = data.mimic_start.data[art + 1];
    for (int row = LANE; row < $PACKETS; row += STRIDE) data.row_contact.data[world * $PACKETS + row] = -1;
$ROWS    }
"""

_MIMIC = """if (mimic_first + index < mimic_end) {
                mimic = data.mimic_list.data[mimic_first + index];
                data.mimic_slot.data[mimic] = -1;
                active = data.mimic_valid.data[mimic] != 0 && data.mimic_enabled.data[mimic];
                if (active) {
                    type = 5;
                    beta = data.beta;
                    d0 = data.mimic_dof0.data[mimic] - first_dof;
                    d1 = data.mimic_dof1.data[mimic] - first_dof;
                    w0 = 1.0f;
                    w1 = -data.mimic_coef1.data[mimic];
                    phi = RowLaw::mimic_phi(data.q.data[data.mimic_q0.data[mimic]], data.q.data[data.mimic_q1.data[mimic]],
                                            data.mimic_coef0.data[mimic], data.mimic_coef1.data[mimic]);
                }
            }"""
"""The mimic family: an enabled, valid mimic row of the primary (``J = e_follower - coef1 e_leader``)."""

_SINK = """            const int sparse = (world * $PACKETS + slot) * 2;
            data.dofs.data[sparse] = d0;
            data.dofs.data[sparse + 1] = d1;
            data.weights.data[sparse] = w0;
            data.weights.data[sparse + 1] = w1;
            data.diag.data[at] = data.cfm;
            data.impulses.data[at] = 0.0f;
            if (mimic >= 0) data.mimic_slot.data[mimic] = slot;"""
"""The packets' sink: a row's sparse coefficients, its diagonal seed and impulse, and its mimic's slot."""


@cache
def get_prefix_kernel(primary: int, mimics: int, packets: int):
    """Build the mimic rows and the candidate limits of one primary articulation per warp.

    Launch tiled ``dim=[ceil(primaries / WARPS_PER_BLOCK)]``, ``block_dim=128``.
    Admission guarantees one primary per world, at most ``mimics`` mimic entries
    per primary, and ``mimics + 2 * primary <= packets`` so every active prefix
    row has a slot. Current q, enabled flags, coefficients and limits are read
    on every invocation.
    """
    if primary <= 0 or mimics < 0 or mimics + 2 * primary > packets:
        raise ValueError("the prefix candidates must fit the packet capacity")
    positions = ((mimics, "", _MIMIC),) if mimics else ()
    rows = prefix_rows(
        ((*positions, limits(tuple(range(primary)), "first_dof + d", "data.")), ()),
        law="data.",
        sink=substitute(_SINK, PACKETS=packets),
        ordered=True,
        rhs=True,
        locals_="int mimic = -1;",
        side="art",
    )
    source = substitute(_PREFIX_SOURCE, ROWS=rows, PACKETS=packets, WARPS=WARPS_PER_BLOCK)
    return KernelTemplate(
        f"row_packet_prefix{primary}_m{mimics}_r{packets}",
        (("data", PrefixData),),
        source,
        preludes=(lane_group(32), *ROW_LAW.preludes()),
        dual=True,
        native_name="prefix_native",
    ).kernel()


@cache
def get_local_factory(primary: int, secondary: int, packets: int):
    """Bind the packet layout to the shared local row-input interface."""
    inputs = CompactLocalRows(
        primary,
        secondary,
        packets,
        packets,
        PREFIX_WIDTH,
        (RowArgument("prefix", PrefixData), RowArgument("packets", ContactPacketData)),
    )
    return partial(sweep_routes_local.local_owned_kernel, row_inputs=inputs)


def validate_prefix_topology(
    groups, worlds, starts, mimic_start, mimic_list, mimic_dof0, mimic_dof1, primary, mimics, mimic_world=None
):
    """Check every prefix entry without per-world Python iteration.

    Notifications can reach this predicate on every batch reset. Keep the
    complete current topology checks, including disabled mimic entries, but
    vectorize over worlds and iterate only over the possible mimic slots.
    """
    art_count = min(len(worlds), len(starts) - 1, len(mimic_start) - 1)
    if np.any(groups < 0) or np.any(groups >= art_count):
        return False
    group_worlds = worlds[groups]
    if len(groups) != np.unique(group_worlds).size or len(groups) != np.unique(worlds).size:
        return False
    first_dof, end_dof = starts[groups], starts[groups + 1]
    first_entry, end_entry = mimic_start[groups], mimic_start[groups + 1]
    counts = end_entry - first_entry
    if (
        np.any(first_dof < 0)
        or np.any(end_dof - first_dof != primary)
        or np.any(first_entry < 0)
        or np.any(end_entry > len(mimic_list))
        or np.any(counts < 0)
        or np.any(counts > mimics)
    ):
        return False
    entry_count = min(len(mimic_dof0), len(mimic_dof1))
    if mimic_world is not None:
        entry_count = min(entry_count, len(mimic_world))
    for slot in range(mimics):
        active = counts > slot
        entries = mimic_list[first_entry[active] + slot]
        if np.any(entries < 0) or np.any(entries >= entry_count):
            return False
        if mimic_world is not None and np.any(mimic_world[entries] != group_worlds[active]):
            return False
        # Even disabled entries must be safe if subsequently enabled.
        lower, upper = first_dof[active], end_dof[active]
        if np.any(mimic_dof0[entries] < lower) or np.any(mimic_dof0[entries] >= upper):
            return False
        if np.any(mimic_dof1[entries] < lower) or np.any(mimic_dof1[entries] >= upper):
            return False
    return True


@wp.kernel
def finalize_and_queue(
    prefix: PrefixData,
    queue: QueueData,
    local_rows: int,
    residual_rows: int,
    residual_mf_rows: int,
    primary_dofs: int,
):
    """Fuse the dense-count clamp, the local ownership decision and the two queues."""
    world = wp.tid()
    rows = reserved_count(
        prefix.slot_counter, world, prefix.row_type.shape[1], prefix.first_rejected_slot, queue.status, 0
    )
    queue.count[world] = rows
    mf_count = queue.mf_count[world]
    primary = queue.primary[world]
    secondary = queue.secondary[world]
    residual_secondary = queue.residual_secondary[world]
    single = prefix.bounds[world, 1] == rows
    local_mf = mf_count > 0 and mf_count <= residual_mf_rows and residual_secondary >= 0
    mf_row = int(0)
    while mf_row < mf_count and local_mf:
        body0, body1 = queue.mf_body0[world, mf_row], queue.mf_body1[world, mf_row]
        if body0 >= 0 and queue.body_art[body0] != residual_secondary:
            local_mf = False
        if body1 >= 0 and queue.body_art[body1] != residual_secondary:
            local_mf = False
        if queue.mf_row_type[world, mf_row] == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT:
            local_mf = False
        mf_row += 1
    owner = int(0)
    if rows > 0 and mf_count == 0:
        if single and rows <= wp.min(local_rows, primary_dofs) and primary >= 0:
            owner = 1
        elif not single and rows <= local_rows and secondary >= 0:
            owner = 2
    if owner == 0 and rows > 0 and rows <= residual_rows and residual_secondary >= 0:
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
    support: RowSupport,
    J: wp.array3d[float],
):
    """Populate the current general worlds' rows at this size's articulations' coordinates, zeros included: their
    prefix rows, and zeros the contact rows' producer (:func:`materialize_fallback`) overwrites at their sides
    (:func:`.dense_rows.jacobian_row`; the packed rows here are dense)."""
    worker, lane = wp.tid()
    for entry in range(worker, queue.general_count[0], workers):
        world = queue.general[entry]
        for item in range(group_start[world], group_start[world + 1]):
            art = group_art[item]
            at = jacobian_row(support, art_group[art], art, wp.vec2i(art, -1))
            for coefficient in range(lane, queue.count[world] * size, warp_stride()):
                row, dof = coefficient // size, coefficient % size
                value = float(0.0)
                if art == queue.primary[world] and row < prefix.bounds[world, 0]:
                    for component in range(PREFIX_WIDTH):
                        if prefix.dofs[world, row, component] == dof:
                            value += prefix.weights[world, row, component]
                J[at[0], row, at[1] + dof] = value


@wp.kernel(enable_backward=False)
def produce_contacts(
    workers: int, data: ContactInput, law: PatchRowLaw, packet: ContactPacketData, primary: int, secondary: int
):
    """Cooperatively project current contacts into physical packets after allocation; a lane per row writes its
    records and rhs (:func:`.row_build.write_contact_row`).

    Tangent rows act at the friction location and follow the row law that the compact producer shares
    (``contact_laws.contact_row_frame`` and ``patch_tangent_row``): under persistent patches at the patch
    friction location with the anchor's tangential phi, ``law.beta`` and the undivided coefficient; under point
    friction (beta 0, a disabled view with a zero phi) at the contact's friction points. Every rhs is the row
    law's (``row_laws``), as the general route's, so the rows equal it bitwise. A bouncing normal row sums its
    incident velocity per body instead of per world coordinate. It agrees with the general route's to rounding,
    unless the incident velocity lies within a rounding of a bounce threshold, where the two routes can
    disagree on whether the row bounces.
    """
    worker, _logical_lane = wp.tid()
    lane = warp_lane()
    stride = warp_stride()
    capacity = packet.jacobian.shape[1]
    width = primary + secondary
    total = data.count[0]
    for contact in range(worker, total, workers):
        if data.path[contact] != 0 or data.slot[contact] < 0:
            continue
        world = data.world[contact]
        slot = data.slot[contact]
        rows = data.slots_needed[contact]
        art0 = data.art0[contact]
        art1 = data.art1[contact]
        body0, body1 = contact_bodies(data.shape0[contact], data.shape1[contact], data.shape_body)
        normal = wp.vec3()
        point0 = wp.vec3()
        point1 = wp.vec3()
        tangent0 = wp.vec3()
        tangent1 = wp.vec3()
        if lane == 0:
            normal, point0, point1, _phi = contact_points(
                data.normal[contact],
                data.point0[contact],
                data.point1[contact],
                data.margin0[contact],
                data.margin1[contact],
                body0,
                body1,
                data.body_q,
            )
            tangent0, tangent1 = contact_tangent_basis(normal)
        normal = warp_broadcast(normal)
        point0 = warp_broadcast(point0)
        point1 = warp_broadcast(point1)
        tangent0 = warp_broadcast(tangent0)
        tangent1 = warp_broadcast(tangent1)
        friction0, friction1 = contact_friction_points(point0, point1, data.shared_anchor, data.friction_shared_anchor)
        if law.patches.enabled != 0:
            friction0, friction1 = patch_friction_location(law.patches, contact)
        for coefficient in range(lane, rows * width, stride):
            row = coefficient // width
            coordinate = coefficient % width
            output_row = slot + row
            if output_row < capacity:
                size = primary
                dof = coordinate
                if coordinate >= primary:
                    size = secondary
                    dof -= primary
                direction, p0, p1 = contact_row_frame(
                    row, data.shared_anchor, normal, tangent0, tangent1, point0, point1, friction0, friction1
                )
                value = endpoint_projection(data, body0, art0, size, dof, row, p0, direction)
                value -= endpoint_projection(data, body1, art1, size, dof, row, p1, direction)
                packet.jacobian[world, output_row, coordinate] = value
        warp_sync()
        # A lane per row writes its records and rhs with the row law the owners share.
        for row in range(lane, rows, stride):
            write_contact_row(data, law, contact, row, True)
            if slot + row < capacity:
                packet.row_contact[world, slot + row] = contact * 3 + row
        warp_sync()


@wp.kernel(enable_backward=False)
def materialize_fallback(
    threads: int,
    data: ContactInput,
    law: PatchRowLaw,
    owner: wp.array[int],
    size_a: int,
    size_b: int,
    groups: wp.array[int],
    support: RowSupport,
    J_a: wp.array3d[float],
    J_b: wp.array3d[float],
):
    """Write the general worlds' contact Jacobian with the masked builder's lanes (:mod:`.row_build`): one thread per
    (contact, coordinate) of the larger size, every row at the row law's friction location."""
    tid = wp.tid()
    lanes = wp.max(size_a, size_b)
    for c in range(tid // lanes, data.count[0], threads // lanes):
        if data.path[c] != 0 or data.slot[c] < 0 or owner[data.world[c]] != 0:
            continue
        a_in, b_in = _contact_in_sizes(data, c, size_a, size_b)
        flags = 1 | wp.where(data.slots_needed[c] >= 3, 2, 0) | wp.where(a_in, 4, 0) | wp.where(b_in, 8, 0)
        flags |= wp.where(data.slots_needed[c] > CONTACT_SPIN_ROW, 16, 0)
        populate_contact_J(c, tid % lanes, data, size_a, groups, law, flags, support, J_a, J_b)


TOPOLOGY = (*JOINT_TOPOLOGY, "constraint_mimic_joint0", "constraint_mimic_joint1", "constraint_mimic_world")
"""The structural model arrays the packets' plan reads; enabled flags, coefficients, q and limits stay live."""


def plan(solver):
    """Derive (primary, secondary, mimic capacity) from the topology, or None.

    The primary is the one articulation size that carries joint limits; every
    world has exactly one of it. A second size, if present, is the world's
    secondary articulation. Mimic rows belong to the primary alone.
    """
    sizes = sorted(int(size) for size in solver.forest.size_groups)
    if not 1 <= len(sizes) <= 2 or len(solver.laws.joint_limit_sizes) != 1:
        return None
    primary = int(next(iter(solver.laws.joint_limit_sizes)))
    if primary not in sizes or solver.forest.n_arts_by_size[primary] != solver.forest.world_count:
        return None
    secondary = next((size for size in sizes if size != primary), 0)
    if secondary and solver.forest.n_arts_by_size[secondary] != solver.forest.world_count:
        return None
    if solver.laws.mimic_sizes - {primary} or solver.laws.connect_count:
        return None
    mimics = 0
    if solver.laws._mimic_art_start is not None and solver.laws._mimic_art_list is not None:
        starts = solver.laws._mimic_art_start.numpy()
        mimics = int(np.max(np.diff(starts))) if len(starts) > 1 else 0
    if mimics + 2 * primary > solver._local_residual_max_rows:
        return None
    return primary, secondary, mimics


def topology_supported(solver, primary, mimics):
    """Check all prefix entries, not only currently enabled or valid mimic rows."""
    if any(
        primary not in getattr(solver, name) for name in ("_local_pair_active_counts", "_local_residual_active_counts")
    ):
        return False
    if np.any(solver._local_primary_articulation.numpy() < 0):
        return False
    if solver.laws._mimic_art_start is None or solver.laws._mimic_art_list is None:
        return mimics == 0
    return validate_prefix_topology(
        solver.forest.group_to_art[primary].numpy(),
        solver.forest.art_to_world.numpy(),
        solver.forest.articulation_dof_start.numpy(),
        solver.laws._mimic_art_start.numpy(),
        solver.laws._mimic_art_list.numpy(),
        solver.laws._mimic_dof0.numpy(),
        solver.laws._mimic_dof1.numpy(),
        primary,
        mimics,
        solver.laws._mimic_world.numpy(),
    )


class LocalRowPackets:
    """Own one bounded prefix/current-row boundary; all general rows stay complete."""

    def __init__(self, solver, primary, secondary, mimics):
        self.solver = solver
        self.primary, self.secondary, self.mimics = int(primary), int(secondary), int(mimics)
        self.packets = int(solver._local_residual_max_rows)
        device = solver.model.device
        worlds = solver.forest.world_count
        self.workers = max(1, min(worlds, int(device.sm_count) * 4))
        self.prefix = PrefixData()
        arrays = SolverArrays(None, device)
        self._packets = SimpleNamespace(**arrays.allocate(PACKETS, owner={}, world=worlds, packet=self.packets))
        sizes = {"world": worlds, "packet": self.packets, "coordinate": self.primary + self.secondary}
        self.packet = field_group.bind(
            ContactPacketData, packets=self._packets, jacobian=arrays.array(CONTACT_PACKETS[0], sizes)
        )
        # A model without mimics binds the prefix's mimic tables to placeholders: no mimic per articulation.
        placeholder = solver._arrays.placeholder
        self._absent_mimics = {
            field.name: placeholder(field.dtype) for field in PREFIX if field.owner in ("mimics", "mimic_model")
        }
        self._absent_mimics["mimic_start"] = placeholder(wp.int32, (solver.model.articulation_count + 1,))
        # The capacity status is allocated after tiled-kernel construction; prepare_prefix binds it.
        size = SimpleNamespace(
            pair_count=solver._local_pair_active_counts[self.primary],
            pair=solver._local_pair_active_candidates[self.primary],
            pair_secondary=solver._local_pair_active_secondaries[self.primary],
            residual_count=solver._local_residual_active_counts[self.primary],
            residual=solver._local_residual_active_candidates[self.primary],
            residual_pair=solver._local_residual_active_secondaries[self.primary],
        )
        self.queue = field_group.bind(
            QueueData, solver=solver, placement=solver.placement, forest=solver.forest, size=size, status=None
        )
        self.contact_input = None
        self.law = None
        self._topology = TopologyPin.read(solver.model, TOPOLOGY)
        self._prefix_kernel = get_prefix_kernel(self.primary, self.mimics, self.packets)

    def local_factory(self):
        return get_local_factory(self.primary, self.secondary, self.packets)

    def validate_notification(self, flags: ModelFlags | int = ModelFlags.ALL):
        """Refuse a structural change the packet topology does not survive (:class:`.step_plan.TopologyPin`)."""
        self._topology.validate(
            self.solver.model, flags, lambda: topology_supported(self.solver, self.primary, self.mimics), "Row-packet"
        )

    def prepare_prefix(self, state_in, dt):
        """Bind current coefficients/state and the current double-buffered row owners."""
        solver = self.solver
        # Sticky status is allocated after tiled-kernel construction.
        self.queue.status = solver.placement.constraint_capacity_status
        data = field_group.bind(
            PrefixData,
            self.prefix,
            solver=solver,
            placement=solver.placement,
            forest=solver.forest,
            factor=solver.factor,
            laws=solver.laws,
            options=solver.options,
            model=solver.model,
            mimics=solver.laws,
            mimic_model=solver.model,
            state_in=state_in,
            packets=self._packets,
            group_to_art=solver.forest.group_to_art[self.primary],
            dt=dt,
            **({} if self.mimics else self._absent_mimics),
        )
        self.contact_input = None
        self.law = None
        wp.launch_tiled(
            self._prefix_kernel,
            dim=[(solver.forest.n_arts_by_size[self.primary] + WARPS_PER_BLOCK - 1) // WARPS_PER_BLOCK],
            inputs=[data],
            block_dim=32 * WARPS_PER_BLOCK,
            device=solver.model.device,
        )

    def produce_contacts(self, contact_input):
        """Publish private current J and complete canonical contact metadata/RHS from the step's bound contact input.

        Persistent patch friction writes its tangent rows with the patch row law. The patch link
        runs after this producer; :meth:`finish_rows` writes neither row parents nor coefficients.
        """
        self.contact_input = contact_input
        # The row law binds the patch view, which contact-pool growth replaces, every step.
        self.law = patch_row_law(self.solver)
        wp.launch_tiled(
            produce_contacts,
            dim=[self.workers],
            inputs=[self.workers, self.contact_input, self.law, self.packet, self.primary, self.secondary],
            block_dim=32,
            device=self.solver.model.device,
        )

    def finish_rows(self):
        """Finalize and route current counts, then reconstruct only the general path."""
        solver, queue = self.solver, self.queue
        queue.general_count.zero_()
        queue.pair_count.zero_()
        queue.residual_count.zero_()
        wp.launch(
            finalize_and_queue,
            dim=solver.forest.world_count,
            inputs=[
                self.prefix,
                queue,
                int(solver._local_solve_max_rows),
                int(solver._local_residual_max_rows),
                int(solver._local_residual_mf_max_rows),
                self.primary,
            ],
            device=solver.model.device,
        )
        support = solver.placement.row_support()
        for size in solver.forest.size_groups:
            wp.launch_tiled(
                materialize_general_prefix,
                dim=[self.workers],
                inputs=[
                    self.workers,
                    self.prefix,
                    queue,
                    solver.forest.world_response_group_art_start[size],
                    solver.forest.world_response_group_to_art[size],
                    solver.forest.art_group_idx,
                    size,
                    support,
                    solver.placement.jacobian_rows(size),
                ],
                block_dim=32,
                device=solver.model.device,
            )
        if self.contact_input is not None:
            size_b = self.secondary or self.primary
            wp.launch(
                materialize_fallback,
                dim=self.workers * 32,
                inputs=[self.workers * 32, self.contact_input, self.law, queue.owner, self.primary, size_b],
                outputs=[
                    solver.forest.art_group_idx,
                    support,
                    solver.placement.jacobian_rows(self.primary),
                    solver.placement.jacobian_rows(size_b),
                ],
                device=solver.model.device,
            )


def decline(solver) -> str | None:
    """Why the local packet law does not admit the solver, or ``None`` when it may (its topology decides next).

    The owner writes the solver's friction law: point rows, or persistent patch rows when friction
    anchors are on; the laws it implements are its capability (``route_table.CAPABILITIES``), and the local
    owners' admission carries CUDA, the matrix-free plain sweep and no velocity limits. It produces the rows of the
    local owners' class (case K3); the rest is its fit.
    """
    return first_failed(
        (
            (solver.routes.case == "K3", "Local packets produce the rows of the local owners' class (case K3)."),
            (solver.placement.hinv_jt_writes_world, "Local packets need the response written in world layout."),
            (
                solver.placement.packed_dense,
                "Local packets write their general worlds' rows at their coordinates: the packed rows are not dense.",
            ),
            (
                solver.placement.dense_max_constraints >= solver._local_residual_max_rows,
                "dense_max_constraints is below the local residual owner's rows.",
            ),
            (
                all(
                    not solver._execution_plan.use_tiled_hinv_jt(size)
                    and not solver._execution_plan.use_diagonal_mass(size)
                    for size in solver.forest.size_groups
                ),
                "A size takes the tiled or diagonal response, which local packets do not write.",
            ),
        )
    )


def create_owner(solver):
    """Admit the local packet law, which :func:`decline` admits, for a topology that fits it, else ``None``."""
    layout = plan(solver)
    if layout is None:
        return None
    primary, secondary, mimics = layout
    if not topology_supported(solver, primary, mimics):
        return None
    return LocalRowPackets(solver, primary, secondary, mimics)
