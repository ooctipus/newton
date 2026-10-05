# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compose native populations, mechanical W/C/D storage and one world directory.

Prepare from one-world templates, reserve descriptors, and publish one executable.
Native Data is authoritative; this root owns no dense Newton State mirror.
"""

from __future__ import annotations

import sys
import traceback
import weakref
from contextlib import nullcontext
from dataclasses import dataclass, field, fields, replace
from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    import mujoco_warp

from gpu_components import backing as backing_ops
from gpu_components import directory as directory_ops
from gpu_components import fields as field_ops
from gpu_components import graph as graph_ops
from gpu_components.directory_data import (
    InstanceBatchResult,
    InstanceCommands,
    InstanceCompaction,
    InstanceDirectoryData,
    InstancePhase,
    InstanceResults,
    InstanceStatus,
    InstanceTransaction,
)
from gpu_components.field_data import FieldRetirement, FieldSpec, FieldStorage

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _guard_health(batch: InstanceBatchResult, t: InstanceTransaction, healthy: wp.array[int], lifecycle: wp.array[int]):
    if healthy[0] == 0:
        batch.consumed[0] = 0
        batch.advance_allowed[0] = 0
        batch.status[0] = int(InstanceStatus.PHASE_INVALID)
        t.phase[0] = int(InstancePhase.IDLE)
    lifecycle[0] = wp.int32(batch.consumed[0] != 0 or batch.status[0] != 0)


@wp.kernel
def _guard_backing_publication(status: wp.array[int], healthy: wp.array[int]):
    if status[0] != int(InstanceStatus.OK):
        healthy[0] = 0


@wp.kernel
def _withdraw_native_counts(
    status: wp.array[int], contact_count: wp.array[int], ccd_count: wp.array[int], contacts: int, ccd: int
):
    if status[0] == int(InstanceStatus.OK):
        contact_count[0] = contacts
        ccd_count[0] = ccd


@wp.kernel
def _guard_graph_updates(
    errors: wp.array[int], count: wp.array[int], healthy: wp.array[int], batch: InstanceBatchResult
):
    index = wp.tid()
    if index < count[0] and errors[index] != 0:
        wp.atomic_min(healthy, 0, 0)
        wp.atomic_min(batch.advance_allowed, 0, 0)
        wp.atomic_max(batch.status, 0, int(InstanceStatus.PHASE_INVALID))


@wp.kernel
def _initialization_ids(
    batch: InstanceBatchResult,
    t: InstanceTransaction,
    prototype: int,
    requests: wp.array[int],
    sources: wp.array[int],
    destinations: wp.array[int],
    count: wp.array[int],
):
    ordinal = wp.tid()
    accepted = int(0)
    if batch.consumed[0] != 0 and t.phase[0] == int(InstancePhase.ADMITTED):
        accepted = t.request_starts[prototype + 1] - t.request_starts[prototype]
    if ordinal == 0:
        count[0] = accepted
    if ordinal < accepted:
        request = t.admitted_requests[t.request_starts[prototype] + ordinal]
        requests[ordinal] = request
        sources[ordinal] = 0
        destinations[ordinal] = t.destination_slot[request]


@wp.kernel
def _acknowledge_initialization(
    batch: InstanceBatchResult, t: InstanceTransaction, prototype: int, status: wp.array[int], caller_ack: int
):
    ordinal = wp.tid()
    if batch.consumed[0] == 0 or t.phase[0] != int(InstancePhase.ADMITTED):
        return
    start = t.request_starts[prototype]
    if ordinal < t.request_starts[prototype + 1] - start:
        request = t.admitted_requests[start + ordinal]
        if status[0] != 0:
            t.initialized_sequence[request] = wp.uint64(0)
        elif caller_ack == 0 and t.status[request] == int(InstanceStatus.OK):
            t.initialized_sequence[request] = batch.sequence[0]


@wp.kernel
def _relocation_count(
    t: InstanceTransaction, moves: InstanceCompaction, prototype: int, healthy: wp.array[int], count: wp.array[int]
):
    count[0] = 0
    if healthy[0] != 0 and t.phase[0] == int(InstancePhase.MOVING):
        count[0] = moves.count[prototype]


@wp.kernel
def _acknowledge_moves(
    t: InstanceTransaction, moves: InstanceCompaction, prototype: int, healthy: wp.array[int], status: wp.array[int]
):
    if healthy[0] != 0 and t.phase[0] == int(InstancePhase.MOVING) and status[0] == 0:
        moves.copied_count[prototype] = moves.count[prototype]


@wp.kernel
def _execution_conditions(
    d: InstanceDirectoryData,
    batch: InstanceBatchResult,
    prototype: int,
    world_ready: wp.array[int],
    contact_ready: wp.array[int],
    transient_world_ready: wp.array[int],
    transient_contact_ready: wp.array[int],
    transient_ccd_ready: wp.array[int],
    contact_count: wp.array[int],
    ccd_count: wp.array[int],
    contact_quota: int,
    ccd_quota: int,
    healthy: wp.array[int],
    permit: wp.array[int],
    step_condition: wp.array[int],
    poses: wp.array[int],
):
    count = d.live_count[prototype]
    ready = wp.int32(
        healthy[0] != 0
        and batch.advance_allowed[0] != 0
        and count > 0
        and count <= world_ready[0]
        and count * contact_quota <= contact_ready[0]
        and count <= transient_world_ready[0]
        and count * contact_quota <= transient_contact_ready[0]
        and count * ccd_quota <= transient_ccd_ready[0]
        and contact_count[0] <= wp.min(contact_ready[0], transient_contact_ready[0])
        and ccd_count[0] <= transient_ccd_ready[0]
    )
    step_condition[0] = wp.int32(ready != 0 and permit[0] != 0)
    poses[0] = ready


def _validate_prototype(m, d):
    """Admit the numerical feature subset qualified for this changing-world program."""
    import mujoco_warp as mjw

    required = (
        m.opt.solver == mjw.SolverType.NEWTON,
        m.opt.integrator == mjw.IntegratorType.IMPLICITFAST,
        m.opt.cone == mjw.ConeType.PYRAMIDAL,
        m.opt.broadphase == mjw.BroadphaseType.NXN,
        bool(m.opt.enableflags & mjw.EnableBit.SLEEP),
        not bool(m.opt.enableflags & mjw.EnableBit.ENERGY),
        not bool(m.opt.disableflags & mjw.DisableBit.ISLAND),
        bool(m.opt.disableflags & mjw.DisableBit.MULTICCD),
        m.opt.run_collision_detection,
        m.opt.graph_conditional,
        not m.has_sdf_geom,
        not m.has_fluid,
    )
    absent = (m.nflex, m.ntendon, m.nsensor, m.ncam, m.nlight, m.neq, m.nacttrnbody, m.nhfield, m.na, m.nhistory)
    if not all(required) or any(absent) or any(getattr(m.callback, f.name) is not None for f in fields(m.callback)):
        raise NotImplementedError("Changing worlds supports native NxN sleeping Newton/implicit-fast rigid models")
    if m.ntree == 0:
        raise NotImplementedError("Changing worlds requires at least one dynamic tree")
    derivative_flags = mjw.DisableBit.ACTUATION | mjw.DisableBit.SPRING | mjw.DisableBit.DAMPER
    if m.body_freeadr.size and (m.opt.disableflags & derivative_flags) != derivative_flags:
        raise NotImplementedError("Changing worlds does not admit implicit-fast free-body solves")
    if not (m.opt.disableflags & mjw.DisableBit.CONSTRAINT):
        if m.jnt_limited_ball_adr.size and not (m.opt.disableflags & mjw.DisableBit.LIMIT):
            raise NotImplementedError("Changing worlds does not admit enabled ball-joint limits")
        if m.flg_surfacevel and not (m.opt.disableflags & mjw.DisableBit.CONTACT):
            raise NotImplementedError("Changing worlds does not admit contact surface velocity")
    if m.opt.run_rne_postconstraint:
        raise NotImplementedError("Changing worlds does not admit postconstraint inverse dynamics")
    passive_flags = mjw.DisableBit.SPRING | mjw.DisableBit.DAMPER
    if (
        m.flg_adhesion
        and m.nv > 0
        and not (m.opt.disableflags & mjw.DisableBit.CONTACT)
        and (m.opt.disableflags & passive_flags) != passive_flags
    ):
        raise NotImplementedError("Changing worlds does not admit passive adhesion")
    if not m.is_sparse and d.nvmax_pad > 50:
        raise NotImplementedError("Changing worlds does not admit dense full Jacobians above 50 padded DOFs")
    if (m.opt.disableflags & derivative_flags) != derivative_flags and (
        any(tile.elemid.size for tile in m.M_tiles) or d.qLD.shape[1] > m.qLD_block_total
    ):
        raise NotImplementedError("Changing worlds does not admit gathered/sparse implicit inertia factorizations")
    mjw.validate_data_layout(m, d)


def _initialize_ready_fields(group, storage, *, start=0):
    """Initialize newly ready native fields using the numerical owner's defaults."""
    for name, value in group.data_defaults.items():
        if name in storage.arrays:
            field_ops.fill(storage, storage.arrays[name], value, count=storage.ready_count, start=start)


def _storage_domains(group):
    """Pair actual state and transient owners with their canonical W/C/D domain."""
    return tuple(
        (owner, domain)
        for owners in ((group.world_storage, group.contact_storage), group.transient_storages)
        for domain, owner in enumerate(owners)
        if owner is not None
    )


def _publish_native_counts(group):
    """Publish one admitted candidate and CCD prefix after every owner is ready."""
    for domain, count in ((1, group.contact_count), (2, group.ccd_count)):
        count.fill_(
            min(
                (owner.ready_rows for owner, index in _storage_domains(group) if index == domain),
                default=wp.upper_bound(group.count_parameters[domain]) if len(group.transient_storages) == 3 else 0,
            )
        )


def _array_signature(array):
    """Describe one borrowed array consistently across public and private relations."""
    return id(array), array.ptr, array.shape, array.strides, array.dtype, array.capacity, array.device


def _population_count_signature(group):
    """Freeze the borrowed count descriptors without visiting private scratch."""
    return (
        _array_signature(group.world_storage.protected_count),
        _array_signature(group.world_storage.ready_count),
        _array_signature(group.contact_count),
        _array_signature(group.ccd_count),
    )


def _binding_signature(group):
    """Freeze this composition's exact storage, parameter and updater relations."""
    return (
        _population_count_signature(group),
        (
            tuple(
                (
                    id(owner),
                    owner.capacity,
                    id(owner.device),
                    id(owner.protected_count),
                    id(owner.ready_count),
                    _array_signature(owner.ready_count) if owner is not group.world_storage else (),
                    tuple((name, _array_signature(array)) for name, array in owner.arrays.items())
                    if any(owner is scratch for scratch in group.transient_storages)
                    else (),
                )
                for owner, _ in _storage_domains(group)
            ),
            tuple(_array_signature(array) for array in group.transient_arrays),
            tuple(id(parameter) for parameter in group.count_parameters),
        ),
        id(group.updates),
        id(group.bindings),
    )


def _validate_population(group):
    """Check native descriptors and this root's frozen storage relation before emission."""
    import mujoco_warp as mjw

    if group.recording_failed:
        raise RuntimeError("The population program has a failed recording")
    signature = _binding_signature(group)
    if signature[0:2] != group.storage_signature:
        raise ValueError("Prepared storage or count descriptors changed")
    if group.binding_signature is not None and signature != group.binding_signature:
        raise ValueError("Prepared storage, count or graph bindings changed")
    for owner, _ in _storage_domains(group):
        if owner.closed or owner.service_failed:
            raise RuntimeError("Native population storage is closed or quarantined")
    if mjw.metadata_signature((group.model, group.data, group.execution_data)) != group.metadata_signature:
        raise ValueError("Prepared native descriptors changed")
    mjw.validate_data_layout(group.model, group.data)
    if group.view is not None:
        mujoco_world_population_validate(group.view)


def _bind_native_program(group, graph, launches):
    """Bind one population's exact native operations to its admitted count sources."""
    _validate_population(group)
    storages = tuple(owner for owner, _ in _storage_domains(group))
    count_sources = tuple(
        zip(
            group.count_parameters,
            (
                group.world_storage.protected_count,
                group.contact_count,
                group.ccd_count,
            ),
            strict=True,
        )
    )
    launch_ids = {id(record) for record in launches}
    operations = tuple(op for op in wp.capture_get_memory_operations(graph) if id(op.launch) in launch_ids)
    field_ops.validate_memory_operations(
        graph,
        operations,
        storages,
        count_sources=count_sources,
        fixed_arrays=tuple(array for array in (*group.global_arrays.values(), *group.transient_arrays) if array.size),
    )
    fixed = tuple(
        i for i, record in enumerate(launches) if not record.extent_parameters and not record.scalar_parameters
    )
    group.bindings.extend(
        graph_ops.adopt_launches(
            group.updates,
            graph,
            launches,
            count_sources=count_sources,
            fixed=fixed,
        )
    )


@dataclass(eq=False)
class MuJoCoWorldPopulation:
    """Borrow native arrays and numeric capacities for one homogeneous population.

    .. experimental::
        Passive borrowed descriptors. Call :func:`mujoco_world_population_validate`
        before preparing work with them; direct array access is scoped to the
        population lifetime. Live count includes sleeping worlds. Readiness is
        separate from membership and may be withdrawn after a service failure.

    Contact records belong to the last completed substep at the current placement.
    A consumed lifecycle batch invalidates them before stepping. Read forces in
    the after-substep callback. Callers must not replace these borrowed fields.
    """

    prototype_index: int
    """Index in the immutable prepared prototype sequence."""
    model: mujoco_warp.Model
    """Borrowed native Model containing this prototype's constants."""
    data: mujoco_warp.Data
    """Borrowed native Data; descriptors stay fixed during the population lifetime."""
    world_capacity: int
    """Reserved world slots, independently of live membership and readiness."""
    contact_capacity: int
    """Reserved candidate rows."""
    ccd_capacity: int
    """Reserved continuous-collision rows."""
    world_live_count: wp.array[wp.int32]
    """Device scalar counting live worlds, including sleeping worlds."""
    world_storage_ready_count: wp.array[wp.int32]
    """Device scalar admitting the accessible world prefix."""
    contact_storage_ready_count: wp.array[wp.int32]
    """Device scalar admitting the accessible candidate prefix."""
    ccd_storage_ready_count: wp.array[wp.int32]
    """Device scalar admitting the accessible continuous-collision prefix."""
    _owner: object = field(repr=False)


@dataclass(eq=False)
class _MuJoCoWorldPopulation:
    """One topology's immutable model and authoritative native fields/program."""

    model: mujoco_warp.Model
    prototype_index: int = 0
    view: MuJoCoWorldPopulation | None = None
    data: object = None
    default_storage: FieldStorage | None = None
    world_storage: FieldStorage | None = None
    contact_storage: FieldStorage | None = None
    transient_storages: tuple = ()
    transient_arrays: tuple = ()
    contact_count: wp.array | None = None
    ccd_count: wp.array | None = None
    contact_quota: int = 0
    ccd_quota: int = 0
    execution_data: object = None
    count_parameters: tuple = ()
    metadata_signature: tuple | None = None
    binding_signature: tuple | None = None
    storage_signature: tuple | None = None
    recording_failed: bool = False
    substeps: int = 1
    before_step: object = None
    after_substep: object = None
    refresh_kinematics: bool = True
    initialization_transfer: object = None
    compaction_transfer: object = None
    updates: object = None
    bindings: list = field(default_factory=list)
    data_defaults: dict = field(default_factory=dict)
    application_ranges: list[tuple[int, int]] | None = None
    request_indices: object = None
    source_rows: object = None
    destination_rows: object = None
    initialization_count: object = None
    move_source_rows: object = None
    move_destination_rows: object = None
    move_count: object = None
    step_condition: object = None
    kinematics_condition: object = None
    global_arrays: dict = field(default_factory=dict)
    empty_fields: dict = field(default_factory=dict)
    retirements: tuple[FieldRetirement, ...] = ()


@dataclass(eq=False)
class MuJoCoWorlds:
    """Prepared native populations and resources; operations are free functions.

    .. experimental::
        Allocate with :func:`mujoco_worlds_prepare`; record with
        :func:`mujoco_worlds_capture`; close after retiring every graph borrower.
        Directory and batch diagnostics remain readable after closure. Raw
        population descriptors do not confer submission or storage access rights.
    """

    device: object = None
    """CUDA device shared by the prepared native prototypes."""
    directory: InstanceDirectoryData | None = None
    """Borrowed numeric identity and placement relations; mutations use the captured lifecycle."""
    batch_result: InstanceBatchResult | None = None
    """Borrowed diagnostics for the last submitted lifecycle batch."""
    populations: tuple[MuJoCoWorldPopulation, ...] = ()
    """Borrowed native populations in prototype order."""
    _backing: object = None
    _directory: object = None
    _populations: list = field(default_factory=list)
    _healthy: wp.array | None = None
    _always_permit: wp.array | None = None
    _lifecycle_needed: wp.array | None = None
    _closed: bool = False
    _service_failed: bool = False
    _capture_attempted: bool = False
    _graph: object = None
    _discovery_memory: dict | None = None
    _recording_passes: int = 0
    _retirement_pending: bool = False
    _retirement_publication: object = None
    _retirement_streams: tuple = ()


def mujoco_worlds_validate(worlds: MuJoCoWorlds) -> None:
    """Reject operation on a closed or quarantined experimental population owner."""
    if worlds._directory is None or worlds._closed or worlds._service_failed:
        raise RuntimeError("Native population is closed or its backing service failed")


def mujoco_world_population_validate(population: MuJoCoWorldPopulation) -> None:
    """Validate experimental borrowed descriptors before preparing their consumers."""
    owner = population._owner()
    if owner is None or owner.data is None:
        raise RuntimeError("The prototype's population is closed")
    if owner.view is not population or owner.data is not population.data or owner.model is not population.model:
        raise ValueError("Borrowed native population descriptors changed")
    if population.prototype_index != owner.prototype_index or type(population.prototype_index) is not int:
        raise ValueError("Borrowed native prototype identity changed")
    if (
        type(population.world_capacity) is not int
        or population.world_capacity != owner.world_storage.capacity
        or type(population.contact_capacity) is not int
        or population.contact_capacity != owner.contact_storage.capacity
        or type(population.ccd_capacity) is not int
        or population.ccd_capacity != wp.upper_bound(owner.count_parameters[2])
    ):
        raise ValueError("Borrowed native capacity changed")
    if (
        population.world_live_count is not owner.world_storage.protected_count
        or population.world_storage_ready_count is not owner.world_storage.ready_count
        or population.contact_storage_ready_count is not owner.contact_count
        or population.ccd_storage_ready_count is not owner.ccd_count
    ):
        raise ValueError("Borrowed native count source changed")
    if owner.storage_signature is not None and _population_count_signature(owner) != owner.storage_signature[0]:
        raise ValueError("Prepared native count descriptors changed")


def _world_ready_capacity(group):
    if len(group.transient_storages) != 3:
        return 0
    quotas = (1, group.contact_quota, group.ccd_quota)
    rows = [owner.ready_rows for owner, _ in _storage_domains(group)]
    for index, retirement in enumerate(group.retirements):
        if retirement.pending:
            rows[index] = min(rows[index], retirement.requested_rows)
    return min(
        (count // quotas[domain] for count, (_, domain) in zip(rows, _storage_domains(group), strict=True)), default=0
    )


def mujoco_world_population_ready_capacity(population: MuJoCoWorldPopulation) -> int:
    """Return the jointly published state, contact, CCD and transient prefix in world units.

    Follow backing-service stream ordering before using the reported capacity.
    Pending withdrawals conservatively limit the result to their requested prefix.
    This host query certifies neither GPU acceptance nor publication completion.
    """
    mujoco_world_population_validate(population)
    return _world_ready_capacity(population._owner())


def _record_physics(group, native_regions):
    """Record conditional native physics and poses on the current stream."""
    import mujoco_warp as mjw

    if group.recording_failed:
        raise RuntimeError("The population program has a failed recording")
    _validate_population(group)

    def record_callback(callback):
        if callback is None:
            return
        graph = graph_ops.current_capture(device=group.data.qpos.device)
        start = wp.capture_launch_count(graph)
        callback(group.view)
        end = wp.capture_launch_count(graph)
        if end != start:
            group.application_ranges.append((start, end))

    def step():
        record_callback(group.before_step)
        for _ in range(group.substeps):
            _validate_population(group)
            region = wp.capture_transient(
                lambda: mjw.step(group.model, group.execution_data),
                assume_nonescaping=True,
                assume_no_indirect_access=True,
            )
            if region is not None:
                native_regions.append(region)
            record_callback(group.after_substep)
        _validate_population(group)

    def step_and_poses():
        wp.capture_if(group.step_condition, on_true=step)
        mjw.kinematics(group.model, group.execution_data)

    try:
        if group.refresh_kinematics:
            wp.capture_if(group.kinematics_condition, step_and_poses)
        else:
            wp.capture_if(group.step_condition, step)
        if group.recording_failed:
            raise RuntimeError("The population program has a failed recording")
    except BaseException:
        group.recording_failed = True
        group.before_step = group.after_substep = None
        raise


def _prepare_transient_storage(worlds, graph, native_regions):
    """Pack canonical numerical requests using exact domain identities and native ordering."""
    requests = wp.capture_get_allocations(graph)
    regions = wp.capture_get_transient_regions(graph)
    owners = {id(region): index for index, recorded in enumerate(native_regions) for region in recorded}
    classified = [[] for _ in worlds._populations]
    bindings = [None] * len(requests)
    for request in requests:
        if request.transient_region is None or id(regions[request.transient_region]) not in owners:
            raise ValueError("Transient allocations must belong to an exact recorded native region")
        if request.strides is not None:
            raise ValueError("Native transient allocations require implicit strides")
        index = owners[id(regions[request.transient_region])]
        group = worlds._populations[index]
        shape = tuple(wp.upper_bound(axis) for axis in request.shape)
        if not all(shape):
            array = wp.empty(shape, dtype=request.dtype, device=worlds.device)
            group.transient_arrays += (array,)
            bindings[request.index] = (request, array)
            continue
        dynamic = tuple(i for i, axis in enumerate(request.shape) if isinstance(axis, wp.CountParameter))
        domain = next(
            (
                i
                for i, parameter in enumerate(group.count_parameters)
                if request.shape and request.shape[0] is parameter
            ),
            None,
        )
        if dynamic and (dynamic != (0,) or domain is None):
            raise ValueError("Native transient dimensions must use the population's exact leading W/C/D identity")
        classified[index].append((request, domain))
    pairs = []
    for allocations in classified:
        for index, (right, domain) in enumerate(allocations):
            shape = right.shape if domain is None else right.shape[1:]
            pairs.extend(
                (left, right)
                for left, left_domain in allocations[:index]
                if left_domain == domain
                and left.dtype == right.dtype
                and (left.shape if domain is None else left.shape[1:]) == shape
            )
    ordering = dict(
        zip(
            ((left.index, right.index) for left, right in pairs), wp.capture_allocation_order(graph, pairs), strict=True
        )
    )
    for group, allocations in zip(worlds._populations, classified, strict=True):
        initial_worlds = min(group.world_storage.ready_rows, group.contact_storage.ready_rows // group.contact_quota)
        quotas = (1, group.contact_quota, group.ccd_quota)
        for domain in (0, 1, 2, None):
            slots, occurrences = [], []
            for request, allocated_domain in allocations:
                if allocated_domain != domain:
                    continue
                slot = len(slots)
                for index, previous in enumerate(slots):
                    if ordering.get((previous.index, request.index), False):
                        slot = index
                        break
                if slot == len(slots):
                    slots.append(request)
                else:
                    slots[slot] = request
                occurrences.append((request, slot))
            if domain is None:
                arrays = tuple(wp.empty(request.shape, dtype=request.dtype, device=worlds.device) for request in slots)
                group.transient_arrays += arrays
            else:
                if not slots:
                    group.transient_storages += (None,)
                    continue
                backed = worlds._backing is not None
                storage = field_ops.allocate(
                    wp.upper_bound(group.count_parameters[domain]),
                    (group.world_storage.protected_count, group.contact_count, group.ccd_count)[domain],
                    fields=tuple(
                        FieldSpec(str(index), request.shape[1:], request.dtype, alignment_bytes=16)
                        for index, request in enumerate(slots)
                    ),
                    backing=worlds._backing if backed else None,
                    initial_ready_count=initial_worlds * quotas[domain] if backed else None,
                )
                group.transient_storages += (storage,)
                arrays = tuple(storage.arrays.values())
            for request, slot in occurrences:
                bindings[request.index] = (request, arrays[slot].view(request.dtype))
        if worlds._backing is not None:
            group.retirements = tuple(field_ops.prepare_retirement(owner) for owner, _ in _storage_domains(group))
        _publish_native_counts(group)
        group.storage_signature = _binding_signature(group)[0:2]
        group.binding_signature = _binding_signature(group)
    return tuple(bindings)


def mujoco_worlds_prepare(
    prepared,
    *,
    world_capacities,
    id_capacity,
    command_capacity,
    contact_capacities=None,
    ccd_capacities=None,
    memory_budget_bytes=None,
    initial_world_ready_capacities=None,
) -> MuJoCoWorlds:
    """Prepare immutable prototypes, native Data and changing world populations.

    Native Data is authoritative. Capacities reserve row limits; an optional
    shared physical budget controls backing. The caller warms each numerical
    program on separate one-world Data before preparation.

    .. experimental::
        This operation and the ``mujoco_worlds_*`` / ``mujoco_world_population_*``
        family may change without the normal deprecation period. The supported
        native feature subset is validated before allocating resources.

    Args:
        prepared: Immutable native ``(Model, one-world Data)`` prototype pairs.
        world_capacities: Reserved world slots per prototype.
        id_capacity: Maximum reusable logical instance identities.
        command_capacity: Maximum requests in one lifecycle batch.
        contact_capacities: Reserved candidate rows, or template quota times world capacity.
        ccd_capacities: Reserved continuous-collision rows, or the corresponding template quota.
        memory_budget_bytes: Shared physical-backing limit in bytes; None selects fixed allocations.
        initial_world_ready_capacities: Initially backed world prefixes; defaults to all reserved slots.

    Returns:
        Passive resource record. Capture once, submit its graph, then retire the
        graph and call :func:`mujoco_worlds_close` with every consumer stream.
    """
    worlds = MuJoCoWorlds()
    if sys.version_info < (3, 11):
        raise RuntimeError("Experimental MuJoCoWorlds requires Python 3.11 or newer")
    import mujoco_warp as mjw

    prepared, world_capacities = tuple(prepared), tuple(world_capacities)
    if not prepared or len(prepared) != len(world_capacities):
        raise ValueError("One virtual world capacity is required per native prototype")
    if any(type(n) is not int or not 1 <= n < 2**31 for n in world_capacities):
        raise ValueError("World capacities must be positive int32 limits")
    contact_capacities = (
        tuple(d.naconmax * n for (_, d), n in zip(prepared, world_capacities, strict=True))
        if contact_capacities is None
        else tuple(contact_capacities)
    )
    ccd_capacities = (
        tuple(d.naccdmax * n for (_, d), n in zip(prepared, world_capacities, strict=True))
        if ccd_capacities is None
        else tuple(ccd_capacities)
    )
    initial_world_ready_capacities = (
        world_capacities if initial_world_ready_capacities is None else tuple(initial_world_ready_capacities)
    )
    if any(
        len(values) != len(prepared) for values in (initial_world_ready_capacities, contact_capacities, ccd_capacities)
    ):
        raise ValueError("Each prototype needs complete W/C/D capacities and initial ready world counts")
    device = prepared[0][1].qpos.device
    for (model, template), capacity, initial, contact_capacity, ccd_capacity in zip(
        prepared, world_capacities, initial_world_ready_capacities, contact_capacities, ccd_capacities, strict=True
    ):
        if template.nworld != 1 or template.qpos.device != device or model.qpos0.device != device:
            raise ValueError("Native defaults must have one world on the model/population device")
        for name, array, shape in mjw.array_fields(model):
            if shape and shape[0] == "*" and array.size and array.shape[0] != 1:
                raise ValueError(f"Prototype model parameters require one broadcast row: {name}")
        if type(initial) is not int or not 0 <= initial <= capacity:
            raise ValueError("Initial world rows exceed capacity")
        if (
            any(type(n) is not int or not 1 <= n < 2**31 for n in (contact_capacity, ccd_capacity))
            or template.naconmax < 1
            or template.naccdmax < 1
            or contact_capacity < initial * template.naconmax
            or ccd_capacity < initial * template.naccdmax
        ):
            raise ValueError("Contact/CCD capacities must support the initial world population")
    if not device.is_cuda:
        raise ValueError("MuJoCoWorlds currently requires a CUDA device")
    worlds.device, worlds._backing, worlds._directory = device, None, None
    worlds._populations, worlds.populations = [], ()
    worlds._healthy = worlds._always_permit = worlds._lifecycle_needed = None
    worlds._closed, worlds._service_failed, worlds._capture_attempted = False, False, False
    worlds._graph = None
    try:
        with wp.ScopedDevice(device):
            if memory_budget_bytes is not None:
                worlds._backing = backing_ops.prepare(
                    memory_budget_bytes, device_ordinal=device.ordinal, expected_uuid=device.uuid
                )
            backing = worlds._backing
            if backing is not None:
                worlds._retirement_publication = wp.Event(device)
            worlds._directory = directory_ops.allocate(
                world_capacities, id_capacity=id_capacity, command_capacity=command_capacity, device=device
            )
            worlds.directory = worlds._directory.data
            worlds.batch_result = worlds._directory.batch_result
            worlds._healthy = wp.ones(1, dtype=int, device=device)
            worlds._always_permit = wp.ones(1, dtype=int, device=device)
            worlds._lifecycle_needed = wp.zeros(1, dtype=int, device=device)
            start = 0
            for prototype, ((model, template), capacity, initial, contact_cap, ccd_cap) in enumerate(
                zip(
                    prepared,
                    world_capacities,
                    initial_world_ready_capacities,
                    contact_capacities,
                    ccd_capacities,
                    strict=True,
                )
            ):
                group = _MuJoCoWorldPopulation(
                    model, prototype_index=prototype, contact_quota=template.naconmax, ccd_quota=template.naccdmax
                )
                worlds._populations.append(group)
                _validate_prototype(model, template)
                counts = tuple(wp.CountParameter(n) for n in (capacity, contact_cap, ccd_cap))
                group.count_parameters = counts
                arrays, world_fields, contact_fields = {}, [], []
                world_sources = {}
                data_alignments = mjw.data_field_alignments(model, template)
                for name, array, shape in mjw.array_fields(template):
                    axis = shape[0] if shape else None
                    if axis in ("nworld", "naconmax"):
                        if array.shape[0] == 0 and not array.size:
                            arrays[name] = array
                            group.empty_fields[name] = {"shape": list(array.shape), "dtype": str(array.dtype)}
                            continue
                        if axis == "nworld":
                            if array.shape[0] != 1:
                                raise ValueError(f"Default world field does not have one row: {name}")
                            world_sources[name] = array
                        destination = world_fields if axis == "nworld" else contact_fields
                        destination.append(
                            FieldSpec(
                                name,
                                array.shape[1:],
                                array.dtype,
                                packed=axis == "naconmax" or array.ndim > 1,
                                alignment_bytes=data_alignments.get(name),
                            )
                        )
                    else:
                        arrays[name] = wp.clone(array)
                        group.global_arrays["data." + name] = arrays[name]
                data_field_names = tuple(spec.name for spec in world_fields)
                group.default_storage = field_ops.allocate(
                    1, wp.ones(1, dtype=int, device=device), fields=tuple(world_fields)
                )
                for name, array in world_sources.items():
                    field_ops.copy(
                        group.default_storage,
                        group.default_storage.arrays[name],
                        array,
                        count=group.default_storage.protected_count,
                    )
                live_count = worlds._directory.data.live_count[prototype : prototype + 1]
                group.world_storage = field_ops.allocate(
                    capacity, live_count, fields=tuple(world_fields), backing=backing, initial_ready_count=initial
                )
                group.contact_count = wp.zeros(1, dtype=int, device=device)
                group.ccd_count = wp.zeros(1, dtype=int, device=device)
                group.contact_storage = field_ops.allocate(
                    contact_cap,
                    group.contact_count,
                    fields=tuple(contact_fields),
                    backing=backing,
                    initial_ready_count=initial * group.contact_quota if backing else None,
                )
                for owner in (group.world_storage, group.contact_storage):
                    arrays.update(owner.arrays)
                _publish_native_counts(group)
                group.data = replace(
                    mjw.replace_arrays(template, arrays), nworld=capacity, naconmax=contact_cap, naccdmax=ccd_cap
                )
                group.execution_data = replace(group.data, nworld=counts[0], naconmax=counts[1], naccdmax=counts[2])
                group.data_defaults = mjw.data_field_defaults(model, group.data)
                for owner in (group.contact_storage,):
                    _initialize_ready_fields(group, owner)
                mjw.invalidate_contact_cache(group.data)
                group.metadata_signature = mjw.metadata_signature((group.model, group.data, group.execution_data))
                group.initialization_transfer = field_ops.prepare_transfer(
                    group.default_storage, group.world_storage, data_field_names
                )
                group.compaction_transfer = field_ops.prepare_transfer(
                    group.world_storage, group.world_storage, data_field_names
                )
                group.request_indices = wp.zeros(command_capacity, dtype=int, device=device)
                group.source_rows = wp.zeros(command_capacity, dtype=int, device=device)
                group.destination_rows = wp.zeros(command_capacity, dtype=int, device=device)
                group.initialization_count = wp.zeros(1, dtype=int, device=device)
                end = start + capacity
                group.move_source_rows = worlds._directory.compaction.source_slots[start:end]
                group.move_destination_rows = worlds._directory.compaction.destination_slots[start:end]
                group.move_count = wp.zeros(1, dtype=int, device=device)
                group.step_condition = wp.zeros(1, dtype=int, device=device)
                group.kinematics_condition = wp.zeros(1, dtype=int, device=device)
                group.view = MuJoCoWorldPopulation(
                    prototype,
                    group.model,
                    group.data,
                    capacity,
                    contact_cap,
                    ccd_cap,
                    group.world_storage.protected_count,
                    group.world_storage.ready_count,
                    group.contact_count,
                    group.ccd_count,
                    weakref.ref(group),
                )
                group.storage_signature = _binding_signature(group)[0:2]
                start = end
            worlds.populations = tuple(group.view for group in worlds._populations)
            directory_ops.publish_admissible_slots(
                worlds._directory, tuple(_world_ready_capacity(group) for group in worlds._populations)
            )
    except BaseException as failure:
        traceback.clear_frames(failure.__traceback__)
        arrays = world_sources = array = owner = None
        try:
            mujoco_worlds_close(worlds, streams=(wp.get_stream(device),))
        except BaseException as cleanup:
            raise BaseExceptionGroup(
                "Native population construction and cleanup failed", [failure, cleanup]
            ) from failure
        raise
    return worlds


def mujoco_worlds_capture(
    worlds,
    commands: InstanceCommands,
    results: InstanceResults,
    *,
    permit=None,
    validate=None,
    initialize=None,
    retain=(),
    substeps=1,
    before_step=None,
    after_substep=None,
    application_bindings=None,
    refresh_kinematics=True,
):
    """Discover bounded transient allocations, then record one retained executable.

    Native Data is allocated during preparation. This operation
    records a non-executable discovery pass, packs ordinary native temporaries into
    exact W/C/D domains, retains fixed arrays, and records the executable with supplied arrays.
    Warp owns discovery addresses; no maximum temporary payload is executed or
    physically committed. Each callback runs twice and must reproduce its operation
    structure without host side effects. Callback-owned allocations are unsupported.
    Callbacks receive supported :class:`MuJoCoWorldPopulation` views.
    Optional preparation callbacks record payload validation before admission
    and payload writes after the default Data copy. The initializer explicitly
    acknowledges successful requests in the supplied initialized_sequence array; missing
    acknowledgements or failed base copies cannot publish a replacement.
    Recording callbacks must use prepared storage and record allocation-free
    operations. Validation and initialization record transaction work, not
    per-frame side effects. Healthy repeated sequences skip that work and
    transfers; new batches and existing batch-error paths retain the complete
    lifecycle. Initialization/move counts and acknowledgements describe their
    transaction stage, not current-frame activity.

    Callers may observe this population's directory, but must submit mutations
    through this captured lifecycle and the backing-service operations.
    Independently invoking the owned directory's mutating methods bypasses native state ownership.

    Args:
        commands: Caller-owned lifecycle buffers with increasing batch sequence.
        results: Caller-owned per-request results.
        permit: Optional GPU int32 scalar permitting physics advancement.
        validate: Preparation callback ``(commands, request_status, consumed)``.
            Guard the read-only consumed scalar before reading requests. Only
            replace existing int(InstanceStatus.OK) request_status values with payload rejection.
        initialize: Preparation callback ``(population, request_indices, destination_rows,
            count, copy_status, initialized_sequence, sequence)``. Write only the
            admitted prefix when copy_status is int(InstanceStatus.OK); acknowledge initialized_sequence
            only after complete payload writes. Other arrays are borrowed read-only.
        retain: External buffers borrowed by any recorded callback kernels.
        substeps: Ordered native steps per graph replay.
        before_step: Callback ``(population)`` recording controls once before ordered native steps.
        after_substep: Callback ``(population)`` recording contact consumers after each native step.
        application_bindings: Preparation operation ``(population, launches)`` returning
            ``(extents, parameters, fixed)`` for :func:`gpu_components.graph.adopt_launches`.
            Launches are the exact records emitted by this population's callbacks, in recording
            order. Check expected kernel identities and declare every local record index;
            repeated kernels may use different count sources. Required for nonempty callback
            work. The operation runs after capture and is never retained by the graph.
            Application and lifecycle callback fills/copies are currently rejected: their
            storage admission is not established by numerical launch-count declarations.
        refresh_kinematics: Refresh body, geometry and site transforms after valid
            reset-only frames or after the final substep, independently of permit.

    Returns:
        One bound Warp graph. The caller owns submissions and must destroy this
        graph before closing the population. Any failed request suppresses
        both physical advancement and pose refresh.
    """

    import mujoco_warp as mjw

    with wp.ScopedDevice(worlds.device):
        mujoco_worlds_validate(worlds)
        if worlds._retirement_pending:
            raise RuntimeError("Resolve pending backing retirement before capturing")
        if worlds._capture_attempted:
            raise RuntimeError("Each native population prepares exactly one graph")
        if any(
            callback is not None and not callable(callback)
            for callback in (validate, initialize, before_step, after_substep, application_bindings)
        ):
            raise TypeError("Payload recording callbacks must be callable")
        if type(substeps) is not int or not 1 <= substeps < 2**31 // 512:
            raise ValueError("Native substeps must be a positive int32-bounded integer")
        if type(refresh_kinematics) is not bool:
            raise TypeError("refresh_kinematics must be a boolean")
        retain = tuple(retain)
        directory_ops.validate_buffers(worlds._directory, commands, results)
        permit = worlds._always_permit if permit is None else permit
        if (
            not isinstance(permit, wp.array)
            or permit.shape != (1,)
            or permit.dtype != wp.int32
            or permit.device != worlds.device
            or not permit.is_contiguous
        ):
            raise ValueError("Native step permit must be one contiguous int32 scalar on the population device")
        worlds._capture_attempted = True
        graph = discovery = None
        allocation_bindings = None
        application_ranges = [[] for _ in worlds._populations]
        native_regions = [[] for _ in worlds._populations]
        population_ranges = [None] * len(worlds._populations)
        invalidation_ranges = []
        try:
            for group, ranges in zip(worlds._populations, application_ranges, strict=True):
                group.substeps, group.refresh_kinematics = substeps, refresh_kinematics
                group.updates = graph_ops.prepare_updates(
                    group.world_storage.protected_count,
                    enable_count_maximum=group.world_storage.capacity,
                    binding_capacity=512 * substeps,
                )
                group.binding_signature = _binding_signature(group)
                group.application_ranges = ranges
                group.before_step, group.after_substep = before_step, after_substep
            # Warmed native programs use separate Data. Load lifecycle and row
            # transfer kernels before conditional capture, avoiding unrelated solvers.
            wp.load_module(module=__name__, device=worlds.device)
            wp.load_module(module=field_ops.__name__, device=worlds.device)
            # Discovery and final recording preserve every conditional ordinal and native branch.
            # The non-executable discovery needs no GPU updater or retained submission rights.
            for discovering in (True, False):
                worlds._recording_passes += 1
                invalidation_ranges.clear()
                for regions in native_regions:
                    regions.clear()
                for group, ranges in zip(worlds._populations, application_ranges, strict=True):
                    ranges.clear()
                    group.application_ranges = ranges
                with wp.ScopedCapture(
                    device=worlds.device,
                    force_module_load=False,
                    capture_mode=wp.CaptureMode.THREAD_LOCAL,
                    record_launches=True,
                    record_memory_operations=True,
                    record_allocations=discovering,
                    allocation_bindings=allocation_bindings,
                ) as capture:
                    if not discovering:
                        _retain_graph(
                            worlds, graph_ops.current_capture(device=worlds.device), commands, results, permit, *retain
                        )
                    directory_ops.begin(worlds._directory, commands)
                    wp.launch(
                        _guard_health,
                        1,
                        [
                            worlds._directory.batch_result,
                            worlds._directory.transaction,
                            worlds._healthy,
                            worlds._lifecycle_needed,
                        ],
                        device=worlds.device,
                    )

                    def record_lifecycle():
                        if validate is not None:
                            validate(
                                commands, worlds._directory.transaction.status, worlds._directory.batch_result.consumed
                            )
                        directory_ops.admit(worlds._directory, commands)

                        def record_initialization(prototype, group):
                            wp.launch(
                                _initialization_ids,
                                worlds._directory.command_capacity,
                                [
                                    worlds._directory.batch_result,
                                    worlds._directory.transaction,
                                    prototype,
                                    group.request_indices,
                                    group.source_rows,
                                    group.destination_rows,
                                    group.initialization_count,
                                ],
                                device=worlds.device,
                            )
                            field_ops.transfer(
                                group.initialization_transfer,
                                group.source_rows,
                                group.destination_rows,
                                group.initialization_count,
                            )
                            if initialize is not None:
                                initialize(
                                    group.view,
                                    group.request_indices,
                                    group.destination_rows,
                                    group.initialization_count,
                                    group.initialization_transfer.status,
                                    worlds._directory.transaction.initialized_sequence,
                                    worlds._directory.batch_result.sequence,
                                )
                            wp.launch(
                                _acknowledge_initialization,
                                worlds._directory.command_capacity,
                                [
                                    worlds._directory.batch_result,
                                    worlds._directory.transaction,
                                    prototype,
                                    group.initialization_transfer.status,
                                    int(initialize is not None),
                                ],
                                device=worlds.device,
                            )

                        graph_ops.capture_parallel(
                            lambda prototype=prototype, group=group: record_initialization(prototype, group)
                            for prototype, group in enumerate(worlds._populations)
                        )
                        directory_ops.publish(worlds._directory, commands, results)
                        directory_ops.plan_compaction(worlds._directory)

                        def record_compaction(prototype, group):
                            wp.launch(
                                _relocation_count,
                                1,
                                [
                                    worlds._directory.transaction,
                                    worlds._directory.compaction,
                                    prototype,
                                    worlds._healthy,
                                    group.move_count,
                                ],
                                device=worlds.device,
                            )
                            field_ops.transfer(
                                group.compaction_transfer,
                                group.move_source_rows,
                                group.move_destination_rows,
                                group.move_count,
                            )
                            wp.launch(
                                _acknowledge_moves,
                                1,
                                [
                                    worlds._directory.transaction,
                                    worlds._directory.compaction,
                                    prototype,
                                    worlds._healthy,
                                    group.compaction_transfer.status,
                                ],
                                device=worlds.device,
                            )

                        graph_ops.capture_parallel(
                            lambda prototype=prototype, group=group: record_compaction(prototype, group)
                            for prototype, group in enumerate(worlds._populations)
                        )
                        directory_ops.publish_compaction(worlds._directory)

                    wp.capture_if(worlds._lifecycle_needed, on_true=record_lifecycle)

                    # Invalidation also runs for a population whose last world was destroyed.
                    # These fixed scalar writes must not be disabled by its live-count updater.
                    def invalidate_contacts(group):
                        graph = graph_ops.current_capture(device=worlds.device)
                        start = wp.capture_launch_count(graph)
                        wp.capture_if(
                            worlds._directory.batch_result.consumed, lambda: mjw.invalidate_contact_cache(group.data)
                        )
                        invalidation_ranges.append((start, wp.capture_launch_count(graph)))

                    graph_ops.capture_parallel(
                        lambda group=group: invalidate_contacts(group) for group in worlds._populations
                    )
                    # Every updater must finish and be checked before any prototype
                    # enters its conditional program, including native solver loops.
                    if not discovering:
                        graph_ops.capture_parallel(
                            [
                                lambda group=group: graph_ops.record_update(group.updates)
                                for group in worlds._populations
                            ]
                        )
                    graph_ops.capture_parallel(
                        lambda group=group: wp.launch(
                            _guard_graph_updates,
                            group.updates.binding_capacity,
                            [
                                group.updates.errors,
                                group.updates.binding_count,
                                worlds._healthy,
                                worlds._directory.batch_result,
                            ],
                            device=worlds.device,
                        )
                        for group in worlds._populations
                    )
                    for prototype, group in enumerate(worlds._populations):
                        wp.launch(
                            _execution_conditions,
                            1,
                            [
                                worlds._directory.data,
                                worlds._directory.batch_result,
                                prototype,
                                group.world_storage.ready_count,
                                group.contact_storage.ready_count,
                                *[
                                    owner.ready_count if owner is not None else fallback
                                    for owner, fallback in zip(
                                        group.transient_storages or (None, None, None),
                                        (
                                            group.world_storage.ready_count,
                                            group.contact_storage.ready_count,
                                            group.ccd_count,
                                        ),
                                        strict=True,
                                    )
                                ],
                                group.contact_count,
                                group.ccd_count,
                                group.contact_quota,
                                group.ccd_quota,
                                worlds._healthy,
                                permit,
                                group.step_condition,
                                group.kinematics_condition,
                            ],
                            device=worlds.device,
                        )

                    def record_population(prototype, group):
                        graph = graph_ops.current_capture(device=worlds.device)
                        start = wp.capture_launch_count(graph)
                        _record_physics(group, native_regions[prototype])
                        population_ranges[prototype] = (start, wp.capture_launch_count(graph))

                    graph_ops.capture_parallel(
                        [
                            lambda prototype=prototype, group=group: record_population(prototype, group)
                            for prototype, group in enumerate(worlds._populations)
                        ]
                    )
                if discovering:
                    discovery = capture.graph
                    worlds._discovery_memory = wp.capture_discovery_memory(discovery)
                    allocation_bindings = _prepare_transient_storage(worlds, discovery, native_regions)
                    for group in worlds._populations:
                        _publish_native_counts(group)
                    directory_ops.publish_admissible_slots(
                        worlds._directory, tuple(_world_ready_capacity(group) for group in worlds._populations)
                    )
            graph = capture.graph
            discovery = None
            del capture
            launches = wp.capture_get_launches(graph)
            memory_launches = {id(operation.launch) for operation in wp.capture_get_memory_operations(graph)}
            population_records = {
                id(launches[index]) for start, end in population_ranges for index in range(start, end)
            }
            invalidation_records = {
                id(launches[index]) for start, end in invalidation_ranges for index in range(start, end)
            }
            if memory_launches - population_records - invalidation_records:
                raise NotImplementedError("Lifecycle fills/copies require explicit storage admission")
            field_ops.validate_memory_operations(
                graph,
                tuple(op for op in wp.capture_get_memory_operations(graph) if id(op.launch) in invalidation_records),
                (),
                fixed_arrays=tuple(
                    array for group in worlds._populations for array in group.global_arrays.values() if array.size
                ),
            )
            for group, (start, end), ranges in zip(
                worlds._populations, population_ranges, application_ranges, strict=True
            ):
                indices = tuple(index for lo, hi in ranges for index in range(lo, hi))
                if any(not start <= index < end for index in indices) or len(set(indices)) != len(indices):
                    raise RuntimeError("Application records must belong exactly once to their population range")
                application_indices = set(indices)
                if indices:
                    if application_bindings is None:
                        raise ValueError("Physics callbacks emitting kernels require application_bindings")
                    records = tuple(launches[index] for index in indices)
                    if any(id(record) in memory_launches for record in records):
                        raise NotImplementedError("Application fills/copies require explicit storage admission")
                    extents, parameters, fixed = application_bindings(group.view, records)
                    group.bindings.extend(
                        graph_ops.adopt_launches(
                            group.updates,
                            graph,
                            records,
                            extents=extents,
                            parameters=parameters,
                            fixed=fixed,
                        )
                    )
                _bind_native_program(
                    group,
                    graph,
                    tuple(launches[index] for index in range(start, end) if index not in application_indices),
                )
                graph_ops.bind(group.updates, graph, group.bindings)
            # The last application's declaration may reference any borrowed population.
            for group in worlds._populations:
                _validate_population(group)
            graph_ops.instantiate_and_upload(worlds._populations[0].updates, graph)
            return graph
        except BaseException as failure:
            # This root cannot silently retry partially marked CUDA graph nodes.
            # Failed finalization already poisons any executable it created;
            # preserve that ownership while quarantining the population.
            worlds._service_failed = True
            if graph is not None and graph.graph_exec is None:
                graph_ops.invalidate(graph)
            try:
                worlds._healthy.fill_(0)
                wp.synchronize_stream(wp.get_stream(worlds.device))
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "Native graph preparation and quarantine failed", [failure, cleanup]
                ) from failure
            finally:
                graph = discovery = allocation_bindings = None
                capture = None  # noqa: F841 - Release graph ownership before propagating failure.
                traceback.clear_frames(failure.__traceback__)
            raise
        finally:
            # Earlier recording failures may leave sibling populations unvisited.
            # Only explicit graph retention owns callback resources after preparation.
            for group in worlds._populations:
                group.before_step = group.after_substep = None
                group.application_ranges = None


def _retain_graph(worlds, graph, *buffers):
    """Retain explicit borrowers on the population's single captured graph."""
    mujoco_worlds_validate(worlds)
    previous = worlds._graph() if worlds._graph is not None else None
    if previous is not None and previous is not graph:
        raise RuntimeError("Native population already belongs to another graph")
    directory_ops.retain_graph(worlds._directory, graph, worlds, *buffers)
    for group in worlds._populations:
        field_ops.retain_transfer_graph(
            group.initialization_transfer,
            graph,
            group.request_indices,
            group.source_rows,
            group.destination_rows,
            group.initialization_count,
        )
        field_ops.retain_transfer_graph(
            group.compaction_transfer, graph, group.move_source_rows, group.move_destination_rows, group.move_count
        )
        for owner, _ in _storage_domains(group):
            field_ops.retain_graph(
                owner, graph, *group.transient_arrays, group.step_condition, group.kinematics_condition
            )
    worlds._graph = weakref.ref(graph)
    return graph


def mujoco_worlds_grow_backing(worlds, world_ready_capacities: tuple[int, ...], *, streams: tuple[wp.Stream, ...]):
    """Grow state and transient W/C/D backing coherently.

    Service requires no active capture on this device. The caller excludes
    new submissions during service. Fresh suffix mapping
    leaves earlier admitted accesses untouched. Ready counts, contact Data initialization, joint C/D counts
    and directory admission are queued on the current stream
    after the supplied reader streams. Subsequent consumers must use this
    stream or wait for it; return does not certify GPU completion. World Data
    still initializes through the captured admission protocol before going live.

    All mappings precede readiness publication. A clean budget failure leaves
    readiness unchanged and retains any newly mapped headroom for a retry.
    Other service failures quarantine the population. Directory publication
    errors latch device health, suppressing every later graph replay. Growth
    that reuses historical addresses joins readers and publication; it still
    only adds mappings, preserving any existing limit on unmapped spares.
    Shrinking and spare trimming remain explicit resize_backing operations.
    """
    with wp.ScopedDevice(worlds.device):
        mujoco_worlds_validate(worlds)
        for group in worlds._populations:
            if group.view is not None:
                if _binding_signature(group)[0:2] != group.storage_signature:
                    raise ValueError("Prepared storage or count descriptors changed")
                mujoco_world_population_validate(group.view)
        if wp.get_device(worlds.device).is_capturing:
            raise RuntimeError("Backing growth requires execution outside graph capture")
        if worlds._backing is None:
            raise RuntimeError("This population has fixed backing")
        if worlds._retirement_pending:
            raise RuntimeError("Resolve pending backing retirement before changing capacity")
        world_ready_capacities = tuple(world_ready_capacities)
        if len(world_ready_capacities) != len(worlds._populations) or any(
            type(n) is not int
            or not _world_ready_capacity(group) <= n <= group.world_storage.capacity
            or n * group.contact_quota > group.contact_storage.capacity
            or n * group.ccd_quota > wp.upper_bound(group.count_parameters[2])
            for n, group in zip(world_ready_capacities, worlds._populations, strict=True)
        ):
            raise ValueError("Growth must preserve ready worlds and fit prepared W/C/D capacities")
        streams = tuple(streams)
        if not streams or any(
            not isinstance(stream, wp.Stream) or stream.device != worlds.device for stream in streams
        ):
            raise ValueError("Backing service requires existing Warp streams on the population device")
        services = [
            (group, owner, target)
            for group, n in zip(worlds._populations, world_ready_capacities, strict=True)
            for owner, domain in _storage_domains(group)
            for target in (n * (1, group.contact_quota, group.ccd_quota)[domain],)
            if target > owner.ready_rows
        ]
        if not services:
            return
        current = wp.get_stream(worlds.device)
        try:
            for stream in streams:
                if stream.cuda_stream != current.cuda_stream:
                    current.wait_stream(stream)
            fresh = all(tuple(field_ops.can_map_backing_without_join(owner, target) for _, owner, target in services))
        except BaseException as failure:
            _quarantine_service(worlds, failure)
            raise
        budget_rejected = False
        try:
            with (
                nullcontext()
                if fresh
                else backing_ops.maintenance(worlds._backing, streams=tuple(stream.cuda_stream for stream in streams))
            ):
                for _, owner, target in services:
                    try:
                        field_ops.map_backing(owner, target)
                    except MemoryError:
                        budget_rejected = not owner.service_failed
                        raise
                for group, owner, target in services:
                    old_ready = owner.ready_rows
                    field_ops.publish_ready(owner, target)
                    if owner is group.contact_storage:
                        _initialize_ready_fields(group, owner, start=old_ready)
                for group in worlds._populations:
                    _publish_native_counts(group)
                status = directory_ops.publish_admissible_slots(
                    worlds._directory, tuple(_world_ready_capacity(group) for group in worlds._populations)
                )
                wp.launch(_guard_backing_publication, 1, [status, worlds._healthy], device=worlds.device)
                if not fresh:
                    wp.synchronize_stream(current)
        except MemoryError as failure:
            if not budget_rejected:
                _quarantine_service(worlds, failure)
            raise
        except BaseException as failure:
            _quarantine_service(worlds, failure)
            raise


def mujoco_worlds_resize_backing(
    worlds, world_ready_capacities: tuple[int, ...], *, streams: tuple[wp.Stream, ...], spare_bytes: int | None = None
):
    """Join once, resize native state and scratch, then publish coherent readiness.

    The caller excludes new submissions until return. Contact/CCD scratch has
    no persistent lifetime after all consumers join. Reclaim mapped ranges,
    including unpublished growth headroom, before acquiring shared backing.
    A clean budget rejection republishes the safely backed partial result;
    driver failures quarantine the complete population, including graphs
    held by external callers.
    If specified, ``spare_bytes`` retains up to its granule-rounded amount
    of available unmapped backing, without allocating reserve. ``None``
    preserves all spare handles. Trimming after successful service uses the
    same maintenance scope and adds no stream synchronization.
    """
    with wp.ScopedDevice(worlds.device):
        mujoco_worlds_validate(worlds)
        for group in worlds._populations:
            if group.view is not None:
                if _binding_signature(group)[0:2] != group.storage_signature:
                    raise ValueError("Prepared storage or count descriptors changed")
                mujoco_world_population_validate(group.view)
        if wp.get_device(worlds.device).is_capturing:
            raise RuntimeError("Backing resize requires execution outside graph capture")
        if worlds._backing is None:
            raise RuntimeError("This population has fixed backing")
        if spare_bytes is not None and (type(spare_bytes) is not int or spare_bytes < 0):
            raise ValueError("Retained spare bytes must be a nonnegative integer or None")
        if worlds._retirement_pending:
            raise RuntimeError("Resolve pending backing retirement before changing capacity")
        world_ready_capacities = tuple(world_ready_capacities)
        if len(world_ready_capacities) != len(worlds._populations) or any(
            type(n) is not int
            or not 0 <= n <= group.world_storage.capacity
            or n * group.contact_quota > group.contact_storage.capacity
            or n * group.ccd_quota > wp.upper_bound(group.count_parameters[2])
            for n, group in zip(world_ready_capacities, worlds._populations, strict=True)
        ):
            raise ValueError("Requested worlds exceed prepared W/C/D capacities")
        streams = tuple(streams)
        if not streams or any(
            not isinstance(stream, wp.Stream) or stream.device != worlds.device for stream in streams
        ):
            raise ValueError("Backing service requires existing Warp streams on the population device")
        budget_rejected = False
        try:
            with backing_ops.maintenance(worlds._backing, streams=tuple(stream.cuda_stream for stream in streams)):
                directory_ops.withdraw_admissible_slots(worlds._directory, world_ready_capacities)
                live_counts = worlds._directory.data.live_count.numpy()
                try:
                    for group in worlds._populations:
                        group.contact_count.zero_()
                        group.ccd_count.zero_()
                    services = [
                        (
                            group,
                            owner,
                            target,
                            int(live_counts[prototype]) if domain == 0 else 0,
                        )
                        for prototype, (group, n) in enumerate(
                            zip(worlds._populations, world_ready_capacities, strict=True)
                        )
                        for owner, domain in _storage_domains(group)
                        for target in (n * (1, group.contact_quota, group.ccd_quota)[domain],)
                        if owner.backing is not None
                    ]
                    # Clean failed growth may retain mapped bytes beyond the published ready prefix.
                    services.sort(
                        key=lambda entry: (
                            entry[1].reservation is None
                            or all(
                                offset + size <= entry[2] * entry[1].row_stride_bytes
                                for offset, size in backing_ops.mapped_ranges(entry[1].backing, entry[1].reservation)
                            )
                        )
                    )
                    for group, owner, target, live_count in services:
                        old_ready = owner.ready_rows
                        try:
                            field_ops.resize_backing(owner, target, protected_count_host=live_count)
                        except MemoryError:
                            budget_rejected = not owner.service_failed
                            raise
                        if owner is group.contact_storage and owner.ready_rows > old_ready:
                            _initialize_ready_fields(group, owner, start=old_ready)
                    for group in worlds._populations:
                        _publish_native_counts(group)
                    directory_ops.publish_admissible_slots(
                        worlds._directory, tuple(_world_ready_capacity(group) for group in worlds._populations)
                    )
                    wp.synchronize_stream(wp.get_stream(worlds.device))
                    if spare_bytes is not None:
                        backing_ops.trim(worlds._backing, keep_bytes=spare_bytes)
                except MemoryError:
                    if budget_rejected:
                        budget_rejected = False
                        for group in worlds._populations:
                            _publish_native_counts(group)
                        directory_ops.publish_admissible_slots(
                            worlds._directory, tuple(_world_ready_capacity(group) for group in worlds._populations)
                        )
                        wp.synchronize_stream(wp.get_stream(worlds.device))
                        budget_rejected = True
                    raise
        except MemoryError as failure:
            if not budget_rejected:
                _quarantine_service(worlds, failure)
            raise
        except BaseException as failure:
            _quarantine_service(worlds, failure)
            raise


def mujoco_worlds_withdraw_backing(
    worlds: MuJoCoWorlds, world_ready_capacities: tuple[int, ...], *, streams: tuple[wp.Stream, ...]
) -> None:
    """Withdraw W/C/D tails in stream order, retaining their physical backing.

    Experimental. Supply every prior reader stream and exclude concurrent
    submissions while this call establishes their dependency cut. Directory
    validation precedes field withdrawal; a rejected directory batch leaves
    every field unchanged. Each receipt latches native health before later
    physics can execute. Return certifies enqueueing, not GPU acceptance.

    The current stream and supplied streams may subsequently run captured
    physics over the surviving prefix. Other streams must wait for the
    current stream. Successful submission performs no host synchronization
    or device readback. Submission failure synchronizes quarantine before
    returning its error, so existing graphs cannot outrun the health update.
    Only one retirement batch may be pending. Growth, resizing and capture
    require completing :func:`mujoco_worlds_reclaim_backing` or :func:`mujoco_worlds_cancel_backing_retirement`.
    Cancellation retains mappings and does not restore withdrawn readiness.
    """
    with wp.ScopedDevice(worlds.device):
        mujoco_worlds_validate(worlds)
        if wp.get_device(worlds.device).is_capturing:
            raise RuntimeError("Backing withdrawal requires execution outside graph capture")
        if worlds._backing is None:
            raise RuntimeError("This population has fixed backing")
        if worlds._retirement_pending:
            raise RuntimeError("Resolve pending backing retirement before another withdrawal")
        for group in worlds._populations:
            if group.view is not None:
                if _binding_signature(group)[0:2] != group.storage_signature:
                    raise ValueError("Prepared storage or count descriptors changed")
                mujoco_world_population_validate(group.view)
            if not group.retirements:
                raise RuntimeError("Capture the complete native program before backing withdrawal")
        targets = tuple(world_ready_capacities)
        if len(targets) != len(worlds._populations) or any(
            type(n) is not int or not 0 <= n <= _world_ready_capacity(group)
            for n, group in zip(targets, worlds._populations, strict=True)
        ):
            raise ValueError("Withdrawal must preserve reserved storage and only reduce ready worlds")
        streams = tuple(streams)
        if not streams or any(
            not isinstance(stream, wp.Stream) or stream.device != worlds.device for stream in streams
        ):
            raise ValueError("Backing withdrawal requires existing reader streams on the population device")
        current = wp.get_stream(worlds.device)
        worlds._retirement_streams = (current, *streams)
        worlds._retirement_pending = True
        try:
            for stream in streams:
                if stream.cuda_stream != current.cuda_stream:
                    backing_ops.record_event(
                        worlds._backing, worlds._retirement_publication.cuda_event, stream=stream.cuda_stream
                    )
                    backing_ops.wait_event(
                        worlds._backing, current.cuda_stream, worlds._retirement_publication.cuda_event
                    )
            status = directory_ops.withdraw_admissible_slots_async(worlds._directory, targets)
            wp.launch(_guard_backing_publication, 1, [status, worlds._healthy], device=worlds.device)
            for group, target in zip(worlds._populations, targets, strict=True):
                counts = (target, target * group.contact_quota, target * group.ccd_quota)
                # C/D protected counts describe intended work, not persistent live rows.
                # Lower them only after directory acceptance and before each storage check.
                wp.launch(
                    _withdraw_native_counts,
                    1,
                    [status, group.contact_count, group.ccd_count, *counts[1:]],
                    device=worlds.device,
                )
                for retirement, (_, domain) in zip(group.retirements, _storage_domains(group), strict=True):
                    receipt = field_ops.withdraw_backing(
                        retirement, counts[domain], streams=(current,), prerequisite=status
                    )
                    wp.launch(_guard_backing_publication, 1, [receipt, worlds._healthy], device=worlds.device)
            backing_ops.record_event(
                worlds._backing, worlds._retirement_publication.cuda_event, stream=current.cuda_stream
            )
            for stream in streams:
                if stream.cuda_stream != current.cuda_stream:
                    backing_ops.wait_event(
                        worlds._backing, stream.cuda_stream, worlds._retirement_publication.cuda_event
                    )
        except BaseException as failure:
            _quarantine_service(worlds, failure)
            raise


def mujoco_worlds_reclaim_backing(worlds: MuJoCoWorlds) -> bool:
    """Poll a withdrawal and unmap its tails after prior readers finish.

    Experimental. Return False while any completion is pending, otherwise
    True. No host wait is introduced. Surviving-prefix physics may continue
    on the streams ordered by :func:`mujoco_worlds_withdraw_backing`. Reclaimed handles
    remain in the shared physical pool. A partial unmap failure retains its
    plan for retry and does not revoke safe surviving-prefix execution.
    GPU rejection is reported after completion and disables future service.
    """
    return _finish_backing_retirement(worlds, cancel=False)


def mujoco_worlds_cancel_backing_retirement(worlds: MuJoCoWorlds) -> bool:
    """Poll completion and retain mappings without restoring withdrawn readiness.

    Experimental. Return False until prior readers finish, otherwise True.
    Cancellation cannot undo an unmap that has started; finish reclamation
    in that case. Later :func:`mujoco_worlds_grow_backing` may republish retained capacity.
    """
    return _finish_backing_retirement(worlds, cancel=True)


def _finish_backing_retirement(worlds, *, cancel):
    with wp.ScopedDevice(worlds.device):
        if wp.get_device(worlds.device).is_capturing:
            raise RuntimeError("Backing retirement requires execution outside graph capture")
        if worlds._closed:
            raise RuntimeError("The native population is closed")
        if worlds._service_failed:
            raise RuntimeError("Failed backing service requires closing retained resources")
        if worlds._backing is None:
            raise RuntimeError("This population has fixed backing")
        if not worlds._retirement_pending:
            return True
        retirements = tuple(retirement for group in worlds._populations for retirement in group.retirements)
        if cancel and any(retirement.reclaim_started for retirement in retirements):
            raise RuntimeError("Cannot cancel backing retirement after any field reclamation has started")
        operation = field_ops.cancel_retirement if cancel else field_ops.reclaim_backing
        failures = []
        for retirement in retirements:
            if retirement.pending:
                try:
                    operation(retirement)
                except BaseException as failure:
                    failures.append(failure)
                    if not retirement.pending:
                        worlds._service_failed = True
        worlds._retirement_pending = any(retirement.pending for retirement in retirements)
        if not worlds._retirement_pending:
            worlds._retirement_streams = ()
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup("Native backing retirement remains incomplete", failures)
        return not worlds._retirement_pending


def _quarantine_service(worlds, failure):
    """Retire service rights while preserving a failed GPU quarantine beside its cause."""
    worlds._service_failed = True
    try:
        worlds._healthy.fill_(0)
        current = wp.get_stream(worlds.device)
        if worlds._backing is None:
            wp.synchronize_stream(current)
        else:
            with backing_ops.maintenance(worlds._backing, streams=(current.cuda_stream,)):
                pass
    except BaseException as cleanup:
        raise BaseExceptionGroup("Native backing service and quarantine failed", [failure, cleanup]) from failure


def mujoco_worlds_memory_report(worlds):
    """Report backing, native arrays and metadata by their authoritative owner.

    Reports remain readable after incomplete retirement. Released subowners
    are named explicitly and their reports are None, never a fabricated zero.
    Remaining storage and backing owners report their own retained resources.
    """
    import mujoco_warp as mjw

    result = {
        "directory": directory_ops.memory_report(worlds._directory),
        "populations": [],
        "shared_backing": backing_ops.memory_report(worlds._backing) if worlds._backing is not None else None,
        "discovery": worlds._discovery_memory,
        "startup_recording_passes": worlds._recording_passes,
        "runtime_recaptures": 0,
        "backing_retirement_pending": worlds._retirement_pending,
        "control_metadata_bytes": (
            worlds._healthy.capacity + worlds._always_permit.capacity + worlds._lifecycle_needed.capacity
        ),
        "scope": "Single-owner native array ledger; external fixture/oracle/JIT/graph/driver memory excluded",
    }
    seen_models = set()
    for group in worlds._populations:
        globals_report = [{"field": name, "bytes": array.capacity} for name, array in group.global_arrays.items()]
        model_arrays = []
        for name, array, _ in mjw.array_fields(group.model):
            if array.size and array.ptr not in seen_models:
                seen_models.add(array.ptr)
                model_arrays.append({"field": name, "bytes": array.capacity})
        result["populations"].append(
            {
                "world_storage": field_ops.memory_report(group.world_storage),
                "contact_storage": field_ops.memory_report(group.contact_storage),
                "transient_storages": {
                    domain: field_ops.memory_report(owner) if owner is not None else None
                    for domain, owner in zip(("world", "candidate", "ccd"), group.transient_storages, strict=False)
                },
                "transient_fixed_bytes": sum(array.capacity for array in group.transient_arrays),
                "default_storage": field_ops.memory_report(group.default_storage),
                "initialization": field_ops.transfer_memory_report(group.initialization_transfer)
                if group.initialization_transfer is not None
                else None,
                "compaction": field_ops.transfer_memory_report(group.compaction_transfer)
                if group.compaction_transfer is not None
                else None,
                "retired_subowners": [
                    name
                    for name in ("initialization_transfer", "compaction_transfer", "updates")
                    if getattr(group, name) is None
                ],
                "graph_updates": graph_ops.memory_report(group.updates) if group.updates is not None else None,
                "execution_metadata_bytes": sum(
                    array.capacity
                    for array in (
                        group.request_indices,
                        group.source_rows,
                        group.destination_rows,
                        group.initialization_count,
                        group.move_count,
                        group.step_condition,
                        group.kinematics_condition,
                        group.default_storage.protected_count,
                        group.contact_count,
                        group.ccd_count,
                    )
                ),
                "global_arrays": globals_report,
                "global_array_bytes": sum(item["bytes"] for item in globals_report),
                "immutable_model_arrays": model_arrays,
                "immutable_model_array_bytes": sum(item["bytes"] for item in model_arrays),
                "empty_fields": group.empty_fields,
                "world_ready_capacity": _world_ready_capacity(group),
                "compacted_fields": list(group.compaction_transfer.field_names)
                if group.compaction_transfer is not None
                else None,
            }
        )
    return result


def mujoco_worlds_close(worlds, *, streams: tuple[wp.Stream, ...]):
    """Join consumers and close owned storage after every graph borrower is gone."""
    streams = tuple(streams)
    if not streams or any(not isinstance(stream, wp.Stream) or stream.device != worlds.device for stream in streams):
        raise ValueError("Retirement requires existing Warp streams on the population device")
    raw_streams = tuple(stream.cuda_stream for stream in streams)
    with wp.ScopedDevice(worlds.device):
        graph = worlds._graph() if worlds._graph is not None else None
        if graph is not None:
            raise RuntimeError("Destroy the native population graph before closing its owners")
        if worlds._retirement_pending:
            with backing_ops.maintenance(
                worlds._backing, streams=tuple(stream.cuda_stream for stream in worlds._retirement_streams)
            ):
                pass
        worlds._closed = True
        if worlds._directory is not None:
            directory_ops.close(worlds._directory, streams=streams)
        for group in worlds._populations:
            group.initialization_transfer = group.compaction_transfer = None
            group.data = group.execution_data = group.updates = None
            group.bindings.clear()
            group.metadata_signature = group.binding_signature = None
            if group.view is not None:
                group.view.data = None
        for group in worlds._populations:
            for owner in (*[owner for owner, _ in _storage_domains(group)], group.default_storage):
                if owner is not None:
                    field_ops.close(owner, streams=raw_streams)
            group.transient_arrays = ()
            group.retirements = ()
        worlds._retirement_pending, worlds._retirement_streams = False, ()
        worlds._retirement_publication = None
        worlds._populations.clear()
        worlds.populations = ()
        if worlds._backing is not None:
            with backing_ops.maintenance(worlds._backing, streams=raw_streams):
                backing_ops.close(worlds._backing)
