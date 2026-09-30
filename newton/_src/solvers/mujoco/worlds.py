# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compose native prototypes, mechanical W/C/D storage and one world directory.

Prepare from one-world templates, reserve descriptors directly, and capture once.
Native Data is authoritative; this root owns no dense Newton State mirror.
"""

import sys
import traceback
import weakref
from dataclasses import dataclass, field, fields, is_dataclass, replace

import warp as wp

from ...sim.worlds import (
    ADMITTED,
    IDLE,
    MOVING,
    OK,
    PHASE_INVALID,
    WorldCommands,
    WorldCompaction,
    WorldDirectory,
    WorldDirectoryData,
    WorldResults,
    WorldTransaction,
    _validate_arrays,
)
from ...utils.cuda_graph import DeviceGraphUpdates, Int32Parameter, KernelBinding, capture_parallel
from ...utils.cuda_vmm import CudaBacking
from ...utils.row_storage import FieldSpec, RowStorage

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _guard_health(d: WorldDirectoryData, t: WorldTransaction, healthy: wp.array[int]):
    if healthy[0] == 0:
        d.flags[0] = 0
        d.flags[1] = 0
        d.flags[2] = PHASE_INVALID
        t.phase[0] = IDLE


@wp.kernel
def _guard_graph_updates(errors: wp.array[int], count: wp.array[int], healthy: wp.array[int], flags: wp.array[int]):
    index = wp.tid()
    if index < count[0] and errors[index] != 0:
        wp.atomic_min(healthy, 0, 0)
        wp.atomic_min(flags, 1, 0)
        wp.atomic_max(flags, 2, PHASE_INVALID)


@wp.kernel
def _initialization_ids(
    d: WorldDirectoryData,
    t: WorldTransaction,
    prototype: int,
    requests: wp.array[int],
    sources: wp.array[int],
    destinations: wp.array[int],
    count: wp.array[int],
):
    ordinal = wp.tid()
    accepted = int(0)
    if d.flags[0] != 0 and t.phase[0] == ADMITTED:
        accepted = t.group_starts[prototype + 1] - t.group_starts[prototype]
    if ordinal == 0:
        count[0] = accepted
    if ordinal < accepted:
        request = t.accepted_requests[t.group_starts[prototype] + ordinal]
        requests[ordinal] = request
        sources[ordinal] = 0
        destinations[ordinal] = t.destination_slot[request]


@wp.kernel
def _acknowledge_initialization(
    d: WorldDirectoryData, t: WorldTransaction, prototype: int, status: wp.array[int], caller_ack: int
):
    ordinal = wp.tid()
    if d.flags[0] == 0 or t.phase[0] != ADMITTED:
        return
    start = t.group_starts[prototype]
    if ordinal < t.group_starts[prototype + 1] - start:
        request = t.accepted_requests[start + ordinal]
        if status[0] != 0:
            t.initialized[request] = wp.uint64(0)
        elif caller_ack == 0 and t.status[request] == OK:
            t.initialized[request] = d.sequence[0]


@wp.kernel
def _relocation_count(
    t: WorldTransaction, moves: WorldCompaction, prototype: int, healthy: wp.array[int], count: wp.array[int]
):
    count[0] = 0
    if healthy[0] != 0 and t.phase[0] == MOVING:
        count[0] = moves.sources[prototype]


@wp.kernel
def _acknowledge_moves(
    t: WorldTransaction, moves: WorldCompaction, prototype: int, healthy: wp.array[int], status: wp.array[int]
):
    if healthy[0] != 0 and t.phase[0] == MOVING and status[0] == 0:
        moves.copied[prototype] = moves.sources[prototype]


@wp.kernel
def _execution_conditions(
    d: WorldDirectoryData,
    prototype: int,
    world_ready: wp.array[int],
    contact_ready: wp.array[int],
    ccd_ready: wp.array[int],
    contact_quota: int,
    ccd_quota: int,
    healthy: wp.array[int],
    permit: wp.array[int],
    condition: wp.array[int],
    poses: wp.array[int],
):
    count = d.active_count[prototype]
    ready = wp.int32(
        healthy[0] != 0
        and d.flags[1] != 0
        and count > 0
        and count <= world_ready[0]
        and count * contact_quota <= contact_ready[0]
        and count * ccd_quota <= ccd_ready[0]
    )
    condition[0] = wp.int32(ready != 0 and permit[0] != 0)
    poses[0] = ready


@dataclass
class _MuJoCoPrototype:
    """One topology's immutable model and authoritative native fields/program."""

    model: object
    data: object = None
    defaults: RowStorage | None = None
    rows: RowStorage | None = None
    contacts: RowStorage | None = None
    ccd: RowStorage | None = None
    contact_quota: int = 0
    ccd_quota: int = 0
    workspace: object = None
    substeps: int = 1
    before_step: object = None
    after_substep: object = None
    refresh_kinematics: bool = True
    initialization: object = None
    relocation: object = None
    updates: DeviceGraphUpdates | None = None
    request_ids: object = None
    source_ids: object = None
    destination_ids: object = None
    initialization_count: object = None
    move_sources: object = None
    move_destinations: object = None
    move_count: object = None
    condition: object = None
    poses_condition: object = None
    bindings: list = field(default_factory=list)
    operations: list = field(default_factory=list)
    global_arrays: dict = field(default_factory=dict)
    absent_fields: dict = field(default_factory=dict)

    @property
    def ready_worlds(self):
        return min(
            self.rows.ready_rows, self.contacts.ready_rows // self.contact_quota, self.ccd.ready_rows // self.ccd_quota
        )

    def observe_launch(self, kernel, dim, extent_domain, extent_axis=0, parameters=None):
        """Record explicit native semantics after the engine emits its own launch."""
        dimensions = (dim,) if isinstance(dim, int) else tuple(dim)
        if not all(dimensions):
            return
        owners = {"world": self.rows, "candidate": self.contacts, "ccd": self.ccd}
        if extent_domain is not None and extent_domain not in owners:
            raise ValueError(f"Unknown native launch domain: {extent_domain}")
        if extent_domain is None and extent_axis is not None:
            raise ValueError("A fixed worker grid must explicitly omit its dynamic axis")
        if extent_domain is not None and extent_axis != 0:
            raise ValueError("Prepared native row domains require explicit leading axis zero")
        sources = {
            name: (owner.count if name == "world" else owner.ready_count, owner.capacity)
            for name, owner in owners.items()
        }
        labels = {argument.label: (index + 1, argument.type) for index, argument in enumerate(kernel.adj.args)}
        scalars = []
        for name, domain in (parameters or {}).items():
            if name not in labels or domain not in sources:
                raise ValueError(f"Unknown native count parameter or domain: {name} -> {domain}")
            index, dtype = labels[name]
            if dtype not in (int, wp.int32):
                raise ValueError(f"Native count parameter must be int32: {name}")
            scalars.append(Int32Parameter(index, *sources[domain]))
        extent = sources[extent_domain][0] if extent_domain is not None else None
        binding = KernelBinding(
            self.updates.capture_tail(),
            rank=kernel.adj.kernel_dim,
            extent_axis=extent_axis,
            extent_source=extent,
            parameters=tuple(scalars),
        )
        self.bindings.append(binding)
        self.operations.append(
            {
                "operation": "launch",
                "kernel": kernel.key,
                "module": kernel.func.__module__,
                "function": kernel.func.__qualname__,
                "dim": list(dimensions),
                "extent_domain": extent_domain,
                "extent_axis": extent_axis,
                "parameters": dict(parameters or {}),
                "rank": binding.rank,
                "node": binding.node,
            }
        )

    def fill(self, array, value, domain):
        """Record a native-declared bounded fill without inferring an array domain."""
        if domain not in ("world", "candidate", "ccd"):
            raise ValueError(f"Unknown native fill domain: {domain}")
        owner = {"world": self.rows, "candidate": self.contacts, "ccd": self.ccd}[domain]
        owner.fill(array, value, count=owner.count if domain == "world" else owner.ready_count)
        self.operations.append({"operation": "fill", "domain": domain, "field": owner.lookup(array).name})
        return array

    def copy(self, destination, source, domain):
        """Record a native-declared bounded copy; RowStorage validates typed ownership."""
        if domain not in ("world", "candidate", "ccd"):
            raise ValueError(f"Unknown native copy domain: {domain}")
        owner = {"world": self.rows, "candidate": self.contacts, "ccd": self.ccd}[domain]
        owner.copy(destination, source, count=owner.count if domain == "world" else owner.ready_count)
        field = owner.lookup(destination) or owner.lookup(source)
        self.operations.append({"operation": "copy", "domain": domain, "field": field.name})

    def record_physics(self):
        """Record conditional native physics and poses on the current stream."""
        import mujoco_warp as mjw

        if self.workspace.observer is not None:
            raise RuntimeError("Native workspace already has a recording observer")
        self.workspace.observer = self

        def step():
            if self.before_step is not None:
                self.before_step(self)
            for _ in range(self.substeps):
                mjw.step(self.model, self.data, workspace=self.workspace)
                if self.after_substep is not None:
                    self.after_substep(self)

        try:
            wp.capture_if(self.condition, on_true=step)
            if self.refresh_kinematics:
                wp.capture_if(
                    self.poses_condition,
                    on_true=lambda: mjw.kinematics(self.model, self.data, workspace=self.workspace),
                )
        finally:
            self.workspace.observer = None
            self.before_step = None
            self.after_substep = None


def _data_arrays(value, prefix=""):
    """Native schema owns the domain of each field; numerical shapes never infer it."""
    for descriptor in fields(value):
        item = getattr(value, descriptor.name)
        name = prefix + descriptor.name
        if isinstance(item, wp.array):
            shape = getattr(descriptor.type, "shape", ())
            yield name, item, shape[0] if shape else None
        elif is_dataclass(item):
            yield from _data_arrays(item, name + ".")


def _replace_data(source, arrays, prefix="", **metadata):
    values = dict(metadata)
    for descriptor in fields(source):
        name = prefix + descriptor.name
        value = getattr(source, descriptor.name)
        if name in arrays:
            values[descriptor.name] = arrays[name]
        elif is_dataclass(value):
            values[descriptor.name] = _replace_data(value, arrays, name + ".")
    return replace(source, **values)


class MuJoCoWorlds:
    """Run changing populations of prepared native MuJoCo world prototypes.

    .. experimental::
        This class and its methods are experimental. The initial feature set is
        native NxN contacts, Newton/implicit-fast integration, pyramidal cones
        and sleeping rigid articulations, as admitted by the prepared workspace.
        Python 3.11 or newer and CUDA are required for this experimental runtime.

    Args:
        prepared: Pairs of immutable native Model and one-world default Data.
            Existing :class:`SolverMuJoCo` supplies one-world preparation.
        capacities: Virtual world row limits, one per prototype.
        id_capacity: Prepared maximum simultaneous logical identities.
        command_capacity: Maximum lifecycle requests in one captured batch.
        contact_capacities: Optional candidate/contact virtual row limits.
        ccd_capacities: Optional CCD scratch virtual row limits.
        memory_budget_bytes: Physical VMM budget; None selects fixed backing.
        initial_rows: Initially backed world rows per prototype.

    Native Data arrays are authoritative. No dense Newton State or replicated
    model is created. The caller warms each native program on separate one-world
    Data before capture, and owns reset policy, command buffers and observations.
    """

    def __init__(
        self,
        prepared,
        *,
        capacities,
        id_capacity,
        command_capacity,
        contact_capacities=None,
        ccd_capacities=None,
        memory_budget_bytes=None,
        initial_rows=None,
    ):
        if sys.version_info < (3, 11):
            raise RuntimeError("Experimental MuJoCoWorlds requires Python 3.11 or newer")
        import mujoco_warp as mjw

        prepared, capacities = tuple(prepared), tuple(capacities)
        if not prepared or len(prepared) != len(capacities):
            raise ValueError("One virtual world capacity is required per native prototype")
        if any(type(n) is not int or not 1 <= n < 2**31 for n in capacities):
            raise ValueError("World capacities must be positive int32 limits")
        contact_capacities = (
            tuple(d.naconmax * n for (_, d), n in zip(prepared, capacities, strict=True))
            if contact_capacities is None
            else tuple(contact_capacities)
        )
        ccd_capacities = (
            tuple(d.naccdmax * n for (_, d), n in zip(prepared, capacities, strict=True))
            if ccd_capacities is None
            else tuple(ccd_capacities)
        )
        initial_rows = capacities if initial_rows is None else tuple(initial_rows)
        if any(len(values) != len(prepared) for values in (initial_rows, contact_capacities, ccd_capacities)):
            raise ValueError("Each prototype needs complete W/C/D capacities and initial rows")
        device = prepared[0][1].qpos.device
        for (model, template), capacity, initial, contacts, ccd in zip(
            prepared, capacities, initial_rows, contact_capacities, ccd_capacities, strict=True
        ):
            if template.nworld != 1 or template.qpos.device != device or model.qpos0.device != device:
                raise ValueError("Native defaults must have one world on the model/population device")
            if type(initial) is not int or not 0 <= initial <= capacity:
                raise ValueError("Initial world rows exceed capacity")
            if (
                any(type(n) is not int or not 1 <= n < 2**31 for n in (contacts, ccd))
                or template.naconmax < 1
                or template.naccdmax < 1
                or contacts < initial * template.naconmax
                or ccd < initial * template.naccdmax
            ):
                raise ValueError("Contact/CCD capacities must support the initial world population")
        if not device.is_cuda:
            raise ValueError("MuJoCoWorlds currently requires a CUDA device")
        self.device, self.backing, self.directory = device, None, None
        self.prototypes = []
        self._healthy = self._always_permit = None
        self._closed, self._service_failed, self._capture_attempted = False, False, False
        self._graph = None
        try:
            with wp.ScopedDevice(device):
                if memory_budget_bytes is not None:
                    self.backing = CudaBacking(
                        memory_budget_bytes, device_ordinal=device.ordinal, expected_uuid=device.uuid
                    )
                backing = self.backing
                self.directory = WorldDirectory(
                    capacities, id_capacity=id_capacity, command_capacity=command_capacity, device=device
                )
                self._healthy = wp.ones(1, dtype=int, device=device)
                self._always_permit = wp.ones(1, dtype=int, device=device)
                start = 0
                for prototype, ((model, template), capacity, initial, contact_cap, ccd_cap) in enumerate(
                    zip(prepared, capacities, initial_rows, contact_capacities, ccd_capacities, strict=True)
                ):
                    group = _MuJoCoPrototype(model, contact_quota=template.naconmax, ccd_quota=template.naccdmax)
                    self.prototypes.append(group)
                    # Native tiled solver consumers require 16-byte field starts and row strides.
                    # Candidate/contact fields use scalar loads and their declared natural alignment.
                    arrays, world_fields, contact_fields = {}, [], []
                    world_sources = {}
                    for name, array, axis in _data_arrays(template):
                        if axis == "nworld":
                            if array.shape[0] == 0 and not array.size:
                                arrays[name] = array
                                group.absent_fields[name] = {"shape": list(array.shape), "dtype": str(array.dtype)}
                            else:
                                if array.shape[0] != 1:
                                    raise ValueError(f"Default world field does not have one row: {name}")
                                world_fields.append(
                                    FieldSpec(name, array.shape[1:], array.dtype, packed=array.ndim > 1, alignment=16)
                                )
                                world_sources[name] = array
                        elif axis == "naconmax":
                            if array.shape[0] == 0 and not array.size:
                                arrays[name] = array
                                group.absent_fields[name] = {"shape": list(array.shape), "dtype": str(array.dtype)}
                            else:
                                contact_fields.append(FieldSpec(name, array.shape[1:], array.dtype))
                        else:
                            arrays[name] = wp.clone(array)
                            group.global_arrays["data." + name] = arrays[name]
                            if name in ("nacon", "ncollision"):
                                arrays[name].zero_()
                    data_field_names = tuple(spec.name for spec in world_fields)
                    group.defaults = RowStorage(1, wp.ones(1, dtype=int, device=device), fields=tuple(world_fields))
                    for name, array in world_sources.items():
                        group.defaults.copy(group.defaults.arrays[name], array)
                    specs = mjw.step_workspace_layout(
                        model, template, world_capacity=capacity, contact_capacity=contact_cap, ccd_capacity=ccd_cap
                    )
                    ccd_fields, scratch = [], {}
                    for spec in specs:
                        name = "workspace." + spec.name
                        if spec.domain == "world":
                            if spec.shape[0] == 0:
                                # Optional solver branches use an absent-array sentinel,
                                # not a capacity-sized field with unspecified values.
                                scratch[spec.name] = wp.empty(spec.shape, dtype=spec.dtype, device=device)
                                group.absent_fields[name] = {"shape": list(spec.shape), "dtype": str(spec.dtype)}
                            else:
                                world_fields.append(
                                    FieldSpec(
                                        name, spec.shape[1:], spec.dtype, packed=len(spec.shape) > 1, alignment=16
                                    )
                                )
                        elif spec.domain == "candidate":
                            contact_fields.append(FieldSpec(name, spec.shape[1:], spec.dtype))
                        elif spec.domain == "ccd":
                            ccd_fields.append(FieldSpec(name, spec.shape[1:], spec.dtype, alignment=16))
                        elif spec.domain == "global_counter":
                            scratch[spec.name] = wp.zeros(spec.shape, dtype=spec.dtype, device=device)
                            group.global_arrays[name] = scratch[spec.name]
                        else:
                            raise ValueError(f"Unknown native workspace domain: {spec.domain}")
                    live_count = self.directory.d.active_count[prototype : prototype + 1]
                    group.rows = RowStorage(
                        capacity, live_count, fields=tuple(world_fields), backing=backing, initial_rows=initial
                    )
                    group.contacts = RowStorage(
                        contact_cap,
                        wp.zeros(1, dtype=int, device=device),
                        fields=tuple(contact_fields),
                        backing=backing,
                        initial_rows=initial * group.contact_quota if backing else None,
                    )
                    group.ccd = RowStorage(
                        ccd_cap,
                        wp.zeros(1, dtype=int, device=device),
                        fields=tuple(ccd_fields),
                        backing=backing,
                        initial_rows=initial * group.ccd_quota if backing else None,
                    )
                    for owner in (group.rows, group.contacts, group.ccd):
                        for name, array in owner.arrays.items():
                            owner.prepare_fill(array, 0)
                            if name.startswith("workspace."):
                                scratch[name.removeprefix("workspace.")] = array
                            else:
                                arrays[name] = array
                    group.contacts.prepare_fill(group.contacts.arrays["contact.efc_address"], -1)
                    for owner in (group.contacts, group.ccd):
                        owner.zero(count=owner.ready_count)
                    group.contacts.fill(
                        group.contacts.arrays["contact.efc_address"], -1, count=group.contacts.ready_count
                    )
                    group.data = _replace_data(
                        template, arrays, nworld=capacity, naconmax=contact_cap, naccdmax=ccd_cap
                    )
                    group.rows.prepare_fill(group.data.island_dofadr, model.nv)
                    group.rows.prepare_fill(scratch["island_can_sleep"], 1)
                    group.workspace = mjw.make_step_workspace(model, group.data, live_count=live_count, arrays=scratch)
                    group.initialization = group.rows.prepare_transfer(group.defaults, fields=data_field_names)
                    group.relocation = group.rows.prepare_transfer(group.rows, fields=data_field_names)
                    group.request_ids = wp.zeros(command_capacity, dtype=int, device=device)
                    group.source_ids = wp.zeros(command_capacity, dtype=int, device=device)
                    group.destination_ids = wp.zeros(command_capacity, dtype=int, device=device)
                    group.initialization_count = wp.zeros(1, dtype=int, device=device)
                    end = start + capacity
                    group.move_sources = self.directory.moves.source[start:end]
                    group.move_destinations = self.directory.moves.destination[start:end]
                    group.move_count = wp.zeros(1, dtype=int, device=device)
                    group.condition = wp.zeros(1, dtype=int, device=device)
                    group.poses_condition = wp.zeros(1, dtype=int, device=device)
                    start = end
                self.directory.publish_ready(tuple(group.ready_worlds for group in self.prototypes))
        except BaseException as failure:
            traceback.clear_frames(failure.__traceback__)
            arrays = scratch = world_sources = array = owner = None
            try:
                self.close(streams=(wp.get_stream(device).cuda_stream,))
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "Native population construction and cleanup failed", [failure, cleanup]
                ) from failure
            raise

    def _ensure_open(self):
        if self._closed or self._service_failed:
            raise RuntimeError("Native population is closed or its backing service failed")

    def capture(
        self,
        commands: WorldCommands,
        results: WorldResults,
        *,
        permit=None,
        validate=None,
        initialize=None,
        retain=(),
        substeps=1,
        before_step=None,
        after_substep=None,
        refresh_kinematics=True,
    ):
        """Capture once, retaining caller buffers and all native owners.

        Optional preparation callbacks record payload validation before admission
        and payload writes after the default Data copy. The initializer explicitly
        acknowledges successful requests in WorldTransaction.initialized; missing
        acknowledgements or failed base copies cannot publish a replacement.

        Args:
            commands: Caller-owned lifecycle buffers with increasing batch sequence.
            results: Caller-owned per-request results.
            permit: Optional GPU int32 scalar permitting physics advancement.
            validate: Preparation callback recording payload validation.
            initialize: Preparation callback recording payload writes and acknowledgement.
            retain: External buffers borrowed by any recorded callback kernels.
            substeps: Ordered native steps per graph replay.
            before_step: Callback recording controls once before ordered native steps.
            after_substep: Callback recording contact consumers after each native step.
            refresh_kinematics: Refresh body, geometry and site transforms after valid
                reset-only frames or after the final substep, independently of permit.

        Returns:
            One bound Warp graph. The caller owns submissions and must destroy this
            graph before closing the population. A failed mandatory reset suppresses
            both physical advancement and pose refresh.
        """

        with wp.ScopedDevice(self.device):
            self._ensure_open()
            if self._capture_attempted:
                raise RuntimeError("Each native population prepares exactly one graph")
            if any(
                callback is not None and not callable(callback)
                for callback in (validate, initialize, before_step, after_substep)
            ):
                raise TypeError("Payload recording callbacks must be callable")
            if type(substeps) is not int or not 1 <= substeps < 2**31 // 512:
                raise ValueError("Native substeps must be a positive int32-bounded integer")
            if type(refresh_kinematics) is not bool:
                raise TypeError("refresh_kinematics must be a boolean")
            retain = tuple(retain)
            _validate_arrays(
                commands, WorldCommands, self.directory.command_capacity, self.device, ("sequence", "count")
            )
            _validate_arrays(results, WorldResults, self.directory.command_capacity, self.device)
            permit = self._always_permit if permit is None else permit
            if (
                not isinstance(permit, wp.array)
                or permit.shape != (1,)
                or permit.dtype != wp.int32
                or permit.device != self.device
                or not permit.is_contiguous
            ):
                raise ValueError("Native step permit must be one contiguous int32 scalar on the population device")
            self._capture_attempted = True
            graph = None
            try:
                for group in self.prototypes:
                    group.substeps, group.before_step = substeps, before_step
                    group.after_substep, group.refresh_kinematics = after_substep, refresh_kinematics
                    group.updates = DeviceGraphUpdates(
                        group.rows.count, maximum=group.rows.capacity, capacity_nodes=512 * substeps
                    )
                # Native programs were warmed on separate Data. Loading only this
                # root avoids compiling every unrelated imported Newton solver.
                wp.load_module(module=__name__, device=self.device)
                with wp.ScopedCapture(
                    device=self.device, force_module_load=False, capture_mode=wp.CaptureMode.THREAD_LOCAL
                ) as capture:
                    self.directory.begin(commands)
                    wp.launch(_guard_health, 1, [self.directory.d, self.directory.t, self._healthy], device=self.device)
                    if validate is not None:
                        validate(commands, self.directory.t)
                    self.directory.admit(commands)
                    for prototype, group in enumerate(self.prototypes):
                        wp.launch(
                            _initialization_ids,
                            self.directory.command_capacity,
                            [
                                self.directory.d,
                                self.directory.t,
                                prototype,
                                group.request_ids,
                                group.source_ids,
                                group.destination_ids,
                                group.initialization_count,
                            ],
                            device=self.device,
                        )
                        group.initialization.record(group.source_ids, group.destination_ids, group.initialization_count)
                        if initialize is not None:
                            initialize(
                                group,
                                group.request_ids,
                                group.destination_ids,
                                group.initialization_count,
                                group.initialization.status,
                                self.directory.t,
                                self.directory.d.sequence,
                            )
                        wp.launch(
                            _acknowledge_initialization,
                            self.directory.command_capacity,
                            [
                                self.directory.d,
                                self.directory.t,
                                prototype,
                                group.initialization.status,
                                int(initialize is not None),
                            ],
                            device=self.device,
                        )
                    self.directory.publish(commands, results)
                    self.directory.plan_moves()
                    for prototype, group in enumerate(self.prototypes):
                        wp.launch(
                            _relocation_count,
                            1,
                            [self.directory.t, self.directory.moves, prototype, self._healthy, group.move_count],
                            device=self.device,
                        )
                        group.relocation.record(group.move_sources, group.move_destinations, group.move_count)
                        wp.launch(
                            _acknowledge_moves,
                            1,
                            [self.directory.t, self.directory.moves, prototype, self._healthy, group.relocation.status],
                            device=self.device,
                        )
                    self.directory.publish_moves()
                    # Every updater must finish and be checked before any prototype
                    # enters its conditional program, including native solver loops.
                    for group in self.prototypes:
                        group.updates.capture_update()
                    for group in self.prototypes:
                        wp.launch(
                            _guard_graph_updates,
                            group.updates.capacity_nodes,
                            [group.updates.errors, group.updates.binding_count, self._healthy, self.directory.d.flags],
                            device=self.device,
                        )
                    for prototype, group in enumerate(self.prototypes):
                        wp.launch(
                            _execution_conditions,
                            1,
                            [
                                self.directory.d,
                                prototype,
                                group.rows.ready_count,
                                group.contacts.ready_count,
                                group.ccd.ready_count,
                                group.contact_quota,
                                group.ccd_quota,
                                self._healthy,
                                permit,
                                group.condition,
                                group.poses_condition,
                            ],
                            device=self.device,
                        )
                    capture_parallel([group.record_physics for group in self.prototypes])
                graph = capture.graph
                del capture
                for group in self.prototypes:
                    group.updates.bind(graph, group.bindings)
                self._retain_graph(graph, commands, results, permit, *retain)
                self.prototypes[0].updates.instantiate_upload(graph)
                return graph
            except BaseException as failure:
                # This root cannot silently retry partially marked CUDA graph nodes.
                self._healthy.fill_(0)
                wp.synchronize_stream(wp.get_stream(self.device))
                graph = None
                capture = None  # noqa: F841 - Release graph ownership before propagating failure.
                traceback.clear_frames(failure.__traceback__)
                raise

    def _retain_graph(self, graph, *buffers):
        """Retain explicit borrowers on the population's single captured graph."""
        self._ensure_open()
        previous = self._graph() if self._graph is not None else None
        if previous is not None and previous is not graph:
            raise RuntimeError("Native population already belongs to another graph")
        self.directory.retain_graph(graph, self, *buffers)
        for group in self.prototypes:
            group.initialization.retain_graph(
                graph, group.request_ids, group.source_ids, group.destination_ids, group.initialization_count
            )
            group.relocation.retain_graph(graph, group.move_sources, group.move_destinations, group.move_count)
            for owner in (group.rows, group.contacts, group.ccd):
                owner.retain_graph(graph, group.workspace, group.condition, group.poses_condition)
        self._graph = weakref.ref(graph)
        return graph

    def resize_backing(self, rows: tuple[int, ...], *, streams, spare_bytes: int | None = None):
        """Join once, service all W/C/D prefixes, then publish one coherent ready set.

        The caller excludes new submissions until return. Contact/CCD scratch has
        no persistent lifetime after all consumers join. A clean budget rejection
        republishes the safely backed partial result; driver failures quarantine
        the complete population, including graphs held by external callers.
        If specified, ``spare_bytes`` retains up to its granule-rounded amount
        of available unmapped backing, without allocating reserve. ``None``
        preserves all spare handles. Trimming after successful service uses the
        same maintenance scope and adds no stream synchronization.
        """
        with wp.ScopedDevice(self.device):
            self._ensure_open()
            if self.backing is None:
                raise RuntimeError("This population has fixed backing")
            if spare_bytes is not None and (type(spare_bytes) is not int or spare_bytes < 0):
                raise ValueError("Retained spare bytes must be a nonnegative integer or None")
            rows = tuple(rows)
            if len(rows) != len(self.prototypes) or any(
                type(n) is not int
                or not 0 <= n <= group.rows.capacity
                or n * group.contact_quota > group.contacts.capacity
                or n * group.ccd_quota > group.ccd.capacity
                for n, group in zip(rows, self.prototypes, strict=True)
            ):
                raise ValueError("Requested worlds exceed prepared W/C/D capacities")
            with self.backing.maintenance(streams=streams):
                self.directory.withdraw_ready(rows)
                live_counts = self.directory.d.active_count.numpy()
                budget_rejected = False
                try:
                    services = [
                        (group, owner, target, int(live_counts[prototype]) if owner is group.rows else 0)
                        for prototype, (group, n) in enumerate(zip(self.prototypes, rows, strict=True))
                        for owner, target in (
                            (group.rows, n),
                            (group.contacts, n * group.contact_quota),
                            (group.ccd, n * group.ccd_quota),
                        )
                    ]
                    # Return every safely retired range before acquiring shared backing.
                    services.sort(key=lambda entry: entry[2] >= entry[1].ready_rows)
                    for group, owner, target, live_count in services:
                        old_ready = owner.ready_rows
                        try:
                            owner.resize_backing(target, live_count=live_count)
                        except MemoryError:
                            budget_rejected = not owner.service_failed
                            raise
                        if owner is not group.rows and owner.ready_rows > old_ready:
                            owner.zero(count=owner.ready_count, start=old_ready)
                            if owner is group.contacts:
                                owner.fill(
                                    owner.arrays["contact.efc_address"], -1, count=owner.ready_count, start=old_ready
                                )
                    self.directory.publish_ready(tuple(group.ready_worlds for group in self.prototypes))
                    wp.synchronize_stream(wp.get_stream(self.device))
                    if spare_bytes is not None:
                        self.backing.trim(keep_bytes=spare_bytes)
                except MemoryError:
                    if not budget_rejected:
                        self._service_failed = True
                        self._healthy.fill_(0)
                        wp.synchronize_stream(wp.get_stream(self.device))
                        raise
                    try:
                        self.directory.publish_ready(tuple(group.ready_worlds for group in self.prototypes))
                        wp.synchronize_stream(wp.get_stream(self.device))
                    except BaseException:
                        self._service_failed = True
                        self._healthy.fill_(0)
                        wp.synchronize_stream(wp.get_stream(self.device))
                        raise
                    raise
                except BaseException:
                    self._service_failed = True
                    self._healthy.fill_(0)
                    wp.synchronize_stream(wp.get_stream(self.device))
                    raise

    def memory_report(self):
        """Report backing, native arrays and metadata by their authoritative owner."""
        result = {
            "directory": self.directory.memory_report(),
            "groups": [],
            "shared_backing": self.backing.memory_report() if self.backing is not None else None,
            "health_metadata_bytes": self._healthy.capacity + self._always_permit.capacity,
            "scope": "Single-owner native array ledger; external fixture/oracle/JIT/graph/driver memory excluded",
        }
        for group in self.prototypes:
            globals_report = [{"field": name, "bytes": array.capacity} for name, array in group.global_arrays.items()]
            model_arrays, seen = [], set()
            for name, array, _ in _data_arrays(group.model):
                if array.size and array.ptr not in seen:
                    seen.add(array.ptr)
                    model_arrays.append({"field": name, "bytes": array.capacity})
            result["groups"].append(
                {
                    "worlds": group.rows.memory_report(),
                    "contacts": group.contacts.memory_report(),
                    "ccd": group.ccd.memory_report(),
                    "defaults": group.defaults.memory_report(),
                    "workspace_borrowed": group.workspace.memory_report(),
                    "initialization": group.initialization.memory_report(),
                    "relocation": group.relocation.memory_report(),
                    "graph_updates": group.updates.memory_report() if group.updates is not None else None,
                    "index_bytes": sum(
                        array.capacity
                        for array in (
                            group.request_ids,
                            group.source_ids,
                            group.destination_ids,
                            group.initialization_count,
                            group.move_count,
                            group.condition,
                            group.poses_condition,
                            group.defaults.count,
                            group.contacts.count,
                            group.ccd.count,
                        )
                    ),
                    "globals": globals_report,
                    "global_bytes": sum(item["bytes"] for item in globals_report),
                    "immutable_model_arrays": model_arrays,
                    "immutable_model_array_bytes": sum(item["bytes"] for item in model_arrays),
                    "absent_fields": group.absent_fields,
                    "ready_worlds": group.ready_worlds,
                    "relocated_fields": list(group.relocation.names),
                }
            )
        return result

    def close(self, *, streams):
        """Join consumers and close owned storage after every graph borrower is gone."""
        with wp.ScopedDevice(self.device):
            graph = self._graph() if self._graph is not None else None
            if graph is not None:
                raise RuntimeError("Destroy the native population graph before closing its owners")
            self._closed = True
            if self.directory is not None:
                self.directory.close(streams=streams)
            for group in self.prototypes:
                group.initialization = group.relocation = None
                group.workspace = group.data = None
            for group in self.prototypes:
                for owner in (group.rows, group.contacts, group.ccd, group.defaults):
                    if owner is not None:
                        owner.close(streams=streams)
            self.prototypes.clear()
            if self.backing is not None:
                with self.backing.maintenance(streams=streams):
                    self.backing.close()
