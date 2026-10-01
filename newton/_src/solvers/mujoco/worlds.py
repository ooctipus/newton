# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compose native populations, mechanical W/C/D storage and one world directory.

Prepare from one-world templates, reserve descriptors directly, and capture once.
Native Data is authoritative; this root owns no dense Newton State mirror.
"""

import sys
import traceback
import weakref
from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    import mujoco_warp

from ...sim.worlds import (
    ADMITTED,
    IDLE,
    MOVING,
    OK,
    PHASE_INVALID,
    WorldBatchResult,
    WorldCommands,
    WorldCompaction,
    WorldDirectory,
    WorldDirectoryData,
    WorldResults,
    WorldTransaction,
    _validate_arrays,
)
from ...utils.cuda_graph import DeviceGraphUpdates, GraphKernelBinding, KernelParameterBinding, capture_parallel
from ...utils.cuda_vmm import MemoryBacking
from ...utils.field_storage import FieldSpec, FieldStorage

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _guard_health(batch: WorldBatchResult, t: WorldTransaction, healthy: wp.array[int], lifecycle: wp.array[int]):
    if healthy[0] == 0:
        batch.consumed[0] = 0
        batch.advance_allowed[0] = 0
        batch.status[0] = PHASE_INVALID
        t.phase[0] = IDLE
    lifecycle[0] = wp.int32(batch.consumed[0] != 0 or batch.status[0] != 0)


@wp.kernel
def _guard_graph_updates(errors: wp.array[int], count: wp.array[int], healthy: wp.array[int], batch: WorldBatchResult):
    index = wp.tid()
    if index < count[0] and errors[index] != 0:
        wp.atomic_min(healthy, 0, 0)
        wp.atomic_min(batch.advance_allowed, 0, 0)
        wp.atomic_max(batch.status, 0, PHASE_INVALID)


@wp.kernel
def _initialization_ids(
    batch: WorldBatchResult,
    t: WorldTransaction,
    prototype: int,
    requests: wp.array[int],
    sources: wp.array[int],
    destinations: wp.array[int],
    count: wp.array[int],
):
    ordinal = wp.tid()
    accepted = int(0)
    if batch.consumed[0] != 0 and t.phase[0] == ADMITTED:
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
    batch: WorldBatchResult, t: WorldTransaction, prototype: int, status: wp.array[int], caller_ack: int
):
    ordinal = wp.tid()
    if batch.consumed[0] == 0 or t.phase[0] != ADMITTED:
        return
    start = t.request_starts[prototype]
    if ordinal < t.request_starts[prototype + 1] - start:
        request = t.admitted_requests[start + ordinal]
        if status[0] != 0:
            t.initialized_sequence[request] = wp.uint64(0)
        elif caller_ack == 0 and t.status[request] == OK:
            t.initialized_sequence[request] = batch.sequence[0]


@wp.kernel
def _relocation_count(
    t: WorldTransaction, moves: WorldCompaction, prototype: int, healthy: wp.array[int], count: wp.array[int]
):
    count[0] = 0
    if healthy[0] != 0 and t.phase[0] == MOVING:
        count[0] = moves.count[prototype]


@wp.kernel
def _acknowledge_moves(
    t: WorldTransaction, moves: WorldCompaction, prototype: int, healthy: wp.array[int], status: wp.array[int]
):
    if healthy[0] != 0 and t.phase[0] == MOVING and status[0] == 0:
        moves.copied_count[prototype] = moves.count[prototype]


@wp.kernel
def _execution_conditions(
    d: WorldDirectoryData,
    batch: WorldBatchResult,
    prototype: int,
    world_ready: wp.array[int],
    contact_ready: wp.array[int],
    ccd_ready: wp.array[int],
    contact_quota: int,
    ccd_quota: int,
    healthy: wp.array[int],
    permit: wp.array[int],
    step_condition: wp.array[int],
    poses: wp.array[int],
    contact_count: wp.array[int],
    collision_count: wp.array[int],
):
    if batch.consumed[0] != 0:
        # Contact world indices refer to the previous placement, even on reset-only frames.
        # The next native collision pass regenerates both transient counters before use.
        contact_count[0] = 0
        collision_count[0] = 0
    count = d.live_count[prototype]
    ready = wp.int32(
        healthy[0] != 0
        and batch.advance_allowed[0] != 0
        and count > 0
        and count <= world_ready[0]
        and count * contact_quota <= contact_ready[0]
        and count * ccd_quota <= ccd_ready[0]
    )
    step_condition[0] = wp.int32(ready != 0 and permit[0] != 0)
    poses[0] = ready


@dataclass(frozen=True, slots=True, eq=False)
class MuJoCoWorldPopulation:
    """Borrow state, capacities and recording for one homogeneous world population.

    .. experimental::
        This type and its attributes are experimental. Instances are supplied by
        :attr:`MuJoCoWorlds.populations` and its preparation callbacks; callers do
        not construct or retire them. Views are invalid after their population
        closes. Native Model is immutable; Data contains authoritative physics
        state. Callers own their control and observation semantics.

    Counts are GPU int32 scalars. Live count includes sleeping worlds. Readiness
    certifies safe access and may be zero after failure while memory stays mapped.
    Current contact records are counted by ``data.nacon`` and belong to the last
    completed substep at the current placement. Every consumed lifecycle batch
    invalidates all population contact records before physics: nacon/ncollision
    become zero until collision runs again. Zero then means no current records,
    not that the geometry is separated. Read contact forces in ``after_substep``.
    Neither readiness nor reserved capacity creates a logical world.
    """

    _prototype_index: int
    _owner: "_MuJoCoWorldPopulation" = field(repr=False, compare=False)

    def _borrow(self):
        if self._owner.data is None:
            raise RuntimeError("The prototype's population is closed")
        return self._owner

    @property
    def prototype_index(self) -> int:
        """Return the stable integer prototype index used by world commands."""
        return self._prototype_index

    @property
    def model(self) -> "mujoco_warp.Model":
        """Return the immutable prepared native Model."""
        return self._borrow().model

    @property
    def data(self) -> "mujoco_warp.Data":
        """Return authoritative native Data; world rows are prototype-local."""
        return self._borrow().data

    @property
    def world_capacity(self) -> int:
        """Return the prepared virtual world-row limit."""
        return self._borrow().world_storage.capacity

    @property
    def contact_capacity(self) -> int:
        """Return the prepared candidate/contact-row limit."""
        return self._borrow().contact_storage.capacity

    @property
    def ccd_capacity(self) -> int:
        """Return the prepared CCD scratch-row limit."""
        return self._borrow().ccd_storage.capacity

    @property
    def world_live_count(self) -> wp.array[wp.int32]:
        """Borrow the directory's authoritative live world count; do not write it."""
        return self._borrow().world_storage.protected_count

    @property
    def world_storage_ready_count(self) -> wp.array[wp.int32]:
        """Borrow certified world-storage readiness; only population service may write it."""
        return self._borrow().world_storage.ready_count

    @property
    def contact_storage_ready_count(self) -> wp.array[wp.int32]:
        """Borrow certified contact-storage readiness; only population service may write it."""
        return self._borrow().contact_storage.ready_count

    @property
    def ccd_storage_ready_count(self) -> wp.array[wp.int32]:
        """Borrow certified CCD-storage readiness; only population service may write it."""
        return self._borrow().ccd_storage.ready_count

    @property
    def world_ready_capacity(self) -> int:
        """Return the last joined W/C/D ready prefix without a device readback."""
        return self._borrow().world_ready_capacity

    def record_launch(
        self,
        kernel,
        dim,
        *,
        inputs=(),
        outputs=(),
        domain,
        extent_axis=0,
        parameter_domains=None,
        block_dim=0,
        tiled=False,
    ):
        """Record an application kernel and its count binding as one operation.

        Call only inside ``before_step`` or ``after_substep``. ``domain`` is
        ``world``, ``candidate`` or ``ccd``; ``None`` with ``extent_axis=None``
        preserves a fixed worker grid. Named int32 ``parameter_domains`` map kernel
        argument names to those same domains. World bounds use the live count;
        candidate/CCD bounds use their ready prefixes. Fixed worker kernels must
        guard their own accesses. All external arrays belong in capture's retain.
        ``tiled=True`` selects Warp's tiled launch with an explicit block size.
        Invalid count declarations fail before emission. Any failure after emission
        invalidates the population program, even when the callback catches it.
        """
        owner = self._borrow()
        if owner.workspace.recorder is not owner or owner.updates is None:
            raise RuntimeError("Record application launches inside a prototype physics callback")
        if not wp.get_stream(owner.data.qpos.device).is_capturing:
            raise RuntimeError("Application recording requires the active population graph capture")
        if type(tiled) is not bool or (tiled and (type(block_dim) is not int or block_dim < 1)):
            raise ValueError("Tiled recording requires an explicit positive block dimension")
        owner._launch_sources(kernel, domain, extent_axis, parameter_domains)
        launch = wp.launch_tiled if tiled else wp.launch
        try:
            launch(kernel, dim=dim, inputs=inputs, outputs=outputs, block_dim=block_dim, device=owner.data.qpos.device)
            owner.bind_launch(kernel, dim, domain, extent_axis=extent_axis, parameter_domains=parameter_domains)
        except BaseException:
            owner.recording_failed = True
            raise


@dataclass
class _MuJoCoWorldPopulation:
    """One topology's immutable model and authoritative native fields/program."""

    model: object
    view: MuJoCoWorldPopulation | None = None
    data: object = None
    default_storage: FieldStorage | None = None
    world_storage: FieldStorage | None = None
    contact_storage: FieldStorage | None = None
    ccd_storage: FieldStorage | None = None
    contact_quota: int = 0
    ccd_quota: int = 0
    workspace: object = None
    substeps: int = 1
    before_step: object = None
    after_substep: object = None
    refresh_kinematics: bool = True
    initialization_transfer: object = None
    compaction_transfer: object = None
    updates: DeviceGraphUpdates | None = None
    request_indices: object = None
    source_rows: object = None
    destination_rows: object = None
    initialization_count: object = None
    move_source_rows: object = None
    move_destination_rows: object = None
    move_count: object = None
    step_condition: object = None
    kinematics_condition: object = None
    recording_failed: bool = False
    bindings: list = field(default_factory=list)
    operations: list = field(default_factory=list)
    global_arrays: dict = field(default_factory=dict)
    empty_fields: dict = field(default_factory=dict)

    @property
    def world_ready_capacity(self):
        return min(
            self.world_storage.ready_rows,
            self.contact_storage.ready_rows // self.contact_quota,
            self.ccd_storage.ready_rows // self.ccd_quota,
        )

    def _launch_sources(self, kernel, extent_domain, extent_axis, parameter_domains):
        """Validate declarations and resolve their count owners before application emission."""
        if self.recording_failed:
            raise RuntimeError("The population program has a failed recording")
        owners = {"world": self.world_storage, "candidate": self.contact_storage, "ccd": self.ccd_storage}
        if extent_domain is not None and extent_domain not in owners:
            raise ValueError(f"Unknown native launch domain: {extent_domain}")
        if extent_domain is None and extent_axis is not None:
            raise ValueError("A fixed worker grid must explicitly omit its dynamic axis")
        if extent_domain is not None and (type(extent_axis) is not int or extent_axis != 0):
            raise ValueError("Prepared native row domains require explicit leading axis zero")
        if parameter_domains is not None and len(parameter_domains) > 4:
            raise ValueError("Prepared kernels support at most four int32 count parameters")
        sources = {
            name: (owner.protected_count if name == "world" else owner.ready_count, owner.capacity)
            for name, owner in owners.items()
        }
        labels = {argument.label: (index + 1, argument.type) for index, argument in enumerate(kernel.adj.args)}
        scalars = []
        for name, domain in (parameter_domains or {}).items():
            if name not in labels or domain not in sources:
                raise ValueError(f"Unknown native count parameter or domain: {name} -> {domain}")
            index, dtype = labels[name]
            if dtype not in (int, wp.int32):
                raise ValueError(f"Native count parameter must be int32: {name}")
            scalars.append(KernelParameterBinding(index, *sources[domain]))
        extent = sources[extent_domain][0] if extent_domain is not None else None
        return extent, tuple(scalars)

    def bind_launch(self, kernel, dim, extent_domain, extent_axis=0, parameter_domains=None):
        """Bind an emitted native node; any failure forbids publishing this program."""
        try:
            extent, scalars = self._launch_sources(kernel, extent_domain, extent_axis, parameter_domains)
            dimensions = (dim,) if isinstance(dim, int) else tuple(dim)
            if not all(dimensions):
                return
            binding = GraphKernelBinding(
                self.updates.register_last_kernel_node(),
                launch_rank=kernel.adj.kernel_dim,
                extent_axis=extent_axis,
                extent_source=extent,
                parameters=scalars,
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
                    "parameter_domains": dict(parameter_domains or {}),
                    "launch_rank": binding.launch_rank,
                    "node": binding.node,
                }
            )
        except BaseException:
            self.recording_failed = True
            raise

    def fill(self, array, value, domain):
        """Record a native-declared bounded fill without inferring an array domain."""
        if domain not in ("world", "candidate", "ccd"):
            raise ValueError(f"Unknown native fill domain: {domain}")
        owner = {"world": self.world_storage, "candidate": self.contact_storage, "ccd": self.ccd_storage}[domain]
        owner.fill(array, value, count=owner.protected_count if domain == "world" else owner.ready_count)
        self.operations.append({"operation": "fill", "domain": domain, "field": owner.lookup(array).name})
        return array

    def copy(self, destination, source, domain):
        """Record a native-declared bounded copy; FieldStorage validates typed ownership."""
        if domain not in ("world", "candidate", "ccd"):
            raise ValueError(f"Unknown native copy domain: {domain}")
        owner = {"world": self.world_storage, "candidate": self.contact_storage, "ccd": self.ccd_storage}[domain]
        owner.copy(destination, source, count=owner.protected_count if domain == "world" else owner.ready_count)
        field = owner.lookup(destination) or owner.lookup(source)
        self.operations.append({"operation": "copy", "domain": domain, "field": field.name})

    def record_physics(self):
        """Record conditional native physics and poses on the current stream."""
        import mujoco_warp as mjw

        if self.recording_failed:
            raise RuntimeError("The population program has a failed recording")
        if self.workspace.recorder is not None and self.workspace.recorder is not self:
            raise RuntimeError("Native workspace already has a recorder")
        self.workspace.recorder = self

        def step():
            if self.before_step is not None:
                self.before_step(self.view)
            for _ in range(self.substeps):
                mjw.step(self.model, self.data, workspace=self.workspace)
                if self.after_substep is not None:
                    self.after_substep(self.view)

        def step_and_poses():
            wp.capture_if(self.step_condition, on_true=step)
            mjw.kinematics(self.model, self.data, workspace=self.workspace)

        try:
            if self.refresh_kinematics:
                wp.capture_if(self.kinematics_condition, on_true=step_and_poses)
            else:
                wp.capture_if(self.step_condition, on_true=step)
            if self.recording_failed:
                raise RuntimeError("The population program has a failed recording")
        except BaseException:
            self.recording_failed = True
            raise
        finally:
            self.workspace.recorder = None
            self.before_step = None
            self.after_substep = None


def _data_arrays(value, prefix="", axis=None):
    """Walk declared array fields; only schema annotations identify capacity axes."""
    if isinstance(value, wp.array):
        yield prefix, value, axis
    elif is_dataclass(value):
        for descriptor in fields(value):
            name = f"{prefix}.{descriptor.name}" if prefix else descriptor.name
            shape = getattr(descriptor.type, "shape", ())
            yield from _data_arrays(getattr(value, descriptor.name), name, shape[0] if shape else None)
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from _data_arrays(child, f"{prefix}[{index}]")


def _replace_data(source, arrays, prefix="", **metadata):
    """Rebind the same dataclass/tuple/list paths discovered by _data_arrays."""
    if not prefix:
        unknown = set(arrays) - {name for name, _, _ in _data_arrays(source)}
        if unknown:
            raise ValueError(f"Cannot bind unknown native array fields: {sorted(unknown)}")
    if prefix in arrays:
        return arrays[prefix]
    if is_dataclass(source):
        values = {
            descriptor.name: _replace_data(
                getattr(source, descriptor.name),
                arrays,
                f"{prefix}.{descriptor.name}" if prefix else descriptor.name,
            )
            for descriptor in fields(source)
        }
        values.update(metadata)
        return replace(source, **values)
    if isinstance(source, (tuple, list)):
        return type(source)(_replace_data(child, arrays, f"{prefix}[{index}]") for index, child in enumerate(source))
    return source


class MuJoCoWorlds:
    """Run changing homogeneous populations of prepared MuJoCo world prototypes.

    .. experimental::
        This class and its methods are experimental. The initial feature set is
        native NxN contacts, Newton/implicit-fast integration, pyramidal cones
        and sleeping rigid articulations, as admitted by the prepared workspace.
        Python 3.11 or newer and CUDA are required for this experimental runtime.

    Args:
        prepared: Pairs of immutable native Model and one-world default Data.
            Every nonempty batched Model field must have one broadcast parameter row;
            slot-dependent model parameters cannot follow identity-preserving compaction.
            Existing :class:`SolverMuJoCo` supplies one-world preparation.
        world_capacities: Virtual world row limits, one per prototype.
        id_capacity: Prepared maximum simultaneous logical identities.
        command_capacity: Maximum lifecycle requests in one captured batch.
        contact_capacities: Candidate/contact virtual row limits; defaults to
            each template quota multiplied by its world capacity.
        ccd_capacities: CCD virtual row limits, with the same quota-based default.
        memory_budget_bytes: Physical VMM budget; None fully backs all capacities.
        initial_world_ready_capacities: Initially ready VMM world rows; None backs every capacity.
            Set an explicit budget and initial ready world counts for partial physical backing.

    Native Data arrays are authoritative. No dense Newton State or replicated
    model is created. The caller warms each native program on separate one-world
    Data before capture, and owns reset policy, command buffers and observations.
    """

    def __init__(
        self,
        prepared,
        *,
        world_capacities,
        id_capacity,
        command_capacity,
        contact_capacities=None,
        ccd_capacities=None,
        memory_budget_bytes=None,
        initial_world_ready_capacities=None,
    ):
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
            len(values) != len(prepared)
            for values in (initial_world_ready_capacities, contact_capacities, ccd_capacities)
        ):
            raise ValueError("Each prototype needs complete W/C/D capacities and initial ready world counts")
        device = prepared[0][1].qpos.device
        for (model, template), capacity, initial, contact_capacity, ccd_capacity in zip(
            prepared, world_capacities, initial_world_ready_capacities, contact_capacities, ccd_capacities, strict=True
        ):
            if template.nworld != 1 or template.qpos.device != device or model.qpos0.device != device:
                raise ValueError("Native defaults must have one world on the model/population device")
            for name, array, axis in _data_arrays(model):
                if axis == "*" and array.size and array.shape[0] != 1:
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
        self.device, self._backing, self._directory = device, None, None
        self._populations, self._views = [], ()
        self._healthy = self._always_permit = self._lifecycle_needed = None
        self._closed, self._service_failed, self._capture_attempted = False, False, False
        self._graph = None
        try:
            with wp.ScopedDevice(device):
                if memory_budget_bytes is not None:
                    self._backing = MemoryBacking(
                        memory_budget_bytes, device_ordinal=device.ordinal, expected_uuid=device.uuid
                    )
                backing = self._backing
                self._directory = WorldDirectory(
                    world_capacities, id_capacity=id_capacity, command_capacity=command_capacity, device=device
                )
                self._healthy = wp.ones(1, dtype=int, device=device)
                self._always_permit = wp.ones(1, dtype=int, device=device)
                self._lifecycle_needed = wp.zeros(1, dtype=int, device=device)
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
                    group = _MuJoCoWorldPopulation(model, contact_quota=template.naconmax, ccd_quota=template.naccdmax)
                    self._populations.append(group)
                    # Native tiled solver consumers require 16-byte field offsets and row strides.
                    # Candidate/contact fields use scalar loads and their declared natural alignment.
                    arrays, world_fields, contact_fields = {}, [], []
                    world_sources = {}
                    for name, array, axis in _data_arrays(template):
                        if axis == "nworld":
                            if array.shape[0] == 0 and not array.size:
                                arrays[name] = array
                                group.empty_fields[name] = {"shape": list(array.shape), "dtype": str(array.dtype)}
                            else:
                                if array.shape[0] != 1:
                                    raise ValueError(f"Default world field does not have one row: {name}")
                                world_fields.append(
                                    FieldSpec(
                                        name, array.shape[1:], array.dtype, packed=array.ndim > 1, alignment_bytes=16
                                    )
                                )
                                world_sources[name] = array
                        elif axis == "naconmax":
                            if array.shape[0] == 0 and not array.size:
                                arrays[name] = array
                                group.empty_fields[name] = {"shape": list(array.shape), "dtype": str(array.dtype)}
                            else:
                                contact_fields.append(FieldSpec(name, array.shape[1:], array.dtype))
                        else:
                            arrays[name] = wp.clone(array)
                            group.global_arrays["data." + name] = arrays[name]
                            if name in ("nacon", "ncollision"):
                                arrays[name].zero_()
                    data_field_names = tuple(spec.name for spec in world_fields)
                    group.default_storage = FieldStorage(
                        1, wp.ones(1, dtype=int, device=device), fields=tuple(world_fields)
                    )
                    for name, array in world_sources.items():
                        group.default_storage.copy(group.default_storage.arrays[name], array)
                    specs = mjw.step_workspace_layout(
                        model, template, world_capacity=capacity, contact_capacity=contact_cap, ccd_capacity=ccd_cap
                    )
                    ccd_fields, scratch = [], {}
                    for spec in specs:
                        name = "workspace." + spec.name
                        if spec.capacity_domain == "world":
                            if spec.shape[0] == 0:
                                # Optional solver branches use an absent-array sentinel,
                                # not a capacity-sized field with unspecified values.
                                scratch[spec.name] = wp.empty(spec.shape, dtype=spec.dtype, device=device)
                                group.empty_fields[name] = {"shape": list(spec.shape), "dtype": str(spec.dtype)}
                            else:
                                world_fields.append(
                                    FieldSpec(
                                        name, spec.shape[1:], spec.dtype, packed=len(spec.shape) > 1, alignment_bytes=16
                                    )
                                )
                        elif spec.capacity_domain == "candidate":
                            contact_fields.append(FieldSpec(name, spec.shape[1:], spec.dtype))
                        elif spec.capacity_domain == "ccd":
                            ccd_fields.append(FieldSpec(name, spec.shape[1:], spec.dtype, alignment_bytes=16))
                        elif spec.capacity_domain == "global_counter":
                            scratch[spec.name] = wp.zeros(spec.shape, dtype=spec.dtype, device=device)
                            group.global_arrays[name] = scratch[spec.name]
                        else:
                            raise ValueError(f"Unknown native workspace domain: {spec.capacity_domain}")
                    live_count = self._directory.data.live_count[prototype : prototype + 1]
                    group.world_storage = FieldStorage(
                        capacity, live_count, fields=tuple(world_fields), backing=backing, initial_ready_count=initial
                    )
                    group.contact_storage = FieldStorage(
                        contact_cap,
                        wp.zeros(1, dtype=int, device=device),
                        fields=tuple(contact_fields),
                        backing=backing,
                        initial_ready_count=initial * group.contact_quota if backing else None,
                    )
                    group.ccd_storage = FieldStorage(
                        ccd_cap,
                        wp.zeros(1, dtype=int, device=device),
                        fields=tuple(ccd_fields),
                        backing=backing,
                        initial_ready_count=initial * group.ccd_quota if backing else None,
                    )
                    for owner in (group.world_storage, group.contact_storage, group.ccd_storage):
                        for name, array in owner.arrays.items():
                            owner.prepare_fill(array, 0)
                            if name.startswith("workspace."):
                                scratch[name.removeprefix("workspace.")] = array
                            else:
                                arrays[name] = array
                    group.contact_storage.prepare_fill(group.contact_storage.arrays["contact.efc_address"], -1)
                    for owner in (group.contact_storage, group.ccd_storage):
                        owner.zero(count=owner.ready_count)
                    group.contact_storage.fill(
                        group.contact_storage.arrays["contact.efc_address"], -1, count=group.contact_storage.ready_count
                    )
                    group.data = _replace_data(
                        template, arrays, nworld=capacity, naconmax=contact_cap, naccdmax=ccd_cap
                    )
                    group.world_storage.prepare_fill(group.data.island_dofadr, model.nv)
                    group.world_storage.prepare_fill(scratch["island_can_sleep"], 1)
                    group.workspace = mjw.make_step_workspace(
                        model, group.data, world_live_count=live_count, arrays=scratch, recorder=group
                    )
                    group.initialization_transfer = group.world_storage.prepare_transfer(
                        group.default_storage, fields=data_field_names
                    )
                    group.compaction_transfer = group.world_storage.prepare_transfer(
                        group.world_storage, fields=data_field_names
                    )
                    group.request_indices = wp.zeros(command_capacity, dtype=int, device=device)
                    group.source_rows = wp.zeros(command_capacity, dtype=int, device=device)
                    group.destination_rows = wp.zeros(command_capacity, dtype=int, device=device)
                    group.initialization_count = wp.zeros(1, dtype=int, device=device)
                    end = start + capacity
                    group.move_source_rows = self._directory.compaction.source_slots[start:end]
                    group.move_destination_rows = self._directory.compaction.destination_slots[start:end]
                    group.move_count = wp.zeros(1, dtype=int, device=device)
                    group.step_condition = wp.zeros(1, dtype=int, device=device)
                    group.kinematics_condition = wp.zeros(1, dtype=int, device=device)
                    group.view = MuJoCoWorldPopulation(prototype, group)
                    start = end
                self._views = tuple(group.view for group in self._populations)
                self._directory.publish_ready_slots(tuple(group.world_ready_capacity for group in self._populations))
        except BaseException as failure:
            traceback.clear_frames(failure.__traceback__)
            arrays = scratch = world_sources = array = owner = None
            try:
                self.close(streams=(wp.get_stream(device),))
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "Native population construction and cleanup failed", [failure, cleanup]
                ) from failure
            raise

    @property
    def directory(self) -> WorldDirectoryData:
        """Borrow published numeric relations without lifecycle mutation operations.

        Callers must not write these arrays. Submit commands through the captured
        population program; only this composition root owns the directory protocol.
        Metadata remains readable after quarantine or close for diagnostics.
        """
        return self._directory.data

    @property
    def batch_result(self) -> WorldBatchResult:
        """Borrow outcomes, including failure diagnostics after quarantine or close."""
        return self._directory.batch_result

    @property
    def populations(self) -> tuple[MuJoCoWorldPopulation, ...]:
        """Return supported borrowed population views in prototype-index order."""
        self._ensure_open()
        return self._views

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
        through this captured lifecycle and resize_backing. Independently invoking
        the owned directory's mutating methods bypasses native state ownership.

        Args:
            commands: Caller-owned lifecycle buffers with increasing batch sequence.
            results: Caller-owned per-request results.
            permit: Optional GPU int32 scalar permitting physics advancement.
            validate: Preparation callback ``(commands, request_status, consumed)``.
                Guard the read-only consumed scalar before reading requests. Only
                replace existing OK request_status values with payload rejection.
            initialize: Preparation callback ``(population, request_indices, destination_rows,
                count, copy_status, initialized_sequence, sequence)``. Write only the
                admitted prefix when copy_status is OK; acknowledge initialized_sequence
                only after complete payload writes. Other arrays are borrowed read-only.
            retain: External buffers borrowed by any recorded callback kernels.
            substeps: Ordered native steps per graph replay.
            before_step: Callback ``(population)`` recording controls once before ordered native steps.
            after_substep: Callback ``(population)`` recording contact consumers after each native step.
            refresh_kinematics: Refresh body, geometry and site transforms after valid
                reset-only frames or after the final substep, independently of permit.

        Returns:
            One bound Warp graph. The caller owns submissions and must destroy this
            graph before closing the population. Any failed request suppresses
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
                commands, WorldCommands, self._directory.command_capacity, self.device, ("sequence", "count")
            )
            _validate_arrays(results, WorldResults, self._directory.command_capacity, self.device)
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
                for group in self._populations:
                    group.workspace.recorder = None
                    group.substeps, group.before_step = substeps, before_step
                    group.after_substep, group.refresh_kinematics = after_substep, refresh_kinematics
                    group.updates = DeviceGraphUpdates(
                        group.world_storage.protected_count,
                        enable_count_maximum=group.world_storage.capacity,
                        binding_capacity=512 * substeps,
                    )
                # Warmed native programs use separate Data. Load lifecycle and row
                # transfer kernels before conditional capture, avoiding unrelated solvers.
                wp.load_module(module=__name__, device=self.device)
                wp.load_module(module=FieldStorage.__module__, device=self.device)
                with wp.ScopedCapture(
                    device=self.device, force_module_load=False, capture_mode=wp.CaptureMode.THREAD_LOCAL
                ) as capture:
                    self._directory.begin(commands)
                    wp.launch(
                        _guard_health,
                        1,
                        [
                            self._directory.batch_result,
                            self._directory.transaction,
                            self._healthy,
                            self._lifecycle_needed,
                        ],
                        device=self.device,
                    )

                    def record_lifecycle():
                        if validate is not None:
                            validate(
                                commands, self._directory.transaction.status, self._directory.batch_result.consumed
                            )
                        self._directory.admit(commands)
                        for prototype, group in enumerate(self._populations):
                            wp.launch(
                                _initialization_ids,
                                self._directory.command_capacity,
                                [
                                    self._directory.batch_result,
                                    self._directory.transaction,
                                    prototype,
                                    group.request_indices,
                                    group.source_rows,
                                    group.destination_rows,
                                    group.initialization_count,
                                ],
                                device=self.device,
                            )
                            group.initialization_transfer.record(
                                group.source_rows, group.destination_rows, group.initialization_count
                            )
                            if initialize is not None:
                                initialize(
                                    group.view,
                                    group.request_indices,
                                    group.destination_rows,
                                    group.initialization_count,
                                    group.initialization_transfer.status,
                                    self._directory.transaction.initialized_sequence,
                                    self._directory.batch_result.sequence,
                                )
                            wp.launch(
                                _acknowledge_initialization,
                                self._directory.command_capacity,
                                [
                                    self._directory.batch_result,
                                    self._directory.transaction,
                                    prototype,
                                    group.initialization_transfer.status,
                                    int(initialize is not None),
                                ],
                                device=self.device,
                            )
                        self._directory.publish(commands, results)
                        self._directory.plan_compaction()
                        for prototype, group in enumerate(self._populations):
                            wp.launch(
                                _relocation_count,
                                1,
                                [
                                    self._directory.transaction,
                                    self._directory.compaction,
                                    prototype,
                                    self._healthy,
                                    group.move_count,
                                ],
                                device=self.device,
                            )
                            group.compaction_transfer.record(
                                group.move_source_rows, group.move_destination_rows, group.move_count
                            )
                            wp.launch(
                                _acknowledge_moves,
                                1,
                                [
                                    self._directory.transaction,
                                    self._directory.compaction,
                                    prototype,
                                    self._healthy,
                                    group.compaction_transfer.status,
                                ],
                                device=self.device,
                            )
                        self._directory.publish_compaction()

                    wp.capture_if(self._lifecycle_needed, on_true=record_lifecycle)
                    # Every updater must finish and be checked before any prototype
                    # enters its conditional program, including native solver loops.
                    capture_parallel([group.updates.record_update for group in self._populations])
                    capture_parallel(
                        lambda group=group: wp.launch(
                            _guard_graph_updates,
                            group.updates.binding_capacity,
                            [
                                group.updates.errors,
                                group.updates.binding_count,
                                self._healthy,
                                self._directory.batch_result,
                            ],
                            device=self.device,
                        )
                        for group in self._populations
                    )
                    for prototype, group in enumerate(self._populations):
                        wp.launch(
                            _execution_conditions,
                            1,
                            [
                                self._directory.data,
                                self._directory.batch_result,
                                prototype,
                                group.world_storage.ready_count,
                                group.contact_storage.ready_count,
                                group.ccd_storage.ready_count,
                                group.contact_quota,
                                group.ccd_quota,
                                self._healthy,
                                permit,
                                group.step_condition,
                                group.kinematics_condition,
                                group.data.nacon,
                                group.data.ncollision,
                            ],
                            device=self.device,
                        )
                    capture_parallel([group.record_physics for group in self._populations])
                graph = capture.graph
                del capture
                for group in self._populations:
                    group.updates.bind(graph, group.bindings)
                self._retain_graph(graph, commands, results, permit, *retain)
                self._populations[0].updates.instantiate_and_upload(graph)
                return graph
            except BaseException as failure:
                # This root cannot silently retry partially marked CUDA graph nodes.
                try:
                    self._healthy.fill_(0)
                    wp.synchronize_stream(wp.get_stream(self.device))
                except BaseException as cleanup:
                    raise BaseExceptionGroup(
                        "Native graph preparation and quarantine failed", [failure, cleanup]
                    ) from failure
                finally:
                    graph = None
                    capture = None  # noqa: F841 - Release graph ownership before propagating failure.
                    traceback.clear_frames(failure.__traceback__)
                raise
            finally:
                # Earlier recording failures may leave sibling populations unvisited.
                # Only explicit graph retention owns callback resources after preparation.
                for group in self._populations:
                    group.workspace.recorder = None
                    group.before_step = group.after_substep = None

    def _retain_graph(self, graph, *buffers):
        """Retain explicit borrowers on the population's single captured graph."""
        self._ensure_open()
        previous = self._graph() if self._graph is not None else None
        if previous is not None and previous is not graph:
            raise RuntimeError("Native population already belongs to another graph")
        self._directory.retain_graph(graph, self, *buffers)
        for group in self._populations:
            group.initialization_transfer.retain_graph(
                graph, group.request_indices, group.source_rows, group.destination_rows, group.initialization_count
            )
            group.compaction_transfer.retain_graph(
                graph, group.move_source_rows, group.move_destination_rows, group.move_count
            )
            for owner in (group.world_storage, group.contact_storage, group.ccd_storage):
                owner.retain_graph(graph, group.workspace, group.step_condition, group.kinematics_condition)
        self._graph = weakref.ref(graph)
        return graph

    def resize_backing(
        self, world_ready_capacities: tuple[int, ...], *, streams: tuple[wp.Stream, ...], spare_bytes: int | None = None
    ):
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
            if self._backing is None:
                raise RuntimeError("This population has fixed backing")
            if spare_bytes is not None and (type(spare_bytes) is not int or spare_bytes < 0):
                raise ValueError("Retained spare bytes must be a nonnegative integer or None")
            world_ready_capacities = tuple(world_ready_capacities)
            if len(world_ready_capacities) != len(self._populations) or any(
                type(n) is not int
                or not 0 <= n <= group.world_storage.capacity
                or n * group.contact_quota > group.contact_storage.capacity
                or n * group.ccd_quota > group.ccd_storage.capacity
                for n, group in zip(world_ready_capacities, self._populations, strict=True)
            ):
                raise ValueError("Requested worlds exceed prepared W/C/D capacities")
            streams = tuple(streams)
            if not streams or any(
                not isinstance(stream, wp.Stream) or stream.device != self.device for stream in streams
            ):
                raise ValueError("Backing service requires existing Warp streams on the population device")
            with self._backing.maintenance(streams=tuple(stream.cuda_stream for stream in streams)):
                self._directory.withdraw_ready_slots(world_ready_capacities)
                live_counts = self._directory.data.live_count.numpy()
                budget_rejected = False
                try:
                    services = [
                        (group, owner, target, int(live_counts[prototype]) if owner is group.world_storage else 0)
                        for prototype, (group, n) in enumerate(
                            zip(self._populations, world_ready_capacities, strict=True)
                        )
                        for owner, target in (
                            (group.world_storage, n),
                            (group.contact_storage, n * group.contact_quota),
                            (group.ccd_storage, n * group.ccd_quota),
                        )
                    ]
                    # Return every safely retired range before acquiring shared backing.
                    services.sort(key=lambda entry: entry[2] >= entry[1].ready_rows)
                    for group, owner, target, live_count in services:
                        old_ready = owner.ready_rows
                        try:
                            owner.resize_backing(target, protected_count_host=live_count)
                        except MemoryError:
                            budget_rejected = not owner.service_failed
                            raise
                        if owner is not group.world_storage and owner.ready_rows > old_ready:
                            owner.zero(count=owner.ready_count, start=old_ready)
                            if owner is group.contact_storage:
                                owner.fill(
                                    owner.arrays["contact.efc_address"], -1, count=owner.ready_count, start=old_ready
                                )
                    self._directory.publish_ready_slots(
                        tuple(group.world_ready_capacity for group in self._populations)
                    )
                    wp.synchronize_stream(wp.get_stream(self.device))
                    if spare_bytes is not None:
                        self._backing.trim(keep_bytes=spare_bytes)
                except MemoryError as failure:
                    if not budget_rejected:
                        self._quarantine_service(failure)
                        raise
                    try:
                        self._directory.publish_ready_slots(
                            tuple(group.world_ready_capacity for group in self._populations)
                        )
                        wp.synchronize_stream(wp.get_stream(self.device))
                    except BaseException as failure:
                        self._quarantine_service(failure)
                        raise
                    raise
                except BaseException as failure:
                    self._quarantine_service(failure)
                    raise

    def _quarantine_service(self, failure):
        """Retire service rights while preserving a failed GPU quarantine beside its cause."""
        self._service_failed = True
        try:
            self._healthy.fill_(0)
            wp.synchronize_stream(wp.get_stream(self.device))
        except BaseException as cleanup:
            raise BaseExceptionGroup("Native backing service and quarantine failed", [failure, cleanup]) from failure

    def memory_report(self):
        """Report backing, native arrays and metadata by their authoritative owner.

        Reports remain readable after incomplete retirement. Released subowners
        are named explicitly and their reports are None, never a fabricated zero.
        Remaining storage and backing owners report their own retained resources.
        """
        result = {
            "directory": self._directory.memory_report(),
            "populations": [],
            "shared_backing": self._backing.memory_report() if self._backing is not None else None,
            "control_metadata_bytes": (
                self._healthy.capacity + self._always_permit.capacity + self._lifecycle_needed.capacity
            ),
            "scope": "Single-owner native array ledger; external fixture/oracle/JIT/graph/driver memory excluded",
        }
        seen_models = set()
        for group in self._populations:
            globals_report = [{"field": name, "bytes": array.capacity} for name, array in group.global_arrays.items()]
            model_arrays = []
            for name, array, _ in _data_arrays(group.model):
                if array.size and array.ptr not in seen_models:
                    seen_models.add(array.ptr)
                    model_arrays.append({"field": name, "bytes": array.capacity})
            result["populations"].append(
                {
                    "world_storage": group.world_storage.memory_report(),
                    "contact_storage": group.contact_storage.memory_report(),
                    "ccd_storage": group.ccd_storage.memory_report(),
                    "default_storage": group.default_storage.memory_report(),
                    "workspace_borrowed": group.workspace.memory_report() if group.workspace is not None else None,
                    "initialization": group.initialization_transfer.memory_report()
                    if group.initialization_transfer is not None
                    else None,
                    "compaction": group.compaction_transfer.memory_report()
                    if group.compaction_transfer is not None
                    else None,
                    "retired_subowners": [
                        name
                        for name in ("workspace", "initialization_transfer", "compaction_transfer")
                        if getattr(group, name) is None
                    ],
                    "graph_updates": group.updates.memory_report() if group.updates is not None else None,
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
                            group.contact_storage.protected_count,
                            group.ccd_storage.protected_count,
                        )
                    ),
                    "global_arrays": globals_report,
                    "global_array_bytes": sum(item["bytes"] for item in globals_report),
                    "immutable_model_arrays": model_arrays,
                    "immutable_model_array_bytes": sum(item["bytes"] for item in model_arrays),
                    "empty_fields": group.empty_fields,
                    "world_ready_capacity": group.world_ready_capacity,
                    "compacted_fields": list(group.compaction_transfer.field_names)
                    if group.compaction_transfer is not None
                    else None,
                }
            )
        return result

    def close(self, *, streams: tuple[wp.Stream, ...]):
        """Join consumers and close owned storage after every graph borrower is gone."""
        streams = tuple(streams)
        if not streams or any(not isinstance(stream, wp.Stream) or stream.device != self.device for stream in streams):
            raise ValueError("Retirement requires existing Warp streams on the population device")
        raw_streams = tuple(stream.cuda_stream for stream in streams)
        with wp.ScopedDevice(self.device):
            graph = self._graph() if self._graph is not None else None
            if graph is not None:
                raise RuntimeError("Destroy the native population graph before closing its owners")
            self._closed = True
            if self._directory is not None:
                self._directory.close(streams=streams)
            for group in self._populations:
                group.initialization_transfer = group.compaction_transfer = None
                if group.workspace is not None:
                    group.workspace.recorder = None
                group.workspace = group.data = None
            for group in self._populations:
                for owner in (group.world_storage, group.contact_storage, group.ccd_storage, group.default_storage):
                    if owner is not None:
                        owner.close(streams=raw_streams)
            self._populations.clear()
            self._views = ()
            if self._backing is not None:
                with self._backing.maintenance(streams=raw_streams):
                    self._backing.close()
