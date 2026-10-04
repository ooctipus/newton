# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compose native populations, mechanical W/C/D storage and one world directory.

Prepare from one-world templates, reserve descriptors directly, and capture once.
Native Data is authoritative; this root owns no dense Newton State mirror.
"""

import sys
import traceback
import weakref
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    import mujoco_warp

from gpu_components import backing as backing_ops
from gpu_components import directory as directory_ops
from gpu_components import fields as field_ops
from gpu_components import graph as graph_ops
from gpu_components import scratch as scratch_ops
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
from gpu_components.field_data import FieldSpec, FieldStorage

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
    """Borrow state and capacities for one homogeneous world population.

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
        """Return the published W/C/D prefix; consumers must follow backing-service stream ordering."""
        return self._borrow().world_ready_capacity


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
    step_bindings: object = None
    scratch_registry: object = None
    scratch_storages: dict = field(default_factory=dict)
    scratch_arrays: tuple = ()
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

    def domains(self, worlds: int):
        """Each storage with its row target for ``worlds`` ready worlds; scratch columns follow their domain."""
        for domain, owner, rows in (
            ("world", self.world_storage, worlds),
            ("contact", self.contact_storage, worlds * self.contact_quota),
            ("ccd", self.ccd_storage, worlds * self.ccd_quota),
        ):
            yield domain, owner, rows
            scratch = self.scratch_storages.get(domain)
            if scratch is not None:
                yield domain, scratch, rows

    @property
    def storages(self):
        """Every capacity-domain storage this population owns, scratch columns included."""
        return tuple(owner for _, owner, _ in self.domains(1))

    @property
    def world_ready_capacity(self):
        return min(owner.ready_rows // rows for _, owner, rows in self.domains(1))

    def record_physics(self, *, gated=True):
        """Record conditional native physics and poses on the current stream.

        ``gated=False`` records the same program without its conditional nodes,
        for a recording that is discarded: allocation during scratch discovery is
        a graph operation, and CUDA refuses it inside a conditional body.
        """
        import mujoco_warp as mjw

        if self.step_bindings.recording_failed:
            raise RuntimeError("The population program has a failed recording")
        if self.workspace.bindings is not None and self.workspace.bindings is not self.step_bindings:
            raise RuntimeError("Native workspace already has step bindings")
        self.workspace.bindings = self.step_bindings
        data = self.workspace.execution_data

        def record_callback(callback):
            if callback is None:
                return
            graph = graph_ops.current_capture(device=self.data.qpos.device)
            start = wp.capture_launch_count(graph)
            callback(self.view)
            end = wp.capture_launch_count(graph)
            if end != start:
                self.application_ranges.append((start, end))

        def step():
            record_callback(self.before_step)
            for _ in range(self.substeps):
                mjw.validate_step_workspace(self.workspace, self.model, self.data)
                mjw.step(self.model, data, scratch=self.workspace.scratch)
                record_callback(self.after_substep)
            mjw.validate_step_workspace(self.workspace, self.model, self.data)

        def guard(condition, on_true):
            if gated:
                wp.capture_if(condition, on_true=on_true)
            else:
                on_true()

        def step_and_poses():
            guard(self.step_condition, step)
            mjw.kinematics(self.model, data)

        try:
            if self.refresh_kinematics:
                guard(self.kinematics_condition, step_and_poses)
            else:
                guard(self.step_condition, step)
            if self.step_bindings.recording_failed:
                raise RuntimeError("The population program has a failed recording")
        except BaseException:
            self.step_bindings.recording_failed = True
            raise
        finally:
            self.workspace.bindings = None
            self.before_step = None
            self.after_substep = None


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
        self.device, self._backing, self._directory = device, None, None
        self._populations, self._views = [], ()
        self._healthy = self._always_permit = self._lifecycle_needed = None
        self._closed, self._service_failed, self._capture_attempted = False, False, False
        self._graph = None
        try:
            with wp.ScopedDevice(device):
                if memory_budget_bytes is not None:
                    self._backing = backing_ops.prepare(
                        memory_budget_bytes, device_ordinal=device.ordinal, expected_uuid=device.uuid
                    )
                backing = self._backing
                self._directory = directory_ops.allocate(
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
                    for name, array, shape in mjw.array_fields(template):
                        axis = shape[0] if shape else None
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
                    group.world_storage = field_ops.allocate(
                        capacity, live_count, fields=tuple(world_fields), backing=backing, initial_ready_count=initial
                    )
                    group.contact_storage = field_ops.allocate(
                        contact_cap,
                        wp.zeros(1, dtype=int, device=device),
                        fields=tuple(contact_fields),
                        backing=backing,
                        initial_ready_count=initial * group.contact_quota if backing else None,
                    )
                    group.ccd_storage = field_ops.allocate(
                        ccd_cap,
                        wp.zeros(1, dtype=int, device=device),
                        fields=tuple(ccd_fields),
                        backing=backing,
                        initial_ready_count=initial * group.ccd_quota if backing else None,
                    )
                    for owner in (group.world_storage, group.contact_storage, group.ccd_storage):
                        for name, array in owner.arrays.items():
                            field_ops.prepare_fill(owner, array, 0)
                            if name.startswith("workspace."):
                                scratch[name.removeprefix("workspace.")] = array
                            else:
                                arrays[name] = array
                    field_ops.prepare_fill(
                        group.contact_storage, group.contact_storage.arrays["contact.efc_address"], -1
                    )
                    for owner in (group.contact_storage, group.ccd_storage):
                        field_ops.zero(owner, count=owner.ready_count)
                    field_ops.fill(
                        group.contact_storage,
                        group.contact_storage.arrays["contact.efc_address"],
                        -1,
                        count=group.contact_storage.ready_count,
                    )
                    group.data = replace(
                        mjw.replace_arrays(template, arrays), nworld=capacity, naconmax=contact_cap, naccdmax=ccd_cap
                    )
                    field_ops.prepare_fill(group.world_storage, group.data.island_dofadr, model.nv)
                    group.step_bindings = mjw.StepBindings(
                        group.world_storage, group.contact_storage, group.ccd_storage
                    )
                    group.scratch_registry = scratch_ops.allocate_registry(device)
                    group.workspace = mjw.make_step_workspace(
                        model, group.data, arrays=scratch, bindings=group.step_bindings, scratch=group.scratch_registry
                    )
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
                    group.move_source_rows = self._directory.compaction.source_slots[start:end]
                    group.move_destination_rows = self._directory.compaction.destination_slots[start:end]
                    group.move_count = wp.zeros(1, dtype=int, device=device)
                    group.step_condition = wp.zeros(1, dtype=int, device=device)
                    group.kinematics_condition = wp.zeros(1, dtype=int, device=device)
                    group.view = MuJoCoWorldPopulation(prototype, group)
                    start = end
                self._views = tuple(group.view for group in self._populations)
                directory_ops.publish_admissible_slots(
                    self._directory, tuple(group.world_ready_capacity for group in self._populations)
                )
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
    def directory(self) -> InstanceDirectoryData:
        """Borrow published numeric relations without lifecycle mutation operations.

        Callers must not write these arrays. Submit commands through the captured
        population program; only this composition root owns the directory protocol.
        Metadata remains readable after quarantine or close for diagnostics.
        """
        return self._directory.data

    @property
    def batch_result(self) -> InstanceBatchResult:
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
        """Capture once, retaining caller buffers and all native owners.

        One discarded recording precedes the capture: the native programs name
        their scratch columns while recording, and that recording is where the
        population learns them (see ``_prepare_scratch_columns``).
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

        with wp.ScopedDevice(self.device):
            self._ensure_open()
            if self._capture_attempted:
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
            directory_ops.validate_buffers(self._directory, commands, results)
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
            application_ranges = [[] for _ in self._populations]
            population_ranges = [None] * len(self._populations)
            try:
                for group in self._populations:
                    group.workspace.bindings = None
                    group.substeps, group.refresh_kinematics = substeps, refresh_kinematics
                    group.step_bindings.updates = graph_ops.prepare_updates(
                        group.world_storage.protected_count,
                        enable_count_maximum=group.world_storage.capacity,
                        binding_capacity=512 * substeps,
                    )
                self._prepare_scratch_columns()
                for group, ranges in zip(self._populations, application_ranges, strict=True):
                    group.application_ranges = ranges
                    group.before_step, group.after_substep = before_step, after_substep
                # Warmed native programs use separate Data. Load lifecycle and row
                # transfer kernels before conditional capture, avoiding unrelated solvers.
                wp.load_module(module=__name__, device=self.device)
                wp.load_module(module=field_ops.__name__, device=self.device)
                with wp.ScopedCapture(
                    device=self.device,
                    force_module_load=False,
                    capture_mode=wp.CaptureMode.THREAD_LOCAL,
                    record_launches=True,
                    record_memory_operations=True,
                ) as capture:
                    directory_ops.begin(self._directory, commands)
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
                        directory_ops.admit(self._directory, commands)

                        def record_initialization(prototype, group):
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

                        graph_ops.capture_parallel(
                            lambda prototype=prototype, group=group: record_initialization(prototype, group)
                            for prototype, group in enumerate(self._populations)
                        )
                        directory_ops.publish(self._directory, commands, results)
                        directory_ops.plan_compaction(self._directory)

                        def record_compaction(prototype, group):
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
                                    self._directory.transaction,
                                    self._directory.compaction,
                                    prototype,
                                    self._healthy,
                                    group.compaction_transfer.status,
                                ],
                                device=self.device,
                            )

                        graph_ops.capture_parallel(
                            lambda prototype=prototype, group=group: record_compaction(prototype, group)
                            for prototype, group in enumerate(self._populations)
                        )
                        directory_ops.publish_compaction(self._directory)

                    wp.capture_if(self._lifecycle_needed, on_true=record_lifecycle)
                    # Every updater must finish and be checked before any prototype
                    # enters its conditional program, including native solver loops.
                    graph_ops.capture_parallel(
                        [
                            lambda group=group: graph_ops.record_update(group.step_bindings.updates)
                            for group in self._populations
                        ]
                    )
                    graph_ops.capture_parallel(
                        lambda group=group: wp.launch(
                            _guard_graph_updates,
                            group.step_bindings.updates.binding_capacity,
                            [
                                group.step_bindings.updates.errors,
                                group.step_bindings.updates.binding_count,
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

                    def record_population(prototype, group):
                        graph = graph_ops.current_capture(device=self.device)
                        start = wp.capture_launch_count(graph)
                        group.record_physics()
                        population_ranges[prototype] = (start, wp.capture_launch_count(graph))

                    graph_ops.capture_parallel(
                        [
                            lambda prototype=prototype, group=group: record_population(prototype, group)
                            for prototype, group in enumerate(self._populations)
                        ]
                    )
                graph = capture.graph
                del capture
                launches = wp.capture_get_launches(graph)
                memory_launches = {id(operation.launch) for operation in wp.capture_get_memory_operations(graph)}
                population_records = {
                    id(launches[index]) for start, end in population_ranges for index in range(start, end)
                }
                if memory_launches - population_records:
                    raise NotImplementedError("Lifecycle fills/copies require explicit storage admission")
                for group, (start, end), ranges in zip(
                    self._populations, population_ranges, application_ranges, strict=True
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
                        group.step_bindings.bindings.extend(
                            graph_ops.adopt_launches(
                                group.step_bindings.updates,
                                graph,
                                records,
                                extents=extents,
                                parameters=parameters,
                                fixed=fixed,
                            )
                        )
                    group.workspace.bindings = group.step_bindings
                    mjw.bind_step_program(
                        group.workspace,
                        graph,
                        tuple(launches[index] for index in range(start, end) if index not in application_indices),
                    )
                    graph_ops.bind(group.step_bindings.updates, graph, group.step_bindings.bindings)
                self._retain_graph(graph, commands, results, permit, *retain)
                graph_ops.instantiate_and_upload(self._populations[0].step_bindings.updates, graph)
                return graph
            except BaseException as failure:
                # This root cannot silently retry partially marked CUDA graph nodes.
                # Failed finalization already poisons any executable it created;
                # preserve that ownership while quarantining the population.
                if graph is not None and graph.graph_exec is None:
                    graph_ops.invalidate(graph)
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
                    group.workspace.bindings = None
                    group.before_step = group.after_substep = None
                    group.application_ranges = None

    def _prepare_scratch_columns(self):
        """Discover each program's scratch columns in a discarded recording, then allocate them.

        Numerical stages name their temporaries where they use them; nothing else
        declares them. One recording with discovering registries collects the
        names, shapes and count domains and is thrown away before anything runs.
        The world domain then gets one storage for its columns, sharing the world
        storage's capacity, live count and ready prefix; count-free temporaries
        are plain fixed arrays. The workspace is finalized on the complete relation,
        and prepared registries answer the real recording by name.
        """
        import mujoco_warp as mjw

        with wp.ScopedCapture(
            device=self.device,
            force_module_load=False,
            capture_mode=wp.CaptureMode.THREAD_LOCAL,
            record_launches=True,
            record_memory_operations=True,
        ) as capture:
            for group in self._populations:
                group.record_physics(gated=False)
        graph_ops.invalidate(capture.graph)
        del capture
        for group in self._populations:
            registry, execution = group.scratch_registry, group.workspace.execution_data
            scratch_ops.freeze(registry)
            for count in (execution.naconmax, execution.naccdmax):
                if scratch_ops.column_specs(registry, count):
                    raise NotImplementedError(
                        "Contact and CCD scratch columns need their domain's ready count as the admitted count; "
                        "only world-domain scratch columns are prepared"
                    )
            columns = {}
            specs = scratch_ops.column_specs(registry, execution.nworld)
            if specs:
                owner = group.world_storage
                storage = field_ops.allocate(
                    owner.capacity,
                    owner.protected_count,
                    fields=specs,
                    backing=self._backing,
                    initial_ready_count=owner.ready_rows if self._backing is not None else None,
                )
                group.scratch_storages["world"] = storage
                columns.update(storage.arrays)
            plain = scratch_ops.plain_declarations(registry)
            group.scratch_arrays = tuple(
                wp.empty(scratch_ops.bounded_shape(declaration.shape), dtype=declaration.dtype, device=self.device)
                for declaration in plain
            )
            columns.update(zip((declaration.name for declaration in plain), group.scratch_arrays, strict=True))
            scratch_ops.prepare(registry, columns)
            group.step_bindings.extra_storages = tuple(group.scratch_storages.values())
            group.step_bindings.extra_fixed_arrays = group.scratch_arrays
            mjw.finalize_step_workspace(group.workspace)

    def _retain_graph(self, graph, *buffers):
        """Retain explicit borrowers on the population's single captured graph."""
        self._ensure_open()
        previous = self._graph() if self._graph is not None else None
        if previous is not None and previous is not graph:
            raise RuntimeError("Native population already belongs to another graph")
        directory_ops.retain_graph(self._directory, graph, self, *buffers)
        for group in self._populations:
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
            for owner in group.storages:
                field_ops.retain_graph(owner, graph, group.workspace, group.step_condition, group.kinematics_condition)
        self._graph = weakref.ref(graph)
        return graph

    def grow_backing(self, world_ready_capacities: tuple[int, ...], *, streams: tuple[wp.Stream, ...]):
        """Grow coherent W/C/D prefixes, joining only for historical address reuse.

        Service requires no active capture on this device. The caller excludes
        new submissions during service. Fresh suffix mapping
        leaves earlier admitted accesses untouched. Ready counts, contact/CCD
        initialization and directory admission are queued on the current stream
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
        with wp.ScopedDevice(self.device):
            self._ensure_open()
            if wp.get_device(self.device).is_capturing:
                raise RuntimeError("Backing growth requires execution outside graph capture")
            if self._backing is None:
                raise RuntimeError("This population has fixed backing")
            world_ready_capacities = tuple(world_ready_capacities)
            if len(world_ready_capacities) != len(self._populations) or any(
                type(n) is not int
                or not group.world_ready_capacity <= n <= group.world_storage.capacity
                or n * group.contact_quota > group.contact_storage.capacity
                or n * group.ccd_quota > group.ccd_storage.capacity
                for n, group in zip(world_ready_capacities, self._populations, strict=True)
            ):
                raise ValueError("Growth must preserve ready worlds and fit prepared W/C/D capacities")
            streams = tuple(streams)
            if not streams or any(
                not isinstance(stream, wp.Stream) or stream.device != self.device for stream in streams
            ):
                raise ValueError("Backing service requires existing Warp streams on the population device")
            services = [
                (group, owner, target)
                for group, n in zip(self._populations, world_ready_capacities, strict=True)
                for _, owner, target in group.domains(n)
                if target > owner.ready_rows
            ]
            if not services:
                return
            current = wp.get_stream(self.device)
            try:
                for stream in streams:
                    if stream.cuda_stream != current.cuda_stream:
                        current.wait_stream(stream)
                fresh = all(
                    tuple(field_ops.can_map_backing_without_join(owner, target) for _, owner, target in services)
                )
            except BaseException as failure:
                self._quarantine_service(failure)
                raise
            budget_rejected = False
            try:
                with (
                    nullcontext()
                    if fresh
                    else backing_ops.maintenance(self._backing, streams=tuple(stream.cuda_stream for stream in streams))
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
                        if owner is group.contact_storage or owner is group.ccd_storage:
                            field_ops.zero(owner, count=owner.ready_count, start=old_ready)
                            if owner is group.contact_storage:
                                field_ops.fill(
                                    owner,
                                    owner.arrays["contact.efc_address"],
                                    -1,
                                    count=owner.ready_count,
                                    start=old_ready,
                                )
                    status = directory_ops.publish_admissible_slots(
                        self._directory, tuple(group.world_ready_capacity for group in self._populations)
                    )
                    wp.launch(_guard_backing_publication, 1, [status, self._healthy], device=self.device)
                    if not fresh:
                        wp.synchronize_stream(current)
            except MemoryError as failure:
                if not budget_rejected:
                    self._quarantine_service(failure)
                raise
            except BaseException as failure:
                self._quarantine_service(failure)
                raise

    def resize_backing(
        self, world_ready_capacities: tuple[int, ...], *, streams: tuple[wp.Stream, ...], spare_bytes: int | None = None
    ):
        """Join once, service all W/C/D prefixes, then publish one coherent ready set.

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
            with backing_ops.maintenance(self._backing, streams=tuple(stream.cuda_stream for stream in streams)):
                directory_ops.withdraw_admissible_slots(self._directory, world_ready_capacities)
                live_counts = self._directory.data.live_count.numpy()
                budget_rejected = False
                try:
                    services = [
                        (group, owner, target, int(live_counts[prototype]) if domain == "world" else 0)
                        for prototype, (group, n) in enumerate(
                            zip(self._populations, world_ready_capacities, strict=True)
                        )
                        for domain, owner, target in group.domains(n)
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
                        if (
                            owner is group.contact_storage or owner is group.ccd_storage
                        ) and owner.ready_rows > old_ready:
                            field_ops.zero(owner, count=owner.ready_count, start=old_ready)
                            if owner is group.contact_storage:
                                field_ops.fill(
                                    owner,
                                    owner.arrays["contact.efc_address"],
                                    -1,
                                    count=owner.ready_count,
                                    start=old_ready,
                                )
                    directory_ops.publish_admissible_slots(
                        self._directory, tuple(group.world_ready_capacity for group in self._populations)
                    )
                    wp.synchronize_stream(wp.get_stream(self.device))
                    if spare_bytes is not None:
                        backing_ops.trim(self._backing, keep_bytes=spare_bytes)
                except MemoryError as failure:
                    if not budget_rejected:
                        self._quarantine_service(failure)
                        raise
                    try:
                        directory_ops.publish_admissible_slots(
                            self._directory, tuple(group.world_ready_capacity for group in self._populations)
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
        import mujoco_warp as mjw

        result = {
            "directory": directory_ops.memory_report(self._directory),
            "populations": [],
            "shared_backing": backing_ops.memory_report(self._backing) if self._backing is not None else None,
            "control_metadata_bytes": (
                self._healthy.capacity + self._always_permit.capacity + self._lifecycle_needed.capacity
            ),
            "scope": "Single-owner native array ledger; external fixture/oracle/JIT/graph/driver memory excluded",
        }
        seen_models = set()
        for group in self._populations:
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
                    "ccd_storage": field_ops.memory_report(group.ccd_storage),
                    "default_storage": field_ops.memory_report(group.default_storage),
                    "scratch_storages": {
                        domain: field_ops.memory_report(owner) for domain, owner in group.scratch_storages.items()
                    },
                    "scratch_arrays_bytes": sum(array.capacity for array in group.scratch_arrays),
                    "workspace_borrowed": mjw.step_workspace_memory_report(group.workspace)
                    if group.workspace is not None
                    else None,
                    "initialization": field_ops.transfer_memory_report(group.initialization_transfer)
                    if group.initialization_transfer is not None
                    else None,
                    "compaction": field_ops.transfer_memory_report(group.compaction_transfer)
                    if group.compaction_transfer is not None
                    else None,
                    "retired_subowners": [
                        name
                        for name in ("workspace", "initialization_transfer", "compaction_transfer", "step_bindings")
                        if getattr(group, name) is None
                    ],
                    "graph_updates": graph_ops.memory_report(group.step_bindings.updates)
                    if group.step_bindings is not None and group.step_bindings.updates is not None
                    else None,
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
                directory_ops.close(self._directory, streams=streams)
            for group in self._populations:
                group.initialization_transfer = group.compaction_transfer = None
                if group.workspace is not None:
                    group.workspace.bindings = None
                # The prepared registry holds the scratch column views; drop them before their storage closes.
                group.workspace = group.data = group.step_bindings = group.scratch_registry = None
                group.scratch_arrays = ()
            for group in self._populations:
                for owner in (*group.storages, group.default_storage):
                    if owner is not None:
                        field_ops.close(owner, streams=raw_streams)
            self._populations.clear()
            self._views = ()
            if self._backing is not None:
                with backing_ops.maintenance(self._backing, streams=raw_streams):
                    backing_ops.close(self._backing)
