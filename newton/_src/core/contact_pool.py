# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Growable storage for per-rigid-contact device fields.

A rigid contact stream sizes several buffers by its contact rows: the
:class:`~newton.Contacts` arrays, the collision pipeline's sort, matching and
reduction buffers, and a solver's per-contact scratch. :class:`ContactPool`
holds them at one *ready* row count. Every rigid contact buffer of a model
shares the model's pool (:func:`model_contact_pool`), so a pipeline and the
solvers that consume its contacts grow together from the moment they exist.
A pool starts at a floor (:data:`MIN_READY_ROWS`) and is sized from measured
demand: :data:`HEADROOM` rows per contact of the state a graph starts from
(:meth:`ContactPool.size`, through
:meth:`newton.CollisionPipeline.service_capacity`), and the same headroom over
the high-water mark when a graph replay's demand outgrows it.
There is no configured contact budget: the only limit, the pool's *bound*, is
the most rows the platform could ever back, the device memory divided by the
bytes one row costs in the pool's fields, at most the rows every field's int32
Warp indices reach (:func:`device_row_limit`, :meth:`ContactField.row_limit`).
Physical memory runs out long before it: a growth that the device cannot back
raises :class:`MemoryError` before changing anything.

Each owner declares its fields once as a :class:`ContactRows`. Views of the
fields always hold the pool's ready rows (times a field's rows per contact,
plus fixed extra rows), so every launch and in-kernel bound that reads an
array's length follows the ready count. Growth keeps the contents of the rows
that existed and gives new rows the values the fields were created with.

Storage:

* On CUDA with the optional ``gpu_components`` package, rows are CUDA virtual
  memory: one reservation per field of the pool's bound when the rows were
  created, of which a prefix is mapped. The bound costs addresses, not memory.
  Growth maps more pages behind the same addresses, so a graph captured before
  a growth still addresses mapped memory. Recapture it before its next replay:
  launch sizes and array lengths are fixed at capture, and a stale replay would
  treat rows that growth made contact rows as its scratch.
* On the CPU, on CUDA without ``gpu_components`` (logged once), for fields with
  gradients and for rows created during a graph capture, rows are ordinary
  arrays of the ready rows that growth reallocates, copying their prefix. The
  superseded arrays stay alive with their rows, so a stale graph replay still
  addresses allocated memory; recapture after a growth all the same.

The demand signal is a sticky device high-water mark of the contact count
(:meth:`ContactPool.record_demand`), never reset by a collision pass, which
the same launch writes into pinned host memory (mapped into the device's
address space, so graphs replay no copy). :meth:`ContactPool.service` reads it without synchronizing and grows
the pool when the high-water mark passes :data:`GROW_THRESHOLD` of the ready
rows. Under graph replay, overflow stays loud: writers drop contacts past the
ready rows and the capacity status reports it; the service then grows and
warns. A collision pass outside capture checks its own count and grows before
anything reads the rows (:meth:`newton.CollisionPipeline.collide`).

A writer that produces contacts of its own (such as MuJoCo's or Kamino's
collision) declares how many it may write with :meth:`ContactPool.reserve`
(``Contacts.reserve_rigid_contacts``): outside graph capture the rows grow to at
least that many right away; during capture the request warns and waits for the
next service. Its kernels stay bounded by the ready rows, keep the demand in
the count and flag the overflow, as the pipeline's writer does. Readers that
keep per-contact buffers of their own grow them with the pool
(:meth:`ContactPool.follow`).
"""

from __future__ import annotations

import logging
import math
import warnings
import weakref
from dataclasses import dataclass
from typing import Any

import warp as wp

logger = logging.getLogger(__name__)

MIN_READY_ROWS = 4096
"""Ready rows of a pool before any demand is measured: a floor, not a guess.

On the 21-task Isaac Lab suite (16,384 worlds) a floor of one 2 MiB granule's
rows (131,072) maps no more memory with CUDA virtual memory but slows the
contact-free tasks through passes that follow the rows (cartpole +6.9% under
patch friction on the GB300), so the floor stays a few thousand rows. A scene
with more contacts is sized from its measured demand: a collision pass outside
capture grows before anything reads the rows, and
:meth:`newton.CollisionPipeline.service_capacity` with a state sizes the rows
before a graph is captured.
"""

HEADROOM = 2.5
"""Ready rows per contact of a measured demand.

A measured demand (a state measured before capture, a collision pass outside
capture that filled the rows, or a graph replay's high-water mark past
:data:`GROW_THRESHOLD` of the rows) sizes the pool to this many rows per
contact, so the demand may rise by ``HEADROOM * GROW_THRESHOLD`` (1.5) before
the pool grows again and graphs are recaptured. Chosen on the 21-task suite's
per-step demand (both friction laws): measured after 20 eager steps, no task
overflowed under graph replay, 7 tasks grew 12 times in 400 environment steps
of random actions as their demand kept rising (anymald from 54,386 to 170,539
contacts), and the pools mapped 4.8 GiB in all (patch), against 3.9 GiB for the
hand-tuned budgets of 19 of the tasks and 11.4 GiB at 64 rows per world.
"""

GROW_THRESHOLD = 0.6
"""Grow when a graph replay's high-water mark passes this fraction of the ready rows.

A graph replay cannot grow the rows, and the service reads the mark a step
late, so the demand may rise by ``1 / GROW_THRESHOLD`` (1.67) over two steps
before a contact is dropped. On the 21-task suite it rose by at most 1.23 over
two steps once 20 eager steps had measured its first contacts."""

ROW_QUANTUM = 64
"""Rows that grown and reserved ready counts round up to.

Launches follow the ready rows, so they stay close to the data; on CUDA the
mapping itself rounds each field up to whole 2 MiB allocation granules.
"""

INDEX_LIMIT = 2**31 - 1
"""Most elements a Warp array may hold: array shapes and kernel indices are int32.

On CUDA, ``gpu_components`` also views each field's whole reservation as one
byte array, so a CUDA virtual-memory field reserves at most this many bytes.
"""


def _round_up(value: float, quantum: int) -> int:
    return math.ceil(value / quantum) * quantum


def reserved_rows(rows: int, bound: int) -> int:
    """Return the ready rows that hold a reservation of ``rows``: rounded up to :data:`ROW_QUANTUM`, at most ``bound``."""
    return min(int(bound), _round_up(rows, ROW_QUANTUM))


def sized_rows(demand: int, ready: int, bound: int) -> int:
    """Return the ready rows a pool should hold for a measured demand of ``demand`` contacts.

    :data:`HEADROOM` rows per contact, rounded up to :data:`ROW_QUANTUM` rows
    and at most ``bound``; never fewer than ``ready``, as a pool never shrinks.

    Args:
        demand: The most contacts measured.
        ready: The rows the pool holds now.
        bound: The most rows the pool may grow to.
    """
    return max(int(ready), min(int(bound), _round_up(HEADROOM * max(int(demand), 0), ROW_QUANTUM)))


def grown_rows(high_water: int, ready: int, bound: int) -> int:
    """Return the ready rows a pool should hold for a graph replay's contact high-water mark.

    Past :data:`GROW_THRESHOLD` of ``ready`` the pool is sized for the mark
    (:func:`sized_rows`); otherwise it keeps ``ready``.

    Args:
        high_water: The most contacts any collision pass counted.
        ready: The rows the pool holds now.
        bound: The most rows the pool may grow to.
    """
    if high_water <= GROW_THRESHOLD * ready:
        return ready
    return sized_rows(high_water, ready, bound)


def device_row_limit(device, row_bytes: int, index_limit: int = INDEX_LIMIT, *, memory: int | None = None) -> int:
    """Return the most rows of ``row_bytes`` bytes that ``device`` could ever back.

    On CUDA that is the device memory divided by the bytes per row; everywhere
    it is at most ``index_limit``, the rows the fields' int32 indices reach
    (:meth:`ContactField.row_limit`).

    Args:
        device: The device of the rows.
        row_bytes: Bytes one contact row costs in every field of the pool.
        index_limit: The most rows every field's arrays can index.
        memory: The device's total memory [bytes], if already known; querying
            it is a driver call, which a pool avoids during graph capture.
    """
    device = wp.get_device(device)
    if device.is_cuda and row_bytes > 0:
        total = int(device.total_memory) if memory is None else int(memory)
        return min(int(index_limit), total // int(row_bytes))
    return int(index_limit)


@dataclass(frozen=True)
class ContactField:
    """One per-contact device field of a :class:`ContactRows`."""

    name: str
    """Field name; :meth:`ContactRows.publish` assigns the view to the target attribute of this name."""
    dtype: Any
    """Warp element type."""
    fill: Any = 0
    """Value of rows that were never written, also given to rows added by growth; ``None`` for the row's index."""
    per_contact: int = 1
    """Rows per contact slot, e.g. 2 for a radix sort's key and scratch halves.

    Growth keeps the first row per contact; the other rows are scratch and take ``fill`` again."""
    extra: int = 0
    """Fixed scratch rows past the per-contact rows; growth moves them and gives them ``fill`` again."""
    requires_grad: bool = False
    """Allocate a gradient; such rows are ordinary arrays that growth reallocates."""

    def length(self, rows: int) -> int:
        """Return the view length for ``rows`` contact rows."""
        return self.per_contact * rows + self.extra

    @property
    def row_bytes(self) -> int:
        """Device bytes one contact row costs in this field, its gradient included."""
        return self.per_contact * wp.types.type_size_in_bytes(self.dtype) * (2 if self.requires_grad else 1)

    def row_limit(self, virtual: bool) -> int:
        """Return the most contact rows the field's arrays can index (:data:`INDEX_LIMIT`).

        Args:
            virtual: Whether the field is CUDA virtual memory, whose reservation
                is also viewed as bytes.
        """
        elements = INDEX_LIMIT // (wp.types.type_size_in_bytes(self.dtype) if virtual else 1)
        return (elements - self.extra) // self.per_contact


@wp.kernel(enable_backward=False)
def _fill_index(start: int, out: wp.array[wp.int32]):
    i = start + wp.tid()
    out[i] = wp.int32(i)


@wp.kernel(enable_backward=False)
def _record_high_water(count: wp.array[wp.int32], high_water: wp.array[wp.int32], mirror: wp.array[wp.int32]):
    demand = count[0]
    mirror[0] = wp.max(wp.atomic_max(high_water, 0, demand), demand)


def capturing(device) -> bool:
    """Whether ``device`` records a graph now: CUDA graph capture, or Warp's CPU graph capture."""
    device = wp.get_device(device)
    return bool(device.is_capturing or (device.is_cuda and wp.get_stream(device).is_capturing))


_VIRTUAL_MEMORY: list[Any] = []
"""After the first look: ``gpu_components``' ``(backing, fields, FieldSpec)``, or ``None`` without the package."""


def _virtual_memory():
    """Return ``gpu_components``' ``(backing, fields, FieldSpec)``, or ``None`` without the optional package.

    Without it, rows on CUDA are ordinary arrays that growth reallocates and copies, as on the CPU.
    """
    if not _VIRTUAL_MEMORY:
        try:
            from gpu_components import backing, fields  # noqa: PLC0415
            from gpu_components.field_data import FieldSpec  # noqa: PLC0415
        except ImportError:
            _VIRTUAL_MEMORY.append(None)
            logger.info(
                "gpu_components is not installed: rigid contact rows on CUDA are ordinary arrays that growth "
                "reallocates and copies, as on the CPU, instead of CUDA virtual memory"
            )
        else:
            _VIRTUAL_MEMORY.append((backing, fields, FieldSpec))
    return _VIRTUAL_MEMORY[0]


_BACKINGS: dict[str, tuple] = {}
"""One CUDA virtual-memory ledger per device, shared by every pool on it, with the ``gpu_components`` modules
that made it: ``(owner, backing, fields)``."""
_RETIRED: list[tuple[str, Any]] = []
"""Virtual-memory storages of collected rows, released at the next pool operation outside capture."""
_RETIRED_HOST: list[wp.array] = []
"""Pinned demand copies of collected pools, released when the next pool is created outside capture.

Freeing pinned memory synchronizes the device, which a graph capture refuses, and a pool may be
collected at any time, during a capture too."""


def _virtual_rows(device, fields):
    """Return the ``gpu_components`` modules that hold ``fields`` in CUDA virtual memory, or ``None``.

    Rows are ordinary arrays on the CPU, without ``gpu_components``, for fields with gradients, and when
    created during a graph capture (a buffer a capture-first loop creates lazily): CUDA virtual memory
    cannot be mapped while a stream captures.
    """
    device = wp.get_device(device)
    if not device.is_cuda or any(field.requires_grad for field in fields) or capturing(device):
        return None
    return _virtual_memory()


def _backing(device, backing_ops, fields_ops) -> tuple:
    """Return ``device``'s virtual-memory ledger ``(owner, backing, fields)``, prepared on first use."""
    key = str(device)
    ledger = _BACKINGS.get(key)
    if ledger is None:
        with wp.ScopedDevice(device):
            owner = backing_ops.prepare(int(device.total_memory), device_ordinal=int(device.ordinal))
        ledger = _BACKINGS[key] = (owner, backing_ops, fields_ops)
    return ledger


def _retire(device_key: str, storages: list) -> None:
    # Runs from a finalizer: queue only, without CUDA calls.
    _RETIRED.extend((device_key, storage) for storage in storages)


def _release_retired(device) -> None:
    """Release the queued storages of ``device`` that no view references any more."""
    key = str(device)
    pending = [storage for owner_key, storage in _RETIRED if owner_key == key]
    if not pending or capturing(device):
        return
    owner, backing_ops, fields_ops = _BACKINGS[key]
    wp.synchronize_device(device)
    stream = wp.get_stream(device).cuda_stream
    released = []
    with wp.ScopedDevice(device):
        for storage in pending:
            try:
                fields_ops.close(storage, streams=(stream,))
                released.append(storage)
            except Exception:
                # A view outside the collected owner still references it; retry later.
                pass
        if released:
            with backing_ops.maintenance(owner, streams=(stream,)):
                backing_ops.trim(owner)
    _RETIRED[:] = [entry for entry in _RETIRED if entry[1] not in released]


def _mark_bound_stale(pool) -> None:
    # Runs when a member is collected: the bound is recomputed at its next use.
    if pool is not None:
        pool._bound_stale = True


def _call(callback, contact_rows):
    if callback is None:
        return
    function = callback() if isinstance(callback, weakref.WeakMethod) else callback
    if function is not None:
        function(contact_rows)


class ContactRows:
    """Per-contact device fields of one owner, whose views hold the same number of contact rows.

    Create through :meth:`ContactPool.rows`. ``on_rows(contact_rows)`` runs
    after the views are created and after every growth; owners assign the
    views there (:meth:`publish`) and refresh values derived from
    :attr:`rows`. A bound method is held weakly.

    Args:
        fields: The fields of the owner.
        rows: Initial contact rows.
        reserve: The most contact rows the storage may hold: in CUDA virtual
            memory, the addresses each field reserves.
        device: The device of the fields.
        on_rows: Called with the rows after creation and every growth.
    """

    def __init__(self, fields, *, rows: int, reserve: int, device, on_rows=None):
        self.fields = tuple(fields)
        self.rows = int(rows)
        self.reserve = int(reserve)
        self.device = wp.get_device(device)
        self.views: dict[str, wp.array] = {}
        self._on_rows = weakref.WeakMethod(on_rows) if hasattr(on_rows, "__self__") else on_rows
        self._storages = []
        # Arrays a reallocating growth replaced; a stale graph replay may still address them.
        self._superseded: list[wp.array] = []
        self._vm = _virtual_rows(self.device, self.fields)
        if self._vm is not None:
            self._mode = "virtual"
            self._allocate_virtual()
        else:
            self._mode = "realloc"
            for field in self.fields:
                self.views[field.name] = self._array(field, field.length(self.rows))
        _call(self._on_rows, self)

    @property
    def row_bytes(self) -> int:
        """Device bytes one contact row costs in these fields."""
        return sum(field.row_bytes for field in self.fields)

    def publish(self, target) -> None:
        """Assign each field's current view to the attribute of ``target`` named after the field."""
        for name, view in self.views.items():
            setattr(target, name, view)

    def mapped_bytes(self) -> int:
        """Return the device bytes the rows hold now."""
        if self._mode != "virtual":
            return sum(view.capacity * (2 if view.requires_grad else 1) for view in self.views.values())
        fields_ops = self._vm[1]
        return sum(fields_ops.memory_report(storage)["mapped_dense_bytes"] for storage, _ in self._storages)

    def reserved_bytes(self) -> int:
        """Return the device addresses the rows reserve (CUDA virtual memory), or the bytes they hold."""
        if self._mode != "virtual":
            return self.mapped_bytes()
        fields_ops = self._vm[1]
        return sum(fields_ops.memory_report(storage)["virtual_dense_bytes"] for storage, _ in self._storages)

    def _array(self, field, length):
        array = wp.empty(length, dtype=field.dtype, device=self.device, requires_grad=field.requires_grad)
        self._initialize(array, field, 0, length)
        return array

    @staticmethod
    def _initialize(array, field, start, end):
        if end <= start:
            return
        if field.fill is None:
            wp.launch(_fill_index, dim=end - start, inputs=[start, array], device=array.device)
        elif field.fill == 0:
            array[start:end].zero_()
        else:
            array[start:end].fill_(field.fill)

    def _allocate_virtual(self):
        backing_ops, fields_ops, FieldSpec = self._vm
        _release_retired(self.device)
        owner, _, _ = _backing(self.device, backing_ops, fields_ops)
        groups: dict[tuple[int, int], list] = {}
        for field in self.fields:
            groups.setdefault((field.per_contact, field.extra), []).append(field)
        with wp.ScopedDevice(self.device):
            for (per_contact, extra), members in groups.items():
                storage = fields_ops.allocate(
                    per_contact * self.reserve + extra,
                    wp.zeros(1, dtype=wp.int32),
                    fields=tuple(FieldSpec(field.name, (), field.dtype, False) for field in members),
                    backing=owner,
                    initial_ready_count=per_contact * self.rows + extra,
                )
                self._storages.append((storage, members))
        weakref.finalize(self, _retire, str(self.device), [storage for storage, _ in self._storages]).atexit = False
        self._bind_virtual_views(0)

    def _bind_virtual_views(self, kept_rows):
        # New pages are uninitialized: every row past the kept contact rows takes its fill.
        for storage, members in self._storages:
            for field in members:
                view = storage.arrays[field.name][: field.length(self.rows)]
                self._initialize(view, field, kept_rows, view.shape[0])
                self.views[field.name] = view

    def _added_bytes(self, rows, granule) -> int:
        """Return the device bytes growing to ``rows`` rows would map or allocate."""
        if self._mode != "virtual":
            return sum(
                (field.length(rows) - field.length(self.rows))
                * wp.types.type_size_in_bytes(field.dtype)
                * (2 if field.requires_grad else 1)
                for field in self.fields
            )
        added = 0
        for field in self.fields:
            size = wp.types.type_size_in_bytes(field.dtype)
            added += _round_up(field.length(rows) * size, granule) - _round_up(field.length(self.rows) * size, granule)
        return added

    def _map(self, rows, fields_ops):
        """Map ``rows`` rows of every virtual field; the caller holds the backing's maintenance scope."""
        for storage, members in self._storages:
            field = members[0]
            fields_ops.resize_backing(storage, field.per_contact * rows + field.extra, protected_count_host=0)

    def _publish_rows(self, rows):
        """Rebuild the views at ``rows`` rows after the storage holds them, then notify the owner."""
        previous = self.rows
        self.rows = rows
        if self._mode == "virtual":
            self._bind_virtual_views(previous)
        else:
            for field in self.fields:
                old = self.views[field.name]
                array = wp.empty(
                    field.length(rows), dtype=field.dtype, device=self.device, requires_grad=field.requires_grad
                )
                if previous:
                    wp.copy(array, old, count=previous)
                self._initialize(array, field, previous, array.shape[0])
                self.views[field.name] = array
                self._superseded.append(old)
        _call(self._on_rows, self)


def _grow_rows(members: list[ContactRows], rows: int) -> None:
    """Bring every member's rows to at least ``rows``: map virtual storage in one maintenance scope, then publish.

    Raises:
        MemoryError: Before changing anything, if the device cannot back the rows.
        RuntimeError: During CUDA graph capture.
    """
    members = [member for member in members if member.rows < rows]
    if not members:
        return
    device = members[0].device
    if capturing(device):
        raise RuntimeError(
            "Rigid contact buffers cannot grow during graph capture; grow them before capturing "
            "(CollisionPipeline.service_capacity(), or one collision pass outside capture)"
        )
    for member in members:
        if rows > member.reserve:
            raise MemoryError(f"Rigid contact buffers reserved for {member.reserve} rows cannot hold {rows}")
    virtual = [member for member in members if member._mode == "virtual"]
    granule = 1
    if virtual:
        _release_retired(device)
        owner, backing_ops, fields_ops = _backing(device, *virtual[0]._vm[:2])
        granule = int(owner.granularity_bytes)
    added = sum(member._added_bytes(rows, granule) for member in members)
    held = members[0].rows
    if device.is_cuda and added > int(device.free_memory):
        raise MemoryError(
            f"Growing rigid contact buffers from {held} to {rows} rows needs {added / 2**20:.1f} MiB more device "
            f"memory; {int(device.free_memory) / 2**20:.1f} MiB are free"
        )
    if virtual:
        # Join every stream that may touch the rows (graph replays, solver side streams) before remapping.
        wp.synchronize_device(device)
        stream = wp.get_stream(device).cuda_stream
        try:
            with wp.ScopedDevice(device), backing_ops.maintenance(owner, streams=(stream,)):
                for member in virtual:
                    member._map(rows, fields_ops)
        except MemoryError as error:
            raise MemoryError(
                f"Growing rigid contact buffers from {held} to {rows} rows ({added / 2**20:.1f} MiB more) exceeds "
                f"the device memory: {error}"
            ) from error
    for member in members:
        member._publish_rows(rows)


class ContactPool:
    """The ready rows of a rigid contact stream's per-contact buffers, and their growth from measured demand.

    Args:
        device: The device of the buffers.
        rows: Initial ready rows; defaults to :data:`MIN_READY_ROWS`.
    """

    def __init__(self, *, device, rows: int | None = None):
        self.device = wp.get_device(device)
        self.ready = MIN_READY_ROWS if rows is None else max(int(rows), 0)
        # Queried once: the bound is recomputed when a member is collected, possibly during capture.
        self._device_memory = int(self.device.total_memory) if self.device.is_cuda else 0
        self._members: list[weakref.ReferenceType] = []
        # The bound of the live members, recomputed after one is collected.
        self._bound = INDEX_LIMIT
        self._bound_stale = False
        self._followers: list[weakref.ReferenceType] = []
        # The demand record, allocated here, outside any graph capture that records demand. A pool
        # created during a capture (for a buffer created lazily in it) records none: it grows by
        # reservations and by following its leader.
        self._high_water = None
        self._high_water_host = None
        if not capturing(self.device):
            self._high_water = wp.zeros(1, dtype=wp.int32, device=self.device)
            self._high_water_host = self._high_water
            if self.device.is_cuda:
                _RETIRED_HOST.clear()
                self._high_water_host = wp.zeros(1, dtype=wp.int32, device="cpu", pinned=True)
                weakref.finalize(self, _RETIRED_HOST.append, self._high_water_host).atexit = False
        # The record kernel writes the mark straight into the host copy, which pinned memory maps into the
        # device's address space: a copy node per collision pass costs a contact-free task 5-8% of its step.
        self._high_water_mirror = self._high_water_host
        if self._high_water_host is not self._high_water:
            self._high_water_mirror = wp.array(
                ptr=self._high_water_host.ptr, dtype=wp.int32, shape=(1,), device=self.device
            )
        self.growth_count = 0
        """Growths so far; each one requires recapturing graphs that use the pool."""
        self._reserved = 0
        # Growths of live rows in this pool (see generation).
        self._changes = 0

    @property
    def bound(self) -> int:
        """The most rows the device could back for the pool's live fields (:func:`device_row_limit`).

        Every live member's storage holds it: a member reserves the bound when it is created, and
        the bound only rises again when a member that lowered it is collected."""
        if self._bound_stale:
            self._bound_stale = False
            self._bound = self._limit(())
        return self._bound

    @property
    def row_bytes(self) -> int:
        """Device bytes one contact row costs in the pool's live fields."""
        return sum(member.row_bytes for member in self._live_members())

    @property
    def growable(self) -> bool:
        """Whether the pool can still grow."""
        return self.ready < self.bound

    def _limit(self, fields) -> int:
        """Return the bound of the live members plus ``fields``: the device limit for all their bytes per row,
        at most what every field can index and what every live member reserved."""
        members = self._live_members()
        groups = [(member.fields, member._mode == "virtual") for member in members]
        fields = tuple(fields)
        if fields:
            groups.append((fields, _virtual_rows(self.device, fields) is not None))
        row_bytes = sum(field.row_bytes for group, _ in groups for field in group)
        index_limit = min([INDEX_LIMIT, *(field.row_limit(virtual) for group, virtual in groups for field in group)])
        limit = device_row_limit(self.device, row_bytes, index_limit, memory=self._device_memory)
        return min([limit, *(member.reserve for member in members)])

    def rows(self, fields, *, on_rows=None) -> ContactRows:
        """Create per-contact fields that follow this pool's ready rows (see :class:`ContactRows`).

        Their bytes per row lower the pool's bound while they live; their storage reserves the new bound.

        Raises:
            MemoryError: If the device cannot back the pool's ready rows of every field.
        """
        fields = tuple(fields)
        bound = self._limit(fields)
        if self.ready > bound:
            row_bytes = self.row_bytes + sum(field.row_bytes for field in fields)
            raise MemoryError(
                f"{self.device} cannot back {self.ready} rigid contact rows of {row_bytes} bytes: it holds at most "
                f"{bound}"
            )
        rows = ContactRows(fields, rows=self.ready, reserve=bound, device=self.device, on_rows=on_rows)
        pool = weakref.ref(self)
        self._members.append(weakref.ref(rows, lambda _, pool=pool: _mark_bound_stale(pool())))
        self._bound = bound
        return rows

    def follow(self, other: ContactPool) -> None:
        """Make ``other`` grow with this pool: its rows stay at least this pool's ready rows.

        Grows ``other`` now if it holds fewer rows, which cannot happen during
        graph capture. Following is idempotent.

        Raises:
            MemoryError: If the device cannot back ``other`` at this pool's ready rows.
            RuntimeError: If ``other`` must grow during graph capture.
        """
        if other is self:
            return
        if all(ref() is not other for ref in self._followers):
            if self.ready > other.ready:
                other.grow(self.ready)
            self._followers.append(weakref.ref(other))

    def grow(self, rows: int) -> bool:
        """Grow the ready rows of this pool and its followers to at least ``rows``.

        Returns:
            Whether any live buffer changed. Graphs that captured them must then
            be recaptured to use the new rows (:attr:`generation` counts it).

        Raises:
            MemoryError: Before changing anything, if the device cannot back the rows.
        """
        rows = int(rows)
        pools = self._closure()
        if all(pool.ready >= rows for pool in pools):
            return False
        for pool in pools:
            if rows > pool.bound and pool.ready < rows:
                raise MemoryError(
                    f"{pool.device} cannot back {rows} rigid contact rows of {pool.row_bytes} bytes: it holds at "
                    f"most {pool.bound}"
                )
        growing = [(pool, pool._live_members()) for pool in pools if pool.ready < rows]
        _grow_rows([member for _, members in growing for member in members], rows)
        for pool, members in growing:
            pool.ready = rows
            pool.growth_count += 1
            pool._changes += int(bool(members))
        return any(members for _, members in growing)

    def reserve(self, rows: int) -> bool:
        """Hold at least ``rows`` ready rows from now on, for a writer that produces its own contacts.

        Outside graph capture the pool and its followers grow now, to
        :func:`reserved_rows`, keeping their contents. During capture the request
        warns and waits for the next :meth:`service` (or a reservation outside
        capture); meanwhile writers stay bounded by the current rows and flag
        their overflow.

        Returns:
            Whether live buffers grew now; graphs that captured them must be
            recaptured (:meth:`service` reports it too).

        Raises:
            MemoryError: If the device cannot back ``rows`` rows of the pool's fields.
        """
        rows = int(rows)
        if rows <= self.ready:
            return False
        if rows > self.bound:
            raise MemoryError(
                f"{self.device} cannot back the {rows} rigid contact rows reserved, of {self.row_bytes} bytes each: "
                f"it holds at most {self.bound}"
            )
        if capturing(self.device):
            if rows > self._reserved:
                warnings.warn(
                    f"{rows} rigid contact rows were reserved during CUDA graph capture, with {self.ready} ready: "
                    "writers drop and flag the rows past them until the pool's service (or a reservation outside "
                    "capture) grows it; recapture then",
                    RuntimeWarning,
                    stacklevel=3,
                )
            self._reserved = max(self._reserved, rows)
            return False
        self._reserved = max(self._reserved, rows)
        return self.grow(reserved_rows(rows, self.bound))

    def fit(self, count: int) -> bool:
        """Grow outside capture when a pass's ``count`` filled the rows, to :func:`sized_rows` for it.

        A collision pass outside capture calls this with its count before
        anything reads the rows. Some writers stop counting once the rows are
        full, so a count at the ready rows counts as an overflow too.

        Returns:
            Whether the rows grew, so the pass must run again.

        Raises:
            MemoryError: If the rows are full at the device bound.
        """
        if count <= 0 or count < self.ready:
            return False
        target = sized_rows(count, self.ready, self.bound)
        if target <= self.ready:
            raise MemoryError(
                f"{count} rigid contacts fill the {self.ready} rows {self.device} can back for these buffers"
            )
        self._grow_logged(target, count)
        return True

    def size(self, demand: int) -> bool:
        """Hold :func:`sized_rows` for a measured demand of ``demand`` contacts, outside capture; never shrink.

        :meth:`newton.CollisionPipeline.service_capacity` calls this with the
        contacts of the state a graph starts from, before the graph is captured.

        Returns:
            Whether live buffers grew; graphs that captured them must be recaptured.

        Raises:
            MemoryError: If the device cannot back ``demand`` rows of the pool's fields.
            RuntimeError: During graph capture.
        """
        if capturing(self.device):
            raise RuntimeError("Rigid contact buffers cannot be sized during graph capture; size them before capturing")
        demand = int(demand)
        if demand > self.bound:
            raise MemoryError(
                f"{self.device} cannot back the {demand} rigid contacts measured, of {self.row_bytes} bytes each: it "
                f"holds at most {self.bound}"
            )
        target = sized_rows(demand, self.ready, self.bound)
        if target <= self.ready:
            return False
        return self._grow_logged(target, demand)

    def record_demand(self, count: wp.array) -> None:
        """Record the contact count in the sticky high-water mark and in its pinned host copy.

        One launch, which graph capture records: it writes the host copy directly.
        """
        if self._high_water is None or not self.growable:
            return
        wp.launch(
            _record_high_water,
            dim=1,
            inputs=[count, self._high_water, self._high_water_mirror],
            device=self.device,
            record_tape=False,
        )

    @property
    def high_water(self) -> int:
        """The most contacts any recorded collision pass counted, as last copied to the host (no synchronization)."""
        return 0 if self._high_water_host is None else int(self._high_water_host.numpy()[0])

    def service(self) -> bool:
        """Grow from the recorded demand and reservations, outside capture; return whether live buffers grew.

        Reads the pinned high-water mark without synchronizing, so it lags the
        device by the passes still in flight. Grows to :func:`grown_rows`, or to
        a reservation made during capture, logging the change. A high-water mark
        at or above the ready rows means a graph replay may have dropped
        contacts: its capacity status stays flagged and this warns.
        """
        if capturing(self.device) or not self.growable:
            return False
        high_water = self.high_water
        target = grown_rows(high_water, self.ready, self.bound)
        if self._reserved > self.ready:
            target = max(target, reserved_rows(self._reserved, self.bound))
        if target <= self.ready:
            return False
        previous = self.ready
        grew = self._grow_logged(target, high_water)
        if high_water >= previous:
            warnings.warn(
                f"Rigid contact demand {high_water} reached the {previous} ready rows in a graph replay, which may "
                f"have dropped contacts (its capacity status stays flagged); the buffers grew to {target} rows, and "
                "graphs that use them must be recaptured. Size them before capturing: "
                "CollisionPipeline.service_capacity(state) with the state the graph starts from.",
                RuntimeWarning,
                stacklevel=3,
            )
        return grew

    @property
    def generation(self) -> int:
        """Growths of live rows in this pool and the pools that follow it.

        A graph captured at another generation must be recaptured before its next replay.
        """
        return sum(pool._changes for pool in self._closure())

    def _grow_logged(self, target: int, demand: int) -> bool:
        previous = self.ready
        before = self.mapped_bytes()
        grew = self.grow(target)
        logger.info(
            "Rigid contact pool grew from %d to %d rows (demand %d, reserved %d, device bound %d): %.1f -> %.1f MiB",
            previous,
            target,
            demand,
            self._reserved,
            self.bound,
            before / 2**20,
            self.mapped_bytes() / 2**20,
        )
        return grew

    def mapped_bytes(self) -> int:
        """Return the device bytes this pool's and its followers' rows hold now."""
        return sum(member.mapped_bytes() for pool in self._closure() for member in pool._live_members())

    def reserved_bytes(self) -> int:
        """Return the device addresses this pool's own rows reserve (diagnostics)."""
        return sum(member.reserved_bytes() for member in self._live_members())

    def _live_members(self) -> list[ContactRows]:
        members = [ref() for ref in self._members]
        self._members = [ref for ref, member in zip(self._members, members, strict=True) if member is not None]
        return [member for member in members if member is not None]

    def _closure(self) -> list[ContactPool]:
        pools, pending = [], [self]
        while pending:
            pool = pending.pop()
            if any(pool is seen for seen in pools):
                continue
            pools.append(pool)
            followers = [ref() for ref in pool._followers]
            pool._followers = [
                ref for ref, follower in zip(pool._followers, followers, strict=True) if follower is not None
            ]
            pending.extend(follower for follower in followers if follower is not None)
        return pools


_MODEL_POOLS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
"""Each model's shared pools, by device alias."""


def model_contact_pool(model, device=None) -> ContactPool:
    """Return the pool that every rigid contact buffer of ``model`` on ``device`` shares.

    The model's collision pipelines with their :class:`~newton.Contacts` and the
    per-contact state of the solvers that step the model hold the same ready
    rows, so they grow together: a solver constructed before the pipeline is
    sized with it when the pipeline measures a state before a graph is captured
    (:meth:`newton.CollisionPipeline.service_capacity`), never during the capture.

    Args:
        model: The model whose rigid contacts the buffers hold.
        device: The device of the buffers; defaults to the model's.
    """
    device = wp.get_device(device if device is not None else model.device)
    try:
        pools = _MODEL_POOLS.setdefault(model, {})
    except TypeError:
        # A model that cannot be weakly referenced keeps a pool per owner.
        return ContactPool(device=device)
    pool = pools.get(device.alias)
    if pool is None:
        pool = pools[device.alias] = ContactPool(device=device)
    return pool
