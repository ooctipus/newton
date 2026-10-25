# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Growable storage for per-rigid-contact device fields.

A rigid contact stream sizes several buffers by its contact rows: the
:class:`~newton.Contacts` arrays, the collision pipeline's sort, matching and
reduction buffers, and a solver's per-contact scratch. :class:`ContactPool`
holds them at one *ready* row count. Every rigid contact buffer of a model
shares the model's pool (:func:`model_contact_pool`), so a pipeline and the
solvers that consume its contacts grow together from the moment they exist.

Sizing: map big, then trim. The collision pipeline plans a *first mapping*
(:meth:`ContactPool.plan_first_mapping`): rows for the model's shape-pair
contact estimate, at most :data:`FIRST_MAPPING_SHARE` of the device memory
free when the pool was created across all its fields, so the first replays
drop nothing. Once the demand has been observed (the high-water mark stayed
within :data:`TRIM_BAND` for :data:`TRIM_AFTER_PASSES` collision passes), :meth:`ContactPool.service`
trims the pool to :data:`HEADROOM` rows per contact of the largest demand ever
recorded, never below :data:`MIN_READY_ROWS`. Past the mapped rows the pool
still grows from measured demand, without a recapture; a replay drops
contacts (and flags it) only when the demand outruns that growth. No configured
contact budget sizes anything: the only limit, the pool's *bound*, is the most
rows the platform could ever back, the device memory divided by the bytes one
row costs in the pool's fields, at most the rows every field's int32 Warp
indices reach (:func:`device_row_limit`, :meth:`ContactField.row_limit`).

Storage:

* On CUDA with the optional ``gpu_components`` package, rows are CUDA virtual
  memory: every field reserves addresses for the pool's bound when it is
  created, as if the whole device held contact rows, and maps physical pages
  for a prefix. Growth maps more pages behind the same addresses without
  joining the GPU: rows past every published count have no reader, so graphs
  keep replaying while the pool grows. The ready rows are published to the
  device too (:attr:`ContactPool.ready_rows`), in stream order after the new
  rows are initialized: rows past the old ones served in-flight replays as
  scratch (second halves, extra rows), so no replay launched before a growth
  may see the new rows. A trim works the other way round, also while graphs
  replay: it lowers the ready rows in stream order first, so no later replay
  addresses the pages past them, records an event, and unmaps those pages
  once the event completed (a later service), joining no stream.
* On the CPU, on CUDA without ``gpu_components`` (logged once), for fields with
  gradients and for rows created during a graph capture, rows are ordinary
  arrays of the ready rows that growth reallocates, copying their prefix. The
  superseded arrays stay alive, so a stale graph replay still addresses
  allocated memory; graphs that captured them must be recaptured.

Counts: the collision pipeline's writers count every contact they find and
drop the rows past :attr:`ContactPool.ready_rows`, which they read on the
device. After each pass :meth:`ContactPool.record_demand` records the count as
demand, in a sticky high-water mark that the same launch writes into pinned
host memory (mapped into the device's address space, so graphs replay no
copy), clamps the count to the ready rows and flags the overflow: a stream's
``rigid_contact_count`` never exceeds its rows, so consumers read every row
below it.

Graph capture: a consumer launches each contact-sized kernel through
:meth:`ContactPool.launch`. In a CUDA graph capture the launch is captured once
at the bound and bound to ``gpu_components``' device graph updater, which
every replay shrinks to the live contact count or the ready rows and uses to
rewrite row-count arguments from :attr:`ContactPool.ready_rows`. The pipeline
records an updater after each pass's narrow phase
(:meth:`ContactPool.begin_updates`), so the pass's consumers, and a solver
that steps its contacts, follow the count it produced; a launch in a capture
without an updater records one first. Such graphs need no recapture when the
pool grows or trims: :meth:`ContactPool.service`, outside capture, reads the
pinned high-water mark without synchronizing, maps more rows when it passes
:data:`GROW_THRESHOLD` of the ready rows and publishes them for the next
replay, or trims once it settled. The replay in which demand first passes
the ready rows drops the excess, and the capacity status flags it.

A capture is *unbound* when something in it sized work by the host's row count
instead: an unconverted consumer reading
:attr:`newton.Contacts.rigid_contact_max` while capturing, a launch with no
virtual memory to bind, or rows that growth reallocates.
:meth:`ContactPool.service` reports that such graphs must be recaptured once
the pool holds more rows than they were captured with.

A writer that produces contacts of its own (such as MuJoCo's or Kamino's
collision) declares how many it may write with :meth:`ContactPool.reserve`
(``Contacts.reserve_rigid_contacts``): outside graph capture the rows grow to at
least that many right away; during capture the request warns and waits for the
next service. Readers that keep per-contact buffers of their own grow them with
the pool (:meth:`ContactPool.follow`).
"""

from __future__ import annotations

import inspect
import logging
import math
import warnings
import weakref
from dataclasses import dataclass
from typing import Any

import warp as wp

logger = logging.getLogger(__name__)

MIN_READY_ROWS = 4096
"""The fewest ready rows a pool holds: a pool no pipeline planned starts here, and a trim stops here.

On the 21-task Isaac Lab suite (16,384 worlds) a floor of one 2 MiB granule's
rows (131,072) maps no more memory with CUDA virtual memory but slows the
contact-free tasks through passes that follow the rows (cartpole +6.9% under
patch friction on the GB300), so the floor stays a few thousand rows.
"""

FIRST_MAPPING_SHARE = 0.25
"""Share of the device memory free at a pool's creation that its first mapping may take, across all its fields.

The first mapping holds rows for the model's shape-pair contact estimate
(:meth:`ContactPool.plan_first_mapping`) unless that would map more than this
share; a field created later lowers the rows it may hold. Chosen on the 21-task
Isaac Lab suite (16,384 worlds, RTX PRO 6000, one task at a time) and the
hardware scene: the estimates exceed the demand 100-250 times where they are
large (keyboard-so101 127M rows for 0.52M contacts, hardware 87M for 1.3M), so
the share only caps those, and a quarter still holds 20 times their demand; at
half, keyboard-so101 under patch friction left 18 GB free after the environment
was created (77 GB start-up peak), little for a training run's own buffers
before the trim.
"""

TRIM_AFTER_PASSES = 200
"""Collision passes the recorded high-water mark must stay within :data:`TRIM_BAND` before :meth:`ContactPool.service` trims.

Counted on the device, as the passes complete: the host's service runs ahead
of the GPU, so service calls would count passes not run yet; no trim happens
before the first contact. Chosen on the demand of the 21-task Isaac Lab suite
(both friction laws) and of the hardware scene (128 worlds) recorded per pass:
the steep start-up rise ends within 80 passes everywhere (hardware's 256 parts
land from pass 13 to 83, 0 to 1.2M contacts; legged robots touch down by pass
40), after which the mark creeps by at most a few percent per 200 passes; the
window then trims every task with contacts within its first 900 passes (kuka
after 200, hardware after 270, anymald's creep after 860), and no recorded
demand later passed 1.17 times the mark it trimmed at (the rows keep 2.5 times).
"""

TRIM_BAND = 0.1
"""Growth of the high-water mark the trim's window tolerates: a mark past ``1 + TRIM_BAND`` times the one the
window started from starts a new window.

Without it a mark that creeps (anymald's from 54K to 170K contacts over 1,600 passes, hardware's by 0.1% per
frame after its parts landed) never holds still, and those tasks would keep their first mapping."""

HEADROOM = 2.5
"""Ready rows per contact of a measured demand.

A measured demand (a state measured before capture, a collision pass outside
capture that filled the rows, or a graph replay's high-water mark past
:data:`GROW_THRESHOLD` of the rows) sizes the pool to this many rows per
contact, so the demand may rise by ``HEADROOM * GROW_THRESHOLD`` (1.5) before
the pool grows again; a trim keeps this many rows per contact of the largest
demand ever recorded. Chosen on the 21-task suite's per-step demand (both
friction laws): measured after 20 eager steps, no task overflowed under graph
replay, 7 tasks grew 12 times in 400 environment steps of random actions as
their demand kept rising (anymald from 54,386 to 170,539 contacts), and the
pools mapped 4.8 GiB in all (patch), against 3.9 GiB for the hand-tuned budgets
of 19 of the tasks and 11.4 GiB at 64 rows per world.
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
"""Most elements a Warp array may hold: array shapes and kernel indices are int32."""

UPDATE_BINDINGS = 128
"""Kernels one device graph updater binds; a capture records another updater when one fills."""

CAPACITY_OVERFLOW = 1024
"""Capacity-status bit a pass sets when its contacts outnumbered the ready rows (the narrow phase's contacts bit)."""


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

    def row_limit(self) -> int:
        """Return the most contact rows the field's arrays can index (:data:`INDEX_LIMIT` elements)."""
        return (INDEX_LIMIT - self.extra) // self.per_contact


@wp.kernel(enable_backward=False)
def _fill_index(start: int, out: wp.array[wp.int32]):
    i = start + wp.tid()
    out[i] = wp.int32(i)


@wp.kernel(enable_backward=False)
def _record_demand(
    count: wp.array[wp.int32],
    ready_rows: wp.array[wp.int32],
    high_water: wp.array[wp.int32],
    passes: wp.array[wp.int32],
    mirror: wp.array[wp.int32],
    rows_used: wp.array[wp.int32],
    status: wp.array[wp.int32],
):
    demand = count[0]
    if high_water.shape[0] > 0:
        mirror[0] = wp.max(wp.atomic_max(high_water, 0, demand), demand)
        mirror[1] = demand
        mirror[2] = wp.atomic_add(passes, 0, 1) + 1
    rows = ready_rows[0]
    held = wp.min(demand, rows)
    if rows_used.shape[0] > 0:
        wp.atomic_max(rows_used, 0, held)
    if demand > rows:
        # Writers dropped the rows past the ready rows: consumers read every row below the count.
        count[0] = rows
        if status.shape[0] > 0:
            wp.atomic_or(status, 0, CAPACITY_OVERFLOW)


def capturing(device) -> bool:
    """Whether ``device`` records a graph now: CUDA graph capture, or Warp's CPU graph capture."""
    device = wp.get_device(device)
    return bool(device.is_capturing or (device.is_cuda and wp.get_stream(device).is_capturing))


def _capture_owner(device):
    """Return the Warp graph recording ``device``'s current stream, or ``None`` (eager, or a capture Warp does not own)."""
    if not device.is_cuda:
        return None
    return getattr(device, "captures", {}).get(wp.get_stream(device))


_VIRTUAL_MEMORY: list[Any] = []
"""After the first look: ``gpu_components``' ``(backing, fields, FieldSpec, graph, KernelParameterBinding)``, or
``None`` without a usable package."""

_GRAPH_FEATURES = {"prepare_updates": ("errors",), "launch": ("clamp_extent", "extent_scale")}
"""The ``gpu_components.graph`` parameters the pool passes; releases without them also predate its Warp 1.17 support."""


def _missing_graph_features(graph) -> list[str]:
    """Return the :data:`_GRAPH_FEATURES` that ``graph`` lacks, as ``function(parameter)``."""
    missing = []
    for name, parameters in _GRAPH_FEATURES.items():
        function = getattr(graph, name, None)
        known = inspect.signature(function).parameters if callable(function) else {}
        missing.extend(f"{name}({parameter})" for parameter in parameters if parameter not in known)
    return missing


def _virtual_memory():
    """Return ``gpu_components``' operation modules and records, or ``None`` without a usable optional package.

    Without it, rows on CUDA are ordinary arrays that growth reallocates and copies, as on the CPU. A release
    older than the one the pool follows counts as absent.
    """
    if not _VIRTUAL_MEMORY:
        try:
            from gpu_components import backing, fields, graph  # noqa: PLC0415
            from gpu_components.field_data import FieldSpec  # noqa: PLC0415
            from gpu_components.graph_data import KernelParameterBinding  # noqa: PLC0415
        except ImportError:
            _VIRTUAL_MEMORY.append(None)
            logger.info(
                "gpu_components is not installed: rigid contact rows on CUDA are ordinary arrays that growth "
                "reallocates and copies, as on the CPU, instead of CUDA virtual memory"
            )
        else:
            missing = _missing_graph_features(graph)
            if missing:
                _VIRTUAL_MEMORY.append(None)
                logger.warning(
                    "gpu_components at %s lacks %s: rigid contact rows on CUDA are ordinary arrays that growth "
                    "reallocates and copies, instead of CUDA virtual memory; install a newer gpu_components",
                    graph.__file__,
                    ", ".join(missing),
                )
            else:
                _VIRTUAL_MEMORY.append((backing, fields, FieldSpec, graph, KernelParameterBinding))
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
    cannot be reserved while a stream captures.
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
    # Retirement unmaps: join every reader first.
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


def _call(callback, contact_rows):
    if callback is None:
        return
    function = callback() if isinstance(callback, weakref.WeakMethod) else callback
    if function is not None:
        function(contact_rows)


class ContactRows:
    """Per-contact device fields of one owner, whose views hold the same number of contact rows.

    Create through :meth:`ContactPool.rows`. ``on_rows(contact_rows)`` runs
    after the views are created and after every growth or trim; owners assign
    the views there (:meth:`publish`) and refresh values derived from
    :attr:`rows`. A bound method is held weakly.

    :attr:`views` hold the ready rows and are safe to read on the host.
    :attr:`reserved` view each field's whole reservation (``reserve`` rows) for
    kernels whose loads are bounded by a view's length, such as tile loads, in
    a graph that outlives growth; only the ready rows of them are mapped.

    Args:
        fields: The fields of the owner.
        rows: Initial contact rows.
        reserve: The most contact rows the storage may hold: in CUDA virtual
            memory, the addresses each field reserves.
        device: The device of the fields.
        on_rows: Called with the rows after creation and every growth or trim.
    """

    def __init__(self, fields, *, rows: int, reserve: int, device, on_rows=None):
        self.fields = tuple(fields)
        self.rows = int(rows)
        self.reserve = int(reserve)
        self.device = wp.get_device(device)
        self.views: dict[str, wp.array] = {}
        self.reserved: dict[str, wp.array] = {}
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
            self.reserved = self.views
        _call(self._on_rows, self)

    @property
    def virtual(self) -> bool:
        """Whether the rows are CUDA virtual memory, which grows behind fixed addresses."""
        return self._mode == "virtual"

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
        backing_ops, fields_ops, FieldSpec = self._vm[:3]
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
                for field in members:
                    self.reserved[field.name] = storage.arrays[field.name]
        weakref.finalize(self, _retire, str(self.device), [storage for storage, _ in self._storages]).atexit = False
        self._bind_virtual_views(0)

    def _bind_virtual_views(self, kept_rows):
        # New pages are uninitialized: every row past the kept contact rows takes its fill, in stream order.
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

    def covered_rows(self) -> int:
        """Return the contact rows every field holds now: what its mapped pages cover, or its arrays' rows."""
        if self._mode != "virtual":
            return self.rows
        return min(
            self.reserve,
            *((storage.ready_rows - members[0].extra) // members[0].per_contact for storage, members in self._storages),
        )

    def covered_at(self, rows: int, granule: int) -> int:
        """Return the contact rows every field holds once its pages cover ``rows`` rows in whole granules."""
        if self._mode != "virtual":
            return rows
        covered = self.reserve
        for field in self.fields:
            size = wp.types.type_size_in_bytes(field.dtype)
            length = _round_up(field.length(rows) * size, granule) // size
            covered = min(covered, (length - field.extra) // field.per_contact)
        return covered

    def _withdraw(self, rows):
        """Shrink the views to ``rows`` rows and withdraw the storage's readiness to them, in stream order.

        Virtual rows only. The pages past them stay mapped for the replays already
        submitted; :meth:`_unmap` returns them once those completed.
        """
        fields_ops = self._vm[1]
        with wp.ScopedDevice(self.device):
            for storage, members in self._storages:
                field = members[0]
                fields_ops.withdraw(storage, field.per_contact * rows + field.extra)
        self._publish_rows(rows)

    def _unmap(self, rows):
        """Unmap the pages past ``rows`` rows that :meth:`_withdraw` left mapped, in a joined maintenance scope."""
        fields_ops = self._vm[1]
        for storage, members in self._storages:
            field = members[0]
            fields_ops.resize_backing(storage, field.per_contact * rows + field.extra, protected_count_host=0)

    def _map(self, rows, fields_ops):
        """Map ``rows`` rows of every virtual field: an append that joins no reader.

        A storage's ready rows are the rows its mapped granules cover, often more than
        the pool's: asking for fewer would be a shrink, so a request never goes below them.
        """
        for storage, members in self._storages:
            field = members[0]
            fields_ops.resize_backing(storage, max(storage.ready_rows, field.per_contact * rows + field.extra))

    def _publish_rows(self, rows):
        """Rebuild the views at ``rows`` rows after the storage holds them, then notify the owner.

        Rows past the kept contact rows (new rows, second halves, extra rows) take their fill.
        """
        previous = self.rows
        self.rows = rows
        if self._mode == "virtual":
            self._bind_virtual_views(min(previous, rows))
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
            self.reserved = self.views
        _call(self._on_rows, self)


def _grow_rows(members: list[ContactRows], rows: int) -> int:
    """Bring every member's rows to at least ``rows`` and return the rows they all hold.

    Virtual storage appends pages behind its fixed addresses without joining the
    GPU and holds every row its pages cover; the views are then published at the
    rows all members hold, the new rows initialized in stream order, ahead of any
    later launch.

    Raises:
        MemoryError: Before changing anything, if the device cannot back the rows.
        RuntimeError: During CUDA graph capture.
    """
    if not members:
        return rows
    held = [member for member in members if member.covered_rows() < rows]
    if not held:
        rows = min(member.covered_rows() for member in members)
        for member in members:
            if member.rows < rows:
                member._publish_rows(rows)
        return rows
    members_to_publish, members = members, held
    device = members[0].device
    if capturing(device):
        raise RuntimeError(
            "Rigid contact buffers cannot grow during graph capture; they grow in "
            "CollisionPipeline.service_capacity() or in a collision pass outside capture"
        )
    for member in members:
        if rows > member.reserve:
            raise MemoryError(f"Rigid contact buffers reserved for {member.reserve} rows cannot hold {rows}")
    virtual = [member for member in members if member._mode == "virtual"]
    granule = 1
    if virtual:
        _release_retired(device)
        owner, _, fields_ops = _backing(device, *virtual[0]._vm[:2])
        granule = int(owner.granularity_bytes)
    added = sum(member._added_bytes(rows, granule) for member in members)
    held = members[0].rows
    if device.is_cuda and added > int(device.free_memory):
        raise MemoryError(
            f"Growing rigid contact buffers from {held} to {rows} rows needs {added / 2**20:.1f} MiB more device "
            f"memory; {int(device.free_memory) / 2**20:.1f} MiB are free"
        )
    if virtual:
        try:
            with wp.ScopedDevice(device):
                for member in virtual:
                    member._map(rows, fields_ops)
        except MemoryError as error:
            raise MemoryError(
                f"Growing rigid contact buffers from {held} to {rows} rows ({added / 2**20:.1f} MiB more) exceeds "
                f"the device memory: {error}"
            ) from error
    rows = max(rows, min(member.covered_rows() for member in members_to_publish))
    for member in members_to_publish:
        if member.rows < rows:
            member._publish_rows(rows)
    return rows


class ContactPool:
    """The ready rows of a rigid contact stream's per-contact buffers, and their growth from measured demand.

    Args:
        device: The device of the buffers.
        rows: Initial ready rows; defaults to :data:`MIN_READY_ROWS`.
    """

    def __init__(self, *, device, rows: int | None = None):
        self.device = wp.get_device(device)
        self.ready = MIN_READY_ROWS if rows is None else max(int(rows), 0)
        # Queried once: the bound is recomputed when members are added, possibly during capture.
        self._device_memory = int(self.device.total_memory) if self.device.is_cuda else 0
        # The free memory the first mapping's share applies to: queried when the pool is created.
        self.free_memory_at_creation = (
            int(self.device.free_memory) if self.device.is_cuda and not capturing(self.device) else 0
        )
        """Device memory free when the pool was created [bytes]; 0 on the CPU."""
        # The model's contact estimate a pipeline planned (plan_first_mapping), while it sizes the rows.
        self._estimate: int | None = None
        self._first_mapping = False
        self._members: list[weakref.ReferenceType] = []
        # The bound of the members so far. It never rises: graphs captured at it keep covering the rows.
        self._bound = INDEX_LIMIT
        self._followers: list[weakref.ReferenceType] = []
        self.ready_rows = None
        """The ready rows on the device: writers' row limit and the rows bound launches follow (int32, shape 1)."""
        # The demand record, allocated here, outside any graph capture that records demand. A pool
        # created during a capture (for a buffer created lazily in it) records none: it grows by
        # reservations and by following its leader.
        self._high_water = None
        self._passes = None
        self._high_water_host = None
        self.rows_used = None
        """The most contacts any recorded pass held, at most the ready rows then (int32, shape 1, on the device):
        the rows a sweep over every used row covers. ``None`` for a pool created during capture."""
        if not capturing(self.device):
            self.ready_rows = wp.full(1, self.ready, dtype=wp.int32, device=self.device)
            self.rows_used = wp.zeros(1, dtype=wp.int32, device=self.device)
            self._high_water = wp.zeros(1, dtype=wp.int32, device=self.device)
            self._passes = wp.zeros(1, dtype=wp.int32, device=self.device)
            # [high-water mark, the last pass's demand, passes recorded]
            self._high_water_host = wp.zeros(3, dtype=wp.int32, device=self.device)
            if self.device.is_cuda:
                _RETIRED_HOST.clear()
                self._high_water_host = wp.zeros(3, dtype=wp.int32, device="cpu", pinned=True)
                weakref.finalize(self, _RETIRED_HOST.append, self._high_water_host).atexit = False
        # The record kernel writes the mark straight into the host copy, which pinned memory maps into the
        # device's address space: a copy node per collision pass costs a contact-free task 5-8% of its step.
        self._high_water_mirror = self._high_water_host
        if self._high_water_host is not None and self._high_water_host.device != self.device:
            self._high_water_mirror = wp.array(
                ptr=self._high_water_host.ptr, dtype=wp.int32, shape=(3,), device=self.device
            )
        self.growth_count = 0
        """Growths so far."""
        self.trim_count = 0
        """Trims completed so far."""
        # The high-water mark the trim watches, and the recorded passes when it last rose.
        self._still = (0, 0)
        # A trim between its withdrawal and its unmap: (event after the withdrawal, rows, rows before, bytes before).
        self._trim = None
        self._reserved = 0
        # Graph binding: the updater recorded last, by capture, and the captures that sized work by the host's rows.
        self._graph_ops = None
        self._update_status = None
        self._recording: tuple[weakref.ReferenceType, Any] | None = None
        # Live captures that sized work by the host's rows, with the ready rows they sized it by.
        self._unbound_captures: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._stale_on_growth = False
        vm = _virtual_memory() if self.device.is_cuda and self.ready_rows is not None else None
        if vm is not None:
            graph_ops = vm[3]
            self._update_status = wp.zeros(UPDATE_BINDINGS, dtype=wp.int32, device=self.device)
            try:
                # One preparation outside capture loads the updater; captures then prepare tables freely.
                graph_ops.prepare_updates(
                    self.ready_rows,
                    enable_count_maximum=INDEX_LIMIT,
                    binding_capacity=UPDATE_BINDINGS,
                    errors=self._update_status,
                )
            except Exception as error:  # No toolkit or bridge: launches stay unbound.
                logger.info("Rigid contact launches are not bound to graph updates: %s", error)
            else:
                self._graph_ops = (graph_ops, vm[4])

    @property
    def estimate(self) -> int | None:
        """The contact estimate a pipeline planned the first mapping for (:meth:`plan_first_mapping`), or ``None``."""
        return self._estimate

    @property
    def bound(self) -> int:
        """The most rows the device could back for the pool's fields (:func:`device_row_limit`).

        Every member's storage reserves at least it: a member reserves the bound
        when it is created, and the bound never rises, so a graph captured at it
        covers every row the pool may grow to."""
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
        """Return the bound with ``fields`` added: the device limit for all live bytes per row, at most every
        field's index limit and the bound so far."""
        members = self._live_members()
        groups = [member.fields for member in members]
        fields = tuple(fields)
        if fields:
            groups.append(fields)
        row_bytes = sum(field.row_bytes for group in groups for field in group)
        index_limit = min([INDEX_LIMIT, *(field.row_limit() for group in groups for field in group)])
        limit = device_row_limit(self.device, row_bytes, index_limit, memory=self._device_memory)
        return min(limit, self._bound)

    def rows(self, fields, *, on_rows=None) -> ContactRows:
        """Create per-contact fields that follow this pool's ready rows (see :class:`ContactRows`).

        Their bytes per row lower the pool's bound; their storage reserves the new bound.
        While the pool holds its first mapping, they also lower the rows it may hold
        (:meth:`plan_first_mapping`): the other fields shrink first.

        Raises:
            MemoryError: If the device cannot back the pool's ready rows of every field.
        """
        fields = tuple(fields)
        bound = self._limit(fields)
        if self._first_mapping and not capturing(self.device):
            self._size_first_mapping(sum(field.row_bytes for field in fields), bound)
        if self.ready > bound:
            row_bytes = self.row_bytes + sum(field.row_bytes for field in fields)
            raise MemoryError(
                f"{self.device} cannot back {self.ready} rigid contact rows of {row_bytes} bytes: it holds at most "
                f"{bound}"
            )
        rows = ContactRows(fields, rows=self.ready, reserve=bound, device=self.device, on_rows=on_rows)
        self._members.append(weakref.ref(rows))
        self._bound = bound
        return rows

    def plan_first_mapping(self, estimate: int) -> None:
        """Hold rows for ``estimate`` contacts from now on, outside capture: the model's shape-pair contact estimate.

        A collision pipeline plans this when it is created, so the first graph
        replays find the rows mapped. The rows are at least :data:`MIN_READY_ROWS`
        and, on CUDA, cost at most :data:`FIRST_MAPPING_SHARE` of the device memory
        free when the pool was created, in every field of the pool: a field created
        later lowers them (:meth:`rows`). They stay until :meth:`service` trims the
        pool to the measured demand. A pool plans once.

        Raises:
            MemoryError: If the device cannot back the rows.
        """
        if self._estimate is not None or capturing(self.device):
            return
        self._estimate = max(int(estimate), 0)
        self._first_mapping = True
        self._size_first_mapping(0, self.bound)

    def _first_rows(self, row_bytes: int, bound: int) -> int:
        """Return the rows of the first mapping for ``row_bytes`` bytes per row in all fields."""
        rows = max(MIN_READY_ROWS, self._estimate)
        if self.device.is_cuda and row_bytes > 0:
            rows = min(rows, max(MIN_READY_ROWS, int(FIRST_MAPPING_SHARE * self.free_memory_at_creation) // row_bytes))
        return min(rows, bound)

    def _size_first_mapping(self, added_row_bytes: int, bound: int) -> None:
        """Hold the first mapping for the live fields and ``added_row_bytes`` more per row, outside capture.

        Growth maps rows; a field that lowers them trims the others first, joining
        the stream (members are created during setup, which joins it anyway). Never
        below the rows a recorded demand or a reservation needs.
        """
        target = self._first_rows(self.row_bytes + added_row_bytes, bound)
        target = max(target, self._kept_rows(self.high_water))
        if target > self.ready:
            previous = self.ready
            self.grow(target)
            logger.info(
                "Rigid contact pool maps %d rows for the model's estimate of %d contacts (from %d; %d bytes per row, "
                "%.1f MiB free at creation): %.1f MiB",
                self.ready,
                self._estimate,
                previous,
                self.row_bytes + added_row_bytes,
                self.free_memory_at_creation / 2**20,
                self.mapped_bytes() / 2**20,
            )
        elif target < self.ready and self._trimmable():
            rows = self._trim_rows(target)
            if rows < self.ready:
                self._begin_trim(rows)
                wp.synchronize_event(self._trim[0])
                self._finish_trim()

    def settle(self) -> None:
        """Publish every row all live members hold, outside capture: CUDA virtual memory maps whole granules.

        The rows past the requested ones cost no memory, so writers fill them
        instead of dropping contacts. Owners call this once their members exist
        (:meth:`newton.CollisionPipeline.contacts`, :meth:`service`); a member
        created later maps as many rows. A buffer sized explicitly keeps its
        rows until it grows. Followers follow.
        """
        members = self._live_members()
        if not members or capturing(self.device):
            return
        covered = min(member.covered_rows() for member in members)
        if covered <= self.ready:
            return
        for member in members:
            member._publish_rows(covered)
        self._publish_ready(covered)
        # More rows to write: a graph that sized work by the old rows must learn of it as of a growth.
        self.growth_count += 1
        for pool in self._closure():
            if pool.ready < covered:
                pool.grow(covered)

    def _publish_ready(self, rows):
        self.ready = int(rows)
        if self.ready_rows is not None:
            self.ready_rows.fill_(self.ready)

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
        """Grow the ready rows of this pool and its followers to at least ``rows``, outside capture.

        Virtual rows grow without joining the GPU and keep their addresses; the
        new ready rows reach the device in stream order.

        Returns:
            Whether graphs that captured the buffers must be recaptured: some
            capture sized work by the host's rows, or rows were reallocated.

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
        for pool in pools:
            if pool.ready >= rows:
                continue
            members = pool._live_members()
            # A trim not yet unmapped is dropped: growth publishes the withdrawn rows again.
            pool._trim = None
            # Virtual rows hold every row their pages cover, often more than asked for.
            pool._publish_ready(_grow_rows(members, rows))
            pool.growth_count += 1
            pool._stale_on_growth |= any(not member.virtual for member in members)
        return self.needs_recapture

    @property
    def generation(self) -> int:
        """Growths of this pool and the pools that follow it, for owners that report each growth once."""
        return sum(pool.growth_count for pool in self._closure())

    @property
    def needs_recapture(self) -> bool:
        """Whether graphs that captured this pool's or its followers' buffers must be recaptured after growth.

        True when a live capture sized work by fewer rows than the pool holds
        now, or when growth reallocated rows; a graph whose contact launches are
        all bound follows the rows without a recapture.
        """
        return any(
            pool._stale_on_growth or any(rows < pool.ready for rows in pool._unbound_captures.values())
            for pool in self._closure()
        )

    def reserve(self, rows: int) -> bool:
        """Hold at least ``rows`` ready rows from now on, for a writer that produces its own contacts.

        Outside graph capture the pool and its followers grow now, to
        :func:`reserved_rows`, keeping their contents. During capture the request
        warns and waits for the next :meth:`service` (or a reservation outside
        capture); meanwhile writers stay bounded by the current rows and flag
        their overflow.

        Returns:
            Whether graphs that captured the buffers must be recaptured (see :meth:`grow`).

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
            # The writer fills the rows of capture time on every replay: its graph must be recaptured after growth.
            self.note_unbound()
            if rows > self._reserved:
                warnings.warn(
                    f"{rows} rigid contact rows were reserved during CUDA graph capture, with {self.ready} ready: "
                    "writers drop and flag the rows past them until the pool's service (or a reservation outside "
                    "capture) grows it",
                    RuntimeWarning,
                    stacklevel=3,
                )
            self._reserved = max(self._reserved, rows)
            return False
        self._reserved = max(self._reserved, rows)
        return self.grow(reserved_rows(rows, self.bound))

    def fit(self) -> bool:
        """Grow outside capture when the last recorded pass filled the rows, to :func:`sized_rows` for its count.

        A collision pass outside capture calls this after :meth:`record_demand`,
        before anything reads the rows. It synchronizes the stream to read the
        pass's count. Some writers stop counting once the rows are full, so a
        count at the ready rows counts as an overflow too.

        Returns:
            Whether the rows grew, so the pass must run again.

        Raises:
            MemoryError: If the rows are full at the device bound.
        """
        if self._high_water_host is None:
            return False
        if self.device.is_cuda:
            wp.synchronize_stream(wp.get_stream(self.device))
        count = int(self._high_water_host.numpy()[1])
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

        Returns:
            Whether graphs that captured the buffers must be recaptured (see :meth:`grow`).

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

    def record_demand(self, count: wp.array, rows: wp.array, status: wp.array | None = None) -> None:
        """Record a pass's contact count as demand, then clamp it to the rows the writers held.

        One launch, which graph capture records: it raises the sticky high-water
        mark and writes it and the pass's count into the pinned host copy (a
        pool created during capture records no demand), and when the count
        exceeds ``rows`` (the writers dropped the rows past them) sets it to
        ``rows`` and :data:`CAPACITY_OVERFLOW` in ``status``.

        Args:
            count: The pass's contact count (int32, shape 1), written by its writers.
            rows: The writers' row limit on the device: :attr:`ready_rows`, or the
                rows a pool created during capture holds.
            status: A capacity-status array (int32) whose element 0 flags the overflow, or ``None``.
        """
        empty = wp.empty(0, dtype=wp.int32, device=self.device)
        wp.launch(
            _record_demand,
            dim=1,
            inputs=[
                count,
                rows,
                empty if self._high_water is None else self._high_water,
                empty if self._high_water is None else self._passes,
                empty if self._high_water is None else self._high_water_mirror,
                empty if self.rows_used is None else self.rows_used,
                empty if status is None else status,
            ],
            device=self.device,
            record_tape=False,
        )

    @property
    def high_water(self) -> int:
        """The most contacts any recorded collision pass counted, as last copied to the host (no synchronization)."""
        return 0 if self._high_water_host is None else int(self._high_water_host.numpy()[0])

    @property
    def recorded_passes(self) -> int:
        """The collision passes recorded so far, as last copied to the host (no synchronization)."""
        return 0 if self._high_water_host is None else int(self._high_water_host.numpy()[2])

    def service(self) -> bool:
        """Grow from the recorded demand and reservations, or trim to it, outside capture, without synchronizing.

        Reads the pinned high-water mark, which lags the device by the passes
        still in flight, and grows to :func:`grown_rows` or to a reservation made
        during capture, logging the change. Bound graphs follow the new rows from
        their next replay. A high-water mark at or above the ready rows means a
        replay dropped contacts, which its capacity status flags; this warns.

        Without a growth, once the mark has stayed within :data:`TRIM_BAND` of
        where it stood :data:`TRIM_AFTER_PASSES` recorded passes earlier, it trims the rows to :data:`HEADROOM` rows per contact
        of the mark, at least :data:`MIN_READY_ROWS`, a reservation and the rows
        of any graph that sized work by the host's rows, in whole granules: it
        lowers the ready rows in stream order and unmaps the pages past them in a
        later call, once the replays submitted before have completed. Graphs keep
        replaying meanwhile, and need no recapture. Only CUDA virtual memory trims.

        Returns:
            Whether graphs that captured the buffers must be recaptured (see :meth:`grow`).
        """
        if capturing(self.device):
            return False
        self._finish_trim()
        self.settle()
        if self.growable:
            high_water = self.high_water
            target = grown_rows(high_water, self.ready, self.bound)
            if self._reserved > self.ready:
                target = max(target, reserved_rows(self._reserved, self.bound))
            if target > self.ready:
                previous = self.ready
                recapture = self._grow_logged(target, high_water)
                if high_water >= previous:
                    warnings.warn(
                        f"Rigid contact demand {high_water} reached the {previous} ready rows in a graph replay, "
                        f"which dropped the contacts past them (its capacity status flags it); the buffers grew to "
                        f"{target} rows for the next replay.",
                        RuntimeWarning,
                        stacklevel=3,
                    )
                return recapture
        self._consider_trim()
        return False

    # -- trim ----------------------------------------------------------------------------------------------------

    def _kept_rows(self, high_water: int) -> int:
        """Return the fewest rows a trim keeps: the demand's headroom, the floor, a reservation, unbound graphs."""
        rows = max(MIN_READY_ROWS, _round_up(HEADROOM * max(int(high_water), 0), ROW_QUANTUM))
        if self._reserved:
            rows = max(rows, reserved_rows(self._reserved, self.bound))
        return max(rows, *self._unbound_captures.values(), 0)

    def _trimmable(self) -> bool:
        """Whether the rows can trim now: virtual rows only, no trim pending, every graph's rows known."""
        members = self._live_members()
        return (
            bool(members)
            and all(member.virtual for member in members)
            and self._trim is None
            and not self._stale_on_growth
            and not capturing(self.device)
        )

    def _trim_rows(self, rows: int) -> int:
        """Return the ready rows of a trim to ``rows``: what the kept whole granules of every field cover."""
        members = self._live_members()
        owner, _, _ = _backing(self.device, *members[0]._vm[:2])
        granule = int(owner.granularity_bytes)
        return min(self.ready, *(member.covered_at(rows, granule) for member in members))

    def _consider_trim(self) -> None:
        """Trim once the high-water mark stayed within :data:`TRIM_BAND` for :data:`TRIM_AFTER_PASSES` recorded passes."""
        if self._high_water_host is None or not self._trimmable():
            return
        mark = self._high_water_host.numpy()
        high_water, passes = int(mark[0]), int(mark[2])
        # The window opens at the first contact: before it, the demand has not been observed.
        if high_water == 0 or high_water > (1.0 + TRIM_BAND) * self._still[0]:
            self._still = (high_water, passes)
            return
        if passes - self._still[1] < TRIM_AFTER_PASSES:
            return
        rows = self._trim_rows(self._kept_rows(high_water))
        if rows < self.ready:
            self._begin_trim(rows)
            self._first_mapping = False

    def _begin_trim(self, rows: int) -> None:
        """Lower the ready rows to ``rows`` in stream order and record the event after which their pages unmap."""
        previous, mapped = self.ready, self.mapped_bytes()
        for member in self._live_members():
            member._withdraw(rows)
        self._publish_ready(rows)
        event = wp.Event(self.device)
        wp.get_stream(self.device).record_event(event)
        self._trim = (event, rows, previous, mapped)

    def _finish_trim(self) -> None:
        """Unmap the pages a trim withdrew once the replays submitted before it completed; return spare memory."""
        if self._trim is None:
            return
        event, rows, previous, mapped = self._trim
        if not event.is_complete:
            return
        members = self._live_members()
        if members:
            owner, backing_ops, _ = _backing(self.device, *members[0]._vm[:2])
            with wp.ScopedDevice(self.device), backing_ops.maintenance(owner, events=(event.cuda_event,)):
                for member in members:
                    member._unmap(rows)
                backing_ops.trim(owner)
        self._trim = None
        self.trim_count += 1
        logger.info(
            "Rigid contact pool trimmed from %d to %d rows (contact high-water mark %d): %.1f -> %.1f MiB",
            previous,
            rows,
            self.high_water,
            mapped / 2**20,
            self.mapped_bytes() / 2**20,
        )

    def _grow_logged(self, target: int, demand: int) -> bool:
        previous = self.ready
        before = self.mapped_bytes()
        recapture = self.grow(target)
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
        return recapture

    # -- graph binding -----------------------------------------------------------------------------------------

    @property
    def bindable(self) -> bool:
        """Whether contact launches in a CUDA graph capture bind to device graph updates.

        Requires ``gpu_components``' graph bridge and virtual rows for every
        member: reallocated rows move, so graphs that use them cannot follow growth.
        """
        return self._graph_ops is not None and all(member.virtual for member in self._live_members())

    def _binding_owner(self):
        """Return the capture that a contact launch now binds into, or ``None`` when it does not bind."""
        owner = _capture_owner(self.device)
        if owner is None:
            return None
        if self.bindable:
            return owner
        self._note(owner)
        return None

    def _note(self, owner) -> None:
        # Growth cannot happen during a capture, so the rows a capture sized work by are the rows of its start.
        self._unbound_captures[owner] = self.ready

    def note_unbound(self) -> None:
        """Note that the current capture sized work by the host's row count: it must be recaptured after growth.

        :attr:`newton.Contacts.rigid_contact_max` calls this when read during a capture."""
        owner = _capture_owner(self.device)
        if owner is not None:
            self._note(owner)
        elif capturing(self.device):
            # A capture Warp does not own: nothing to follow, so every growth asks for a recapture.
            self._stale_on_growth = True

    def launch_rows(self) -> int:
        """Return the rows a contact-sized launch is sized for now.

        The bound in a capture whose launches bind (each replay then shrinks
        them, see :meth:`launch`), otherwise the ready rows.
        """
        return self.bound if self._binding_owner() is not None else self.ready

    def begin_updates(self) -> None:
        """Record a device graph updater, in a capture whose launches bind, for the launches that follow.

        A collision pass calls this right after its writers and :meth:`record_demand`,
        so its consumers and the solvers that step its contacts follow the count it
        produced. Outside capture this does nothing.
        """
        owner = self._binding_owner()
        if owner is not None:
            self._open_updates(owner)

    def _open_updates(self, owner):
        graph_ops, _ = self._graph_ops
        updates = graph_ops.prepare_updates(
            self.ready_rows,
            enable_count_maximum=self.bound,
            binding_capacity=UPDATE_BINDINGS,
            errors=self._update_status,
        )
        graph_ops.record_update(updates)
        self._recording = (weakref.ref(owner), updates)
        return updates

    def _updates_for(self, owner):
        recording = self._recording
        if recording is not None and recording[0]() is owner:
            updates = recording[1]
            if len(updates.bindings) < updates.binding_capacity:
                return updates
        return self._open_updates(owner)

    def launch(
        self,
        kernel,
        dim,
        *,
        inputs=(),
        outputs=(),
        extent: wp.array | None = None,
        axis: int = 0,
        scale: int = 1,
        clamp: bool = False,
        row_args: tuple[int, ...] = (),
        tiled: bool = False,
        block_dim: int = 0,
        max_blocks: int = 0,
    ) -> None:
        """Launch a kernel sized by contact rows, binding it in a capture so replays follow growth.

        Size ``dim`` (and any thread-count argument) for :meth:`launch_rows`,
        called right before. Outside capture this is an ordinary launch. In a
        capture whose launches bind, the launch is captured at the bound and
        bound to the current device graph updater: each replay runs axis ``axis``
        of ``dim`` at ``extent * scale`` (``scale`` launch items per counted
        item, e.g. 3 threads per contact), where ``extent`` is a device count of
        at most the ready rows (the live contact count, or :attr:`ready_rows` for
        work over every row), and rewrites the int32 arguments at ``row_args``
        (indices from 1, the first kernel argument) with the ready rows.
        ``clamp`` serves kernels that stride over their own count with the
        launched threads as their stride: a replay then runs at most the
        captured threads. A zero extent skips the kernel. A sweep over every
        row the stream used takes ``extent=rows_used``, and one over every row
        ``extent=ready_rows``. ``extent=None`` keeps the captured size, so
        ``dim`` must then not depend on the rows.
        """
        owner = self._binding_owner()
        if owner is None:
            # Contact bookkeeping is forward-only: an active autodiff tape records none of it.
            options = {"block_dim": block_dim} if block_dim else {}
            if tiled:
                wp.launch_tiled(
                    kernel,
                    dim=dim,
                    inputs=list(inputs),
                    outputs=list(outputs),
                    device=self.device,
                    record_tape=False,
                    **(options or {"block_dim": 256}),
                )
            else:
                if max_blocks:
                    options["max_blocks"] = max_blocks
                wp.launch(
                    kernel,
                    dim=dim,
                    inputs=list(inputs),
                    outputs=list(outputs),
                    device=self.device,
                    record_tape=False,
                    **options,
                )
            return
        graph_ops, parameter = self._graph_ops
        updates = self._updates_for(owner)
        graph_ops.launch(
            updates,
            kernel,
            dim,
            inputs=list(inputs),
            outputs=list(outputs),
            extent_axis=None if extent is None else axis,
            extent_source=extent,
            extent_scale=scale,
            clamp_extent=clamp,
            parameters=tuple(parameter(index, self.ready_rows, self.bound) for index in row_args),
            tiled=tiled,
            block_dim=block_dim,
            max_blocks=max_blocks,
            device=self.device,
        )

    # ----------------------------------------------------------------------------------------------------------

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
    rows, so they grow together.

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
