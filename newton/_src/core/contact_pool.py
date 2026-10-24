# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Growable storage for per-rigid-contact device fields.

A rigid contact stream sizes several buffers by its contact capacity: the
:class:`~newton.Contacts` arrays, the collision pipeline's sort, matching and
reduction buffers, and a solver's per-contact scratch. :class:`ContactPool`
holds them at one *ready* row count, starts it small and grows it from the
measured demand, up to a *bound* (the resolved ``rigid_contact_max``).

Each owner declares its fields once as a :class:`ContactRows`. Views of the
fields always hold the pool's ready rows (times a field's rows per contact,
plus fixed extra rows), so every launch and in-kernel bound that reads an
array's length follows the ready count. Growth keeps the contents of the rows
that existed and gives new rows the values the fields were created with.

Storage depends on whether the rows can grow:

* Rows that start at the bound never grow and are ordinary arrays.
* On CUDA, growable rows are CUDA virtual memory: one reservation per field,
  sized by the bound, of which a prefix is mapped (``gpu_components``). The
  bound costs addresses, not memory. Growth maps more pages behind the same
  addresses, so a graph captured before a growth still addresses mapped
  memory. Recapture it before its next replay: launch sizes and array lengths
  are fixed at capture, and a stale replay would treat rows that growth made
  contact rows as its scratch.
* On the CPU, growable rows are reallocated and their prefix copied.

The demand signal is a sticky device high-water mark of the contact count
(:meth:`ContactPool.record_demand`), never reset by a collision pass and
copied each pass into pinned host memory by an asynchronous copy that graphs
capture. :meth:`ContactPool.service` reads it without synchronizing and grows
the pool when the high-water mark passes half the ready rows. Overflow stays
loud: writers drop contacts past the ready rows as before and the capacity
status reports it; the service then grows and warns.

A writer with a fixed contact capacity (a solver that writes up to N contacts of
its own into the buffer, such as MuJoCo's or Kamino's collision) declares N with
:meth:`ContactPool.reserve` (``Contacts.reserve_rigid_contacts``): outside graph
capture the rows grow to at least N right away; during capture the request warns
and waits for the next service. Its kernels stay bounded by the ready rows, keep
the demand in the count and flag the overflow, as the pipeline's writer does.
Readers that keep per-contact buffers of their own either grow them with the
pool (:meth:`ContactPool.follow`) or size them by the bound.
"""

from __future__ import annotations

import logging
import warnings
import weakref
from dataclasses import dataclass
from typing import Any

import warp as wp

logger = logging.getLogger(__name__)

READY_ROWS_PER_WORLD = 64
"""Initial ready rows per world.

Measured live rigid contacts per world on the 21-task Isaac Lab suite at
16,384 worlds peak at about 30 (keyboard-so101 29.2, anymald 10.7,
franka-cabinet 11.6, franka 0.12), so 64 per world holds every task without
growth while franka-cabinet's estimated 25.2M-slot bound shrinks to 1M ready
rows.
"""

GROW_THRESHOLD = 0.5
"""Grow when the high-water mark passes this fraction of the ready rows."""

GROW_FACTOR = 2
"""Grow to this multiple of the high-water mark."""

ROW_QUANTUM = 524_288
"""Rows that the initial and grown ready counts round up to.

One 2 MiB CUDA allocation granule holds this many 4-byte rows, so a ready
count that is a multiple of it maps whole granules for every field whose
element size is a multiple of 4 bytes. Pools whose bound is at most this many
rows start at their bound and never grow.
"""


def _round_up(value: int, quantum: int) -> int:
    return -(-int(value) // quantum) * quantum


def initial_rows(bound: int, world_count: int) -> int:
    """Return the initial ready rows of a pool: :data:`READY_ROWS_PER_WORLD` per world, rounded, at most ``bound``.

    Args:
        bound: The most rows the pool may grow to.
        world_count: Number of worlds of the model.
    """
    return min(int(bound), _round_up(READY_ROWS_PER_WORLD * max(int(world_count), 1), ROW_QUANTUM))


def reserved_rows(rows: int, bound: int) -> int:
    """Return the ready rows that hold a reservation of ``rows``: rounded up to :data:`ROW_QUANTUM`, at most ``bound``."""
    return min(int(bound), _round_up(rows, ROW_QUANTUM))


def grown_rows(high_water: int, ready: int, bound: int) -> int:
    """Return the ready rows a pool should hold for a contact high-water mark.

    Past :data:`GROW_THRESHOLD` of ``ready`` the pool grows to
    :data:`GROW_FACTOR` times the high-water mark, rounded up to
    :data:`ROW_QUANTUM` rows and at most ``bound``; otherwise it keeps ``ready``.
    The pool never shrinks.

    Args:
        high_water: The most contacts any collision pass counted.
        ready: The rows the pool holds now.
        bound: The most rows the pool may grow to.
    """
    if high_water <= GROW_THRESHOLD * ready:
        return ready
    return max(ready, min(int(bound), _round_up(GROW_FACTOR * high_water, ROW_QUANTUM)))


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
    """Allocate a gradient; only rows that cannot grow support it."""

    def length(self, rows: int) -> int:
        """Return the view length for ``rows`` contact rows."""
        return self.per_contact * rows + self.extra


@wp.kernel(enable_backward=False)
def _fill_index(start: int, out: wp.array[wp.int32]):
    i = start + wp.tid()
    out[i] = wp.int32(i)


@wp.kernel(enable_backward=False)
def _record_high_water(count: wp.array[wp.int32], high_water: wp.array[wp.int32]):
    wp.atomic_max(high_water, 0, count[0])


def _gpu_components():
    try:
        from gpu_components import backing, fields  # noqa: PLC0415
        from gpu_components.field_data import FieldSpec  # noqa: PLC0415
    except ImportError as error:
        raise ImportError(
            "Growable rigid contact buffers on CUDA need the gpu_components package (CUDA virtual memory): "
            "install a checkout with `uv pip install --no-deps -e <gpu-components>`. Buffers whose bound "
            f"(rigid_contact_max) is at most {READY_ROWS_PER_WORLD} contacts per world, rounded up to "
            f"{ROW_QUANTUM} rows, need no growth and no package"
        ) from error
    return backing, fields, FieldSpec


_BACKINGS: dict[str, Any] = {}
"""One CUDA virtual-memory ledger per device, shared by every growable pool on it."""
_RETIRED: list[tuple[str, Any]] = []
"""Virtual-memory storages of collected rows, released at the next pool operation outside capture."""


def _backing(device):
    key = str(device)
    owner = _BACKINGS.get(key)
    if owner is None:
        backing_ops, _, _ = _gpu_components()
        with wp.ScopedDevice(device):
            owner = backing_ops.prepare(int(device.total_memory), device_ordinal=int(device.ordinal))
        _BACKINGS[key] = owner
    return owner


def _retire(device_key: str, storages: list) -> None:
    # Runs from a finalizer: queue only, without CUDA calls.
    _RETIRED.extend((device_key, storage) for storage in storages)


def _release_retired(device) -> None:
    """Release the queued storages of ``device`` that no view references any more."""
    key = str(device)
    pending = [storage for owner_key, storage in _RETIRED if owner_key == key]
    if not pending or (device.is_cuda and wp.get_stream(device).is_capturing):
        return
    backing_ops, fields_ops, _ = _gpu_components()
    owner = _BACKINGS[key]
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
    after the views are created and after every growth; owners assign the
    views there (:meth:`publish`) and refresh values derived from
    :attr:`rows`. A bound method is held weakly.
    """

    def __init__(self, fields, *, rows: int, bound: int, device, on_rows=None):
        self.fields = tuple(fields)
        self.rows = int(rows)
        self.bound = int(bound)
        self.device = wp.get_device(device)
        self.views: dict[str, wp.array] = {}
        self._on_rows = weakref.WeakMethod(on_rows) if hasattr(on_rows, "__self__") else on_rows
        self._storages = []
        if self.rows < self.bound and self.device.is_cuda:
            self._mode = "virtual"
            self._allocate_virtual()
        else:
            self._mode = "fixed" if self.rows >= self.bound else "realloc"
            for field in self.fields:
                self.views[field.name] = self._array(field, field.length(self.rows))
        if any(field.requires_grad for field in self.fields) and self._mode != "fixed":
            raise ValueError("Per-contact fields with gradients cannot grow; size them at their bound")
        _call(self._on_rows, self)

    @property
    def growable(self) -> bool:
        """Whether the rows can still grow."""
        return self.rows < self.bound

    def publish(self, target) -> None:
        """Assign each field's current view to the attribute of ``target`` named after the field."""
        for name, view in self.views.items():
            setattr(target, name, view)

    def mapped_bytes(self) -> int:
        """Return the device bytes the rows hold now."""
        if self._mode != "virtual":
            return sum(view.capacity for view in self.views.values())
        _, fields_ops, _ = _gpu_components()
        return sum(fields_ops.memory_report(storage)["mapped_dense_bytes"] for storage, _ in self._storages)

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
        if wp.get_stream(self.device).is_capturing:
            raise RuntimeError("Growable per-contact buffers cannot be created during CUDA graph capture")
        _, fields_ops, FieldSpec = _gpu_components()
        _release_retired(self.device)
        owner = _backing(self.device)
        groups: dict[tuple[int, int], list] = {}
        for field in self.fields:
            groups.setdefault((field.per_contact, field.extra), []).append(field)
        with wp.ScopedDevice(self.device):
            for (per_contact, extra), members in groups.items():
                storage = fields_ops.allocate(
                    per_contact * self.bound + extra,
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
                array = wp.empty(field.length(rows), dtype=field.dtype, device=self.device)
                if previous:
                    wp.copy(array, self.views[field.name], count=previous)
                self._initialize(array, field, previous, array.shape[0])
                self.views[field.name] = array
        _call(self._on_rows, self)


def _grow_rows(members: list[ContactRows], rows: int) -> None:
    """Bring every member's rows to at least ``rows``: map virtual storage in one maintenance scope, then publish."""
    members = [member for member in members if member.rows < rows]
    for member in members:
        if rows > member.bound:
            raise ValueError(
                f"Per-contact buffers bounded by {member.bound} rigid contacts cannot hold {rows}; "
                "construct the solver with model.rigid_contact_max at least the collision pipeline's rigid_contact_max"
            )
    if not members:
        return
    device = members[0].device
    if device.is_cuda and wp.get_stream(device).is_capturing:
        raise RuntimeError(
            "Per-contact buffers cannot grow during CUDA graph capture; grow them before capturing "
            "(CollisionPipeline.service_capacity(), or one eager step)"
        )
    virtual = [member for member in members if member._mode == "virtual"]
    if virtual:
        backing_ops, fields_ops, _ = _gpu_components()
        _release_retired(device)
        owner = _backing(device)
        # Join every stream that may touch the rows (graph replays, solver side streams) before remapping.
        wp.synchronize_device(device)
        stream = wp.get_stream(device).cuda_stream
        with wp.ScopedDevice(device), backing_ops.maintenance(owner, streams=(stream,)):
            for member in virtual:
                member._map(rows, fields_ops)
    for member in members:
        member._publish_rows(rows)


class ContactPool:
    """The ready rows of a rigid contact stream's per-contact buffers, and their growth from measured demand.

    Args:
        bound: The most rows the pool may grow to.
        device: The device of the buffers.
        world_count: Number of worlds of the model, for the initial rows.
        rows: Initial ready rows; defaults to :func:`initial_rows`. Rows equal
            to ``bound`` never grow.
    """

    def __init__(self, bound: int, *, device, world_count: int = 1, rows: int | None = None):
        self.bound = int(bound)
        self.device = wp.get_device(device)
        self.ready = min(self.bound, initial_rows(self.bound, world_count) if rows is None else int(rows))
        self._members: list[weakref.ReferenceType] = []
        self._followers: list[weakref.ReferenceType] = []
        self._high_water = None
        self._high_water_host = None
        if self.growable:
            # Allocated here, outside any graph capture that records demand.
            self._high_water = wp.zeros(1, dtype=wp.int32, device=self.device)
            self._high_water_host = (
                wp.zeros(1, dtype=wp.int32, device="cpu", pinned=True) if self.device.is_cuda else self._high_water
            )
        self.growth_count = 0
        """Growths so far; each one requires recapturing graphs that use the pool."""
        self._reserved = 0
        # Growths of live rows in this pool, and the total over the pools that follow it
        # that service() last reported.
        self._changes = 0
        self._reported = 0

    @property
    def growable(self) -> bool:
        """Whether the pool can still grow."""
        return self.ready < self.bound

    def rows(self, fields, *, on_rows=None) -> ContactRows:
        """Create per-contact fields that follow this pool's ready rows (see :class:`ContactRows`)."""
        rows = ContactRows(fields, rows=self.ready, bound=self.bound, device=self.device, on_rows=on_rows)
        self._members.append(weakref.ref(rows))
        return rows

    def follow(self, other: ContactPool) -> None:
        """Make ``other`` grow with this pool: its rows stay at least this pool's ready rows.

        Grows ``other`` now if it holds fewer rows, which cannot happen during
        graph capture. Following is idempotent.

        Raises:
            ValueError: If ``other`` is bounded below this pool's ready rows.
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
            be recaptured to use the new rows; :meth:`service` reports it too.
        """
        pools = self._closure()
        if all(pool.ready >= rows for pool in pools):
            return False
        for pool in pools:
            if rows > pool.bound and pool.ready < rows:
                raise ValueError(
                    f"A contact pool bounded by {pool.bound} rows cannot grow to {rows}; construct the solver with "
                    "model.rigid_contact_max at least the collision pipeline's rigid_contact_max"
                )
        growing = [(pool, pool._live_members()) for pool in pools if pool.ready < rows]
        _grow_rows([member for _, members in growing for member in members], rows)
        for pool, members in growing:
            pool.ready = rows
            pool.growth_count += 1
            pool._changes += int(bool(members))
        return any(members for _, members in growing)

    def reserve(self, rows: int) -> bool:
        """Hold at least ``rows`` ready rows from now on, for a writer with a fixed contact capacity.

        Outside graph capture the pool and its followers grow now, to
        :func:`reserved_rows`, keeping their contents. During capture the request
        warns and waits for the next :meth:`service` (or a reservation outside
        capture); meanwhile writers stay bounded by the current rows and flag
        their overflow.

        Returns:
            Whether live buffers grew now; graphs that captured them must be
            recaptured (:meth:`service` reports it too).

        Raises:
            ValueError: If ``rows`` exceeds the bound.
        """
        rows = int(rows)
        if rows <= self.ready:
            return False
        if rows > self.bound:
            raise ValueError(f"A contact pool bounded by {self.bound} rows cannot hold the {rows} required")
        if self.device.is_cuda and wp.get_stream(self.device).is_capturing:
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

    def record_demand(self, count: wp.array) -> None:
        """Record the contact count in the sticky high-water mark and copy it to pinned host memory.

        Graph capture records both launches. Pools that cannot grow record nothing.
        """
        if self._high_water is None or not self.growable:
            return
        wp.launch(_record_high_water, dim=1, inputs=[count, self._high_water], device=self.device, record_tape=False)
        if self._high_water_host is not self._high_water:
            wp.copy(self._high_water_host, self._high_water)

    @property
    def high_water(self) -> int:
        """The most contacts any recorded collision pass counted, as last copied to the host (no synchronization)."""
        return 0 if self._high_water_host is None else int(self._high_water_host.numpy()[0])

    def service(self) -> bool:
        """Grow from the recorded demand and reservations; return whether graphs using the pool must be recaptured.

        Reads the pinned high-water mark without synchronizing, so it lags the
        device by the passes still in flight. Grows to :func:`grown_rows`, or to
        a reservation made during capture, logging the change. A high-water mark
        at or above the ready rows means a pass may have dropped contacts: its
        capacity status stays flagged and this warns. Returns ``True`` when live
        buffers of this pool or its followers grew since the last call, also by
        a :meth:`reserve` or :meth:`follow` outside it.
        """
        if self.device.is_cuda and wp.get_stream(self.device).is_capturing:
            return False
        if self.growable:
            high_water = self.high_water
            target = grown_rows(high_water, self.ready, self.bound)
            if self._reserved > self.ready:
                target = max(target, reserved_rows(self._reserved, self.bound))
            if target > self.ready:
                previous = self.ready
                before = self.mapped_bytes()
                self.grow(target)
                logger.info(
                    "Rigid contact pool grew from %d to %d rows (high-water %d, reserved %d, bound %d): "
                    "%.1f -> %.1f MiB",
                    previous,
                    target,
                    high_water,
                    self._reserved,
                    self.bound,
                    before / 2**20,
                    self.mapped_bytes() / 2**20,
                )
                if high_water >= previous:
                    warnings.warn(
                        f"Rigid contact demand {high_water} reached the {previous} ready rows: a collision pass may "
                        f"have dropped contacts (its capacity status stays flagged); the pool grew to {target} rows",
                        RuntimeWarning,
                        stacklevel=2,
                    )
        changes = sum(pool._changes for pool in self._closure())
        changed = changes != self._reported
        self._reported = changes
        return changed

    def mapped_bytes(self) -> int:
        """Return the device bytes this pool's and its followers' rows hold now."""
        return sum(member.mapped_bytes() for pool in self._closure() for member in pool._live_members())

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
