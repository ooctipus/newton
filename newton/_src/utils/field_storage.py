# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Typed field storage, bounded transfers, backing readiness and safe retirement.

Callers declare fields and packing explicitly. This mechanical owner knows no
physics schema, world identity, defaults, scheduler or graph-node binding policy.
"""

import math
import sys
import traceback
import weakref
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import warp as wp


@wp.kernel
def _copy_rows(count: wp.array[int], dst: wp.array2d[wp.uint32], src: wp.array2d[wp.uint32], workers: int):
    index = wp.int64(wp.tid())
    width = wp.int64(dst.shape[1])
    total = wp.int64(count[0]) * width
    while index < total:
        row, column = index // width, index % width
        dst[row, column] = src[row, column]
        index += wp.int64(workers)


@wp.kernel
def _fill_rows(
    count: wp.array[int], dst: wp.array2d[wp.uint32], pattern: wp.array[wp.uint32], workers: int, start: int
):
    width = wp.int64(dst.shape[1])
    index = wp.int64(start) * width + wp.int64(wp.tid())
    total = wp.int64(count[0]) * width
    while index < total:
        row, column = index // width, index % width
        dst[row, column] = pattern[column % wp.int64(pattern.shape[0])]
        index += wp.int64(workers)


@wp.kernel
def _copy_bytes(count: wp.array[int], dst: wp.array2d[wp.uint8], src: wp.array2d[wp.uint8], workers: int):
    index = wp.int64(wp.tid())
    width = wp.int64(dst.shape[1])
    total = wp.int64(count[0]) * width
    while index < total:
        row, column = index // width, index % width
        dst[row, column] = src[row, column]
        index += wp.int64(workers)


@wp.kernel
def _fill_bytes(count: wp.array[int], dst: wp.array2d[wp.uint8], pattern: wp.array[wp.uint8], workers: int, start: int):
    width = wp.int64(dst.shape[1])
    index = wp.int64(start) * width + wp.int64(wp.tid())
    total = wp.int64(count[0]) * width
    while index < total:
        row, column = index // width, index % width
        dst[row, column] = pattern[column % wp.int64(pattern.shape[0])]
        index += wp.int64(workers)


@wp.kernel
def _zero_rows(count: wp.array[int], dst: wp.array2d[wp.uint32], workers: int, start: int):
    width = wp.int64(dst.shape[1])
    index = wp.int64(start) * width + wp.int64(wp.tid())
    total = wp.int64(count[0]) * width
    while index < total:
        dst[index // width, index % width] = wp.uint32(0)
        index += wp.int64(workers)


@wp.struct
class _FieldTransferDescriptor:
    source_words: wp.array2d[wp.uint32]
    destination_words: wp.array2d[wp.uint32]
    source_bytes: wp.array2d[wp.uint8]
    destination_bytes: wp.array2d[wp.uint8]
    unit_count: int
    use_words: int


@wp.kernel
def _clear_transfer(
    count: wp.array[int],
    source_indices: wp.array[int],
    destination_indices: wp.array[int],
    source_seen: wp.array[int],
    destination_seen: wp.array[int],
    status: wp.array[int],
):
    index = wp.tid()
    if index == 0:
        status[0] = 0
        if count[0] < 0 or count[0] > source_indices.shape[0] or count[0] > destination_indices.shape[0]:
            status[0] = 1
    if index < source_seen.shape[0]:
        source_seen[index] = 0
    if index < destination_seen.shape[0]:
        destination_seen[index] = 0


@wp.kernel
def _validate_transfer(
    count: wp.array[int],
    source_indices: wp.array[int],
    destination_indices: wp.array[int],
    source_ready: wp.array[int],
    destination_ready: wp.array[int],
    source_seen: wp.array[int],
    destination_seen: wp.array[int],
    status: wp.array[int],
    same_owner: int,
    source_capacity: int,
    destination_capacity: int,
):
    index = wp.tid()
    if (
        count[0] < 0
        or count[0] > source_indices.shape[0]
        or count[0] > destination_indices.shape[0]
        or index >= count[0]
    ):
        return
    source, destination = source_indices[index], destination_indices[index]
    if (
        source_ready[0] < 0
        or source_ready[0] > source_capacity
        or destination_ready[0] < 0
        or destination_ready[0] > destination_capacity
        or source < 0
        or source >= source_ready[0]
        or destination < 0
        or destination >= destination_ready[0]
    ):
        wp.atomic_max(status, 0, 2)
        return
    if wp.atomic_add(destination_seen, destination, 1) != 0:
        wp.atomic_max(status, 0, 3)
    if same_owner != 0:
        wp.atomic_max(source_seen, source, 1)


@wp.kernel
def _validate_transfer_overlap(source_seen: wp.array[int], destination_seen: wp.array[int], status: wp.array[int]):
    index = wp.tid()
    if source_seen[index] != 0 and destination_seen[index] != 0:
        wp.atomic_max(status, 0, 4)


@wp.kernel
def _transfer_fields(
    fields: wp.array[_FieldTransferDescriptor],
    count: wp.array[int],
    source_indices: wp.array[int],
    destination_indices: wp.array[int],
    status: wp.array[int],
    workers: int,
):
    field_index, thread = wp.tid()
    if status[0] != 0:
        return
    field = fields[field_index]
    width = wp.int64(field.unit_count)
    index = wp.int64(thread)
    total = wp.int64(count[0]) * width
    while index < total:
        ordinal, column = index // width, index % width
        source, destination = source_indices[ordinal], destination_indices[ordinal]
        if field.use_words != 0:
            field.destination_words[destination, column] = field.source_words[source, column]
        else:
            field.destination_bytes[destination, column] = field.source_bytes[source, column]
        index += wp.int64(workers)


@dataclass(frozen=True)
class FieldView:
    """Borrowed field descriptor returned by :meth:`FieldStorage.lookup`.

    ``array`` has one outer storage index and the declared inner dimensions.
    ``row_payload_bytes`` excludes inter-field padding. ``packed_offset_bytes``
    is the field's offset within each shared packed row, or None for a separate
    dense allocation. ``transfer_view`` aliases the same payload as uint32 words
    when aligned, otherwise bytes; it does not own another allocation.
    Retain the storage owner and obey its readiness and retirement boundaries.
    """

    name: str
    array: wp.array
    row_payload_bytes: int
    packed_offset_bytes: int | None
    transfer_view: wp.array


def _array_key(array):
    return array.ptr, array.dtype, array.shape, array.strides


@dataclass(frozen=True)
class FieldSpec:
    """Declare one named typed field before allocating any population storage.

    ``inner_shape`` excludes the outer capacity axis. Names identify fields in
    this schema, not world identities or symbolic scene paths. Packed fields
    share one strided reservation; ``packed=False`` allocates a fully backed dense
    array. ``alignment_bytes`` strengthens natural scalar alignment for consumers
    such as vectorized matrix tiles. It must be a power of two at most 16.
    """

    name: str
    inner_shape: tuple[int, ...]
    dtype: object
    packed: bool = True
    alignment_bytes: int | None = None


class FieldStorage:
    """Own explicitly declared typed rows behind stable descriptors.

    ``protected_count`` is a borrowed prefix length that shrinking must preserve;
    it is also the default extent for memory operations. It is not a population
    count: disposable scratch may protect zero rows while using ``ready_count``.
    Callers keep recorded operation counts in [0, ready_count] for every replay.
    ``ready_count`` certifies accessible rows, not initialized values or lifetimes;
    quarantine withdraws readiness even when physical mappings remain. Callers own
    initialization and schema composition. Dense fields remain fully backed;
    nonempty packed fields share one virtual reservation.
    Packed fields use scalar alignment unless the consumer requests a stronger
    alignment, up to the 16-byte row stride alignment. Dense fields keep Warp's
    native layout. A scalar-aligned field need not satisfy vectorized tile loads.
    """

    def __init__(self, capacity, protected_count, *, fields, backing=None, initial_ready_count=None):
        if sys.version_info < (3, 11):
            raise RuntimeError("Experimental CUDA field storage requires Python 3.11 or newer")
        if (
            not isinstance(protected_count, wp.array)
            or protected_count.dtype != wp.int32
            or protected_count.shape != (1,)
            or not protected_count.is_contiguous
        ):
            raise ValueError("Protected count must be one contiguous int32 scalar")
        self.capacity, self.protected_count = capacity, protected_count
        self.device, self.backing = protected_count.device, backing
        if (
            not self.device.is_cuda
            or self.device.is_capturing
            or type(capacity) is not int
            or not 1 <= capacity < 2**31
        ):
            raise ValueError("Field storage requires CUDA, positive prepared capacity and preparation outside capture")
        initial_ready_count = capacity if initial_ready_count is None else initial_ready_count
        if type(initial_ready_count) is not int or not 0 <= initial_ready_count <= capacity:
            raise ValueError("Initial ready rows exceed the prepared capacity")
        if backing is None and initial_ready_count != capacity:
            raise ValueError("Partial backing requires the VMM owner")
        specifications, layout, row_bytes = {}, {}, 0
        for field in fields:
            if (
                not isinstance(field, FieldSpec)
                or not isinstance(field.name, str)
                or not field.name
                or field.name in specifications
            ):
                raise ValueError("Field specs require distinct nonempty names")
            if (
                not isinstance(field.inner_shape, tuple)
                or len(field.inner_shape) > 3
                or any(type(length) is not int or not 0 <= length < 2**31 for length in field.inner_shape)
                or type(field.packed) is not bool
            ):
                raise ValueError("Field specs require at most three nonnegative inner dimensions and boolean packing")
            try:
                dtype = {int: wp.int32, float: wp.float32, bool: wp.bool}.get(field.dtype, field.dtype)
                if not wp.types.type_is_value(dtype):
                    raise TypeError("not a value type")
                width = wp.types.type_size_in_bytes(dtype) * math.prod(field.inner_shape)
            except (TypeError, AttributeError) as error:
                raise ValueError(f"Unsupported field dtype: {field.name}") from error
            if width >= 2**31:
                raise ValueError(f"Field row width exceeds the int32 stride limit: {field.name}")
            scalar_alignment = wp.types.type_size_in_bytes(getattr(dtype, "_wp_scalar_type_", dtype))
            alignment = scalar_alignment if field.alignment_bytes is None else field.alignment_bytes
            if (
                type(alignment) is not int
                or alignment < scalar_alignment
                or alignment > 16
                or alignment & (alignment - 1)
            ):
                raise ValueError(
                    f"Field alignment must be a power of two from scalar alignment through 16: {field.name}"
                )
            specifications[field.name] = FieldSpec(field.name, field.inner_shape, dtype, field.packed, alignment)
            if width and field.packed:
                row_bytes = (row_bytes + alignment - 1) // alignment * alignment
                layout[field.name] = row_bytes
                row_bytes += width
        if not specifications:
            raise ValueError("Field storage requires at least one field")
        self.row_stride_bytes = (row_bytes + 15) // 16 * 16
        if self.row_stride_bytes >= 2**31:
            raise ValueError("Packed row stride exceeds the int32 stride limit")
        if backing is not None and not self.row_stride_bytes:
            raise ValueError("VMM field storage requires a nonempty packed field")
        self.specifications, self.arrays, self.fields, self._by_array, self._patterns = specifications, {}, {}, {}, {}
        self._graphs, self._transfers = weakref.WeakSet(), weakref.WeakSet()
        self._closed, self._service_failed, self.reservation, self._storage = False, False, None, None
        self._packed_words = None
        self.ready_rows, self.ready_count = 0, None
        array = None
        current_stream = wp.get_stream(self.device).cuda_stream
        try:
            self.ready_count = wp.array([0], dtype=int, device=self.device)
            address = 0
            if backing is None:
                if self.row_stride_bytes:
                    self._storage = wp.empty(capacity * self.row_stride_bytes, dtype=wp.uint8, device=self.device)
                    address = self._storage.ptr
                self.ready_rows = capacity
            else:
                self.reservation = backing.reserve(capacity * self.row_stride_bytes)
                with backing.maintenance(streams=(current_stream,)):
                    mapped = self._rounded_bytes(initial_ready_count)
                    if mapped:
                        backing.map(self.reservation, 0, mapped)
                address = self.reservation.address
                self.ready_rows = min(capacity, mapped // self.row_stride_bytes)
            for name, field in specifications.items():
                shape = (capacity, *field.inner_shape)
                item_bytes = wp.types.type_size_in_bytes(field.dtype)
                row_width = item_bytes * math.prod(field.inner_shape)
                offset = layout.get(name)
                if offset is None:
                    array = wp.empty(shape, dtype=field.dtype, device=self.device)
                else:
                    deleter = None
                    if backing is not None:
                        backing.acquire_reference(self.reservation)
                        reference_live = [True]

                        def deleter(
                            ptr=None, size=None, owner=backing, reservation=self.reservation, live=reference_live
                        ):
                            if live[0]:
                                owner.release_reference(reservation)
                                live[0] = False

                    strides = (
                        self.row_stride_bytes,
                        *(item_bytes * math.prod(field.inner_shape[i + 1 :]) for i in range(len(field.inner_shape))),
                    )
                    try:
                        array = wp.array(
                            ptr=address + offset,
                            shape=shape,
                            dtype=field.dtype,
                            strides=strides,
                            device=self.device,
                            deleter=deleter,
                        )
                    except BaseException:
                        if deleter:
                            deleter()
                        raise
                    if backing is None:
                        array._storage_owner = self._storage
                self.arrays[name] = array
                aligned_words = row_width and row_width % 4 == 0 and array.ptr % 4 == 0 and array.strides[0] % 4 == 0
                word_dtype = wp.uint32 if aligned_words else wp.uint8
                word_bytes = wp.types.type_size_in_bytes(word_dtype)
                words = wp.array(
                    ptr=array.ptr,
                    shape=(capacity, row_width // word_bytes),
                    strides=(array.strides[0], word_bytes),
                    dtype=word_dtype,
                    device=self.device,
                )
                words._view_owner = array
                descriptor = FieldView(name, array, row_width, offset, words)
                self.fields[name] = descriptor
                self._by_array[_array_key(array)] = descriptor
                words = descriptor = None
            if self.row_stride_bytes:
                self._packed_words = wp.array(
                    ptr=address, shape=(capacity, self.row_stride_bytes // 4), dtype=wp.uint32, device=self.device
                )
                self._packed_words._view_owner = next(
                    field.array for field in self.fields.values() if field.packed_offset_bytes is not None
                )
            self._publish_ready(self.ready_rows)
        except BaseException as failure:
            self._closed = True
            traceback.clear_frames(failure.__traceback__)
            try:
                if backing is not None:
                    with backing.maintenance(streams=(current_stream,)):
                        array = None
                        self._packed_words = None
                        self.arrays.clear()
                        self.fields.clear()
                        self._by_array.clear()
                        if self.reservation is not None:
                            backing.release_reservation(self.reservation)
                            self.reservation = None
                else:
                    wp.synchronize_stream(wp.get_stream(self.device))
                    array = None
                    self._packed_words = None
                    self.arrays.clear()
                    self.fields.clear()
                    self._by_array.clear()
                    self._storage = None
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "Field storage construction and cleanup failed", [failure, cleanup]
                ) from failure
            raise

    @property
    def service_failed(self):
        """Whether joined backing service quarantined this owner permanently."""
        return self._service_failed

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("Field storage is closed")
        if self._service_failed:
            raise RuntimeError("Field backing service failed; retire this owner before further replay")

    def lookup(self, array):
        """Return the registered field descriptor for an exact array view, or ``None``."""
        self._ensure_open()
        if not isinstance(array, wp.array) or array.device != self.device:
            raise ValueError("Lookup requires an array on the storage device")
        return self._by_array.get(_array_key(array))

    def _transfer_view(self, array, reference):
        if array.shape != reference.array.shape or array.dtype != reference.array.dtype or array.device != self.device:
            raise ValueError("Row copy requires identical typed shapes and device")
        if array.strides[1:] != reference.array.strides[1:]:
            raise ValueError("Row copy requires matching contiguous inner layout")
        if array.size and array.strides[0] < reference.row_payload_bytes:
            raise ValueError("Row copy requires nonoverlapping rows")
        if reference.transfer_view.dtype == wp.uint32 and (array.ptr % 4 or array.strides[0] % 4):
            raise ValueError("Word row copy requires four-byte aligned pointers and row strides")
        owned = self._by_array.get(_array_key(array))
        if owned is not None and owned.transfer_view.dtype == reference.transfer_view.dtype:
            return owned.transfer_view
        width = reference.transfer_view.shape[1]
        view = wp.array(
            ptr=array.ptr,
            shape=(self.capacity, width),
            dtype=reference.transfer_view.dtype,
            strides=(array.strides[0], wp.types.type_size_in_bytes(reference.transfer_view.dtype)),
            device=self.device,
        )
        view._view_owner = array
        return view

    def copy(self, destination, source, *, count=None):
        """Copy a prefix between disjoint fields; an exact self-copy records no work.

        At least one side must be registered here. Distinct registered fields have
        disjoint payloads even when packed rows interleave their address spans.
        External arrays must have disjoint spans; partial aliases are rejected.
        The caller certifies accessibility of an external operand's count prefix;
        this owner cannot infer another allocation's backing or readiness.
        """
        self._ensure_open()
        count = self.protected_count if count is None else count
        if (
            not isinstance(count, wp.array)
            or count.dtype != wp.int32
            or count.shape != (1,)
            or not count.is_contiguous
            or count.device != self.device
        ):
            raise ValueError("Copy count must be one contiguous int32 scalar on the storage device")
        if (
            not isinstance(destination, wp.array)
            or not isinstance(source, wp.array)
            or destination.shape != source.shape
            or destination.dtype != source.dtype
            or destination.device != self.device
            or source.device != self.device
        ):
            raise ValueError("Row copy requires identical typed shapes and device")
        destination_field, source_field = self.lookup(destination), self.lookup(source)
        field = destination_field or source_field
        if field is None:
            raise ValueError("Row copy requires an owned field")
        if not destination.size:
            return
        dst, src = self._transfer_view(destination, field), self._transfer_view(source, field)
        if _array_key(destination) == _array_key(source):
            return dst, src
        if destination_field is None or source_field is None:
            dst_end = destination.ptr + (self.capacity - 1) * destination.strides[0] + field.row_payload_bytes
            src_end = source.ptr + (self.capacity - 1) * source.strides[0] + field.row_payload_bytes
            if destination.ptr < src_end and source.ptr < dst_end:
                raise ValueError("Prefix copy rejects overlapping external array spans")
        workers = min(65536, self.capacity * dst.shape[1])
        kernel = _copy_rows if dst.dtype == wp.uint32 else _copy_bytes
        wp.launch(kernel, workers, [count, dst, src, workers], device=self.device)
        return dst, src

    def prepare_fill(self, array, value):
        """Prepare a typed fill pattern without writing the destination."""
        self._ensure_open()
        field = self.lookup(array)
        if field is None:
            raise ValueError("Row fill requires an owned field")
        key = array.dtype, repr(value), field.transfer_view.dtype
        pattern = self._patterns.get(key)
        if pattern is None:
            if self.device.is_capturing:
                raise RuntimeError("Fill pattern must be prepared before graph capture")
            one = wp.full(1, value, dtype=array.dtype, device="cpu")
            raw = one.numpy().tobytes()
            dtype = np.uint32 if field.transfer_view.dtype == wp.uint32 else np.uint8
            unit_bytes = np.dtype(dtype).itemsize
            # Whole transfer words must preserve the typed element's byte period,
            # including subword scalars and vectors with odd byte widths.
            raw *= unit_bytes // math.gcd(len(raw), unit_bytes)
            pattern = wp.array(np.frombuffer(raw, dtype=dtype), dtype=field.transfer_view.dtype, device=self.device)
            self._patterns[key] = pattern
        return pattern

    def fill(self, array, value, *, count=None, start=0):
        """Fill rows in [start, count); its typed pattern must exist before capture."""
        self._ensure_open()
        field = self.lookup(array)
        if field is None:
            raise ValueError("Row fill requires an owned field")
        count = self.protected_count if count is None else count
        if (
            not isinstance(count, wp.array)
            or count.dtype != wp.int32
            or count.shape != (1,)
            or not count.is_contiguous
            or count.device != self.device
        ):
            raise ValueError("Fill count must be one contiguous int32 scalar on the storage device")
        if type(start) is not int or not 0 <= start <= self.capacity:
            raise ValueError("Fill start must be within the prepared row capacity")
        if not array.size or start == self.capacity:
            return
        pattern = self.prepare_fill(array, value)
        workers = min(65536, (self.capacity - start) * field.transfer_view.shape[1])
        kernel = _fill_rows if field.transfer_view.dtype == wp.uint32 else _fill_bytes
        wp.launch(kernel, workers, [count, field.transfer_view, pattern, workers, start], device=self.device)

    def zero(self, *, count=None, start=0):
        """Zero a row suffix, fusing packed fields and preserving earlier rows."""
        self._ensure_open()
        count = self.protected_count if count is None else count
        if (
            not isinstance(count, wp.array)
            or count.dtype != wp.int32
            or count.shape != (1,)
            or not count.is_contiguous
            or count.device != self.device
        ):
            raise ValueError("Zero count must be one contiguous int32 scalar on the storage device")
        if type(start) is not int or not 0 <= start <= self.capacity:
            raise ValueError("Zero start must be within the prepared row capacity")
        if start == self.capacity:
            return
        if self._packed_words is not None:
            workers = min(65536, (self.capacity - start) * (self.row_stride_bytes // 4))
            wp.launch(_zero_rows, workers, [count, self._packed_words, workers, start], device=self.device)
        for field in self.fields.values():
            if field.packed_offset_bytes is None and field.row_payload_bytes:
                self.fill(field.array, 0, count=count, start=start)

    def _rounded_bytes(self, rows):
        granularity = self.backing.granularity_bytes
        return min(
            self.reservation.size_bytes, (rows * self.row_stride_bytes + granularity - 1) // granularity * granularity
        )

    def _publish_ready(self, rows, *, synchronize=True):
        self.ready_count.fill_(rows)
        if synchronize:
            wp.synchronize_stream(wp.get_stream(self.device))
        self.ready_rows = rows

    def prepare_transfer(self, source, *, fields):
        """Prepare typed row transfers between owned, compatible row domains."""
        return FieldTransfer(source, self, fields)

    def resize_backing(self, rows, *, streams=None, protected_count_host=None):
        """Service readiness inside a joined scope or after joining the supplied streams.

        ``streams`` contains raw CUDA stream handles, not Warp Stream objects.
        A composition owner may supply the protected count from its single joined
        batch readback. This requires an already active backing maintenance scope;
        no submissions may occur between that readback and completing the batch.
        Joined growth publications are ordered on the current stream; the caller
        joins that stream before allowing submissions on other streams.
        """
        self._ensure_open()
        if self.backing is None:
            raise RuntimeError("Backing service requires an open VMM field owner")
        if type(rows) is not int or not 0 <= rows <= self.capacity:
            raise ValueError("Requested rows exceed prepared capacity")
        if protected_count_host is not None and (streams is not None or type(protected_count_host) is not int):
            raise ValueError("A joined protected count requires active maintenance and an integer")
        if streams is None:
            self.backing._check(maintenance=True)
        with self.backing.maintenance(streams=streams) if streams is not None else nullcontext():
            protected = int(self.protected_count.numpy()[0]) if protected_count_host is None else protected_count_host
            if not 0 <= protected <= self.ready_rows or rows < protected:
                raise RuntimeError("Cannot retire protected rows or service an invalid protected count")
            ranges = self.backing.mapped_ranges(self.reservation)
            if ranges and (len(ranges) != 1 or ranges[0][0] != 0):
                self._service_failed = True
                self._publish_ready(0)
                raise RuntimeError("Field backing is not a contiguous prefix; retire this owner")
            old = ranges[0][1] if ranges else 0
            new = self._rounded_bytes(rows)
            if new == old:
                return
            clean_budget_rejection = False
            try:
                if new < old:
                    self._publish_ready(min(self.ready_rows, new // self.row_stride_bytes))
                    self.backing.unmap(self.reservation, new, old - new)
                else:
                    try:
                        self.backing.map(self.reservation, old, new - old)
                    except MemoryError:
                        # This call rejects its byte budget before driver mutation.
                        clean_budget_rejection = True
                        raise
                    self._publish_ready(
                        min(self.capacity, new // self.row_stride_bytes), synchronize=streams is not None
                    )
            except BaseException as failure:
                if clean_budget_rejection:
                    raise
                self._service_failed = True
                try:
                    self._publish_ready(0)
                except BaseException as quarantine:
                    raise BaseExceptionGroup("Row service and quarantine failed", [failure, quarantine]) from failure
                raise

    def retain_graph(self, graph, *owners):
        """Retain this storage and additional owners until graph retirement; return the graph."""
        self._ensure_open()
        if graph.device != self.device:
            raise ValueError("Graph and field storage must share a device")
        graph._resource_owners = (*getattr(graph, "_resource_owners", ()), self, *owners)
        self._graphs.add(graph)
        return graph

    def memory_report(self):
        """Report this owner's current allocations and prepared layout, not process memory.

        After closure, ordinary allocations retained only by external array views
        are outside this report. A VMM reservation whose retirement failed remains
        in the mapped-byte ledger until a later close succeeds.
        """
        payload = sum(
            field.row_payload_bytes for field in self.fields.values() if field.packed_offset_bytes is not None
        )
        return {
            "capacity": self.capacity,
            "ready_rows": self.ready_rows,
            "row_stride_bytes": self.row_stride_bytes,
            "packed_payload_bytes_per_row": payload,
            "padding_bytes_per_row": self.row_stride_bytes - payload,
            "ready_metadata_bytes": self.ready_count.capacity,
            "fill_pattern_bytes": sum(pattern.capacity for pattern in self._patterns.values()),
            "packed_fields": sum(field.packed_offset_bytes is not None for field in self.fields.values()),
            "dense_field_bytes": sum(
                field.array.size * wp.types.type_size_in_bytes(field.array.dtype)
                for field in self.fields.values()
                if field.packed_offset_bytes is None
            ),
            "virtual_packed_bytes": self.capacity * self.row_stride_bytes,
            "mapped_packed_bytes": sum(size for _, size in self.backing.mapped_ranges(self.reservation))
            if self.backing and self.reservation is not None
            else self._storage.capacity
            if self._storage is not None
            else 0,
            "service_failed": self._service_failed,
            "closed": self._closed,
            "scope": "owned typed fields, readiness and fill patterns; callers account other allocations separately",
        }

    def close(self, *, streams):
        """Retire storage after consumers join; retained graphs or transfer plans block close.

        ``streams`` contains raw CUDA stream handles. VMM joins the supplied
        streams; fixed storage joins the whole device. Exclude new submissions.
        Ordinary arrays may keep allocations alive outside this owner's report
        after close. Packed VMM views block reservation release after readiness is
        withdrawn; drop those views and retry close to complete retirement.
        """
        if self._closed and self.reservation is None:
            return
        if self._graphs:
            raise RuntimeError("Field storage remains retained by graphs")
        if self._transfers:
            raise RuntimeError("Field storage remains retained by transfer plans")
        if self.backing:
            with self.backing.maintenance(streams=streams):
                self._publish_ready(0)
                self._closed = True
                self._packed_words = None
                self.arrays.clear()
                self.fields.clear()
                self._by_array.clear()
                self._patterns.clear()
                self.backing.release_reservation(self.reservation)
                self.reservation = None
        else:
            if not streams:
                raise ValueError("Fixed field storage retirement requires explicit consuming streams")
            wp.synchronize_device(self.device)
            self._publish_ready(0)
            self._closed = True
            self._packed_words = None
            self.arrays.clear()
            self.fields.clear()
            self._by_array.clear()
            self._patterns.clear()
            self._storage = None
        self._closed = True


class FieldTransfer:
    """Prepared typed field copies between owned rows; all requests validate before writes.

    Source repeats are legal for prototype initialization. Destination repeats and
    any source/destination overlap in an in-place transfer are rejected. The domain
    supplies the retained/init field list and proves logical lifetime membership;
    this plan proves descriptor compatibility and physical readiness only.

    ``status`` is 0 on success, 1 for count, 2 for unready/out-of-range indices, 3 for
    duplicate destinations, or 4 for overlapping in-place source/destination sets.
    A plan may not execute concurrently with itself. The graph must retain it and
    both field owners for its complete lifetime.
    """

    def __init__(self, source, destination, field_names):
        if not isinstance(source, FieldStorage) or not isinstance(destination, FieldStorage):
            raise TypeError("Transfers require explicit FieldStorage owners on both ends")
        source._ensure_open()
        destination._ensure_open()
        if source.device != destination.device or source.device.is_capturing:
            raise ValueError("Transfer preparation requires matching devices outside capture")
        if isinstance(field_names, str):
            raise ValueError("Transfer fields must be an explicit sequence of field names")
        field_names = tuple(field_names)
        if (
            not field_names
            or any(not isinstance(name, str) for name in field_names)
            or len(set(field_names)) != len(field_names)
        ):
            raise ValueError("Transfer fields must be nonempty distinct field names")
        source_fields, destination_fields = source.arrays, destination.arrays
        active = []
        for name in field_names:
            if name not in source_fields or name not in destination_fields:
                raise ValueError(f"Transfer field is not owned by both field stores: {name}")
            src, dst = source_fields[name], destination_fields[name]
            if src.dtype != dst.dtype or src.shape[1:] != dst.shape[1:]:
                raise ValueError(f"Transfer field typed inner layout differs: {name}")
            if bool(src.size) != bool(dst.size):
                raise ValueError(f"Transfer field is absent on only one side: {name}")
            if src.size:
                active.append((source.lookup(src), destination.lookup(dst)))
        if not active or any(src is None or dst is None for src, dst in active):
            raise ValueError("Transfer requires nonempty fields registered by both owners")
        self.source, self.destination, self.device = source, destination, source.device
        self.field_names, self.nonempty_field_names = field_names, tuple(src.name for src, _ in active)
        self.same_owner = source is destination
        # Distinct FieldStorage owners construct independent typed fields. Explicitly
        # reject a subsequently injected alias rather than assume that identity.
        if not self.same_owner:
            source_ranges = [
                (
                    field.array.ptr,
                    field.array.ptr + (source.capacity - 1) * field.array.strides[0] + field.row_payload_bytes,
                )
                for field, _ in active
            ]
            destination_ranges = [
                (
                    field.array.ptr,
                    field.array.ptr + (destination.capacity - 1) * field.array.strides[0] + field.row_payload_bytes,
                )
                for _, field in active
            ]
            if any(
                lo < other_hi and other_lo < hi for lo, hi in source_ranges for other_lo, other_hi in destination_ranges
            ):
                raise ValueError("Distinct source and destination owners have overlapping storage")
        self._views, descriptors = [], []
        empty_words = wp.empty((0, 0), dtype=wp.uint32, device=self.device)
        empty_bytes = wp.empty((0, 0), dtype=wp.uint8, device=self.device)
        for src, dst in active:
            descriptor = _FieldTransferDescriptor()
            descriptor.use_words = int(src.transfer_view.dtype == wp.uint32 and dst.transfer_view.dtype == wp.uint32)
            if descriptor.use_words:
                descriptor.source_words, descriptor.destination_words = src.transfer_view, dst.transfer_view
                descriptor.source_bytes = descriptor.destination_bytes = empty_bytes
                descriptor.unit_count = src.transfer_view.shape[1]
                self._views.extend((src.transfer_view, dst.transfer_view))
            else:
                views = []
                for field in (src, dst):
                    view = wp.array(
                        ptr=field.array.ptr,
                        shape=(field.array.shape[0], field.row_payload_bytes),
                        dtype=wp.uint8,
                        strides=(field.array.strides[0], 1),
                        device=self.device,
                    )
                    view._view_owner = field.array
                    views.append(view)
                descriptor.source_bytes, descriptor.destination_bytes = views
                descriptor.source_words = descriptor.destination_words = empty_words
                descriptor.unit_count = src.row_payload_bytes
                self._views.extend(views)
            descriptors.append(descriptor)
        self._views.extend((empty_words, empty_bytes))
        self._descriptors = wp.array(descriptors, dtype=_FieldTransferDescriptor, device=self.device)
        self._source_seen = wp.zeros(source.capacity if self.same_owner else 0, dtype=int, device=self.device)
        self._destination_seen = wp.zeros(destination.capacity, dtype=int, device=self.device)
        self.status = wp.zeros(1, dtype=int, device=self.device)
        self._worker_count = min(8192, destination.capacity * max(field.unit_count for field in descriptors))
        source._transfers.add(self)
        destination._transfers.add(self)

    def record(self, source_indices, destination_indices, count):
        """Record allocation-free validation and field transfer from GPU index lists."""
        self.source._ensure_open()
        self.destination._ensure_open()
        for name, value in (
            ("source indices", source_indices),
            ("destination indices", destination_indices),
            ("count", count),
        ):
            if (
                not isinstance(value, wp.array)
                or value.dtype != wp.int32
                or value.ndim != 1
                or value.device != self.device
                or not value.is_contiguous
            ):
                raise ValueError(f"Transfer {name} must be contiguous one-dimensional int32 on the storage device")
        if count.shape != (1,) or not source_indices.size or not destination_indices.size:
            raise ValueError("Transfer needs a scalar count and nonempty prepared index buffers")
        capacity = min(source_indices.size, destination_indices.size)
        wp.launch(
            _clear_transfer,
            max(self._source_seen.size, self._destination_seen.size, 1),
            [count, source_indices, destination_indices, self._source_seen, self._destination_seen, self.status],
            device=self.device,
        )
        wp.launch(
            _validate_transfer,
            capacity,
            [
                count,
                source_indices,
                destination_indices,
                self.source.ready_count,
                self.destination.ready_count,
                self._source_seen,
                self._destination_seen,
                self.status,
                int(self.same_owner),
                self.source.capacity,
                self.destination.capacity,
            ],
            device=self.device,
        )
        if self.same_owner:
            wp.launch(
                _validate_transfer_overlap,
                self.source.capacity,
                [self._source_seen, self._destination_seen, self.status],
                device=self.device,
            )
        wp.launch(
            _transfer_fields,
            (self._descriptors.size, self._worker_count),
            [self._descriptors, count, source_indices, destination_indices, self.status, self._worker_count],
            device=self.device,
        )

    def retain_graph(self, graph, *buffers):
        """Retain descriptors, field owners and caller index buffers until graph retirement."""
        self.source._ensure_open()
        self.destination._ensure_open()
        self.source.retain_graph(graph, self, *buffers)
        if self.destination is not self.source:
            self.destination.retain_graph(graph)
        return graph

    def memory_report(self):
        """Report owned transfer descriptors and validation bytes, excluding both field stores."""
        return {
            "fields": list(self.nonempty_field_names),
            "source_capacity": self.source.capacity,
            "destination_capacity": self.destination.capacity,
            "metadata_bytes": self._descriptors.capacity
            + self._source_seen.capacity
            + self._destination_seen.capacity
            + self.status.capacity,
            "scope": "transfer descriptors/validation scratch only; field storage counted by its owners",
        }
