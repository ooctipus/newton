# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CUDA virtual addresses and a bounded pool of physical allocation granules.

This owner knows bytes and CUDA dependencies, never worlds, tensors or kernels.
The caller stops submissions before entering maintenance and joins every stream
that can reference these regions. Pins retain virtual addresses across remaps;
they do not authorize touching an unmapped range.
"""

import ctypes as ct
import sys
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass

_MAX_SIZE = ct.c_size_t(-1).value


class _Location(ct.Structure):
    _fields_ = [("type", ct.c_int), ("id", ct.c_int)]


class _AllocationFlags(ct.Structure):
    _fields_ = [("compression", ct.c_ubyte), ("rdma", ct.c_ubyte), ("usage", ct.c_ushort), ("reserved", ct.c_ubyte * 4)]


class _AllocationProp(ct.Structure):
    _fields_ = [
        ("type", ct.c_int),
        ("handles", ct.c_int),
        ("location", _Location),
        ("win32", ct.c_void_p),
        ("flags", _AllocationFlags),
    ]


class _Access(ct.Structure):
    _fields_ = [("location", _Location), ("flags", ct.c_int)]


@dataclass(frozen=True)
class Region:
    address: int
    size_bytes: int
    requested_bytes: int


class CudaBacking:
    """Own VMM reservations and handles in the already-current CUDA context.

    ``driver`` is an optional callable(name, *ctypes_arguments) for failure tests;
    it must raise on CUDA failures, exactly as the real driver binding does.
    No method pushes, sets, retains or creates a CUDA context. The caller owns
    that context and must keep it alive until this backing owner is closed.
    """

    def __init__(self, budget_bytes: int, *, device_ordinal: int = 0, expected_uuid: str | None = None, driver=None):
        if sys.version_info < (3, 11):
            raise RuntimeError("Experimental CUDA backing requires Python 3.11 or newer")
        if not isinstance(budget_bytes, int) or not 0 <= budget_bytes <= _MAX_SIZE:
            raise ValueError("budget_bytes must fit a nonnegative CUDA size_t")
        if not isinstance(device_ordinal, int) or not 0 <= device_ordinal < 2**31:
            raise ValueError("device_ordinal must fit a nonnegative CUDA device index")
        self._budget_bytes = budget_bytes
        self._lock = threading.RLock()
        self._regions: dict[int, Region] = {}
        self._pins: dict[int, int] = {}
        self._pages: dict[int, int] = {}
        self._handles: set[int] = set()
        self._free: list[int] = []
        self._maintenance_thread = None
        self._closed = False
        self._driver = driver if driver is not None else self._load_driver()
        self._driver("cuInit", 0)
        context, device, current_device = ct.c_void_p(), ct.c_int(), ct.c_int()
        self._driver("cuCtxGetCurrent", ct.byref(context))
        if not context.value:
            raise RuntimeError("CudaBacking requires an already-current CUDA context")
        self._context = context.value
        context_id = ct.c_uint64()
        self._driver("cuCtxGetId", context, ct.byref(context_id))
        self._context_id = context_id.value
        self._driver("cuDeviceGet", ct.byref(device), device_ordinal)
        self._driver("cuCtxGetDevice", ct.byref(current_device))
        if device.value != current_device.value:
            raise RuntimeError("Current CUDA context does not belong to device_ordinal")
        raw_uuid = (ct.c_ubyte * 16)()
        self._driver("cuDeviceGetUuid_v2", ct.byref(raw_uuid), device.value)
        self._device_uuid = "GPU-" + str(uuid.UUID(bytes=bytes(raw_uuid)))
        if expected_uuid is not None and self.device_uuid != expected_uuid:
            raise RuntimeError(f"Expected {expected_uuid}, found {self.device_uuid}")
        self._prop = _AllocationProp(type=1, location=_Location(1, device.value))
        self._access = _Access(_Location(1, device.value), 3)
        granularity = ct.c_size_t()
        self._driver("cuMemGetAllocationGranularity", ct.byref(granularity), ct.byref(self._prop), 0)
        self._granularity_bytes = granularity.value
        if not self.granularity_bytes:
            raise RuntimeError("CUDA reported zero VMM allocation granularity")

    @staticmethod
    def _load_driver():
        lib = ct.CDLL("libcuda.so.1")
        u64, size, ptr = ct.c_uint64, ct.c_size_t, ct.c_void_p
        signatures = {
            "cuInit": [ct.c_uint],
            "cuCtxGetCurrent": [ct.POINTER(ptr)],
            "cuCtxGetId": [ptr, ct.POINTER(u64)],
            "cuCtxGetDevice": [ct.POINTER(ct.c_int)],
            "cuDeviceGet": [ct.POINTER(ct.c_int), ct.c_int],
            "cuDeviceGetUuid_v2": [ptr, ct.c_int],
            "cuMemGetAllocationGranularity": [ct.POINTER(size), ct.POINTER(_AllocationProp), ct.c_int],
            "cuMemAddressReserve": [ct.POINTER(u64), size, size, u64, u64],
            "cuMemAddressFree": [u64, size],
            "cuMemCreate": [ct.POINTER(u64), size, ct.POINTER(_AllocationProp), u64],
            "cuMemRelease": [u64],
            "cuMemMap": [u64, size, size, u64, u64],
            "cuMemUnmap": [u64, size],
            "cuMemSetAccess": [u64, size, ct.POINTER(_Access), size],
            "cuStreamSynchronize": [ptr],
            "cuStreamGetCtx": [ptr, ct.POINTER(ptr)],
            "cuEventSynchronize": [ptr],
            "cuGetErrorName": [ct.c_int, ct.POINTER(ct.c_char_p)],
        }
        for name, args in signatures.items():
            getattr(lib, name).argtypes = args
            getattr(lib, name).restype = ct.c_int

        def call(name, *args):
            result = getattr(lib, name)(*args)
            if result:
                error = ct.c_char_p()
                lib.cuGetErrorName(result, ct.byref(error))
                label = error.value.decode() if error.value else "unknown CUDA error"
                raise RuntimeError(f"{name}: {result} ({label})")

        return call

    @property
    def budget_bytes(self):
        return self._budget_bytes

    @property
    def granularity_bytes(self):
        return self._granularity_bytes

    @property
    def device_uuid(self):
        return self._device_uuid

    def _check(self, maintenance=False):
        if self._closed:
            raise RuntimeError("CudaBacking is closed")
        context = ct.c_void_p()
        self._driver("cuCtxGetCurrent", ct.byref(context))
        if context.value != self._context:
            raise RuntimeError("CudaBacking requires its original CUDA context to be current")
        context_id = ct.c_uint64()
        self._driver("cuCtxGetId", context, ct.byref(context_id))
        if context_id.value != self._context_id:
            raise RuntimeError("The original CUDA context was destroyed and its handle was reused")
        if maintenance and self._maintenance_thread != threading.get_ident():
            raise RuntimeError("Mapping and retirement require an explicit maintenance scope")

    def _validate_region(self, region):
        if not isinstance(region, Region) or self._regions.get(region.address) is not region:
            raise ValueError("Region is foreign or has already been released")

    def _range(self, region, offset, nbytes):
        self._validate_region(region)
        if not isinstance(offset, int) or not isinstance(nbytes, int):
            raise ValueError("Mapping offsets and sizes must be integers")
        unit = self.granularity_bytes
        if offset < 0 or nbytes <= 0 or offset % unit or nbytes % unit or offset + nbytes > region.size_bytes:
            raise ValueError("Range must be positive, granule aligned and contained in its region")
        return range(region.address + offset, region.address + offset + nbytes, unit)

    @contextmanager
    def maintenance(self, *, streams=(), events=()):
        """Wait for supplied dependencies while the caller excludes new submissions.

        A nonempty tuple containing stream 0 is valid. An empty dependency set is
        rejected even during initialization so there is no implicit safe mode.
        CUDA graph objects and external views may remain alive during this scope.
        """
        streams, events = tuple(streams), tuple(events)
        if not streams and not events:
            raise ValueError("Maintenance requires explicit stream or event dependencies")
        with self._lock:
            self._check()
            if self._maintenance_thread is not None:
                raise RuntimeError("Maintenance scopes cannot be nested")
            for stream in streams:
                context = ct.c_void_p()
                self._driver("cuStreamGetCtx", stream, ct.byref(context))
                if context.value != self._context:
                    raise ValueError("Maintenance stream belongs to another CUDA context")
                self._driver("cuStreamSynchronize", stream)
            for event in events:
                self._driver("cuEventSynchronize", event)
            self._maintenance_thread = threading.get_ident()
            try:
                yield self
            finally:
                self._maintenance_thread = None

    def reserve(self, nbytes: int) -> Region:
        """Reserve virtual bytes only; this operation consumes no physical budget."""
        with self._lock:
            self._check()
            if not isinstance(nbytes, int) or not 0 < nbytes <= _MAX_SIZE:
                raise ValueError("Reservation size must fit a positive CUDA size_t")
            unit = self.granularity_bytes
            size = ((nbytes + unit - 1) // unit) * unit
            if size > _MAX_SIZE:
                raise ValueError("Rounded reservation size exceeds CUDA size_t")
            address = ct.c_uint64()
            self._driver("cuMemAddressReserve", ct.byref(address), size, unit, 0, 0)
            region = Region(address.value, size, nbytes)
            self._regions[region.address] = region
            self._pins[region.address] = 0
            return region

    def pin(self, region: Region):
        """Retain the VA while an external view or graph can still refer to it."""
        with self._lock:
            self._check()
            self._validate_region(region)
            self._pins[region.address] += 1

    def unpin(self, region: Region):
        # A Python view deleter can run without a current CUDA context. Decrement
        # metadata only; actual CUDA retirement still requires maintenance.
        with self._lock:
            self._validate_region(region)
            if self._pins[region.address] == 0:
                raise RuntimeError("Region has no pin to release")
            self._pins[region.address] -= 1

    def map(self, region: Region, offset: int, nbytes: int):
        """Back an entirely unmapped range, rolling back on allocation/map failure.

        Physical memory is uninitialized. The caller initializes new state before
        publishing it. If rollback itself fails, the exception includes cleanup
        errors and the ledger continues to own every surviving map and handle.
        """
        self._check(maintenance=True)
        addresses = self._range(region, offset, nbytes)
        if any(address in self._pages for address in addresses):
            raise ValueError("Map range overlaps an existing mapping")
        unit = self.granularity_bytes
        new_count = max(0, len(addresses) - len(self._free))
        if (len(self._handles) + new_count) * unit > self.budget_bytes:
            raise MemoryError("Mapping would exceed the physical backing budget")
        acquired, created, mapped = [], set(), []
        try:
            for _ in addresses:
                if self._free:
                    handle = self._free.pop()
                else:
                    output = ct.c_uint64()
                    self._driver("cuMemCreate", ct.byref(output), unit, ct.byref(self._prop), 0)
                    handle = output.value
                    self._handles.add(handle)
                    created.add(handle)
                acquired.append(handle)
            for address, handle in zip(addresses, acquired, strict=True):
                self._driver("cuMemMap", address, unit, 0, handle, 0)
                self._pages[address] = handle
                mapped.append(address)
                self._driver("cuMemSetAccess", address, unit, ct.byref(self._access), 1)
        except Exception as failure:
            cleanup_errors = []
            for address in reversed(mapped):
                try:
                    self._driver("cuMemUnmap", address, unit)
                    del self._pages[address]
                except Exception as error:
                    cleanup_errors.append(error)
            still_mapped = set(self._pages.values())
            for handle in acquired:
                if handle in still_mapped:
                    continue
                if handle in created:
                    try:
                        self._driver("cuMemRelease", handle)
                        self._handles.remove(handle)
                        continue
                    except Exception as error:
                        cleanup_errors.append(error)
                self._free.append(handle)
            if cleanup_errors:
                raise ExceptionGroup(
                    "Mapping failed and rollback was incomplete", [failure, *cleanup_errors]
                ) from failure
            raise

    def unmap(self, region: Region, offset: int, nbytes: int):
        """Return mapped granules to the shared physical pool; retain the VA.

        A driver error can leave a partially unmapped range. Completed operations
        remain reflected in the ledger; no handle is lost or released prematurely.
        Keep affected ranges withdrawn from work, query mapped_ranges(), and repair
        or retry within maintenance before publishing those ranges as ready again.
        """
        self._check(maintenance=True)
        addresses = self._range(region, offset, nbytes)
        if any(address not in self._pages for address in addresses):
            raise ValueError("Unmap range contains an unmapped granule")
        for address in addresses:
            self._driver("cuMemUnmap", address, self.granularity_bytes)
            self._free.append(self._pages.pop(address))

    def release(self, region: Region):
        """Release one unpinned reservation, retaining its backing in the pool."""
        self._check(maintenance=True)
        self._validate_region(region)
        if self._pins[region.address]:
            raise RuntimeError("Cannot release a region retained by external views or graphs")
        end = region.address + region.size_bytes
        addresses = tuple(address for address in self._pages if region.address <= address < end)
        for address in addresses:
            self.unmap(region, address - region.address, self.granularity_bytes)
        self._driver("cuMemAddressFree", region.address, region.size_bytes)
        del self._regions[region.address]
        del self._pins[region.address]

    def trim(self):
        """Release all currently unmapped physical handles back to CUDA."""
        self._check(maintenance=True)
        errors = []
        for index in range(len(self._free) - 1, -1, -1):
            handle = self._free[index]
            try:
                self._driver("cuMemRelease", handle)
                del self._free[index]
                self._handles.remove(handle)
            except Exception as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup("Some spare CUDA handles could not be released", errors)

    def mapped_ranges(self, region: Region) -> tuple[tuple[int, int], ...]:
        """Return coalesced (byte offset, size) mappings from the owner's ledger.

        This host query does not imply initialized contents, successfully granted
        access after a failed transaction, or eligibility for simulation work.
        """
        with self._lock:
            self._validate_region(region)
            end = region.address + region.size_bytes
            addresses = sorted(address for address in self._pages if region.address <= address < end)
            ranges = []
            for address in addresses:
                offset = address - region.address
                if ranges and ranges[-1][0] + ranges[-1][1] == offset:
                    start, size = ranges[-1]
                    ranges[-1] = (start, size + self.granularity_bytes)
                else:
                    ranges.append((offset, self.granularity_bytes))
            return tuple(ranges)

    def memory_report(self) -> dict:
        """Exact requested handle bytes and VA, excluding CUDA internal overhead.

        Caller-owned metadata, solver bytes, context/JIT storage and driver page
        tables are separate from this physical payload budget.
        """
        with self._lock:
            free, mapped = set(self._free), set(self._pages.values())
            assert len(free) == len(self._free) and len(mapped) == len(self._pages)
            assert not free.intersection(mapped) and free.union(mapped) == self._handles
            unit = self.granularity_bytes
            return {
                "budget_bytes": self.budget_bytes,
                "granularity_bytes": unit,
                "virtual_reserved_bytes": sum(region.size_bytes for region in self._regions.values()),
                "physical_retained_bytes": len(self._handles) * unit,
                "mapped_bytes": len(self._pages) * unit,
                "spare_bytes": len(self._free) * unit,
                "regions": len(self._regions),
                "pins": sum(self._pins.values()),
                "closed": self._closed,
            }

    def close(self):
        """Retire all reservations and handles after views/graphs release their pins.

        Must run inside maintenance. Idempotent after successful closure. Driver
        errors preserve remaining ownership so a later maintenance scope can retry.
        """
        if self._closed:
            return
        self._check(maintenance=True)
        if any(self._pins.values()):
            raise RuntimeError("Cannot close backing while external views or graphs retain regions")
        errors = []
        for region in tuple(self._regions.values()):
            try:
                self.release(region)
            except Exception as error:
                errors.append(error)
        try:
            self.trim()
        except Exception as error:
            errors.append(error)
        if errors:
            raise ExceptionGroup("CUDA backing cleanup was incomplete; ownership is retained", errors)
        self._closed = True
