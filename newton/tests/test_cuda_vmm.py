# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU driver-fault tests for byte ownership, context identity and retirement.

The fake driver owns an independent reservation/mapping/handle ledger and fails
selected CUDA calls before they mutate it. These tests do not establish actual
CUDA ordering or hardware behavior; GPU remapping probes cover those separately.
"""

import ctypes as ct
import threading
import unittest
import uuid
from collections import Counter

from newton._src.utils.cuda_vmm import CudaBacking, Region

FAKE_UUID = "GPU-00000000-0000-0000-0000-000000000001"


class FakeDriver:
    """Model CUDA resources independently of the owner's Python bookkeeping."""

    def __init__(self):
        self.context = 0xCAFE
        self.context_id = 37
        self.device = 0
        self.current_device = 0
        self.device_uuid = FAKE_UUID
        self.granularity = 4096
        self.stream_contexts = {11: self.context, 22: 0xBAD}
        self.calls = Counter()
        self.history = []
        self.failures = {}
        self.reservations = {}
        self.handles = set()
        self.mappings = {}
        self.access = {}
        self.next_address = 0x10000000
        self.next_handle = 100

    def fail(self, name, *, after=1):
        self.failures[(name, self.calls[name] + after)] = RuntimeError(f"injected {name}")

    @staticmethod
    def put(pointer, dtype, value):
        ct.cast(pointer, ct.POINTER(dtype))[0] = value

    @staticmethod
    def scalar(value):
        return value.value if hasattr(value, "value") else value

    def __call__(self, name, *args):
        self.calls[name] += 1
        self.history.append((name, args))
        failure = self.failures.pop((name, self.calls[name]), None)
        if failure is not None:
            raise failure
        if name == "cuInit":
            return
        if name == "cuCtxGetCurrent":
            self.put(args[0], ct.c_void_p, self.context)
        elif name == "cuCtxGetId":
            assert self.scalar(args[0]) == self.context
            self.put(args[1], ct.c_uint64, self.context_id)
        elif name == "cuDeviceGet":
            assert args[1] == 0
            self.put(args[0], ct.c_int, self.device)
        elif name == "cuCtxGetDevice":
            self.put(args[0], ct.c_int, self.current_device)
        elif name == "cuDeviceGetUuid_v2":
            ct.memmove(args[0], uuid.UUID(self.device_uuid.removeprefix("GPU-")).bytes, 16)
        elif name == "cuMemGetAllocationGranularity":
            self.put(args[0], ct.c_size_t, self.granularity)
        elif name == "cuStreamGetCtx":
            stream = self.scalar(args[0]) or 0
            context = self.context if stream == 0 else self.stream_contexts[stream]
            self.put(args[1], ct.c_void_p, context)
        elif name in ("cuStreamSynchronize", "cuEventSynchronize"):
            return
        elif name == "cuMemAddressReserve":
            _, size, alignment, address_hint, flags = args
            assert size > 0 and size % self.granularity == 0
            assert alignment == self.granularity
            assert address_hint == 0 and flags == 0
            address = self.next_address
            self.next_address += size + self.granularity
            self.reservations[address] = size
            self.put(args[0], ct.c_uint64, address)
        elif name == "cuMemAddressFree":
            address, size = args
            assert self.reservations[address] == size
            assert not any(address <= mapped < address + size for mapped in self.mappings)
            del self.reservations[address]
        elif name == "cuMemCreate":
            assert args[1] == self.granularity
            handle = self.next_handle
            self.next_handle += 1
            self.handles.add(handle)
            self.put(args[0], ct.c_uint64, handle)
        elif name == "cuMemRelease":
            (handle,) = args
            assert handle in self.handles and handle not in self.mappings.values()
            self.handles.remove(handle)
        elif name == "cuMemMap":
            address, size, offset, handle, flags = args
            assert size == self.granularity and offset == 0 and flags == 0
            assert address % self.granularity == 0 and address not in self.mappings
            assert any(
                base <= address and address + size <= base + length for base, length in self.reservations.items()
            )
            assert handle in self.handles
            self.mappings[address] = handle
        elif name == "cuMemSetAccess":
            address, size, _, count = args
            assert size == self.granularity and address in self.mappings and count == 1
            self.access[address] = 3
        elif name == "cuMemUnmap":
            address, size = args
            assert size == self.granularity and address in self.mappings
            del self.mappings[address]
            self.access.pop(address, None)
        else:
            raise AssertionError(f"Unexpected CUDA operation: {name}")


class BackingTests(unittest.TestCase):
    def setUp(self):
        self.owners = []

    def tearDown(self):
        for owner, driver in reversed(self.owners):
            driver.failures.clear()
            driver.context = 0xCAFE
            driver.context_id = 37
            if not owner.memory_report()["closed"]:
                with owner.maintenance(streams=(0,)):
                    owner.close()
            self.assertEqual(driver.reservations, {})
            self.assertEqual(driver.handles, set())
            self.assertEqual(driver.mappings, {})

    def make_owner(self, granules=4, *, budget=None):
        driver = FakeDriver()
        owner = CudaBacking(granules * driver.granularity if budget is None else budget, driver=driver)
        self.owners.append((owner, driver))
        return owner, driver

    def assert_ledger(self, owner, driver):
        report = owner.memory_report()
        mapped_handles = set(driver.mappings.values())
        self.assertEqual(len(mapped_handles), len(driver.mappings), "owner must not alias a retained granule")
        self.assertTrue(mapped_handles <= driver.handles)
        self.assertEqual(report["virtual_reserved_bytes"], sum(driver.reservations.values()))
        self.assertEqual(report["physical_retained_bytes"], len(driver.handles) * driver.granularity)
        self.assertEqual(report["mapped_bytes"], len(driver.mappings) * driver.granularity)
        self.assertEqual(report["spare_bytes"], len(driver.handles - mapped_handles) * driver.granularity)
        self.assertEqual(report["regions"], len(driver.reservations))
        self.assertLessEqual(report["physical_retained_bytes"], report["budget_bytes"])
        return report

    def test_constructor_requires_current_context_device_uuid_and_nonzero_granularity(self):
        """Verify constructor requires current context device uuid and nonzero granularity."""
        cases = (
            ("context", None),
            ("current_device", 1),
            ("device_uuid", "GPU-00000000-0000-0000-0000-000000000000"),
            ("granularity", 0),
        )
        for field, value in cases:
            with self.subTest(field=field):
                driver = FakeDriver()
                setattr(driver, field, value)
                with self.assertRaises(RuntimeError):
                    CudaBacking(4096, expected_uuid=FAKE_UUID, driver=driver)
                self.assertFalse(driver.reservations or driver.handles or driver.mappings)
        for value in (-1, 1.5, "4096"):
            with self.subTest(budget=value), self.assertRaises(ValueError):
                CudaBacking(value, driver=FakeDriver())

    def test_reservation_rounds_size_without_consuming_physical_budget(self):
        """Verify reservation rounds size without consuming physical budget."""
        owner, driver = self.make_owner(granules=0)
        region = owner.reserve(10 * driver.granularity + 1)
        self.assertEqual(region.requested_bytes, 10 * driver.granularity + 1)
        self.assertEqual(region.size_bytes, 11 * driver.granularity)
        self.assertEqual(self.assert_ledger(owner, driver)["physical_retained_bytes"], 0)
        self.assertEqual(driver.calls["cuMemCreate"], 0)
        for value in (0, -1, 0.5, "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                owner.reserve(value)

    def test_reservation_failure_does_not_create_owner_resources(self):
        """Verify reservation failure does not create owner resources."""
        owner, driver = self.make_owner()
        driver.fail("cuMemAddressReserve")
        with self.assertRaisesRegex(RuntimeError, "cuMemAddressReserve"):
            owner.reserve(1)
        self.assertEqual(self.assert_ledger(owner, driver)["regions"], 0)
        owner.reserve(1)
        self.assertEqual(self.assert_ledger(owner, driver)["regions"], 1)

    def test_size_limits_and_read_only_properties_prevent_silent_reconfiguration(self):
        """Verify size limits and read only properties prevent silent reconfiguration."""
        maximum = (1 << (8 * ct.sizeof(ct.c_size_t))) - 1
        driver = FakeDriver()
        with self.assertRaises(ValueError):
            CudaBacking(maximum + 1, driver=driver)
        for ordinal in (-1, 2**31, 1.5, "0"):
            with self.subTest(ordinal=ordinal), self.assertRaises(ValueError):
                CudaBacking(4096, device_ordinal=ordinal, driver=driver)
        self.assertEqual(driver.calls, {})
        owner, driver = self.make_owner()
        for size in (maximum, maximum + 1):
            with self.subTest(size=size), self.assertRaises(ValueError):
                owner.reserve(size)
        self.assertEqual(driver.calls["cuMemAddressReserve"], 0)
        for attribute, value in (("budget_bytes", maximum), ("granularity_bytes", 1), ("device_uuid", "changed")):
            with self.subTest(attribute=attribute), self.assertRaises(AttributeError):
                setattr(owner, attribute, value)
        region = owner.reserve(1)
        with self.assertRaises(AttributeError):
            region.address = 0
        self.assert_ledger(owner, driver)

    def test_current_context_is_checked_without_changing_it(self):
        """Verify current context is checked without changing it."""
        owner, driver = self.make_owner()
        driver.context = 0xBAD
        with self.assertRaisesRegex(RuntimeError, "context"):
            owner.reserve(1)
        self.assertFalse(driver.reservations)
        driver.context = 0xCAFE
        driver.context_id += 1
        with self.assertRaisesRegex(RuntimeError, "context"):
            owner.reserve(1)
        self.assertFalse(driver.reservations)
        forbidden = {"cuCtxSetCurrent", "cuCtxCreate", "cuCtxPushCurrent", "cuDevicePrimaryCtxRetain"}
        self.assertFalse(forbidden.intersection(driver.calls))

    def test_maintenance_requires_explicit_dependencies_and_preserves_barrier_order(self):
        """Verify maintenance requires explicit dependencies and preserves barrier order."""
        owner, driver = self.make_owner()
        region = owner.reserve(driver.granularity)
        with self.assertRaises(ValueError):
            with owner.maintenance():
                self.fail("empty dependencies must not enter maintenance")
        for operation in (
            lambda: owner.map(region, 0, region.size_bytes),
            lambda: owner.unmap(region, 0, region.size_bytes),
            lambda: owner.release(region),
            owner.trim,
            owner.close,
        ):
            with self.subTest(operation=operation), self.assertRaisesRegex(RuntimeError, "maintenance"):
                operation()
        start = len(driver.history)
        with owner.maintenance(streams=(0, 11), events=(50, 51)):
            owner.map(region, 0, region.size_bytes)
        operations = [name for name, _ in driver.history[start:]]
        self.assertEqual(operations.count("cuStreamSynchronize"), 2)
        self.assertEqual(operations.count("cuEventSynchronize"), 2)
        self.assertLess(
            max(i for i, name in enumerate(operations) if name.endswith("Synchronize")), operations.index("cuMemMap")
        )
        self.assert_ledger(owner, driver)

    def test_event_only_maintenance_and_default_stream_are_explicit_dependencies(self):
        """Verify event only maintenance and default stream are explicit dependencies."""
        owner, driver = self.make_owner()
        region = owner.reserve(driver.granularity)
        with owner.maintenance(events=(50,)):
            owner.map(region, 0, region.size_bytes)
        with owner.maintenance(streams=(0,)):
            owner.unmap(region, 0, region.size_bytes)
        self.assertEqual(driver.calls["cuEventSynchronize"], 1)
        self.assertEqual(driver.calls["cuStreamSynchronize"], 1)
        self.assert_ledger(owner, driver)

    def test_foreign_stream_and_failed_dependencies_never_authorize_mutation(self):
        """Verify foreign stream and failed dependencies never authorize mutation."""
        for name in ("foreign_stream", "cuStreamSynchronize", "cuEventSynchronize"):
            with self.subTest(case=name):
                owner, driver = self.make_owner()
                region = owner.reserve(driver.granularity)
                if name != "foreign_stream":
                    driver.fail(name)
                streams = (22,) if name == "foreign_stream" else (0,)
                expected_error = ValueError if name == "foreign_stream" else RuntimeError
                with self.assertRaises(expected_error):
                    with owner.maintenance(streams=streams, events=(50,)):
                        self.fail("invalid dependency must not enter maintenance")
                with self.assertRaisesRegex(RuntimeError, "maintenance"):
                    owner.map(region, 0, region.size_bytes)
                self.assertEqual(self.assert_ledger(owner, driver)["mapped_bytes"], 0)
                with owner.maintenance(streams=(0,)):
                    owner.map(region, 0, region.size_bytes)
                self.assert_ledger(owner, driver)

    def test_nested_scope_rejection_does_not_revoke_outer_maintenance(self):
        """Verify nested scope rejection does not revoke outer maintenance."""
        owner, driver = self.make_owner()
        region = owner.reserve(driver.granularity)
        with owner.maintenance(streams=(0,)):
            with self.assertRaisesRegex(RuntimeError, "nested"):
                with owner.maintenance(events=(50,)):
                    self.fail("nested scope must not enter")
            owner.map(region, 0, region.size_bytes)
        self.assertEqual(driver.calls["cuEventSynchronize"], 0)
        self.assert_ledger(owner, driver)

    def test_maintenance_is_thread_bound_and_context_is_rechecked(self):
        """Verify maintenance is thread bound and context is rechecked."""
        owner, driver = self.make_owner()
        region = owner.reserve(driver.granularity)
        errors = []
        with owner.maintenance(streams=(0,)):

            def other_thread():
                try:
                    owner.map(region, 0, region.size_bytes)
                except Exception as error:
                    errors.append(error)

            thread = threading.Thread(target=other_thread)
            thread.start()
            thread.join(timeout=1)
            self.assertFalse(thread.is_alive())
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], RuntimeError)
            driver.context = 0xBAD
            with self.assertRaisesRegex(RuntimeError, "context"):
                owner.map(region, 0, region.size_bytes)
            driver.context = 0xCAFE
            owner.map(region, 0, region.size_bytes)
        self.assert_ledger(owner, driver)

    def test_user_exception_ends_scope_but_keeps_successful_mutations_owned(self):
        """Verify user exception ends scope but keeps successful mutations owned."""
        owner, driver = self.make_owner()
        region = owner.reserve(driver.granularity)
        with self.assertRaisesRegex(ValueError, "user failure"):
            with owner.maintenance(streams=(0,)):
                owner.map(region, 0, region.size_bytes)
                raise ValueError("user failure")
        with self.assertRaisesRegex(RuntimeError, "maintenance"):
            owner.unmap(region, 0, region.size_bytes)
        self.assertEqual(self.assert_ledger(owner, driver)["mapped_bytes"], driver.granularity)

    def test_region_identity_and_range_checks_are_preflight(self):
        """Verify region identity and range checks are preflight."""
        owner, driver = self.make_owner()
        unit = driver.granularity
        region = owner.reserve(2 * unit)
        clone = Region(region.address, region.size_bytes, region.requested_bytes)
        for malformed in (clone, None, {}, region.address):
            with self.subTest(region=malformed), self.assertRaises(ValueError):
                owner.pin(malformed)
        with owner.maintenance(streams=(0,)):
            for offset, size in (
                (-unit, unit),
                (0, 0),
                (1, unit),
                (0, unit + 1),
                (unit, 2 * unit),
                (0.0, unit),
                (0, "1"),
            ):
                with self.subTest(offset=offset, size=size), self.assertRaises(ValueError):
                    owner.map(region, offset, size)
            with self.assertRaises(ValueError):
                owner.map(clone, 0, unit)
            self.assertEqual(driver.calls["cuMemCreate"], 0)
            owner.map(region, unit, unit)
            created = driver.calls["cuMemCreate"]
            with self.assertRaises(ValueError):
                owner.map(region, 0, 2 * unit)
            with self.assertRaises(ValueError):
                owner.unmap(region, 0, 2 * unit)
            self.assertEqual(driver.calls["cuMemCreate"], created)
            self.assertEqual(driver.calls["cuMemUnmap"], 0)
        self.assert_ledger(owner, driver)

    def test_mapping_query_coalesces_ranges_and_remains_a_context_free_snapshot(self):
        """Verify mapping query coalesces ranges and remains a context free snapshot."""
        owner, driver = self.make_owner(granules=6)
        unit = driver.granularity
        region = owner.reserve(6 * unit)
        self.assertEqual(owner.mapped_ranges(region), ())
        with owner.maintenance(streams=(0,)):
            # Deliberately insert ranges out of order.
            owner.map(region, 4 * unit, 2 * unit)
            owner.map(region, 0, 2 * unit)
            owner.map(region, 2 * unit, unit)
        expected = ((0, 3 * unit), (4 * unit, 2 * unit))
        snapshot = owner.mapped_ranges(region)
        self.assertEqual(snapshot, expected)
        driver.context = None
        self.assertEqual(owner.mapped_ranges(region), expected)
        driver.context = 0xCAFE
        with owner.maintenance(streams=(0,)):
            owner.unmap(region, unit, unit)
            self.assertEqual(owner.mapped_ranges(region), ((0, unit), (2 * unit, unit), (4 * unit, 2 * unit)))
            self.assertEqual(snapshot, expected)
            owner.release(region)
        for invalid in (region, None):
            with self.subTest(region=invalid), self.assertRaises(ValueError):
                owner.mapped_ranges(invalid)
        self.assert_ledger(owner, driver)

    def test_budget_counts_retained_handles_and_reuses_shared_spares(self):
        """Verify budget counts retained handles and reuses shared spares."""
        owner, driver = self.make_owner(budget=2 * 4096 + 2048)
        unit = driver.granularity
        first, second = owner.reserve(3 * unit), owner.reserve(2 * unit)
        with owner.maintenance(streams=(0,)):
            with self.assertRaises(MemoryError):
                owner.map(first, 0, 3 * unit)
            self.assertEqual(driver.calls["cuMemCreate"], 0)
            owner.map(first, 0, 2 * unit)
            handles = set(driver.handles)
            with self.assertRaises(MemoryError):
                owner.map(second, 0, unit)
            owner.unmap(first, 0, 2 * unit)
            self.assertEqual(self.assert_ledger(owner, driver)["spare_bytes"], 2 * unit)
            owner.map(second, 0, 2 * unit)
            self.assertEqual(driver.handles, handles)
            self.assertEqual(driver.calls["cuMemCreate"], 2)
            owner.release(first)
        self.assert_ledger(owner, driver)

    def test_pins_retain_addresses_but_allow_remapping_and_context_free_unpin(self):
        """Verify pins retain addresses but allow remapping and context free unpin."""
        owner, driver = self.make_owner()
        unit = driver.granularity
        region = owner.reserve(unit)
        owner.pin(region)
        owner.pin(region)
        with owner.maintenance(streams=(0,)):
            owner.map(region, 0, unit)
            owner.unmap(region, 0, unit)
            owner.map(region, 0, unit)
            for operation in (lambda: owner.release(region), owner.close):
                with self.assertRaisesRegex(RuntimeError, "retain"):
                    operation()
        self.assertEqual(owner.memory_report()["pins"], 2)
        driver.context = None
        owner.unpin(region)
        owner.unpin(region)
        with self.assertRaisesRegex(RuntimeError, "no pin"):
            owner.unpin(region)
        driver.context = 0xCAFE
        self.assert_ledger(owner, driver)

    def test_create_map_and_access_failures_roll_back_all_new_resources(self):
        """Verify create map and access failures roll back all new resources."""
        for operation, fail_after in (
            ("cuMemCreate", 1),
            ("cuMemCreate", 2),
            ("cuMemCreate", 3),
            ("cuMemMap", 1),
            ("cuMemMap", 2),
            ("cuMemSetAccess", 1),
            ("cuMemSetAccess", 2),
        ):
            with self.subTest(operation=operation, fail_after=fail_after):
                owner, driver = self.make_owner()
                region = owner.reserve(3 * driver.granularity)
                driver.fail(operation, after=fail_after)
                with owner.maintenance(streams=(0,)):
                    with self.assertRaisesRegex(RuntimeError, operation):
                        owner.map(region, 0, region.size_bytes)
                    report = self.assert_ledger(owner, driver)
                    self.assertEqual(report["physical_retained_bytes"], 0)
                    self.assertEqual(report["mapped_bytes"], 0)
                    owner.map(region, 0, region.size_bytes)
                self.assert_ledger(owner, driver)

    def test_failed_map_returns_existing_spares_without_destroying_them(self):
        """Verify failed map returns existing spares without destroying them."""
        owner, driver = self.make_owner()
        unit = driver.granularity
        region = owner.reserve(3 * unit)
        with owner.maintenance(streams=(0,)):
            owner.map(region, 0, unit)
            owner.unmap(region, 0, unit)
            spare = set(driver.handles)
            driver.fail("cuMemCreate")
            with self.assertRaisesRegex(RuntimeError, "cuMemCreate"):
                owner.map(region, 0, 3 * unit)
            self.assertEqual(driver.handles, spare)
            self.assertEqual(self.assert_ledger(owner, driver)["spare_bytes"], unit)
            driver.fail("cuMemMap")
            with self.assertRaisesRegex(RuntimeError, "cuMemMap"):
                owner.map(region, 0, 2 * unit)
            self.assertEqual(driver.handles, spare)
            self.assertEqual(self.assert_ledger(owner, driver)["spare_bytes"], unit)

    def test_failed_rollback_unmap_keeps_surviving_mapping_owned_for_retry(self):
        """Verify failed rollback unmap keeps surviving mapping owned for retry."""
        owner, driver = self.make_owner()
        unit = driver.granularity
        region = owner.reserve(2 * unit)
        driver.fail("cuMemSetAccess", after=2)
        driver.fail("cuMemUnmap")
        with owner.maintenance(streams=(0,)):
            with self.assertRaisesRegex(ExceptionGroup, "rollback") as error:
                owner.map(region, 0, 2 * unit)
            self.assertEqual(len(error.exception.exceptions), 2)
            self.assertEqual(set(driver.mappings), {region.address + unit})
            self.assertNotIn(region.address + unit, driver.access)
            self.assertEqual(owner.mapped_ranges(region), ((unit, unit),))
            report = self.assert_ledger(owner, driver)
            self.assertEqual(report["mapped_bytes"], unit)
            self.assertEqual(report["physical_retained_bytes"], unit)
            with self.assertRaises(ValueError):
                owner.map(region, 0, 2 * unit)
            for offset, size in owner.mapped_ranges(region):
                owner.unmap(region, offset, size)
            owner.map(region, 0, 2 * unit)
        self.assert_ledger(owner, driver)

    def test_failed_rollback_release_keeps_surviving_handle_as_spare(self):
        """Verify failed rollback release keeps surviving handle as spare."""
        owner, driver = self.make_owner()
        unit = driver.granularity
        region = owner.reserve(2 * unit)
        driver.fail("cuMemMap")
        driver.fail("cuMemRelease")
        with owner.maintenance(streams=(0,)):
            with self.assertRaisesRegex(ExceptionGroup, "rollback"):
                owner.map(region, 0, 2 * unit)
            report = self.assert_ledger(owner, driver)
            self.assertEqual(report["physical_retained_bytes"], unit)
            self.assertEqual(report["mapped_bytes"], 0)
            self.assertEqual(report["spare_bytes"], unit)
            owner.trim()
            self.assertEqual(self.assert_ledger(owner, driver)["physical_retained_bytes"], 0)

    def test_multiple_cleanup_failures_retain_both_mapping_and_spare_handle(self):
        """Verify multiple cleanup failures retain both mapping and spare handle."""
        owner, driver = self.make_owner()
        unit = driver.granularity
        region = owner.reserve(3 * unit)
        driver.fail("cuMemSetAccess", after=2)
        driver.fail("cuMemUnmap")
        driver.fail("cuMemRelease")
        with owner.maintenance(streams=(0,)):
            with self.assertRaisesRegex(ExceptionGroup, "rollback") as error:
                owner.map(region, 0, 3 * unit)
            self.assertEqual(len(error.exception.exceptions), 3)
            report = self.assert_ledger(owner, driver)
            self.assertEqual(report["mapped_bytes"], unit)
            self.assertEqual(report["spare_bytes"], unit)
            self.assertEqual(report["physical_retained_bytes"], 2 * unit)
            self.assertEqual(owner.mapped_ranges(region), ((unit, unit),))
            owner.close()
        self.assertTrue(self.assert_ledger(owner, driver)["closed"])

    def test_partial_unmap_failure_preserves_completed_work_and_remaining_range(self):
        """Verify partial unmap failure preserves completed work and remaining range."""
        owner, driver = self.make_owner()
        unit = driver.granularity
        region = owner.reserve(3 * unit)
        with owner.maintenance(streams=(0,)):
            owner.map(region, 0, 3 * unit)
            driver.fail("cuMemUnmap", after=2)
            with self.assertRaisesRegex(RuntimeError, "cuMemUnmap"):
                owner.unmap(region, 0, 3 * unit)
            report = self.assert_ledger(owner, driver)
            self.assertEqual(report["mapped_bytes"], 2 * unit)
            self.assertEqual(report["spare_bytes"], unit)
            self.assertEqual(owner.mapped_ranges(region), ((unit, 2 * unit),))
            with self.assertRaises(ValueError):
                owner.unmap(region, 0, 3 * unit)
            for offset, size in owner.mapped_ranges(region):
                owner.unmap(region, offset, size)
        self.assertEqual(self.assert_ledger(owner, driver)["spare_bytes"], 3 * unit)

    def test_release_retries_after_partial_unmap_or_address_free_failure(self):
        """Verify release retries after partial unmap or address free failure."""
        for operation in ("cuMemUnmap", "cuMemAddressFree"):
            with self.subTest(operation=operation):
                owner, driver = self.make_owner()
                unit = driver.granularity
                region = owner.reserve(3 * unit)
                with owner.maintenance(streams=(0,)):
                    owner.map(region, 0, 3 * unit)
                    driver.fail(operation, after=2 if operation == "cuMemUnmap" else 1)
                    with self.assertRaisesRegex(RuntimeError, operation):
                        owner.release(region)
                    self.assertEqual(self.assert_ledger(owner, driver)["regions"], 1)
                    owner.release(region)
                    report = self.assert_ledger(owner, driver)
                    self.assertEqual(report["regions"], 0)
                    self.assertEqual(report["spare_bytes"], 3 * unit)
                    with self.assertRaises(ValueError):
                        owner.release(region)
                    with self.assertRaises(ValueError):
                        owner.pin(region)

    def test_trim_releases_independent_handles_and_retains_failed_ones(self):
        """Verify trim releases independent handles and retains failed ones."""
        owner, driver = self.make_owner()
        unit = driver.granularity
        region = owner.reserve(4 * unit)
        with owner.maintenance(streams=(0,)):
            owner.map(region, 0, 4 * unit)
            owner.unmap(region, 0, 4 * unit)
            driver.fail("cuMemRelease")
            driver.fail("cuMemRelease", after=3)
            with self.assertRaises(ExceptionGroup) as error:
                owner.trim()
            self.assertEqual(len(error.exception.exceptions), 2)
            self.assertEqual(self.assert_ledger(owner, driver)["spare_bytes"], 2 * unit)
            self.assertEqual(driver.calls["cuMemRelease"], 4)
            owner.trim()
        self.assertEqual(self.assert_ledger(owner, driver)["physical_retained_bytes"], 0)

    def test_close_retries_surviving_resources_after_each_retirement_failure(self):
        """Verify close retries surviving resources after each retirement failure."""
        for operation in ("cuMemUnmap", "cuMemAddressFree", "cuMemRelease"):
            with self.subTest(operation=operation):
                owner, driver = self.make_owner()
                first, second = owner.reserve(driver.granularity), owner.reserve(driver.granularity)
                with owner.maintenance(streams=(0,)):
                    owner.map(first, 0, first.size_bytes)
                    owner.map(second, 0, second.size_bytes)
                    driver.fail(operation)
                    with self.assertRaisesRegex(ExceptionGroup, "ownership is retained"):
                        owner.close()
                    self.assertFalse(self.assert_ledger(owner, driver)["closed"])
                with owner.maintenance(events=(50,)):
                    owner.close()
                self.assertTrue(self.assert_ledger(owner, driver)["closed"])
                self.assertFalse(driver.reservations or driver.handles or driver.mappings)

    def test_closed_owner_is_idempotent_and_rejects_new_operations(self):
        """Verify closed owner is idempotent and rejects new operations."""
        owner, driver = self.make_owner()
        region = owner.reserve(driver.granularity)
        with owner.maintenance(streams=(0,)):
            owner.map(region, 0, region.size_bytes)
            owner.close()
        operations = len(driver.history)
        owner.close()
        self.assertEqual(len(driver.history), operations)
        for operation in (
            lambda: owner.reserve(1),
            lambda: owner.pin(region),
            lambda: owner.map(region, 0, region.size_bytes),
        ):
            with self.subTest(operation=operation), self.assertRaisesRegex(RuntimeError, "closed"):
                operation()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            with owner.maintenance(streams=(0,)):
                self.fail("closed owner cannot enter maintenance")
        self.assertTrue(self.assert_ledger(owner, driver)["closed"])

    def test_failed_region_growth_preserves_other_regions_and_global_handle_ownership(self):
        """Failed region growth preserves other regions and global handle ownership."""
        owner, driver = self.make_owner()
        unit = driver.granularity
        first, second = owner.reserve(2 * unit), owner.reserve(2 * unit)
        with owner.maintenance(streams=(0,)):
            owner.map(first, 0, 2 * unit)
            original = dict(driver.mappings)
            driver.fail("cuMemSetAccess", after=2)
            driver.fail("cuMemUnmap")
            with self.assertRaisesRegex(ExceptionGroup, "rollback"):
                owner.map(second, 0, 2 * unit)
            self.assertEqual(owner.mapped_ranges(first), ((0, 2 * unit),))
            self.assertEqual(owner.mapped_ranges(second), ((unit, unit),))
            for address, handle in original.items():
                self.assertEqual(driver.mappings[address], handle)
            self.assertEqual(self.assert_ledger(owner, driver)["mapped_bytes"], 3 * unit)
            owner.release(second)
            self.assertEqual(owner.mapped_ranges(first), ((0, 2 * unit),))
            self.assertEqual(self.assert_ledger(owner, driver)["mapped_bytes"], 2 * unit)


if __name__ == "__main__":
    unittest.main(verbosity=2)
