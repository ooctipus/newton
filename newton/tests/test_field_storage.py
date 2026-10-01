# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU descriptor and fault tests for field storage; no CUDA execution.

A fake Warp allocation/view surface retains actual Python lifetimes, while the
mechanical backing owner uses the independent CUDA resource ledger from its tests.
GPU arithmetic, real views and captured replay are qualified by the storage probe.
"""

import ast
import gc
import math
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as real_wp

from newton._src.utils import field_storage as storage
from newton._src.utils.cuda_vmm import MemoryBacking
from newton.tests.test_cuda_vmm import FakeDriver


class FakeArray:
    api = None

    def __init__(self, data=None, *, shape=None, dtype=None, device=None, ptr=None, strides=None, deleter=None):
        api = self.api
        self._deleter = None
        api.check("view" if ptr is not None else "array")
        dtype = {int: real_wp.int32, float: real_wp.float32, bool: real_wp.bool}.get(dtype, dtype)
        self.dtype = dtype or real_wp.float32
        self.device = api.device if device is None else device
        if data is not None:
            values = np.asarray(data, dtype=api.numpy_type(self.dtype))
            shape = values.shape
        else:
            shape = (shape,) if isinstance(shape, int) else tuple(shape)
            values = np.zeros(shape, dtype=api.numpy_type(self.dtype)) if ptr is None else None
        self.shape, self.ndim = tuple(shape), len(shape)
        self.size = math.prod(shape)
        width = real_wp.types.type_size_in_bytes(self.dtype)
        expected = tuple(width * math.prod(shape[i + 1 :]) for i in range(len(shape)))
        self.strides = tuple(strides) if strides is not None else expected
        self.is_contiguous = self.strides == expected
        self.capacity = self.size * width
        if ptr is None:
            ptr = api.next_address
            api.next_address += ((self.capacity + 255) // 256 + 1) * 256
        self.ptr = ptr
        self._values = values
        self._deleter = deleter

    def numpy(self):
        self.api.events.append("read")
        return self._values.copy()

    def fill_(self, value):
        self.api.check("fill")
        self.api.events.append(("fill", self.ptr, value))
        self._values.fill(value)
        return self

    def __del__(self):
        if getattr(self, "_deleter", None):
            self.api.events.append("release_reference")
            self._deleter()
            self._deleter = None


class FakeWarp:
    array = FakeArray
    types = real_wp.types
    int32, uint32, uint8 = real_wp.int32, real_wp.uint32, real_wp.uint8
    float32, bool = real_wp.float32, real_wp.bool

    def __init__(self):
        self.device = SimpleNamespace(is_cuda=True, is_capturing=False, label="device0")
        self.next_address = 0x80000000
        self.events, self.failures, self.calls = [], {}, {}
        FakeArray.api = self

    @staticmethod
    def numpy_type(dtype):
        return np.dtype(getattr(dtype, "_wp_scalar_type_", dtype).__name__)

    def check(self, call):
        self.calls[call] = self.calls.get(call, 0) + 1
        self.events.append(call)
        failure = self.failures.pop((call, self.calls[call]), None)
        if failure is not None:
            raise failure

    def fail(self, call, *, after=1, failure=None):
        self.failures[(call, self.calls.get(call, 0) + after)] = failure or RuntimeError(f"injected {call}")

    def empty(self, shape, *, dtype, device):
        self.check("empty")
        return FakeArray(shape=shape, dtype=dtype, device=device)

    def clone(self, array):
        self.check("clone")
        return FakeArray(array.numpy(), dtype=array.dtype, device=array.device)

    def full(self, count, value, *, dtype, device):
        return FakeArray(np.full(count, value, dtype=self.numpy_type(dtype)), dtype=dtype, device=device)

    def launch(self, kernel, dim, inputs, *, device):
        self.check("launch")
        # Do not retain temporary views: a real launch only retains raw addresses.

    @staticmethod
    def get_stream(device):
        return SimpleNamespace(cuda_stream=11)

    def synchronize_stream(self, stream):
        self.check("synchronize")

    def synchronize_device(self, device):
        self.check("synchronize")


class FakeGraph:
    def __init__(self, device):
        self.device = device


@real_wp.kernel
def _read_typed_fields(
    basis: real_wp.array2d[real_wp.vec3],
    frame: real_wp.array[real_wp.mat33],
    half: real_wp.array2d[real_wp.float16],
    wide: real_wp.array[real_wp.int64],
    out: real_wp.array2d[float],
):
    row = real_wp.tid()
    out[row, 0] = basis[row, 0][1]
    out[row, 1] = frame[row][1, 2]
    out[row, 2] = float(half[row, 0])
    out[row, 3] = float(wide[row])


@real_wp.kernel
def _read_aligned_tiles(count: real_wp.array[int], source: real_wp.array3d[float], output: real_wp.array3d[float]):
    row = real_wp.tid()
    if row < count[0]:
        tile = real_wp.tile_load(source[row], shape=(16, 16), storage="shared", bounds_check=False, aligned=True)
        real_wp.tile_store(output[row], tile, bounds_check=False, aligned=True)


class StorageBoundaryTests(unittest.TestCase):
    """Enforce the mechanical storage composition and precise ownership vocabulary."""

    def test_only_field_storage_module_owns_typed_fields(self):
        """Forbid the rejected row abstraction and compatibility aliases."""
        source = Path(storage.__file__).parent
        self.assertFalse((source / "row_storage.py").exists())
        self.assertFalse((source.parents[1] / "tests" / "test_row_storage.py").exists())
        for name in ("RowStorage", "RowTransfer", "RowField", "TransferField"):
            self.assertFalse(hasattr(storage, name), name)
        self.assertEqual(
            set(storage.FieldSpec.__dataclass_fields__), {"name", "inner_shape", "dtype", "packed", "alignment_bytes"}
        )
        self.assertEqual(
            set(storage.FieldView.__dataclass_fields__),
            {"name", "array", "row_payload_bytes", "packed_offset_bytes", "transfer_view"},
        )

    def test_mechanical_owners_have_no_world_or_solver_dependency(self):
        """Keep topology and lifecycle imports above memory and graph mechanics."""
        source = Path(storage.__file__).parent
        for name in ("field_storage.py", "cuda_vmm.py", "cuda_graph.py"):
            tree = ast.parse((source / name).read_text())
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.append(node.module or "")
            for imported in imports:
                with self.subTest(source=name, imported=imported):
                    self.assertFalse(
                        imported.startswith(("newton._src.sim", "newton._src.solvers", "isaaclab", "mujoco"))
                    )


class FieldStorageTests(unittest.TestCase):
    def setUp(self):
        self.wp = FakeWarp()
        self.patch = patch.object(storage, "wp", self.wp)
        self.patch.start()
        self.rows, self.owners = [], []

    def tearDown(self):
        self.wp.failures.clear()
        for _owner, driver in self.owners:
            driver.failures.clear()
        gc.collect()
        for rows in reversed(self.rows):
            rows.close(streams=(11,))
        for owner, driver in reversed(self.owners):
            with owner.maintenance(streams=(11,)):
                owner.close()
            self.assertFalse(driver.reservations or driver.mappings or driver.handles)
        self.patch.stop()

    def source(self, capacity=257):
        def arr(shape, dtype):
            return FakeArray(shape=shape, dtype=dtype, device=self.wp.device)

        return SimpleNamespace(
            q=arr((capacity, 3), float),
            v=arr((capacity, 3), float),
            flags=arr((capacity, 3), bool),
            time=arr((capacity,), float),
            empty=arr((capacity, 0), float),
            ints=arr((capacity, 3), int),
        )

    def specs(self, *, with_ints=False):
        fields = (
            storage.FieldSpec("q", (3,), real_wp.float32),
            storage.FieldSpec("v", (3,), real_wp.float32),
            storage.FieldSpec("flags", (3,), real_wp.bool),
            storage.FieldSpec("time", (), real_wp.float32, False),
            storage.FieldSpec("empty", (0,), real_wp.float32),
        )
        return (*fields, storage.FieldSpec("ints", (3,), real_wp.int32)) if with_ints else fields

    def backing(self, pages=16):
        driver = FakeDriver()
        with patch.object(MemoryBacking, "_load_driver", return_value=driver):
            owner = MemoryBacking(pages * driver.granularity)
        self.owners.append((owner, driver))
        return owner, driver

    def create(self, *, vmm=True, initial=0, capacity=257, with_ints=False):
        source, count = self.source(capacity), FakeArray([0], dtype=int)
        owner, driver = self.backing() if vmm else (None, None)
        rows = storage.FieldStorage(
            capacity,
            count,
            fields=self.specs(with_ints=with_ints),
            backing=owner,
            initial_ready_count=initial if vmm else capacity,
        )
        self.rows.append(rows)
        return rows, source, owner, driver

    def test_explicit_field_layout_and_real_granule_accounting(self):
        """Verify explicit field layout and real granule accounting."""
        rows, source, owner, _ = self.create(initial=1)
        self.assertEqual(rows.row_stride_bytes, 32)
        self.assertEqual(rows.ready_rows, 128)
        self.assertEqual(rows.arrays["q"].strides, (32, 4))
        self.assertEqual(rows.arrays["v"].ptr - rows.arrays["q"].ptr, 12)
        self.assertEqual(rows.arrays["flags"].ptr - rows.arrays["q"].ptr, 24)
        self.assertEqual(rows.memory_report()["packed_payload_bytes_per_row"], 27)
        self.assertEqual(rows.memory_report()["padding_bytes_per_row"], 5)
        self.assertIsNot(rows.arrays["time"], source.time)
        self.assertNotIn("queue", rows.fields)
        self.assertEqual(rows.memory_report()["mapped_packed_bytes"], 4096)
        self.assertEqual(owner.memory_report()["references"], 3)
        rows.resize_backing(257, streams=(11,))
        self.assertEqual(rows.ready_rows, 257)
        self.assertEqual(rows.memory_report()["mapped_packed_bytes"], 12288)
        rows.resize_backing(0, streams=(11,))
        self.assertEqual(owner.memory_report()["mapped_bytes"], 0)
        self.assertEqual(rows.ready_rows, 0)

    def test_scalar_alignment_covers_vectors_matrices_and_mixed_scalar_widths(self):
        """Verify scalar alignment covers vectors matrices and mixed scalar widths."""
        wp = real_wp
        specs = (
            storage.FieldSpec("flag", (3,), wp.bool),
            storage.FieldSpec("half", (), wp.float16),
            storage.FieldSpec("basis", (), wp.vec3),
            storage.FieldSpec("frame", (), wp.mat33),
            storage.FieldSpec("wide", (), wp.int64),
            storage.FieldSpec("small", (), wp.uint16),
            storage.FieldSpec("orientation", (), wp.quat),
            storage.FieldSpec("pose", (), wp.transform),
        )
        rows = storage.FieldStorage(257, FakeArray([0], dtype=int), fields=specs)
        self.rows.append(rows)
        expected = {
            "flag": 0,
            "half": 4,
            "basis": 8,
            "frame": 20,
            "wide": 56,
            "small": 64,
            "orientation": 68,
            "pose": 84,
        }
        self.assertEqual(rows.row_stride_bytes, 112)
        self.assertEqual({name: field.packed_offset_bytes for name, field in rows.fields.items()}, expected)
        self.assertEqual(rows.memory_report()["packed_payload_bytes_per_row"], 107)
        self.assertEqual(rows.memory_report()["padding_bytes_per_row"], 5)
        for spec in specs:
            array = rows.arrays[spec.name]
            alignment = wp.types.type_size_in_bytes(getattr(spec.dtype, "_wp_scalar_type_", spec.dtype))
            self.assertEqual(array.ptr % alignment, 0, spec.name)
            self.assertEqual(array.strides[0] % alignment, 0, spec.name)
        self.assertEqual(rows.fields["half"].transfer_view.dtype, wp.uint8)
        self.assertEqual(rows.fields["basis"].transfer_view.dtype, wp.uint32)

    def test_transfer_field_names_validate_before_allocating_scratch(self):
        """Reject malformed schema names before constructing a copy plan."""
        rows, _, _, _ = self.create(vmm=False)
        for names in ("q", (), ("q", "q"), (["q"],), (4,)):
            before = dict(self.wp.calls)
            with self.subTest(names=names), self.assertRaises(ValueError):
                rows.prepare_transfer(rows, fields=names)
            self.assertEqual(self.wp.calls, before)

    def test_protected_prefix_is_independent_from_readiness(self):
        """Permit disposable ready scratch while protecting no logical rows."""
        rows, _, _, _ = self.create(initial=257)
        self.assertEqual(rows.ready_rows, 257)
        self.assertEqual(int(rows.protected_count.numpy()[0]), 0)
        rows.resize_backing(0, streams=(11,))
        self.assertEqual(rows.ready_rows, 0)
        rows.resize_backing(257, streams=(11,))
        rows.protected_count.fill_(129)
        with self.assertRaisesRegex(RuntimeError, "protected"):
            rows.resize_backing(128, streams=(11,))
        self.assertEqual(rows.ready_rows, 257)
        for obsolete in ("count", "row_bytes", "region"):
            self.assertFalse(hasattr(rows, obsolete), obsolete)

    def test_empty_fields_preserve_explicit_shape_without_payload(self):
        """Verify empty fields preserve explicit shape without payload."""
        rows, _, _, _ = self.create()
        self.assertEqual(rows.arrays["empty"].shape, (257, 0))
        self.assertEqual(rows.fields["empty"].row_payload_bytes, 0)
        self.assertEqual(rows.memory_report()["dense_field_bytes"], 257 * 4)
        self.assertFalse(hasattr(rows, "data"))
        self.assertFalse(hasattr(rows, "register_world_array"))

    def test_constructor_failures_release_partial_views_and_reservation(self):
        """Verify constructor failures release_reservation partial views and reservation."""
        for call in ("view", "empty", "fill", "synchronize"):
            with self.subTest(call=call):
                owner, driver = self.backing()
                count = FakeArray([0], dtype=int)
                # The fourth view fails after at least one pinned field is registered.
                self.wp.fail(call, after=4 if call == "view" else 1)
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    storage.FieldStorage(257, count, fields=self.specs(), backing=owner, initial_ready_count=1)
                self.assertEqual(owner.memory_report()["references"], 0)
                self.assertFalse(driver.reservations or driver.mappings)

    def test_constructor_map_failure_releases_reservation(self):
        """Verify constructor map failure releases reservation."""
        for call in ("cuMemMap", "cuMemSetAccess"):
            with self.subTest(call=call):
                owner, driver = self.backing()
                driver.fail(call)
                with self.assertRaisesRegex(RuntimeError, call):
                    storage.FieldStorage(
                        257, FakeArray([0], dtype=int), fields=self.specs(), backing=owner, initial_ready_count=1
                    )
                self.assertFalse(driver.reservations or driver.mappings)
                self.assertEqual(owner.memory_report()["references"], 0)

    def test_constructor_preserves_baseexception_if_cleanup_also_fails(self):
        """Verify constructor preserves baseexception if cleanup also fails."""
        owner, driver = self.backing()
        self.wp.fail("fill", failure=KeyboardInterrupt("stop"))
        driver.fail("cuMemAddressFree")
        rows = storage.FieldStorage.__new__(storage.FieldStorage)
        self.rows.append(rows)
        with self.assertRaises(BaseExceptionGroup) as caught:
            rows.__init__(257, FakeArray([0], dtype=int), fields=self.specs(), backing=owner, initial_ready_count=1)
        self.assertIsInstance(caught.exception.exceptions[0], KeyboardInterrupt)
        self.assertIn("cuMemAddressFree", str(caught.exception.exceptions[1]))
        self.assertEqual(owner.memory_report()["references"], 0)
        rows.close(streams=(11,))
        self.assertFalse(driver.reservations)

    def test_fixed_constructor_joins_before_dropping_last_storage_reference(self):
        """Verify fixed constructor joins before dropping last storage reference."""
        self.wp.fail("fill")
        with self.assertRaisesRegex(RuntimeError, "injected fill"):
            storage.FieldStorage(257, FakeArray([0], dtype=int), fields=self.specs())
        self.assertGreater(self.wp.events.index("synchronize"), self.wp.events.index("fill"))

    def test_external_typed_or_word_view_keeps_region_pinned_until_retry_close(self):
        """Verify external typed or word view keeps region pinned until retry close."""
        for word_view in (False, True):
            with self.subTest(word_view=word_view):
                rows, _, owner, driver = self.create(initial=1)
                escaped = rows.fields["q"].transfer_view if word_view else rows.arrays["q"]
                with self.assertRaisesRegex(RuntimeError, "external views"):
                    rows.close(streams=(11,))
                self.assertEqual(owner.memory_report()["references"], 1)
                self.assertTrue(driver.reservations)
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    rows.fill(escaped, 0)
                del escaped
                gc.collect()
                rows.close(streams=(11,))
                self.assertFalse(driver.reservations)

    def test_close_joins_before_unpin_and_fixed_view_retains_allocation(self):
        """Verify close joins before release_reference and fixed view retains allocation."""
        rows, _, owner, driver = self.create(initial=1)
        observed = []
        original = owner.release_reference

        def release_reference(region):
            observed.append(driver.calls["cuStreamSynchronize"])
            original(region)

        owner.release_reference = release_reference
        before = driver.calls["cuStreamSynchronize"]
        rows.close(streams=(11,))
        self.assertEqual(observed, [before + 1] * 3)
        fixed, _, _, _ = self.create(vmm=False)
        escaped = fixed.arrays["q"]
        allocation = weakref.ref(fixed._storage)
        fixed.close(streams=(11,))
        self.assertIsNotNone(allocation())
        del escaped
        gc.collect()
        self.assertIsNone(allocation())

    def test_graph_retention_prevents_close_without_clearing_fields(self):
        """Verify graph retention prevents close without clearing fields."""
        rows, _, _, _ = self.create()
        graph = rows.retain_graph(FakeGraph(self.wp.device))
        with self.assertRaisesRegex(RuntimeError, "graphs"):
            rows.close(streams=(11,))
        self.assertIn("q", rows.fields)
        with self.assertRaises(ValueError):
            rows.retain_graph(FakeGraph("foreign"))
        del graph
        gc.collect()
        rows.close(streams=(11,))

    def test_resize_rejects_live_retirement_invalid_count_and_missing_dependencies(self):
        """Verify resize rejects live retirement invalid count and missing dependencies."""
        rows, _, _, _ = self.create(initial=1)
        rows.protected_count._values[0] = 5
        with self.assertRaisesRegex(RuntimeError, "protected"):
            rows.resize_backing(4, streams=(11,))
        rows.protected_count._values[0] = rows.ready_rows + 1
        with self.assertRaisesRegex(RuntimeError, "invalid"):
            rows.resize_backing(100, streams=(11,))
        rows.protected_count._values[0] = -1
        with self.assertRaises(RuntimeError):
            rows.resize_backing(0, streams=(11,))
        rows.protected_count._values[0] = 0
        with self.assertRaises(ValueError):
            rows.resize_backing(0, streams=())
        for bad in (-1, 258, True, 1.5):
            with self.subTest(rows=bad), self.assertRaises(ValueError):
                rows.resize_backing(bad, streams=(11,))

    def test_successful_shrink_publishes_readiness_only_before_unmap(self):
        """Avoid a second identical publication while retaining the unmap barrier."""
        for joined in (False, True):
            with self.subTest(joined=joined):
                rows, _, owner, driver = self.create(initial=257)
                original = owner._driver

                def record(name, *args, driver=original):
                    if name == "cuMemUnmap":
                        self.wp.events.append(name)
                    return driver(name, *args)

                owner._driver = record
                self.wp.events.clear()
                if joined:
                    with owner.maintenance(streams=(11,)):
                        rows.resize_backing(128, protected_count_host=0)
                else:
                    rows.resize_backing(128, streams=(11,))
                publication = ("fill", rows.ready_count.ptr, 128)
                self.assertEqual(self.wp.events.count(publication), 1)
                self.assertEqual(self.wp.events.count("synchronize"), 1)
                self.assertLess(self.wp.events.index(publication), self.wp.events.index("cuMemUnmap"))
                self.assertLess(self.wp.events.index("synchronize"), self.wp.events.index("cuMemUnmap"))
                self.assertEqual(rows.ready_rows, 128)
                self.assertEqual(owner.mapped_ranges(rows.reservation), ((0, driver.granularity),))
                owner._driver = original

    def test_partial_unmap_withdraws_ready_tail_and_quarantines_until_retirement(self):
        """Verify partial unmap withdraws ready tail and quarantines until retirement."""
        rows, _, owner, driver = self.create(initial=257)
        driver.fail("cuMemUnmap", after=2)
        with self.assertRaisesRegex(RuntimeError, "cuMemUnmap"):
            rows.resize_backing(0, streams=(11,))
        self.assertEqual(rows.ready_rows, 0)
        self.assertEqual(int(rows.ready_count.numpy()[0]), 0)
        self.assertEqual(owner.mapped_ranges(rows.reservation), ((4096, 8192),))
        self.assertTrue(rows.memory_report()["service_failed"])
        with self.assertRaisesRegex(RuntimeError, "service failed"):
            rows.resize_backing(257, streams=(11,))
        with self.assertRaisesRegex(RuntimeError, "service failed"):
            rows.copy(rows.arrays["q"], rows.arrays["v"])
        rows.close(streams=(11,))
        self.assertFalse(driver.reservations or driver.mappings)

    def test_ready_device_capacity_publishes_after_map_and_withdraws_before_unmap(self):
        """Verify ready device capacity publishes after map and withdraws before unmap."""
        rows, _, owner, driver = self.create()

        def record(name, *args):
            if name in ("cuMemMap", "cuMemSetAccess", "cuMemUnmap"):
                self.wp.events.append(name)
            return driver(name, *args)

        owner._driver = record
        self.wp.events.clear()
        rows.resize_backing(1, streams=(11,))
        published = ("fill", rows.ready_count.ptr, 128)
        self.assertLess(self.wp.events.index("cuMemSetAccess"), self.wp.events.index(published))
        self.assertEqual(int(rows.ready_count.numpy()[0]), 128)
        self.wp.events.clear()
        rows.resize_backing(0, streams=(11,))
        withdrawn = ("fill", rows.ready_count.ptr, 0)
        self.assertLess(self.wp.events.index(withdrawn), self.wp.events.index("synchronize"))
        self.assertLess(self.wp.events.index("synchronize"), self.wp.events.index("cuMemUnmap"))
        self.assertEqual(int(rows.ready_count.numpy()[0]), 0)

    def test_map_access_or_rollback_failure_preserves_ledger_for_retirement(self):
        """Verify map access or rollback failure preserves ledger for retirement."""
        for rollback_failure in (False, True):
            with self.subTest(rollback=rollback_failure):
                rows, _, owner, driver = self.create()
                driver.fail("cuMemSetAccess")
                if rollback_failure:
                    driver.fail("cuMemUnmap")
                with self.assertRaises((RuntimeError, ExceptionGroup)):
                    rows.resize_backing(1, streams=(11,))
                self.assertTrue(rows.memory_report()["service_failed"])
                self.assertEqual(rows.ready_rows, 0)
                self.assertEqual(bool(owner.mapped_ranges(rows.reservation)), rollback_failure)
                rows.close(streams=(11,))
                self.assertFalse(driver.reservations or driver.mappings)

    def test_clean_budget_rejection_preserves_owner_for_retry_after_other_region_retires(self):
        """Verify clean budget rejection preserves owner for retry after other region retires."""
        owner, driver = self.backing(pages=1)
        other = owner.reserve(4096)
        with owner.maintenance(streams=(11,)):
            owner.map(other, 0, 4096)
        rows = storage.FieldStorage(
            257, FakeArray([0], dtype=int), fields=self.specs(), backing=owner, initial_ready_count=0
        )
        self.rows.append(rows)
        mapping_calls = driver.calls["cuMemMap"]
        with self.assertRaises(MemoryError):
            rows.resize_backing(1, streams=(11,))
        self.assertEqual(driver.calls["cuMemMap"], mapping_calls)
        self.assertFalse(rows.memory_report()["service_failed"])
        self.assertEqual(rows.ready_rows, 0)
        with owner.maintenance(streams=(11,)):
            owner.release_reservation(other)
        rows.resize_backing(1, streams=(11,))
        self.assertEqual(rows.ready_rows, 128)
        self.assertEqual(driver.calls["cuMemCreate"], 1, "Freed backing must be reused without allocation")

    def test_publication_memory_error_quarantines_after_successful_mapping(self):
        """Verify publication memory error quarantines after successful mapping."""
        rows, _, owner, _ = self.create()
        self.wp.fail("fill", failure=MemoryError("publication failed"))
        with self.assertRaisesRegex(MemoryError, "publication failed"):
            rows.resize_backing(1, streams=(11,))
        self.assertTrue(rows.service_failed)
        with self.assertRaises(AttributeError):
            rows.service_failed = False
        self.assertEqual(rows.ready_rows, 0)
        self.assertTrue(owner.mapped_ranges(rows.reservation))
        with self.assertRaisesRegex(RuntimeError, "service failed"):
            rows.resize_backing(1, streams=(11,))

    def test_nonprefix_mapping_is_not_mistaken_for_ready_capacity(self):
        """Verify nonprefix mapping is not mistaken for ready capacity."""
        rows, _, owner, _ = self.create(initial=257)
        with owner.maintenance(streams=(11,)):
            owner.unmap(rows.reservation, 4096, 4096)
        with self.assertRaisesRegex(RuntimeError, "contiguous prefix"):
            rows.resize_backing(257, streams=(11,))
        self.assertTrue(rows.memory_report()["service_failed"])

    def test_close_retry_after_partial_unmap_or_address_release_failure(self):
        """Verify close retry after partial unmap or address release_reservation failure."""
        for call in ("cuMemUnmap", "cuMemAddressFree"):
            with self.subTest(call=call):
                rows, _, _, driver = self.create(initial=257)
                driver.fail(call, after=2 if call == "cuMemUnmap" else 1)
                with self.assertRaisesRegex(RuntimeError, call):
                    rows.close(streams=(11,))
                self.assertTrue(rows.memory_report()["closed"])
                self.assertIsNotNone(rows.reservation)
                rows.close(streams=(11,))
                self.assertIsNone(rows.reservation)
                self.assertFalse(driver.reservations or driver.mappings)

    def test_copy_reuses_owned_transfer_views_without_allocating_descriptors(self):
        """Reuse prepared byte and word views while preserving external view ownership."""
        rows, source, _, _ = self.create(vmm=False)
        for name in ("q", "flags"):
            before = self.wp.calls.get("view", 0)
            destination, origin = rows.copy(rows.arrays[name], rows.arrays[name])
            self.assertIs(destination, rows.fields[name].transfer_view)
            self.assertIs(origin, rows.fields[name].transfer_view)
            self.assertEqual(self.wp.calls.get("view", 0), before)
            destination, origin = rows.copy(rows.arrays[name], getattr(source, name))
            self.assertIs(destination, rows.fields[name].transfer_view)
            self.assertIs(origin._view_owner, getattr(source, name))
            self.assertEqual(self.wp.calls.get("view", 0), before + 1)
        destination, origin = rows.copy(rows.arrays["q"], rows.arrays["v"])
        self.assertIs(destination, rows.fields["q"].transfer_view)
        self.assertIs(origin, rows.fields["v"].transfer_view)

    def test_copy_checks_count_device_shape_inner_layout_alignment_and_overlap(self):
        """Verify copy checks count device shape inner layout alignment and overlap."""
        rows, source, _, _ = self.create()
        valid = rows.arrays["q"]
        bad_arrays = [
            FakeArray(shape=(257, 3), dtype=float, device="foreign"),
            FakeArray(shape=(257, 2), dtype=float),
            FakeArray(shape=(257, 3), dtype=int),
            FakeArray(shape=(257, 3), dtype=float, strides=(24, 8)),
            FakeArray(shape=(257, 3), dtype=float, strides=(8, 4)),
            FakeArray(ptr=source.q.ptr + 1, shape=(257, 3), dtype=float),
        ]
        for bad in bad_arrays:
            with self.subTest(shape=bad.shape, strides=bad.strides), self.assertRaises(ValueError):
                rows.copy(valid, bad)
        for count in (
            FakeArray([0], dtype=int, device="foreign"),
            FakeArray([0, 1], dtype=int),
            FakeArray([0], dtype=float),
            None,
        ):
            if count is not None:
                with self.assertRaises(ValueError):
                    rows.copy(valid, source.q, count=count)
        with self.assertRaises(ValueError):
            rows.copy(None, valid)
        before = self.wp.calls.get("launch", 0)
        rows.copy(rows.arrays["empty"], source.empty)
        self.assertEqual(self.wp.calls.get("launch", 0), before)
        empty_foreign = FakeArray(shape=(257, 0), dtype=float, device="foreign")
        with self.assertRaises(ValueError):
            rows.copy(empty_foreign, empty_foreign)
        with self.assertRaises(ValueError):
            rows.fill(empty_foreign, 0)

    def test_copy_rejects_partial_external_aliases_before_emission(self):
        """A prefix copy must not race with shifted or differently strided input views."""
        rows, _, _, _ = self.create(vmm=False)
        for name in ("q", "flags"):
            destination = rows.arrays[name]
            for offset, stride in ((1, destination.strides[0]), (0, destination.strides[0] + 4)):
                width = real_wp.types.type_size_in_bytes(destination.dtype)
                source = FakeArray(
                    ptr=destination.ptr + offset * width,
                    shape=destination.shape,
                    dtype=destination.dtype,
                    strides=(stride, *destination.strides[1:]),
                )
                before = self.wp.calls.get("launch", 0)
                with (
                    self.subTest(name=name, offset=offset, stride=stride),
                    self.assertRaisesRegex(ValueError, "overlap"),
                ):
                    rows.copy(destination, source)
                self.assertEqual(self.wp.calls.get("launch", 0), before)

    def test_copy_allows_disjoint_packed_fields_and_skips_exact_self_copy(self):
        """Registered fields have disjoint row payloads despite overlapping address spans."""
        rows, _, _, _ = self.create(vmm=False)
        destination, source = rows.arrays["q"], rows.arrays["v"]
        self.assertGreater(destination.ptr + destination.strides[0] * rows.capacity, source.ptr)
        before = self.wp.calls.get("launch", 0)
        rows.copy(destination, source)
        self.assertEqual(self.wp.calls.get("launch", 0), before + 1)
        rows.copy(destination, destination)
        self.assertEqual(self.wp.calls.get("launch", 0), before + 1)

    def test_specs_validate_before_allocating_and_forbid_late_registration(self):
        """Verify specs validate before allocating and forbid late registration."""
        malformed = [
            (),
            (storage.FieldSpec("", (3,), real_wp.float32),),
            (storage.FieldSpec("q", (3,), real_wp.float32),) * 2,
            (storage.FieldSpec("q", [3], real_wp.float32),),
            (storage.FieldSpec("q", (-1,), real_wp.float32),),
            (storage.FieldSpec("q", (True,), real_wp.float32),),
            (storage.FieldSpec("q", (3,), object),),
            (storage.FieldSpec("q", (3,), real_wp.float32, 1),),
        ]
        count = FakeArray([0], dtype=int)
        before = self.wp.calls.get("empty", 0)
        for fields in malformed:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                storage.FieldStorage(257, count, fields=fields)
        self.assertEqual(self.wp.calls.get("empty", 0), before)
        rows, _, _, _ = self.create()
        self.assertFalse(hasattr(rows, "register_world_array"))
        with self.assertRaises(ValueError):
            rows.lookup(FakeArray(shape=(257, 3), dtype=float, device="foreign"))

    def test_all_dense_fixed_storage_is_valid_and_has_no_packed_allocation(self):
        """Verify all dense fixed storage is valid and has no packed allocation."""
        rows = storage.FieldStorage(
            3, FakeArray([0], dtype=int), fields=(storage.FieldSpec("status", (), real_wp.int32, False),)
        )
        self.rows.append(rows)
        self.assertEqual(rows.ready_rows, 3)
        self.assertEqual(rows.row_stride_bytes, 0)
        self.assertEqual(rows.memory_report()["dense_field_bytes"], 12)
        self.assertEqual(rows.memory_report()["mapped_packed_bytes"], 0)

    def test_fill_patterns_preserve_typed_bits_and_require_preparation_before_capture(self):
        """Verify fill patterns preserve typed bits and require preparation before capture."""
        rows, _, _, _ = self.create(with_ints=True)
        pattern = rows.prepare_fill(rows.arrays["q"], -1.5)
        self.assertEqual(pattern.numpy().tobytes(), np.float32(-1.5).tobytes())
        flags = rows.prepare_fill(rows.arrays["flags"], True)
        self.assertEqual(flags.dtype, real_wp.uint8)
        self.assertEqual(flags.numpy().tobytes(), b"\x01")
        integers = rows.prepare_fill(rows.fields["ints"].array, -2147483648)
        self.assertEqual(integers.numpy().tobytes(), np.int32(-2147483648).tobytes())
        self.wp.device.is_capturing = True
        self.assertIs(rows.prepare_fill(rows.arrays["q"], -1.5), pattern)
        with self.assertRaisesRegex(RuntimeError, "before graph capture"):
            rows.prepare_fill(rows.arrays["q"], 42)
        rows.fill(rows.arrays["q"], -1.5)
        self.wp.device.is_capturing = False

    def test_fill_pattern_identity_uses_typed_bytes_not_value_repr(self):
        """Distinct vector bytes cannot alias; equivalent typed values share capture-ready storage."""
        rows = storage.FieldStorage(
            2, FakeArray([2], dtype=int), fields=(storage.FieldSpec("value", (), real_wp.vec3d),)
        )
        self.rows.append(rows)
        first = np.array([1.000000001, 2.0, 3.0], dtype=np.float64)
        second = np.array([1.000000002, 2.0, 3.0], dtype=np.float64)
        with np.printoptions(precision=8):
            self.assertEqual(repr(first), repr(second))
        with patch.object(self.wp, "full", side_effect=real_wp.full):
            a = rows.prepare_fill(rows.arrays["value"], first)
            b = rows.prepare_fill(rows.arrays["value"], second)
            self.assertEqual(a.numpy().tobytes(), first.tobytes())
            self.assertEqual(b.numpy().tobytes(), second.tobytes())
            self.assertIsNot(a, b)
            allocations = self.wp.calls.get("array", 0)
            self.wp.device.is_capturing = True
            self.assertNotEqual(repr(first), repr(tuple(first)))
            self.assertIs(rows.prepare_fill(rows.arrays["value"], tuple(first)), a)
            self.assertIs(rows.prepare_fill(rows.arrays["value"], list(second)), b)
            self.assertEqual(self.wp.calls.get("array", 0), allocations)
            self.wp.device.is_capturing = False

    def test_fill_validates_explicit_count_and_overrides_live_count(self):
        """Verify fill validates explicit count and overrides live count."""
        rows, _, _, _ = self.create()
        count = FakeArray([0], dtype=int)
        rows.fill(rows.arrays["q"], 0, count=count)
        for invalid in (
            FakeArray([0], dtype=float),
            FakeArray([0, 1], dtype=int),
            FakeArray([0], dtype=int, device="foreign"),
        ):
            with self.assertRaises(ValueError):
                rows.fill(rows.arrays["q"], 0, count=invalid)
            with self.assertRaises(ValueError):
                rows.fill(rows.arrays["empty"], 0, count=invalid)

    def test_subword_fill_patterns_preserve_element_period_across_transfer_words(self):
        """Packed byte, half and odd-byte vector fields retain their typed fill period."""
        byte_vector = real_wp.types.vector(length=3, dtype=real_wp.uint8)
        cases = (
            (real_wp.bool, (), True, b"\x01", real_wp.uint8),
            (real_wp.bool, (4,), True, b"\x01" * 4, real_wp.uint32),
            (real_wp.uint8, (3,), 173, b"\xad", real_wp.uint8),
            (real_wp.uint8, (4,), 173, b"\xad" * 4, real_wp.uint32),
            (real_wp.float16, (3,), -1.5, np.float16(-1.5).tobytes(), real_wp.uint8),
            (real_wp.float16, (2,), -1.5, np.float16(-1.5).tobytes() * 2, real_wp.uint32),
            (byte_vector, (1,), (7, 131, 255), b"\x07\x83\xff", real_wp.uint8),
            (byte_vector, (4,), (7, 131, 255), b"\x07\x83\xff" * 4, real_wp.uint32),
        )
        for dtype, inner_shape, value, expected, transfer_dtype in cases:
            with self.subTest(dtype=dtype, inner_shape=inner_shape):
                rows = storage.FieldStorage(
                    7, FakeArray([7], dtype=int), fields=(storage.FieldSpec("value", inner_shape, dtype),)
                )
                self.rows.append(rows)
                # Use the real CPU scalar/vector conversion, keeping allocation
                # and launch behavior independent of CUDA through FakeWarp.
                with patch.object(self.wp, "full", side_effect=real_wp.full):
                    pattern = rows.prepare_fill(rows.arrays["value"], value)
                self.assertEqual(pattern.dtype, transfer_dtype)
                self.assertEqual(pattern.numpy().tobytes(), expected)

    def test_suffix_memory_operations_validate_start_and_count_before_launch(self):
        """Reject invalid row intervals even for empty fields without recording writes."""
        rows, _, _, _ = self.create()
        for start in (-1, 258, True, 0.5):
            with self.subTest(start=start), self.assertRaises(ValueError):
                rows.zero(start=start)
            with self.subTest(start=start), self.assertRaises(ValueError):
                rows.fill(rows.arrays["empty"], 0, start=start)
        for count in (FakeArray([0], dtype=float), FakeArray([0, 1], dtype=int)):
            with self.assertRaises(ValueError):
                rows.zero(count=count)
        before = self.wp.calls.get("launch", 0)
        rows.zero(start=257)
        rows.fill(rows.arrays["q"], 0, start=257)
        self.assertEqual(self.wp.calls.get("launch", 0), before)

    def test_closed_owner_rejects_operations_even_for_empty_arrays(self):
        """Verify closed owner rejects operations even for empty arrays."""
        rows, source, _, _ = self.create()
        rows.close(streams=(11,))
        actions = (
            lambda: rows.copy(source.empty, source.empty),
            lambda: rows.fill(source.empty, 0),
            rows.zero,
            lambda: rows.prepare_fill(source.empty, 0),
            lambda: rows.lookup(source.empty),
            lambda: rows.resize_backing(0, streams=(11,)),
            lambda: rows.retain_graph(FakeGraph(self.wp.device)),
        )
        for action in actions:
            with self.assertRaisesRegex(RuntimeError, "closed"):
                action()
        self.assertEqual(rows.memory_report()["mapped_packed_bytes"], 0)

    def test_consumer_alignment_is_explicit_validated_and_preserved_per_row(self):
        """Verify consumer alignment is explicit validated and preserved per row."""
        fields = (
            storage.FieldSpec("prefix", (), real_wp.uint8),
            storage.FieldSpec("matrix", (16, 16), real_wp.float32, alignment_bytes=16),
        )
        rows = storage.FieldStorage(257, FakeArray([0], dtype=int), fields=fields)
        self.rows.append(rows)
        self.assertEqual(rows.fields["matrix"].packed_offset_bytes, 16)
        self.assertEqual(rows.row_stride_bytes, 1040)
        self.assertEqual(rows.arrays["matrix"].ptr % 16, 0)
        self.assertEqual(rows.arrays["matrix"].strides[0] % 16, 0)
        for alignment in (0, -1, 1, 2, 3, 12, 32, True, 4.0):
            count = FakeArray([0], dtype=int)
            before = dict(self.wp.calls)
            with self.subTest(alignment_bytes=alignment), self.assertRaisesRegex(ValueError, "alignment"):
                storage.FieldStorage(
                    3, count, fields=(storage.FieldSpec("x", (), real_wp.float32, alignment_bytes=alignment),)
                )
            self.assertEqual(self.wp.calls, before)

    def test_same_granule_resize_checks_liveness_without_republishing_readiness(self):
        """Same granule resize checks liveness without republishing readiness."""
        rows, _, _, _ = self.create(initial=1)
        self.wp.events.clear()
        rows.resize_backing(20, streams=(11,))
        self.assertIn("read", self.wp.events)
        self.assertNotIn("fill", self.wp.events)
        self.assertNotIn("synchronize", self.wp.events)
        self.assertEqual(rows.ready_rows, 128)
        rows.protected_count._values[0] = 21
        with self.assertRaisesRegex(RuntimeError, "protected"):
            rows.resize_backing(20, streams=(11,))

    def test_borrowed_maintenance_requires_the_current_backing_scope(self):
        """Borrowed maintenance requires the current backing scope."""
        rows, _, owner, driver = self.create(initial=1)
        with self.assertRaisesRegex(RuntimeError, "maintenance"):
            rows.resize_backing(20)
        before = driver.calls.get("cuStreamSynchronize", 0)
        with owner.maintenance(streams=(11,)):
            rows.resize_backing(20)
            rows.resize_backing(257)
            rows.resize_backing(0)
        self.assertEqual(driver.calls["cuStreamSynchronize"] - before, 1)
        self.assertEqual(rows.ready_rows, 0)

    def test_joined_batch_count_avoids_duplicate_readback_but_preserves_retirement_checks(self):
        """Joined batch count avoids duplicate readback but preserves retirement checks."""
        rows, _, owner, _ = self.create(initial=1)
        with self.assertRaisesRegex(ValueError, "joined"):
            rows.resize_backing(20, streams=(11,), protected_count_host=0)
        with self.assertRaisesRegex(RuntimeError, "maintenance"):
            rows.resize_backing(20, protected_count_host=0)
        with owner.maintenance(streams=(11,)):
            self.wp.events.clear()
            rows.resize_backing(20, protected_count_host=10)
            self.assertNotIn("read", self.wp.events)
            for live in (-1, 21, rows.ready_rows + 1):
                with self.subTest(live=live), self.assertRaisesRegex(RuntimeError, "protected|invalid"):
                    rows.resize_backing(20, protected_count_host=live)


class FieldStorageGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        real_wp.init()
        if not real_wp.get_cuda_devices():
            raise unittest.SkipTest("Field storage requires CUDA")

    def test_fill_replay_distinguishes_equal_repr_vectors_and_reuses_equivalent_values(self):
        """Capture keeps distinct typed vectors and reuses patterns without device allocation."""
        wp = real_wp
        first = np.array([1.000000001, 2.0, 3.0], dtype=np.float64)
        second = np.array([1.000000002, 2.0, 3.0], dtype=np.float64)
        with np.printoptions(precision=8):
            self.assertEqual(repr(first), repr(second))
        with wp.ScopedDevice("cuda:0"):
            count = wp.array([3], dtype=int)
            fields = tuple(storage.FieldSpec(name, (), wp.vec3d) for name in ("first", "second"))
            rows = storage.FieldStorage(3, count, fields=fields)
            a = rows.prepare_fill(rows.arrays["first"], first)
            b = rows.prepare_fill(rows.arrays["second"], second)
            self.assertIsNot(a, b)
            wp.load_module(module=storage.__name__)
            with wp.ScopedCapture(capture_mode=wp.CaptureMode.THREAD_LOCAL) as capture:
                self.assertIs(rows.prepare_fill(rows.arrays["first"], list(first)), a)
                self.assertIs(rows.prepare_fill(rows.arrays["second"], tuple(second)), b)
                rows.fill(rows.arrays["first"], list(first))
                rows.fill(rows.arrays["second"], tuple(second))
            graph = rows.retain_graph(capture.graph)
            capture.graph = None
            wp.capture_launch(graph)
            for name, expected in (("first", first), ("second", second)):
                self.assertEqual(rows.arrays[name].numpy().tobytes(), np.tile(expected, (3, 1)).tobytes())
            graph = a = b = None
            gc.collect()
            rows.close(streams=(wp.get_stream().cuda_stream,))

    def test_explicit_alignment_supports_vectorized_tiles_after_unmap_and_regrow(self):
        """Verify explicit alignment supports vectorized tiles after unmap and regrow."""
        wp = real_wp
        with wp.ScopedDevice("cuda:0"):
            stream = wp.get_stream().cuda_stream
            count = wp.array([7], dtype=int)
            backing = MemoryBacking(2 * 1024**2)
            fields = (
                storage.FieldSpec("prefix", (), wp.uint8),
                storage.FieldSpec("matrix", (16, 16), wp.float32, alignment_bytes=16),
            )
            rows = storage.FieldStorage(7, count, fields=fields, backing=backing)
            output = wp.zeros((7, 16, 16), dtype=float)
            matrix = rows.arrays["matrix"]
            rows.prepare_fill(matrix, 3.5)
            rows.fill(matrix, 3.5)
            wp.launch_tiled(_read_aligned_tiles, dim=7, inputs=[count, matrix, output], block_dim=32)
            with wp.ScopedCapture(capture_mode=wp.CaptureMode.THREAD_LOCAL) as capture:
                wp.launch_tiled(_read_aligned_tiles, dim=7, inputs=[count, matrix, output], block_dim=32)
            graph = rows.retain_graph(capture.graph, output)
            capture.graph = None
            for active in (7, 0, 3):
                count.fill_(0)
                rows.resize_backing(active, streams=(stream,))
                count.fill_(active)
                rows.fill(matrix, 3.5)
                output.zero_()
                wp.capture_launch(graph)
                expected = np.zeros((7, 16, 16), np.float32)
                expected[:active] = 3.5
                np.testing.assert_array_equal(output.numpy(), expected)
            graph = matrix = None
            gc.collect()
            rows.close(streams=(stream,))
            with backing.maintenance(streams=(stream,)):
                backing.close()
            self.assertEqual(backing.memory_report()["physical_retained_bytes"], 0)

    def test_fixed_vmm_and_inplace_gathers_keep_one_graph_and_reject_before_write(self):
        """Verify fixed vmm and inplace gathers keep one graph and reject before write."""
        wp = real_wp
        fields = (
            storage.FieldSpec("q", (1024,), wp.float32),
            storage.FieldSpec("v", (7,), wp.int32),
            storage.FieldSpec("flags", (3,), wp.bool),
            storage.FieldSpec("time", (), wp.float32, False),
            storage.FieldSpec("empty", (0,), wp.float32),
            storage.FieldSpec("half", (1,), wp.float16),
            storage.FieldSpec("basis", (2,), wp.vec3),
            storage.FieldSpec("frame", (), wp.mat33),
            storage.FieldSpec("wide", (), wp.int64),
        )
        with wp.ScopedDevice("cuda:0"):
            stream = wp.get_stream().cuda_stream
            for inplace, vmm in ((False, False), (True, False), (False, True)):
                with self.subTest(inplace=inplace, vmm=vmm):
                    n = 257
                    backing = MemoryBacking(4 * 1024**2) if vmm else None
                    src = storage.FieldStorage(n, wp.zeros(1, dtype=int), fields=fields)
                    dst = (
                        src
                        if inplace
                        else storage.FieldStorage(
                            n,
                            wp.zeros(1, dtype=int),
                            fields=fields,
                            backing=backing,
                            initial_ready_count=0 if vmm else n,
                        )
                    )

                    def initialize(owner, offset):
                        count = wp.array([owner.ready_rows], dtype=int)
                        for name, array in owner.arrays.items():
                            if not array.size:
                                continue
                            shape = (*array.shape, *getattr(array.dtype, "_shape_", ()))
                            values = np.arange(math.prod(shape)).reshape(shape) + offset
                            if name == "flags":
                                values = values % 2 == 0
                            source = wp.array(values, dtype=array.dtype)
                            owner.copy(array, source, count=count)

                    def read(owner):
                        values = {}
                        for name, array in owner.arrays.items():
                            shape = (owner.ready_rows, *array.shape[1:])
                            values[name] = (
                                wp.array(
                                    ptr=array.ptr,
                                    shape=shape,
                                    strides=array.strides,
                                    dtype=array.dtype,
                                    device=array.device,
                                ).numpy()
                                if owner.ready_rows and array.size
                                else np.empty(shape)
                            )
                        if owner.ready_rows:
                            out = wp.empty((owner.ready_rows, 4), dtype=float)
                            wp.launch(
                                _read_typed_fields,
                                owner.ready_rows,
                                [owner.arrays[name] for name in ("basis", "frame", "half", "wide")] + [out],
                            )
                            expected = np.stack(
                                (
                                    values["basis"][:, 0, 1],
                                    values["frame"][:, 1, 2],
                                    values["half"][:, 0],
                                    values["wide"],
                                ),
                                axis=1,
                            ).astype(np.float32)
                            np.testing.assert_array_equal(out.numpy(), expected)
                        return values

                    initialize(src, 1000)
                    if dst is not src and not vmm:
                        initialize(dst, -1000)
                    plan = dst.prepare_transfer(src, fields=tuple(f.name for f in fields))
                    source_indices, destination_indices, count = (
                        wp.zeros(n, dtype=int),
                        wp.zeros(n, dtype=int),
                        wp.zeros(1, dtype=int),
                    )
                    plan.record(source_indices, destination_indices, count)
                    with wp.ScopedCapture(capture_mode=wp.CaptureMode.THREAD_LOCAL) as capture:
                        plan.record(source_indices, destination_indices, count)
                    graph = plan.retain_graph(capture.graph, source_indices, destination_indices, count)
                    capture.graph = None
                    pointers = {name: array.ptr for name, array in dst.arrays.items()}
                    with self.assertRaisesRegex(RuntimeError, "graphs"):
                        dst.close(streams=(stream,))
                    cases = [
                        ([], [], 0, 0),
                        ([200, 201], [1, 3], 2, 0),
                        ([210, 210], [4, 5], 2, 0),
                        ([211, 212], [6, 6], 2, 3),
                        ([-1], [7], 1, 2),
                        ([257], [7], 1, 2),
                        ([211], [257], 1, 2),
                        ([], [], -1, 1),
                        ([], [], 258, 1),
                        ([213], [8], 1, 0),
                    ]
                    if inplace:
                        cases += [([9], [9], 1, 4), ([10, 11], [11, 12], 2, 4), ([214, 215], [10, 11], 2, 0)]
                    if vmm:
                        cases = [([1], [0], 1, 2), ([], [], 0, 0), *cases]
                    for ordinal, (sources, destinations, num, status) in enumerate(cases):
                        if vmm and ordinal == 2:
                            dst.resize_backing(n, streams=(stream,))
                            initialize(dst, -1000)
                        a, b = np.zeros(n, np.int32), np.zeros(n, np.int32)
                        a[: len(sources)], b[: len(destinations)] = sources, destinations
                        source_indices.assign(a)
                        destination_indices.assign(b)
                        count.fill_(num)
                        before, source = read(dst), read(src)
                        wp.capture_launch(graph)
                        self.assertEqual(int(plan.status.numpy()[0]), status)
                        after = read(dst)
                        for name in before:
                            expected = before[name].copy()
                            if status == 0:
                                for source_row, destination_row in zip(sources, destinations, strict=True):
                                    expected[destination_row] = source[name][source_row]
                            np.testing.assert_array_equal(after[name], expected, err_msg=f"{ordinal=} {name=}")
                        self.assertEqual(pointers, {name: array.ptr for name, array in dst.arrays.items()})
                    if vmm:
                        dst.resize_backing(0, streams=(stream,))
                        count.fill_(1)
                        source_indices.zero_()
                        destination_indices.zero_()
                        wp.capture_launch(graph)
                        self.assertEqual(int(plan.status.numpy()[0]), 2)
                        dst.resize_backing(n, streams=(stream,))
                        wp.capture_launch(graph)
                        self.assertEqual(int(plan.status.numpy()[0]), 0)
                        np.testing.assert_array_equal(read(dst)["q"][0], read(src)["q"][0])
                    ref, graph = weakref.ref(graph), None
                    gc.collect()
                    self.assertIsNone(ref())
                    with self.assertRaisesRegex(RuntimeError, "transfer plans"):
                        dst.close(streams=(stream,))
                    plan = None
                    gc.collect()
                    dst.close(streams=(stream,))
                    if src is not dst:
                        src.close(streams=(stream,))
                    if backing:
                        with backing.maintenance(streams=(stream,)):
                            backing.close()
                        self.assertEqual(backing.memory_report()["physical_retained_bytes"], 0)

    def test_suffix_fills_preserve_prior_rows_for_word_and_byte_fields(self):
        """Suffix fills preserve prior rows for word and byte fields."""
        wp = real_wp
        with wp.ScopedDevice("cuda:0"):
            count = wp.array([7], dtype=int)
            specs = (
                storage.FieldSpec("q", (3,), wp.float32),
                storage.FieldSpec("flag", (3,), wp.bool),
                storage.FieldSpec("basis", (2,), wp.vec3),
                storage.FieldSpec("time", (), wp.float32, False),
            )
            rows = storage.FieldStorage(7, count, fields=specs)
            for array in rows.arrays.values():
                rows.fill(array, 1)
            rows.zero(start=4)
            for array in rows.arrays.values():
                expected = np.ones_like(array.numpy())
                expected[4:] = 0
                np.testing.assert_array_equal(array.numpy(), expected)
                rows.fill(array, 1)
                rows.fill(array, 0, start=6)
                expected[...] = 1
                expected[6:] = 0
                np.testing.assert_array_equal(array.numpy(), expected)
            array = None
            rows.close(streams=(wp.get_stream().cuda_stream,))

    def test_subword_fill_replay_preserves_values_and_row_bounds(self):
        """Fill subword scalar/vector fields across byte and word transfer boundaries."""
        wp = real_wp
        byte_vector = wp.types.vector(length=3, dtype=wp.uint8)
        cases = (
            ("bool_scalar", (), wp.bool, True),
            ("bool_words", (4,), wp.bool, True),
            ("byte_bytes", (3,), wp.uint8, 173),
            ("byte_words", (4,), wp.uint8, 173),
            ("half_bytes", (3,), wp.float16, -1.5),
            ("half_words", (2,), wp.float16, -1.5),
            ("vector_bytes", (1,), byte_vector, (7, 131, 255)),
            ("vector_words", (4,), byte_vector, (7, 131, 255)),
        )
        fields = tuple(storage.FieldSpec(name, shape, dtype, alignment_bytes=4) for name, shape, dtype, _ in cases)
        with wp.ScopedDevice("cuda:0"):
            stream = wp.get_stream().cuda_stream
            for vmm in (False, True):
                with self.subTest(vmm=vmm):
                    backing = MemoryBacking(4 * 1024**2) if vmm else None
                    count = wp.array([7], dtype=int)
                    rows = storage.FieldStorage(7, count, fields=fields, backing=backing)
                    for name, _, _, value in cases:
                        rows.prepare_fill(rows.arrays[name], value)
                    wp.load_module(module=storage.__name__)
                    with wp.ScopedCapture(capture_mode=wp.CaptureMode.THREAD_LOCAL) as capture:
                        for name, _, _, value in cases:
                            rows.fill(rows.arrays[name], value, start=1)
                    graph = rows.retain_graph(capture.graph)
                    capture.graph = None
                    for active in (0, 1, 4, 7):
                        rows.zero(count=rows.ready_count)
                        count.fill_(active)
                        wp.capture_launch(graph)
                        for name, _, _, value in cases:
                            actual = rows.arrays[name].numpy()
                            expected = np.zeros_like(actual)
                            expected[1:active] = value
                            np.testing.assert_array_equal(actual, expected, err_msg=f"{name}, active={active}")
                    graph = None
                    gc.collect()
                    rows.close(streams=(stream,))
                    if backing is not None:
                        with backing.maintenance(streams=(stream,)):
                            backing.close()


if __name__ == "__main__":
    unittest.main()
