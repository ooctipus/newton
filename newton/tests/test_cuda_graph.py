# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check prepared CUDA graph ownership, failures, count sources and launch bounds."""

import ctypes as ct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.utils import cuda_graph as bridge
from newton._src.utils.cuda_graph import DeviceGraphUpdates, Int32Parameter, KernelBinding


class DeviceGraphTests(unittest.TestCase):
    """Exercise the binding owner's preparation gates without CUDA or a compiler."""

    class Graph:
        pass

    class Stream:
        pass

    class Array:
        def __init__(self, size, **kwargs):
            self.ptr, self.capacity = id(self), size
            self.assigned = None

        def assign(self, data):
            self.assigned = data.copy()

        def fill_(self, value):
            self.assigned = value

    class Library:
        def __init__(self):
            self.calls, self.nodes = [], {11: 100, 12: 200}
            self.tail, self.prepare_failure, self.upload_failure = 11, None, 0

        def binding_size(self):
            return 128

        def capture_tail(self, stream, node, graph):
            self.calls.append(("tail", stream))
            node._obj.value, graph._obj.value = self.tail, self.nodes[self.tail]
            return 0

        def validate_node_owner(self, node, graph):
            self.calls.append(("validate", node))
            return 0 if self.nodes.get(node) == graph else -112

        def prepare_binding(self, node, rank, axis, extent, scalars, sources, maxima, count, output):
            self.calls.append(("prepare", node))
            return -999 if node == self.prepare_failure else 0

        def launch_update(self, *args):
            self.calls.append(("update", args[0]))
            return 0

        def instantiate_upload(self, graph, stream, flags, executable):
            self.calls.append(("instantiate", graph, stream, flags))
            if not self.upload_failure:
                executable._obj.value = 900
            return self.upload_failure

    def setUp(self):
        self.bridge = bridge
        self.original_load_library = bridge.DeviceGraphUpdates._load_library
        self.device = SimpleNamespace(is_cuda=True, captures={}, arch=120)
        self.stream = self.Stream()
        self.stream.device, self.stream.cuda_stream = self.device, 5
        self.count = SimpleNamespace(shape=(1,), dtype=bridge.wp.int32, device=self.device, is_contiguous=True, ptr=10)
        self.library = self.Library()
        self.addCleanup(patch.stopall)
        patch.object(bridge.DeviceGraphUpdates, "_load_library", return_value=self.library).start()
        patch.object(bridge.wp, "zeros", side_effect=self.Array).start()
        patch.object(bridge.wp, "get_stream", return_value=self.stream).start()
        self.owner = bridge.DeviceGraphUpdates(self.count, maximum=31, capacity_nodes=4)
        self.graph = self.Graph()
        self.graph.device, self.graph.graph_exec, self.graph.graph = self.device, None, 100

    def capture(self, *, child=False):
        self.device.captures[self.stream] = self.graph
        self.owner.capture_update()
        first = self.owner.capture_tail()
        nodes = [self.bridge.KernelBinding(first, rank=2)]
        if child:
            self.graph.graph, self.library.tail = 200, 12
            nodes.append(self.bridge.KernelBinding(self.owner.capture_tail(), rank=1))
            self.graph.graph = 100
        self.device.captures.clear()
        return nodes

    def test_descriptor_validation_precedes_allocation(self):
        """Verify descriptor validation precedes allocation."""
        for field, value in (("shape", (2,)), ("dtype", self.bridge.wp.float32), ("is_contiguous", False)):
            with self.subTest(field=field):
                bad = SimpleNamespace(**vars(self.count))
                setattr(bad, field, value)
                with self.assertRaises(ValueError):
                    self.bridge.DeviceGraphUpdates(bad, maximum=31, capacity_nodes=4)
        for capacity in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                self.bridge.DeviceGraphUpdates(self.count, maximum=31, capacity_nodes=capacity)

    def test_same_graph_conditional_children_bind_and_retain_one_owner(self):
        """Verify same graph conditional children bind and retain one owner."""
        bindings = self.capture(child=True)
        self.owner.bind(self.graph, bindings)
        self.assertEqual(self.owner.binding_count.assigned, 2)
        self.assertIn(self.owner, self.graph.world_owners)
        self.assertIn(self.count, self.graph.world_owners)
        self.owner.instantiate_upload(self.graph)
        self.assertEqual(self.graph.graph_exec.value, 900)
        with self.assertRaises(RuntimeError):
            self.owner.instantiate_upload(self.graph)

    def test_all_explicit_stream_entrypoints_reject_foreign_device(self):
        """Verify all explicit stream entrypoints reject foreign device."""
        foreign = self.Stream()
        foreign.device, foreign.cuda_stream = object(), 99
        for method in (self.owner.capture_tail, self.owner.capture_update):
            with self.assertRaises(ValueError):
                method(stream=foreign)
        self.assertEqual(self.library.calls, [])
        self.owner.bind(self.graph, self.capture())
        with self.assertRaises(ValueError):
            self.owner.instantiate_upload(self.graph, stream=foreign)
        self.assertFalse(any(call[0] == "instantiate" for call in self.library.calls))

    def test_capture_must_be_managed_ordered_and_recorded_once(self):
        """Verify capture must be managed ordered and recorded once."""
        with self.assertRaises(RuntimeError):
            self.owner.capture_update()
        self.device.captures[self.stream] = self.graph
        with self.assertRaises(RuntimeError):
            self.owner.capture_tail()
        self.owner.capture_update()
        with self.assertRaises(RuntimeError):
            self.owner.capture_update()
        other = self.Graph()
        self.device.captures[self.stream] = other
        with self.assertRaises(RuntimeError):
            self.owner.capture_tail()

    def test_foreign_graph_and_unrecorded_nodes_fail_before_marking(self):
        """Verify foreign graph and unrecorded nodes fail before marking."""
        bindings = self.capture()
        other = self.Graph()
        other.device, other.graph, other.graph_exec = self.device, 300, None
        with self.assertRaises(ValueError):
            self.owner.bind(other, bindings)
        with self.assertRaises(ValueError):
            self.owner.bind(self.graph, [self.bridge.KernelBinding(999, rank=1)])
        self.assertFalse(any(call[0] == "prepare" for call in self.library.calls))

    def test_native_node_ownership_is_rechecked_before_irreversible_marking(self):
        """Verify native node ownership is rechecked before irreversible marking."""
        bindings = self.capture()
        self.library.nodes[11] = 777
        with self.assertRaisesRegex(RuntimeError, "validate captured node owner"):
            self.owner.bind(self.graph, bindings)
        self.assertIsNone(self.owner._graph)
        self.assertFalse(any(call[0] == "prepare" for call in self.library.calls))

    def test_duplicate_scalar_indices_rank_types_and_duplicate_nodes_rejected(self):
        """Verify duplicate scalar indices rank types and duplicate nodes rejected."""
        bindings = self.capture()
        for binding in (
            self.bridge.KernelBinding(11, 2, parameters=(self.bridge.Int32Parameter(1, self.count, 31),) * 2),
            self.bridge.KernelBinding(11, True),
            self.bridge.KernelBinding(11, 2, extent_axis=True),
            self.bridge.KernelBinding(11, 2, parameters=(self.bridge.Int32Parameter(True, self.count, 31),)),
            self.bridge.KernelBinding(11, 2, parameters=(self.bridge.Int32Parameter(0, self.count, 31),)),
        ):
            with self.subTest(binding=binding), self.assertRaises(ValueError):
                self.owner.bind(self.graph, [binding])
        with self.assertRaises(ValueError):
            self.owner.bind(self.graph, bindings * 2)
        self.assertFalse(any(call[0] == "prepare" for call in self.library.calls))

    def test_independent_parameter_sources_are_validated_and_retained(self):
        """Verify independent parameter sources are validated and retained."""
        self.capture()
        contact_count = SimpleNamespace(**{**vars(self.count), "ptr": 88})
        ccd_count = SimpleNamespace(**{**vars(self.count), "ptr": 99})
        parameter = self.bridge.Int32Parameter(2, ccd_count, 127)
        binding = self.bridge.KernelBinding(11, 2, extent_source=contact_count, parameters=(parameter,))
        self.owner.bind(self.graph, [binding])
        self.assertIn(contact_count, self.graph.world_owners)
        self.assertIn(ccd_count, self.graph.world_owners)

    def test_parameter_sources_and_limits_fail_before_node_marking(self):
        """Verify parameter sources and limits fail before node marking."""
        self.capture()
        for field, value in (
            ("shape", (2,)),
            ("dtype", self.bridge.wp.float32),
            ("ptr", 0),
            ("device", object()),
            ("is_contiguous", False),
        ):
            source = SimpleNamespace(**vars(self.count))
            setattr(source, field, value)
            for binding in (
                self.bridge.KernelBinding(11, 2, extent_source=source),
                self.bridge.KernelBinding(11, 2, parameters=(self.bridge.Int32Parameter(1, source, 31),)),
            ):
                with self.subTest(field=field), self.assertRaises(ValueError):
                    self.owner.bind(self.graph, [binding])
        for maximum in (-1, True, 2**31, 1.5):
            with self.subTest(maximum=maximum), self.assertRaises(ValueError):
                self.owner.bind(
                    self.graph,
                    [
                        self.bridge.KernelBinding(
                            11, 2, parameters=(self.bridge.Int32Parameter(1, self.count, maximum),)
                        )
                    ],
                )
        with self.assertRaises(ValueError):
            self.owner.bind(self.graph, [self.bridge.KernelBinding(11, 2, extent_axis=None, extent_source=self.count)])
        self.assertFalse(any(call[0] == "prepare" for call in self.library.calls))

    def test_enable_count_requires_an_explicit_int32_limit(self):
        """Verify enable count requires an explicit int32 limit."""
        for maximum in (0, -1, True, 2**31, 1.5):
            with self.subTest(maximum=maximum), self.assertRaises(ValueError):
                self.bridge.DeviceGraphUpdates(self.count, maximum=maximum, capacity_nodes=4)

    def test_two_updaters_cannot_claim_the_same_node_even_after_failed_preparation(self):
        """Verify two updaters cannot claim the same node even after failed preparation."""
        bindings = self.capture()
        other = self.bridge.DeviceGraphUpdates(self.count, maximum=31, capacity_nodes=4)
        self.device.captures[self.stream] = self.graph
        other.capture_update()
        other.capture_tail()
        self.device.captures.clear()
        self.library.prepare_failure = 11
        with self.assertRaisesRegex(RuntimeError, "prepare binding"):
            self.owner.bind(self.graph, bindings)
        self.assertIs(self.graph.device_node_owners[11], self.owner)
        with self.assertRaisesRegex(ValueError, "already has"):
            other.bind(self.graph, bindings)
        self.assertIsNone(other._graph)
        self.assertEqual(sum(call[0] == "prepare" for call in self.library.calls), 1)

    def test_partial_preparation_failure_retains_owner_and_cannot_instantiate_or_rebind(self):
        """Verify partial preparation failure retains owner and cannot instantiate or rebind."""
        bindings = self.capture(child=True)
        self.library.prepare_failure = 12
        with self.assertRaisesRegex(RuntimeError, "prepare binding 1"):
            self.owner.bind(self.graph, bindings)
        self.assertIn(self.owner, self.graph.world_owners)
        with self.assertRaises(RuntimeError):
            self.owner.instantiate_upload(self.graph)
        with self.assertRaises(RuntimeError):
            self.owner.bind(self.graph, bindings)

    def test_failed_instantiation_does_not_allow_another_attempt(self):
        """Verify failed instantiation does not allow another attempt."""
        self.owner.bind(self.graph, self.capture())
        self.library.upload_failure = 999
        with self.assertRaisesRegex(RuntimeError, "instantiate/upload"):
            self.owner.instantiate_upload(self.graph)
        self.assertIsNone(self.graph.graph_exec)
        with self.assertRaises(RuntimeError):
            self.owner.instantiate_upload(self.graph)
        self.assertEqual(sum(call[0] == "instantiate" for call in self.library.calls), 1)

    def test_invalid_flags_do_not_consume_the_one_instantiation_attempt(self):
        """Verify invalid flags do not consume the one instantiation attempt."""
        self.owner.bind(self.graph, self.capture())
        for flags in (-1, 2**64, True, 1.5):
            with self.assertRaises(ValueError):
                self.owner.instantiate_upload(self.graph, flags=flags)
        self.assertFalse(self.owner._instantiation_attempted)
        self.owner.instantiate_upload(self.graph)

    def test_compiler_diagnostic_preserves_stderr_and_removes_partial_artifact(self):
        """Verify compiler diagnostic preserves stderr and removes partial artifact."""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "cuda_graph.cu"
            source.write_text("// fixture\n")
            outputs = []

            def fail(command, **kwargs):
                output = Path(command[command.index("-o") + 1])
                output.write_text("incomplete binary")
                outputs.append(output)
                raise self.bridge.subprocess.CalledProcessError(
                    2, command, output="compiler stdout\n", stderr="missing cuda.h"
                )

            with (
                patch.object(self.bridge, "__file__", str(source.with_suffix(".py"))),
                patch.object(self.bridge.subprocess, "run", side_effect=fail),
            ):
                with self.assertRaisesRegex(RuntimeError, "missing cuda.h") as raised:
                    self.original_load_library(self.owner, None)
            self.assertIn("compiler stdout", str(raised.exception))
            self.assertIn("nvcc", str(raised.exception))
            self.assertTrue(outputs)
            self.assertFalse(any(path.exists() for path in outputs))


class ParallelCaptureTests(unittest.TestCase):
    """Independent fake CUDA frontier model: ownership, DAG joins and failed graphs."""

    def setUp(self):
        self.ct, self.bridge = ct, bridge
        self.device = SimpleNamespace(is_cuda=True, captures={})
        self.stream = DeviceGraphTests.Stream()
        self.stream.device, self.stream.cuda_stream = self.device, 5
        self.graph = SimpleNamespace(device=self.device, capture_id=7, graph_exec=None)
        self.device.captures[self.stream] = self.graph
        self.native, self.identity, self.status, self.dependencies = 100, 7, 1, (11,)
        self.set_calls, self.get_failure, self.set_failure = [], 0, 0

        class Function:
            def __init__(self, implementation):
                self.implementation = implementation

            def __call__(self, *args):
                return self.implementation(*args)

        def get_info(stream, status, identity, graph, nodes, count):
            self.assertEqual(stream, 5)
            if self.get_failure:
                return self.get_failure
            status._obj.value, identity._obj.value, graph._obj.value = self.status, self.identity, self.native
            count._obj.value = len(self.dependencies)
            self.node_array = (ct.c_void_p * len(self.dependencies))(*self.dependencies)
            ct.cast(nodes, ct.POINTER(ct.POINTER(ct.c_void_p)))[0] = self.node_array
            return 0

        def set_dependencies(stream, nodes, count, flags):
            self.assertEqual((stream, flags), (5, 1))
            self.set_calls.append(tuple(nodes[i] for i in range(count)))
            if self.set_failure:
                return self.set_failure
            self.dependencies = self.set_calls[-1]
            return 0

        self.library = SimpleNamespace(
            cuStreamGetCaptureInfo_v2=Function(get_info), cuStreamUpdateCaptureDependencies=Function(set_dependencies)
        )
        self.addCleanup(patch.stopall)
        patch.object(bridge.ct, "CDLL", return_value=self.library).start()
        patch.object(bridge.wp, "get_stream", return_value=self.stream).start()

    def test_siblings_share_prefix_and_join_every_branch(self):
        """Verify siblings share prefix and join every branch."""
        observations = []

        def first():
            observations.append(self.dependencies)
            self.dependencies = (21, 22)

        def second():
            observations.append(self.dependencies)
            # Warp conditional resume may include an older parent's leaf.
            self.dependencies = (21, 31)
            self.identity = self.graph.capture_id = 8

        self.bridge.capture_parallel([first, second])
        self.assertEqual(observations, [(11,), (11,)])
        self.assertEqual(self.dependencies, (21, 22, 31))
        self.assertEqual(self.set_calls, [(11,), (11,), (21, 22, 31)])

        observations.clear()

        def first_guard():
            observations.append(self.dependencies)
            self.dependencies = (41,)

        def second_guard():
            observations.append(self.dependencies)
            self.dependencies = (42,)

        self.bridge.capture_parallel([first_guard, second_guard])
        self.assertEqual(observations, [(21, 22, 31), (21, 22, 31)])
        self.assertEqual(self.dependencies, (41, 42))
        self.assertEqual(self.set_calls[-3:], [(21, 22, 31), (21, 22, 31), (41, 42)])

    def test_empty_prefix_empty_branches_and_noop_callbacks(self):
        """Verify empty prefix empty branches and noop callbacks."""
        for prefix in ((), (11, 12)):
            self.dependencies = prefix
            self.bridge.capture_parallel([])
            self.assertEqual(self.dependencies, prefix)
            self.bridge.capture_parallel([lambda: None, lambda: None])
            self.assertEqual(self.dependencies, prefix)

    def test_nested_fork_restores_parent_and_outer_join(self):
        """Verify nested fork restores parent and outer join."""

        def leaf(node):
            def record():
                self.dependencies = (node,)

            return record

        self.bridge.capture_parallel([lambda: self.bridge.capture_parallel([leaf(21), leaf(22)]), leaf(31)])
        self.assertEqual(self.dependencies, (21, 22, 31))
        self.assertIn((21, 22), self.set_calls)

    def test_callback_failure_quarantines_all_later_binding_and_instantiation(self):
        """Verify callback failure quarantines all later binding and instantiation."""

        def fail():
            self.dependencies = (99,)
            raise ArithmeticError("branch failed")

        with self.assertRaisesRegex(ArithmeticError, "branch failed"):
            self.bridge.capture_parallel([fail])
        self.assertTrue(self.graph._parallel_capture_failed)
        with self.assertRaisesRegex(RuntimeError, "failed parallel"):
            self.bridge.capture_parallel([])
        owner = object.__new__(self.bridge.DeviceGraphUpdates)
        for operation in (lambda: owner.bind(self.graph, []), lambda: owner.instantiate_upload(self.graph)):
            with self.assertRaisesRegex(RuntimeError, "failed parallel"):
                operation()

    def test_native_parent_change_and_python_owner_change_are_rejected(self):
        """Verify native parent change and python owner change are rejected."""
        for changed in ("native", "python"):
            self.graph._parallel_capture_failed = False
            self.native = 100
            self.device.captures[self.stream] = self.graph

            def callback(changed=changed):
                if changed == "native":
                    self.native = 200
                else:
                    self.device.captures[self.stream] = object()

            with self.subTest(changed=changed), self.assertRaises(RuntimeError):
                self.bridge.capture_parallel([callback])
            self.assertTrue(self.graph._parallel_capture_failed)

    def test_caught_inner_failure_cannot_publish_successful_outer_join(self):
        """Verify caught inner failure cannot publish successful outer join."""

        def fail():
            raise ArithmeticError("nested failure")

        def catch():
            try:
                self.bridge.capture_parallel([fail])
            except ArithmeticError:
                pass

        with self.assertRaisesRegex(RuntimeError, "failed parallel"):
            self.bridge.capture_parallel([catch])
        self.assertTrue(self.graph._parallel_capture_failed)

    def test_missing_active_capture_identity_mismatch_and_null_graph(self):
        """Verify missing active capture identity mismatch and null graph."""
        for attribute, value in (("status", 0), ("identity", 99), ("native", 0)):
            original = getattr(self, attribute)
            setattr(self, attribute, value)
            self.graph._parallel_capture_failed = False
            with self.subTest(attribute=attribute), self.assertRaisesRegex(RuntimeError, "ownership disagree"):
                self.bridge.capture_parallel([])
            self.assertTrue(self.graph._parallel_capture_failed)
            setattr(self, attribute, original)
        self.assertEqual(self.set_calls, [])

    def test_query_and_edit_driver_failures_remain_failed(self):
        """Verify query and edit driver failures remain failed."""
        for attribute in ("get_failure", "set_failure"):
            self.graph._parallel_capture_failed = False
            setattr(self, attribute, 999)
            with self.subTest(attribute=attribute), self.assertRaisesRegex(RuntimeError, "999"):
                self.bridge.capture_parallel([lambda: None])
            self.assertTrue(self.graph._parallel_capture_failed)
            setattr(self, attribute, 0)

    def test_invalid_preconditions_do_not_edit_graph(self):
        """Verify invalid preconditions do not edit graph."""
        with self.assertRaises(TypeError):
            self.bridge.capture_parallel([42])
        self.graph.apic = True
        with self.assertRaisesRegex(ValueError, "APIC"):
            self.bridge.capture_parallel([])
        self.graph.apic = False
        self.device.captures.clear()
        with self.assertRaisesRegex(RuntimeError, "Warp-managed"):
            self.bridge.capture_parallel([])
        self.assertEqual(self.set_calls, [])


@wp.kernel(grid_stride=False)
def _lean_grid(output: wp.array3d[wp.uint8]):
    world, row, column = wp.tid()
    output[world, row, column] = wp.uint8(world % 251)


@wp.kernel(grid_stride=True)
def _capped_grid(output: wp.array[int]):
    index = wp.tid()
    output[index] = index + 1


@wp.kernel
def _count_worlds(worlds: int, contacts: int, ccd: int, output: wp.array[int]):
    output[wp.tid()] = worlds * 100000 + contacts * 1000 + ccd


@wp.kernel
def _count_contacts(ccd: int, output: wp.array[int]):
    output[wp.tid()] = ccd + 7


@wp.kernel
def _count_workers(contacts: int, ccd: int, stride: int, output: wp.array[int]):
    index = wp.tid()
    while index < contacts:
        output[index] = 1000 + ccd + index
        index += stride


class DeviceGraphGPU(unittest.TestCase):
    """Check actual graph replay separately from the fake preparation ledger."""

    @classmethod
    def setUpClass(cls):
        wp.init()
        if not wp.get_cuda_devices():
            raise unittest.SkipTest("Device-updated graphs require CUDA")

    def test_large_lean_grid_and_capped_grid_stride_keep_one_graph(self):
        """Verify all items of a large 3D grid and capped loops under changing GPU counts."""

        class Parameters(ct.Structure):
            _fields_ = [
                ("function", ct.c_void_p),
                ("gx", ct.c_uint),
                ("gy", ct.c_uint),
                ("gz", ct.c_uint),
                ("bx", ct.c_uint),
                ("by", ct.c_uint),
                ("bz", ct.c_uint),
                ("shared", ct.c_uint),
                ("params", ct.c_void_p),
                ("extra", ct.c_void_p),
            ]

        with wp.ScopedDevice("cuda:0"):
            capacity = 4096
            count = wp.zeros(1, dtype=int)
            output = wp.empty((capacity, 128, 128), dtype=wp.uint8)
            other = wp.empty(capacity, dtype=int)
            wp.launch(_lean_grid, (1, 128, 128), [output])
            wp.launch(_capped_grid, 1, [other], max_blocks=2)
            updater = DeviceGraphUpdates(count, maximum=capacity, capacity_nodes=2)
            with wp.ScopedCapture(capture_mode=wp.CaptureMode.THREAD_LOCAL) as capture:
                updater.capture_update()
                wp.launch(_lean_grid, (capacity, 128, 128), [output])
                first = KernelBinding(updater.capture_tail(), rank=3)
                wp.launch(_capped_grid, capacity, [other], max_blocks=2)
                second = KernelBinding(updater.capture_tail(), rank=1)
            graph = capture.graph
            driver = ct.CDLL("libcuda.so.1")
            driver.cuGraphKernelNodeGetParams.argtypes = [ct.c_void_p, ct.POINTER(Parameters)]
            grids = []
            for binding in (first, second):
                parameters = Parameters()
                self.assertEqual(driver.cuGraphKernelNodeGetParams(binding.node, ct.byref(parameters)), 0)
                grids.append((parameters.gx, parameters.gy, parameters.gz))
            self.assertGreater(grids[0][1], 1, "The test must exercise a captured grid with multiple Y blocks")
            self.assertEqual(grids[1], (2, 1, 1))
            updater.bind(graph, [first, second])
            updater.instantiate_upload(graph)
            graph.world_owners += (output, other)
            identity = graph.graph_exec.value
            for live in (4096, 1, 1023, 1024, 1025, 3073, 0, 4096, -1, 4097):
                with self.subTest(live=live):
                    output.fill_(255)
                    other.fill_(-1)
                    count.fill_(live)
                    wp.capture_launch(graph)
                    valid = 0 <= live <= capacity
                    active = live if valid else 0
                    expected = np.full(capacity, 255, np.uint8)
                    expected[:active] = np.arange(active) % 251
                    np.testing.assert_array_equal(
                        output.numpy(), np.broadcast_to(expected[:, None, None], output.shape)
                    )
                    expected_other = np.full(capacity, -1, np.int32)
                    expected_other[:active] = np.arange(1, active + 1, dtype=np.int32)
                    np.testing.assert_array_equal(other.numpy(), expected_other)
                    np.testing.assert_array_equal(updater.errors.numpy(), [0 if valid else -1] * 2)
                    self.assertEqual(graph.graph_exec.value, identity)

    def test_independent_enable_extent_and_scalar_counts_fail_closed(self):
        """Verify independent domain capacities and fixed worker grids without recapture."""
        with wp.ScopedDevice("cuda:0"):
            count, contacts, ccd = (wp.zeros(1, dtype=int) for _ in range(3))
            worlds, candidates, workers = (wp.full(n, -1, dtype=int) for n in (31, 127, 127))
            wp.launch(_count_worlds, 31, [31, 127, 511, worlds])
            wp.launch(_count_contacts, 127, [511, candidates])
            wp.launch(_count_workers, 16, [127, 511, 16, workers])
            for array in (worlds, candidates, workers):
                array.fill_(-1)
            updater = DeviceGraphUpdates(count, maximum=31, capacity_nodes=3)
            bindings = []
            with wp.ScopedCapture(capture_mode=wp.CaptureMode.THREAD_LOCAL) as capture:
                updater.capture_update()
                wp.launch(_count_worlds, 31, [31, 127, 511, worlds])
                bindings.append(
                    KernelBinding(
                        updater.capture_tail(),
                        rank=1,
                        parameters=(
                            Int32Parameter(1, count, 31),
                            Int32Parameter(2, contacts, 127),
                            Int32Parameter(3, ccd, 511),
                        ),
                    )
                )
                wp.launch(_count_contacts, 127, [511, candidates])
                bindings.append(
                    KernelBinding(
                        updater.capture_tail(),
                        rank=1,
                        extent_source=contacts,
                        parameters=(Int32Parameter(1, ccd, 511),),
                    )
                )
                wp.launch(_count_workers, 16, [127, 511, 16, workers])
                bindings.append(
                    KernelBinding(
                        updater.capture_tail(),
                        rank=1,
                        extent_axis=None,
                        parameters=(Int32Parameter(1, contacts, 127), Int32Parameter(2, ccd, 511)),
                    )
                )
            graph = capture.graph
            updater.bind(graph, bindings)
            updater.instantiate_upload(graph)
            graph.world_owners += (worlds, candidates, workers)
            identity = graph.graph_exec.value
            expected = [np.full(n, -1, np.int32) for n in (31, 127, 127)]
            for live, cap, ccd_cap in (
                (0, 0, 0),
                (1, 7, 2),
                (3, 0, 2),
                (5, 17, 300),
                (31, 127, 511),
                (32, 7, 2),
                (3, 128, 2),
                (4, 9, 512),
                (4, -1, 2),
                (-1, 10, 2),
                (7, 33, 4),
                (0, 0, 0),
            ):
                with self.subTest(worlds=live, contacts=cap, ccd=ccd_cap):
                    count.fill_(live)
                    contacts.fill_(cap)
                    ccd.fill_(ccd_cap)
                    wp.capture_launch(graph)
                    valid = 0 <= live <= 31 and 0 <= cap <= 127 and 0 <= ccd_cap <= 511
                    if valid and live:
                        expected[0][:live] = live * 100000 + cap * 1000 + ccd_cap
                        expected[1][:cap] = ccd_cap + 7
                        expected[2][:cap] = 1000 + ccd_cap + np.arange(cap, dtype=np.int32)
                    for actual, wanted in zip((worlds, candidates, workers), expected, strict=True):
                        np.testing.assert_array_equal(actual.numpy(), wanted)
                    np.testing.assert_array_equal(updater.errors.numpy(), [0 if valid else -1] * 3)
                    self.assertEqual(graph.graph_exec.value, identity)


if __name__ == "__main__":
    unittest.main()
