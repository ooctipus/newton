# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise generic world ownership with a raster-like compute consumer.

This is a test composition, not a renderer integration or an alternative physics
solver. Two prototypes retain differently sized primitive state and generate
independent sample work. No physics-backend schema or lifecycle adapter is used.
"""

import ast
import gc
import unittest
import weakref
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import warp as wp

from newton import worlds
from newton.worlds import (
    DeviceGraphUpdates,
    FieldSpec,
    FieldStorage,
    GraphKernelBinding,
    MemoryBacking,
    capture_parallel,
)

_ADMITTED, _MOVING = int(worlds.WorldPhase.ADMITTED), int(worlds.WorldPhase.MOVING)


@wp.kernel
def _prepare_primitives(
    position: wp.array2d[wp.vec3], velocity: wp.array2d[wp.vec3], seed: wp.array[int], prototype: int
):
    primitive = wp.tid()
    position[0, primitive] = wp.vec3(float(primitive + 1), float(prototype + 1), 0.0)
    velocity[0, primitive] = wp.vec3(1.0, float(prototype + 1), 2.0)
    if primitive == 0:
        seed[0] = 0


@wp.kernel
def _initialization_requests(
    batch: worlds.WorldBatchResult,
    transaction: worlds.WorldTransaction,
    prototype: int,
    request_indices: wp.array[int],
    destinations: wp.array[int],
    count: wp.array[int],
):
    ordinal = wp.tid()
    admitted = int(0)
    if batch.consumed[0] != 0 and transaction.phase[0] == _ADMITTED:
        admitted = transaction.request_starts[prototype + 1] - transaction.request_starts[prototype]
    if ordinal == 0:
        count[0] = admitted
    if ordinal < admitted:
        request = transaction.admitted_requests[transaction.request_starts[prototype] + ordinal]
        request_indices[ordinal] = request
        destinations[ordinal] = transaction.destination_slot[request]


@wp.kernel
def _initialize_seeds(
    batch: worlds.WorldBatchResult,
    transaction: worlds.WorldTransaction,
    request_indices: wp.array[int],
    destinations: wp.array[int],
    count: wp.array[int],
    copy_status: wp.array[int],
    seed: wp.array[int],
):
    ordinal = wp.tid()
    if ordinal < count[0] and copy_status[0] == 0:
        request = request_indices[ordinal]
        seed[destinations[ordinal]] = transaction.destination_world_id[request] + 1
        transaction.initialized_sequence[request] = batch.sequence[0]


@wp.kernel
def _compaction_count(
    transaction: worlds.WorldTransaction, moves: worlds.WorldCompaction, prototype: int, out: wp.array[int]
):
    out[0] = 0
    if transaction.phase[0] == _MOVING:
        out[0] = moves.count[prototype]


@wp.kernel
def _acknowledge_compaction(moves: worlds.WorldCompaction, prototype: int, count: wp.array[int], status: wp.array[int]):
    if status[0] == 0:
        moves.copied_count[prototype] = count[0]


@wp.kernel
def _move_primitives(position: wp.array2d[wp.vec3], velocity: wp.array2d[wp.vec3]):
    world, primitive = wp.tid()
    position[world, primitive] += 0.125 * velocity[world, primitive]


@wp.kernel
def _begin_samples(count: wp.array[int], valid: wp.array[int], completed: wp.array[int]):
    count[0] = 0
    valid[0] = 1
    completed[0] = 0


@wp.kernel
def _generate_samples(
    seed: wp.array[int],
    primitive_count: int,
    limit: wp.array[int],
    ready: wp.array[int],
    count: wp.array[int],
    valid: wp.array[int],
    primitive_location: wp.array[wp.vec2i],
):
    world, primitive = wp.tid()
    if primitive < 1 + seed[world] % primitive_count:
        sample = wp.atomic_add(count, 0, 1)
        if sample < wp.min(limit[0], ready[0]):
            primitive_location[sample] = wp.vec2i(world, primitive)
        else:
            wp.atomic_min(valid, 0, 0)


@wp.kernel
def _shade_samples(
    position: wp.array2d[wp.vec3],
    primitive_location: wp.array[wp.vec2i],
    value: wp.array[float],
    completed: wp.array[int],
):
    sample = wp.tid()
    world, primitive = primitive_location[sample][0], primitive_location[sample][1]
    p = position[world, primitive]
    value[sample] = p[0] + p[1] + p[2]
    wp.atomic_add(completed, 0, 1)


@dataclass
class _SamplePopulation:
    """Own one test prototype's retained fields, derived samples and count bindings."""

    primitive_count: int
    default_storage: FieldStorage
    state_storage: FieldStorage
    sample_storage: FieldStorage
    initialization_transfer: object
    compaction_transfer: object
    request_indices: wp.array
    source_rows: wp.array
    destination_rows: wp.array
    initialization_count: wp.array
    move_source_rows: wp.array
    move_destination_rows: wp.array
    move_count: wp.array
    sample_count: wp.array
    samples_valid: wp.array
    completed_samples: wp.array
    sample_limit: wp.array
    world_updates: DeviceGraphUpdates
    sample_updates: DeviceGraphUpdates
    world_bindings: list = field(default_factory=list)
    sample_bindings: list = field(default_factory=list)

    @property
    def world_ready_capacity(self):
        return min(self.state_storage.ready_rows, self.sample_storage.ready_rows // self.primitive_count)

    def record(self):
        state, samples = self.state_storage.arrays, self.sample_storage.arrays
        self.world_updates.record_update()
        wp.launch(
            _move_primitives,
            (self.state_storage.capacity, self.primitive_count),
            [state["position"], state["velocity"]],
        )
        self.world_bindings.append(GraphKernelBinding(self.world_updates.register_last_kernel_node(), 2))
        wp.launch(_begin_samples, 1, [self.sample_count, self.samples_valid, self.completed_samples])
        wp.launch(
            _generate_samples,
            (self.state_storage.capacity, self.primitive_count),
            [
                state["seed"],
                self.primitive_count,
                self.sample_limit,
                self.sample_storage.ready_count,
                self.sample_count,
                self.samples_valid,
                samples["primitive_location"],
            ],
        )
        self.world_bindings.append(GraphKernelBinding(self.world_updates.register_last_kernel_node(), 2))
        # The producer has now supplied this independent work domain's exact count.
        self.sample_updates.record_update()

        def shade():
            wp.launch(
                _shade_samples,
                self.sample_storage.capacity,
                [state["position"], samples["primitive_location"], samples["value"], self.completed_samples],
            )
            self.sample_bindings.append(GraphKernelBinding(self.sample_updates.register_last_kernel_node(), 1))

        wp.capture_if(self.samples_valid, on_true=shade)


class _SampleWorlds:
    """Test composition root; the directory and backing remain the sole authorities."""

    def __init__(self, *, vmm):
        self.device, self.stream = wp.get_device(), wp.get_stream()
        self.backing, self.directory, self.populations = None, None, []
        self._graph = None
        defaults_by_prototype = []
        if vmm:
            # Query the real allocation granularity without reserving any memory.
            query = MemoryBacking(0, device_ordinal=self.device.ordinal, expected_uuid=self.device.uuid)
            granularity = query.granularity_bytes
            with query.maintenance(streams=(self.stream.cuda_stream,)):
                query.close()
            self.backing = MemoryBacking(4 * granularity, device_ordinal=self.device.ordinal)
        capacities = []
        specifications = []
        for prototype, primitive_count in enumerate((2, 5)):
            specs = (
                FieldSpec("position", (primitive_count,), wp.vec3),
                FieldSpec("velocity", (primitive_count,), wp.vec3),
                FieldSpec("seed", (), wp.int32),
            )
            count = wp.ones(1, dtype=int)
            defaults = FieldStorage(1, count, fields=specs)
            defaults_by_prototype.append(defaults)
            specifications.append(specs)
            wp.launch(
                _prepare_primitives,
                primitive_count,
                [defaults.arrays["position"], defaults.arrays["velocity"], defaults.arrays["seed"], prototype],
            )
            capacities.append(2 * granularity // defaults.row_stride_bytes if vmm else 12)
        self.directory = worlds.WorldDirectory(
            tuple(capacities), id_capacity=sum(capacities), command_capacity=max(capacities)
        )
        self.commands = worlds.create_world_commands(max(capacities))
        self.results = worlds.create_world_results(max(capacities))
        self.sequence = 0
        start = 0
        for prototype, (capacity, specs, defaults) in enumerate(
            zip(capacities, specifications, defaults_by_prototype, strict=True)
        ):
            primitive_count = (2, 5)[prototype]
            state = FieldStorage(
                capacity,
                self.directory.data.live_count[prototype : prototype + 1],
                fields=specs,
                backing=self.backing,
                initial_ready_count=1 if vmm else capacity,
            )
            disposable = wp.zeros(1, dtype=int)
            sample_capacity = capacity * primitive_count
            samples = FieldStorage(
                sample_capacity,
                disposable,
                fields=(FieldSpec("primitive_location", (), wp.vec2i), FieldSpec("value", (), wp.float32)),
                backing=self.backing,
                initial_ready_count=primitive_count if vmm else sample_capacity,
            )
            requests, source_rows, destination_rows = (wp.zeros(max(capacities), dtype=int) for _ in range(3))
            initialization_count, move_count, sample_count, valid, completed = (
                wp.zeros(1, dtype=int) for _ in range(5)
            )
            limit = wp.array([sample_capacity], dtype=int)
            population = _SamplePopulation(
                primitive_count,
                defaults,
                state,
                samples,
                state.prepare_transfer(defaults, fields=("position", "velocity", "seed")),
                state.prepare_transfer(state, fields=("position", "velocity", "seed")),
                requests,
                source_rows,
                destination_rows,
                initialization_count,
                self.directory.compaction.source_slots[start : start + capacity],
                self.directory.compaction.destination_slots[start : start + capacity],
                move_count,
                sample_count,
                valid,
                completed,
                limit,
                DeviceGraphUpdates(state.protected_count, enable_count_maximum=capacity, binding_capacity=2),
                DeviceGraphUpdates(sample_count, enable_count_maximum=sample_capacity, binding_capacity=1),
            )
            self.populations.append(population)
            start += capacity
        self.publish_ready()

    def publish_ready(self):
        self.directory.publish_ready_slots(tuple(p.world_ready_capacity for p in self.populations))

    def memory_report(self):
        """Account each allocation once; graph execution, JIT and driver overhead are excluded."""
        storage_reports = [
            storage.memory_report()
            for p in self.populations
            for storage in (p.state_storage, p.sample_storage, p.default_storage)
        ]
        backing = self.backing.memory_report() if self.backing is not None else None
        # The directory owns live-count views and compaction indices. These values
        # are borrowed by populations and must not be counted again below.
        metadata = self.directory.memory_report()["directory_metadata_bytes"]
        metadata += sum(
            getattr(record, name).capacity for record in (self.commands, self.results) for name in record._cls.vars
        )
        for p in self.populations:
            metadata += sum(
                array.capacity
                for array in (
                    p.request_indices,
                    p.source_rows,
                    p.destination_rows,
                    p.initialization_count,
                    p.move_count,
                    p.sample_count,
                    p.samples_valid,
                    p.completed_samples,
                    p.sample_limit,
                    p.default_storage.protected_count,
                    p.sample_storage.protected_count,
                )
            )
            metadata += p.initialization_transfer.memory_report()["metadata_bytes"]
            metadata += p.compaction_transfer.memory_report()["metadata_bytes"]
            metadata += p.world_updates.memory_report()["device_payload_bytes"]
            metadata += p.sample_updates.memory_report()["device_payload_bytes"]
        metadata += sum(report["ready_metadata_bytes"] + report["fill_pattern_bytes"] for report in storage_reports)
        fixed_payload = sum(report["dense_field_bytes"] for report in storage_reports)
        fixed_payload += sum(p.default_storage.memory_report()["mapped_packed_bytes"] for p in self.populations)
        if backing is None:
            fixed_payload += sum(
                p.state_storage.memory_report()["mapped_packed_bytes"]
                + p.sample_storage.memory_report()["mapped_packed_bytes"]
                for p in self.populations
            )
        return {"metadata_bytes": metadata, "fixed_payload_bytes": fixed_payload, "backing": backing}

    def capture(self):
        wp.load_module(module=__name__)
        wp.load_module(module=FieldStorage.__module__)
        wp.load_module(module=worlds.WorldDirectory.__module__)
        with wp.ScopedCapture(force_module_load=False, capture_mode=wp.CaptureMode.THREAD_LOCAL) as capture:
            directory = self.directory
            directory.begin(self.commands)
            directory.admit(self.commands)
            for prototype, p in enumerate(self.populations):
                wp.launch(
                    _initialization_requests,
                    directory.command_capacity,
                    [
                        directory.batch_result,
                        directory.transaction,
                        prototype,
                        p.request_indices,
                        p.destination_rows,
                        p.initialization_count,
                    ],
                )
                p.initialization_transfer.record(p.source_rows, p.destination_rows, p.initialization_count)
                wp.launch(
                    _initialize_seeds,
                    directory.command_capacity,
                    [
                        directory.batch_result,
                        directory.transaction,
                        p.request_indices,
                        p.destination_rows,
                        p.initialization_count,
                        p.initialization_transfer.status,
                        p.state_storage.arrays["seed"],
                    ],
                )
            directory.publish(self.commands, self.results)
            directory.plan_compaction()
            for prototype, p in enumerate(self.populations):
                wp.launch(_compaction_count, 1, [directory.transaction, directory.compaction, prototype, p.move_count])
                p.compaction_transfer.record(p.move_source_rows, p.move_destination_rows, p.move_count)
                wp.launch(
                    _acknowledge_compaction,
                    1,
                    [directory.compaction, prototype, p.move_count, p.compaction_transfer.status],
                )
            directory.publish_compaction()
            capture_parallel(
                [
                    lambda p=p: wp.capture_if(directory.batch_result.advance_allowed, on_true=p.record)
                    for p in self.populations
                ]
            )
        graph = capture.graph
        for p in self.populations:
            p.world_updates.bind(graph, p.world_bindings)
            p.sample_updates.bind(graph, p.sample_bindings)
            p.initialization_transfer.retain_graph(graph)
            p.compaction_transfer.retain_graph(graph)
            p.sample_storage.retain_graph(graph)
        self.directory.retain_graph(graph, self)
        self.populations[0].world_updates.instantiate_and_upload(graph)
        self._graph = weakref.ref(graph)
        return graph

    def submit(self, graph, commands):
        self.sequence += 1
        capacity = self.directory.command_capacity
        if len(commands) > capacity:
            raise ValueError("Test batch exceeds its prepared command capacity")
        payload = np.zeros((capacity, 4), dtype=np.int64)
        if commands:
            payload[: len(commands)] = commands
        for name, column in (("operation", 0), ("world_id", 1), ("generation", 2), ("prototype", 3)):
            dtype = np.uint64 if name == "generation" else np.int32
            getattr(self.commands, name).assign(payload[:, column].astype(dtype))
        self.commands.count.fill_(len(commands))
        self.commands.sequence.fill_(self.sequence)
        wp.capture_launch(graph)
        return list(
            zip(
                self.results.world_id.numpy()[: len(commands)],
                self.results.generation.numpy()[: len(commands)],
                strict=True,
            )
        )

    def snapshot(self):
        data = self.directory.data
        slots, generations = data.slot_id.numpy(), data.generation.numpy()
        starts, counts = data.slot_starts.numpy(), data.live_count.numpy()
        result = {}
        for prototype, p in enumerate(self.populations):
            count = int(counts[prototype])
            positions = p.state_storage.arrays["position"][:count].numpy() if count else ()
            seeds = p.state_storage.arrays["seed"][:count].numpy() if count else ()
            for row in range(count):
                identity = int(slots[int(starts[prototype]) + row])
                result[(identity, int(generations[identity]))] = (prototype, positions[row].copy(), int(seeds[row]))
        return result

    def close(self):
        if self._graph is not None and self._graph() is not None:
            raise RuntimeError("Destroy the retained graph before closing the test composition")
        self.directory.close(streams=(self.stream,))
        for p in self.populations:
            p.initialization_transfer = p.compaction_transfer = None
        for p in self.populations:
            for storage in (p.state_storage, p.sample_storage, p.default_storage):
                storage.close(streams=(self.stream.cuda_stream,))
        if self.backing is not None:
            with self.backing.maintenance(streams=(self.stream.cuda_stream,)):
                self.backing.close()


class RuntimeCompositionArchitectureTests(unittest.TestCase):
    def test_public_owner_operations_are_documented(self):
        """Keep supported operations visible to the public API documentation filter."""
        owners = (
            worlds.WorldDirectory,
            worlds.FieldStorage,
            worlds.FieldTransfer,
            worlds.MemoryBacking,
            worlds.DeviceGraphUpdates,
        )
        for owner in owners:
            for name, member in vars(owner).items():
                if name.startswith("_") or not (
                    callable(member) or isinstance(member, (property, staticmethod, classmethod))
                ):
                    continue
                with self.subTest(owner=owner.__name__, member=name):
                    self.assertTrue((member.__doc__ or "").strip(), "Public operations must have docstrings")

    def test_generated_work_uses_its_own_count_and_capacity(self):
        """Execute the independent compute stages with dense CPU arrays before capture."""
        for primitives in (2, 5):
            with self.subTest(primitives=primitives), wp.ScopedDevice("cpu"):
                position = wp.zeros((2, primitives), dtype=wp.vec3)
                velocity = wp.full((2, primitives), wp.vec3(1.0), dtype=wp.vec3)
                seed = wp.array([0, primitives - 1], dtype=int)
                count, completed = wp.zeros(1, dtype=int), wp.zeros(1, dtype=int)
                valid = wp.ones(1, dtype=int)
                ready = wp.array([2 * primitives], dtype=int)
                limit = wp.clone(ready)
                locations = wp.full(2 * primitives, wp.vec2i(-1), dtype=wp.vec2i)
                value = wp.full(2 * primitives, -1.0)
                wp.launch(_move_primitives, (2, primitives), [position, velocity])
                wp.launch(_generate_samples, (2, primitives), [seed, primitives, limit, ready, count, valid, locations])
                self.assertEqual(int(count.numpy()[0]), primitives + 1)
                wp.launch(_shade_samples, primitives + 1, [position, locations, value, completed])
                self.assertEqual(int(completed.numpy()[0]), primitives + 1)
                np.testing.assert_array_equal(value.numpy()[: primitives + 1], np.full(primitives + 1, 0.375))
                np.testing.assert_array_equal(value.numpy()[primitives + 1 :], np.full(primitives - 1, -1.0))
                prior_locations = locations.numpy().copy()
                limit.zero_()
                wp.launch(_begin_samples, 1, [count, valid, completed])
                wp.launch(_generate_samples, (2, primitives), [seed, primitives, limit, ready, count, valid, locations])
                self.assertEqual(int(count.numpy()[0]), primitives + 1)
                self.assertEqual(int(valid.numpy()[0]), 0)
                np.testing.assert_array_equal(locations.numpy(), prior_locations)

    def test_consumer_uses_only_public_domain_independent_contracts(self):
        """Keep extensions free from private implementation, solver and task dependencies."""
        tree = ast.parse(Path(__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imports = [node.module or ""]
            else:
                continue
            for imported in imports:
                self.assertFalse(imported.startswith(("mujoco", "isaaclab", "newton._src", "newton.solvers")))
        self.assertFalse(
            any(isinstance(node, ast.ClassDef) and node.name.endswith("Adapter") for node in ast.walk(tree))
        )


class RuntimeCompositionGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        if not wp.get_cuda_devices():
            raise unittest.SkipTest("Independent captured composition requires CUDA")

    def assert_state_and_samples(self, runtime, before):
        current = runtime.snapshot()
        advance = int(runtime.directory.batch_result.advance_allowed.numpy()[0])
        for handle, (prototype, actual, seed) in current.items():
            count = runtime.populations[prototype].primitive_count
            initial = np.array([[i + 1, prototype + 1, 0] for i in range(count)], dtype=np.float32)
            expected = before[handle][1] if handle in before else initial
            expected = expected + advance * 0.125 * np.array([1, prototype + 1, 2], dtype=np.float32)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(seed, handle[0] + 1)
        if advance:
            for p in runtime.populations:
                self.assertFalse(np.any(p.world_updates.errors.numpy()))
                self.assertFalse(np.any(p.sample_updates.errors.numpy()))
                produced, completed = int(p.sample_count.numpy()[0]), int(p.completed_samples.numpy()[0])
                if not int(p.samples_valid.numpy()[0]):
                    self.assertEqual(completed, 0)
                    continue
                self.assertEqual(produced, completed)
                live = int(p.state_storage.protected_count.numpy()[0])
                seeds = p.state_storage.arrays["seed"][:live].numpy() if live else ()
                expected_owners = [
                    (row, primitive)
                    for row, seed in enumerate(seeds)
                    for primitive in range(1 + int(seed) % p.primitive_count)
                ]
                self.assertEqual(produced, len(expected_owners))
                if produced:
                    owners = p.sample_storage.arrays["primitive_location"][:produced].numpy()
                    values = p.sample_storage.arrays["value"][:produced].numpy()
                    positions = p.state_storage.arrays["position"][:live].numpy()
                    self.assertEqual(sorted(map(tuple, owners.tolist())), expected_owners)
                    np.testing.assert_array_equal(
                        values, [positions[row, primitive].sum() for row, primitive in owners]
                    )
        return current

    def test_fixed_and_vmm_compose_state_generated_work_and_lifetimes(self):
        """Preserve retained state and reject stale handles under one captured program."""
        with wp.ScopedDevice("cuda:0"):
            for vmm in (False, True):
                with self.subTest(vmm=vmm):
                    runtime = _SampleWorlds(vmm=vmm)
                    graph = runtime.capture()
                    try:
                        retained = weakref.ref(runtime)
                        del runtime
                        gc.collect()
                        self.assertIsNotNone(retained())
                        runtime = retained()
                        executable = graph.graph_exec
                        accounting = runtime.memory_report()
                        self.assertGreater(accounting["metadata_bytes"], 0)
                        self.assertGreater(accounting["fixed_payload_bytes"], 0)
                        self.assertEqual(accounting["backing"] is not None, vmm)
                        pointers = [
                            (p.state_storage.arrays["position"].ptr, p.sample_storage.arrays["primitive_location"].ptr)
                            for p in runtime.populations
                        ]
                        before = {}
                        handles = runtime.submit(graph, [(worlds.WorldOperation.CREATE, -1, 0, p) for p in (0, 0, 1)])
                        before = self.assert_state_and_samples(runtime, before)
                        first, _second, third = handles
                        replacement = runtime.submit(
                            graph,
                            [(worlds.WorldOperation.RESET, *first, 1), (worlds.WorldOperation.DESTROY, *third, -1)],
                        )[0]
                        before = self.assert_state_and_samples(runtime, before)
                        self.assertNotIn(tuple(first), before)
                        self.assertIn(tuple(replacement), before)
                        runtime.submit(graph, [(worlds.WorldOperation.DESTROY, *first, -1)])
                        before = self.assert_state_and_samples(runtime, before)
                        self.assertEqual(int(runtime.results.status.numpy()[0]), int(worlds.WorldStatus.STALE))
                        # This is a rendering-work failure: retained motion can advance,
                        # but the incomplete generated sample frame must not be shaded.
                        for p in runtime.populations:
                            p.sample_limit.fill_(0)
                        runtime.submit(graph, [])
                        before = self.assert_state_and_samples(runtime, before)
                        for p in runtime.populations:
                            self.assertEqual(int(p.samples_valid.numpy()[0]), 0)
                            p.sample_limit.fill_(p.sample_storage.capacity)
                        runtime.submit(graph, [])
                        self.assert_state_and_samples(runtime, before)
                        self.assertEqual(graph.graph_exec, executable)
                        self.assertEqual(
                            [
                                (
                                    p.state_storage.arrays["position"].ptr,
                                    p.sample_storage.arrays["primitive_location"].ptr,
                                )
                                for p in runtime.populations
                            ],
                            pointers,
                        )
                        with self.assertRaisesRegex(RuntimeError, "graph"):
                            runtime.close()
                    finally:
                        del graph
                        gc.collect()
                        runtime.close()

    def test_shared_budget_rejection_shrink_and_regrow_keep_one_graph(self):
        """Reuse backing across prototypes without replacing state addresses or the graph."""
        with wp.ScopedDevice("cuda:0"):
            runtime = _SampleWorlds(vmm=True)
            graph = runtime.capture()
            try:
                first, second = runtime.populations
                initial_ready = first.state_storage.ready_rows
                pointer, executable = first.state_storage.arrays["position"].ptr, graph.graph_exec
                handles = runtime.submit(
                    graph, [(worlds.WorldOperation.CREATE, -1, 0, 0), (worlds.WorldOperation.CREATE, -1, 0, 1)]
                )
                before = self.assert_state_and_samples(runtime, {})
                original = runtime.backing.memory_report()
                self.assertEqual(original["mapped_bytes"], original["budget_bytes"])
                with self.assertRaises(MemoryError):
                    first.state_storage.resize_backing(
                        first.state_storage.capacity, streams=(runtime.stream.cuda_stream,)
                    )
                self.assertFalse(first.state_storage.service_failed)
                self.assertEqual(first.state_storage.ready_rows, initial_ready)
                runtime.submit(graph, [(worlds.WorldOperation.DESTROY, *handles[1], -1)])
                before = self.assert_state_and_samples(runtime, before)
                runtime.directory.withdraw_ready_slots((first.world_ready_capacity, 0))
                second.state_storage.resize_backing(0, streams=(runtime.stream.cuda_stream,))
                second.sample_storage.resize_backing(0, streams=(runtime.stream.cuda_stream,))
                first.state_storage.resize_backing(first.state_storage.capacity, streams=(runtime.stream.cuda_stream,))
                runtime.publish_ready()
                runtime.submit(graph, [(worlds.WorldOperation.CREATE, -1, 0, 0)] * initial_ready)
                before = self.assert_state_and_samples(runtime, before)
                self.assertGreater(int(first.state_storage.protected_count.numpy()[0]), initial_ready)
                self.assertEqual(first.state_storage.arrays["position"].ptr, pointer)
                self.assertEqual(graph.graph_exec, executable)
                # Drop all but one world, then give the second prototype its old
                # virtual range back. Derived sample storage needs no lifetime copy.
                keep = next(iter(before))
                runtime.submit(
                    graph, [(worlds.WorldOperation.DESTROY, *handle, -1) for handle in before if handle != keep]
                )
                before = self.assert_state_and_samples(runtime, before)
                runtime.directory.withdraw_ready_slots((initial_ready, 0))
                first.state_storage.resize_backing(initial_ready, streams=(runtime.stream.cuda_stream,))
                second.state_storage.resize_backing(1, streams=(runtime.stream.cuda_stream,))
                second.sample_storage.resize_backing(second.primitive_count, streams=(runtime.stream.cuda_stream,))
                runtime.publish_ready()
                runtime.submit(graph, [(worlds.WorldOperation.CREATE, -1, 0, 1)])
                self.assert_state_and_samples(runtime, before)
                report = runtime.backing.memory_report()
                self.assertEqual(report["physical_retained_bytes"], original["physical_retained_bytes"])
                self.assertEqual(report["physical_retained_bytes"], report["mapped_bytes"] + report["spare_bytes"])
                self.assertEqual(
                    report["mapped_bytes"],
                    sum(
                        p.state_storage.memory_report()["mapped_packed_bytes"]
                        + p.sample_storage.memory_report()["mapped_packed_bytes"]
                        for p in runtime.populations
                    ),
                )
                self.assertEqual(first.state_storage.arrays["position"].ptr, pointer)
                self.assertEqual(graph.graph_exec, executable)
            finally:
                del graph
                gc.collect()
                runtime.close()
