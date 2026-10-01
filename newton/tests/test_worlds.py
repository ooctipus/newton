# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check world-directory ownership and lifecycle semantics independently of physics."""

import ast
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

from newton import worlds


@wp.kernel
def _locations(
    data: worlds.WorldDirectoryData,
    identities: wp.array[int],
    generations: wp.array[wp.uint64],
    prototypes: wp.array[int],
    rows: wp.array[int],
    valid: wp.array[bool],
):
    i = wp.tid()
    prototype, row, found = worlds.world_location(data, identities[i], generations[i])
    prototypes[i] = prototype
    rows[i] = row
    valid[i] = found


@wp.kernel
def _handles(
    data: worlds.WorldDirectoryData,
    prototypes: wp.array[int],
    rows: wp.array[int],
    identities: wp.array[int],
    generations: wp.array[wp.uint64],
    valid: wp.array[bool],
):
    i = wp.tid()
    identity, generation, found = worlds.world_handle_at(data, prototypes[i], rows[i])
    identities[i] = identity
    generations[i] = generation
    valid[i] = found


@wp.kernel
def _initialize_reference_payload(
    d: worlds.WorldDirectoryData,
    c: worlds.WorldCommands,
    t: worlds.WorldTransaction,
    acknowledge: wp.array[int],
    payload: wp.array[wp.int64],
):
    request = wp.tid()
    if t.phase[0] == int(worlds.WorldPhase.ADMITTED) and request < c.count[0]:
        op = c.operation[request]
        if t.status[request] == int(worlds.WorldStatus.OK) and (op == 1 or op == 2) and acknowledge[request] != 0:
            destination = d.slot_starts[c.prototype[request]] + t.destination_slot[request]
            payload[destination] = wp.int64(c.sequence[0]) * wp.int64(c.operation.shape[0]) + wp.int64(request)
            t.initialized_sequence[request] = c.sequence[0]


@wp.kernel
def _move_reference_payload(
    d: worlds.WorldDirectoryData, t: worlds.WorldTransaction, m: worlds.WorldCompaction, payload: wp.array[wp.int64]
):
    prototype = wp.tid()
    if t.phase[0] == int(worlds.WorldPhase.MOVING):
        start = d.slot_starts[prototype]
        for rank in range(m.count[prototype]):
            payload[start + m.destination_slots[start + rank]] = payload[start + m.source_slots[start + rank]]
        m.copied_count[prototype] = m.count[prototype]


class WorldArchitectureTests(unittest.TestCase):
    """Guard the intended ownership graph rather than historical module names."""

    root = Path(__file__).resolve().parents[1]
    owners = (
        "_src/utils/cuda_vmm.py",
        "_src/utils/field_storage.py",
        "_src/utils/cuda_graph.py",
        "_src/utils/cuda_graph.cu",
        "_src/sim/worlds.py",
        "_src/solvers/mujoco/worlds.py",
        "worlds.py",
    )

    def test_expected_owners_exist_without_laboratory_modules_or_facades(self):
        """Verify the complete runtime tree has package-owned implementations only."""
        forbidden = {
            "native_dynamic",
            "native_bindings",
            "native_population",
            "native_storage",
            "warp_runtime",
            "synthetic_domain",
            "backing",
            "device_graph",
        }
        for filename in self.owners:
            path = self.root / filename
            self.assertTrue(path.is_file(), filename)
            source = path.read_text()
            self.assertNotIn("newton-worlds-lab", source)
            if path.suffix != ".py":
                continue
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                self.assertFalse(forbidden.intersection(name.split(".")[0] for name in names), filename)
        self.assertFalse(any(self.root.rglob("native_bindings.py")))
        self.assertFalse(any(self.root.rglob("native_dynamic.py")))

    def test_mechanical_owners_and_directory_do_not_import_physics_or_each_other(self):
        """Verify lower owners do not acquire domain state, schema or scheduling policy."""
        for filename in (*self.owners[:3], "_src/sim/worlds.py"):
            tree = ast.parse((self.root / filename).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertEqual(node.level, 0, filename)
                    names = [node.module or ""]
                elif isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                else:
                    continue
                self.assertFalse(
                    {"newton", "mujoco", "mujoco_warp"}.intersection(name.split(".")[0] for name in names), filename
                )
        row_source = (self.root / "_src/utils/field_storage.py").read_text()
        for symbol in ("NativeRows", "NativeScratch", "register_world_array", "_replace_data", "nworld"):
            self.assertNotIn(symbol, row_source)
        directory = ast.parse((self.root / "_src/sim/worlds.py").read_text())
        structs = {node.name: node for node in directory.body if isinstance(node, ast.ClassDef)}
        for name in ("WorldDirectoryData", "WorldTransaction", "WorldCompaction"):
            fields = {node.target.id for node in structs[name].body if isinstance(node, ast.AnnAssign)}
            self.assertFalse(fields.intersection(("qpos", "qvel", "body_q", "joint_q", "state", "history")))

    def test_public_world_api_has_one_canonical_name_and_no_legacy_aliases(self):
        """Verify explicit public exports do not preserve discarded laboratory aliases."""
        self.assertEqual(len(worlds.__all__), len(set(worlds.__all__)))
        for name in worlds.__all__:
            self.assertTrue(hasattr(worlds, name), name)
        for name in (
            "Commands",
            "Results",
            "Directory",
            "Transaction",
            "WorldPool",
            "WorldBatch",
            "NativePopulation",
            "make_commands",
            "make_results",
        ):
            self.assertFalse(hasattr(worlds, name), name)
        self.assertEqual([int(value) for value in worlds.WorldOperation], [0, 1, 2, 3])
        self.assertEqual(worlds.WorldStatus.INITIALIZATION_MISSING, 11)

    def test_composition_mechanisms_have_one_public_namespace(self):
        """Expose the reusable contracts without private imports or duplicate aliases."""
        expected = {
            "DeviceGraphUpdates",
            "GraphKernelBinding",
            "KernelParameterBinding",
            "capture_parallel",
            "MemoryBacking",
            "VirtualReservation",
            "FieldSpec",
            "FieldStorage",
            "FieldTransfer",
            "FieldView",
        }
        self.assertTrue(expected.issubset(worlds.__all__))
        for module in self.root.glob("*.py"):
            if module.name == "worlds.py":
                continue
            tree = ast.parse(module.read_text())
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    targets = node.targets
                elif isinstance(node, ast.AugAssign):
                    targets = (node.target,)
                else:
                    continue
                if any(isinstance(target, ast.Name) and target.id == "__all__" for target in targets):
                    exports = {value.value for value in ast.walk(node.value) if isinstance(value, ast.Constant)}
                    self.assertFalse(expected.intersection(exports), module)

    def test_directory_free_count_is_canonical_without_readiness_alias(self):
        """Keep free admission slots distinct from mechanically backed row prefixes."""
        self.assertIn("free_slot_count", worlds.WorldDirectoryData.vars)
        self.assertNotIn("ready_count", worlds.WorldDirectoryData.vars)
        self.assertFalse(hasattr(worlds.WorldDirectoryData(), "ready_count"))

    def test_public_records_exclude_allocator_scratch_and_positional_flags(self):
        """Expose numeric relations, named outcomes and domain acknowledgements only."""
        self.assertEqual(
            set(worlds.WorldDirectoryData.vars),
            {
                "prototype",
                "slot",
                "generation",
                "slot_starts",
                "slot_id",
                "slot_live_rank",
                "live_slots",
                "live_count",
                "free_slot_count",
            },
        )
        self.assertEqual(
            set(worlds.WorldBatchResult.vars),
            {"sequence", "consumed", "advance_allowed", "status", "slot_rejection_count"},
        )
        self.assertEqual(
            set(worlds.WorldTransaction.vars),
            {
                "phase",
                "status",
                "initialized_sequence",
                "destination_world_id",
                "destination_slot",
                "admitted_requests",
                "request_starts",
            },
        )
        self.assertEqual(
            set(worlds.WorldCompaction.vars), {"source_slots", "destination_slots", "count", "copied_count"}
        )
        self.assertNotIn("_WorldScratch", worlds.__all__)
        self.assertFalse(hasattr(worlds, "_WorldScratch"))
        for record in (
            worlds.WorldDirectoryData,
            worlds.WorldBatchResult,
            worlds.WorldTransaction,
            worlds.WorldCompaction,
        ):
            self.assertNotIn("flags", record.vars)
        self.assertIn("world_location", worlds.__all__)
        self.assertIn("world_handle_at", worlds.__all__)

    def test_capacity_service_is_batched_without_legacy_scalar_aliases(self):
        """Forbid directory rebuilds inside per-prototype native service loops."""
        tree = ast.parse((self.root / "_src/solvers/mujoco/worlds.py").read_text())
        owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MuJoCoWorlds")
        resize = next(
            node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "resize_backing"
        )
        self.assertEqual([arg.arg for arg in resize.args.args], ["self", "world_ready_capacities"])
        for method in owner.body:
            if not isinstance(method, ast.FunctionDef):
                continue
            for loop in (node for node in ast.walk(method) if isinstance(node, (ast.For, ast.While))):
                for call in (node for node in ast.walk(loop) if isinstance(node, ast.Call)):
                    if isinstance(call.func, ast.Attribute):
                        self.assertNotIn(call.func.attr, ("publish_ready_slots", "withdraw_ready_slots"), method.name)

    def test_public_import_does_not_initialize_optional_backend_or_cuda_runtime(self):
        """Verify importing world contracts keeps optional engines and CUDA uninitialized."""
        source = """
import sys
from types import SimpleNamespace
import warp as wp
def forbidden(*args, **kwargs):
    raise AssertionError('import initialized Warp')
wp.init = forbidden
import newton.worlds
assert 'mujoco' not in sys.modules
assert 'mujoco_warp' not in sys.modules
assert 'newton._src.solvers.mujoco.worlds' not in sys.modules
"""
        subprocess.run([sys.executable, "-c", source], check=True, capture_output=True, text=True)


class WorldDirectoryCPUTests(unittest.TestCase):
    """Exercise identities and acknowledgements with no physical state allocation."""

    def setUp(self):
        self.directory = worlds.WorldDirectory((2, 2), id_capacity=4, command_capacity=4, device="cpu")
        self.commands = worlds.create_world_commands(4, device="cpu")
        self.results = worlds.create_world_results(4, device="cpu")
        self.sequence = 0
        self.directory.publish_ready_slots((2, 2))

    def tearDown(self):
        self.directory.close(streams=())

    def submit(self, requests, *, initialize=True):
        self.sequence += 1
        self.commands.sequence.fill_(self.sequence)
        self.commands.count.fill_(len(requests))
        for column, name in enumerate(("operation", "world_id", "generation", "prototype")):
            values = np.zeros(4, np.uint64 if name == "generation" else np.int32)
            values[: len(requests)] = [request[column] for request in requests]
            getattr(self.commands, name).assign(values)
        self.directory.begin(self.commands)
        self.directory.admit(self.commands)
        if initialize:
            self.directory.transaction.initialized_sequence.fill_(self.sequence)
        self.directory.publish(self.commands, self.results)
        return self.results.status.numpy()[: len(requests)], self.results.world_id.numpy()[: len(requests)]

    def assert_relations(self):
        """Check independent cardinality and forward/inverse membership laws."""
        d = self.directory.data
        arrays = {name: getattr(d, name).numpy() for name in d._cls.vars}
        live = np.flatnonzero(arrays["prototype"] >= 0)
        self.assertEqual(int(arrays["live_count"].sum()), len(live))
        occupied = np.flatnonzero(arrays["slot_id"] >= 0)
        self.assertEqual(len(occupied), len(live))
        self.assertEqual(len(set(map(int, arrays["slot_id"][occupied]))), len(live))
        for prototype, count in enumerate(arrays["live_count"]):
            start, end = arrays["slot_starts"][prototype : prototype + 2]
            rows = arrays["live_slots"][start : start + count]
            self.assertEqual(len(set(map(int, rows))), count)
            self.assertTrue(np.all((rows >= 0) & (rows < end - start)))
            self.assertLessEqual(count + arrays["free_slot_count"][prototype], end - start)
            for rank, row in enumerate(rows):
                identity = arrays["slot_id"][start + row]
                self.assertGreaterEqual(identity, 0)
                self.assertEqual(arrays["prototype"][identity], prototype)
                self.assertEqual(arrays["slot"][identity], row)
                self.assertEqual(arrays["slot_live_rank"][start + row], rank)
        return arrays

    def locations(self, identities, generations):
        """Run public handle resolution against the current directory."""
        n = len(identities)
        p, row = wp.empty(n, dtype=int, device="cpu"), wp.empty(n, dtype=int, device="cpu")
        valid = wp.empty(n, dtype=bool, device="cpu")
        wp.launch(
            _locations,
            n,
            [
                self.directory.data,
                wp.array(identities, dtype=int, device="cpu"),
                wp.array(generations, dtype=wp.uint64, device="cpu"),
            ],
            [p, row, valid],
            device="cpu",
        )
        return p.numpy(), row.numpy(), valid.numpy()

    def handles(self, prototypes, rows):
        """Run public inverse resolution against the current directory."""
        n = len(rows)
        identities = wp.empty(n, dtype=int, device="cpu")
        generations, valid = wp.empty(n, dtype=wp.uint64, device="cpu"), wp.empty(n, dtype=bool, device="cpu")
        wp.launch(
            _handles,
            n,
            [
                self.directory.data,
                wp.array(prototypes, dtype=int, device="cpu"),
                wp.array(rows, dtype=int, device="cpu"),
            ],
            [identities, generations, valid],
            device="cpu",
        )
        return identities.numpy(), generations.numpy(), valid.numpy()

    def test_owner_has_no_legacy_record_aliases(self):
        """Reject alternate mutation paths left over from the old directory boundary."""
        for name in ("d", "t", "moves", "flags"):
            self.assertFalse(hasattr(self.directory, name), name)

    def test_public_relations_resolve_invalid_stale_reset_and_destroyed_handles(self):
        """Verify lookup inverses and lifetime invalidation without any physical state."""
        op, status = worlds.WorldOperation, worlds.WorldStatus
        result, identities = self.submit([(op.CREATE, -1, 0, 0), (op.CREATE, -1, 0, 1)])
        np.testing.assert_array_equal(result, [status.OK, status.OK])
        first, second = map(int, identities)
        arrays = self.assert_relations()
        p, rows, valid = self.locations([first, second, -1, 4, first], [1, 1, 1, 1, 0])
        np.testing.assert_array_equal(p, [0, 1, -1, -1, -1])
        np.testing.assert_array_equal(valid, [True, True, False, False, False])
        np.testing.assert_array_equal(rows[:2], arrays["slot"][[first, second]])
        ids, gens, inverse_valid = self.handles([0, 1, -1, 2, 0, 1], [int(rows[0]), int(rows[1]), 0, 0, -1, 2])
        np.testing.assert_array_equal(ids, [first, second, -1, -1, -1, -1])
        np.testing.assert_array_equal(gens, [1, 1, 0, 0, 0, 0])
        np.testing.assert_array_equal(inverse_valid, [True, True, False, False, False, False])
        self.submit([(op.RESET, first, 1, 1)])
        self.assert_relations()
        p, _, valid = self.locations([first, first], [1, 2])
        np.testing.assert_array_equal(p, [-1, 1])
        np.testing.assert_array_equal(valid, [False, True])
        old_row = int(self.directory.data.slot.numpy()[first])
        self.submit([(op.DESTROY, first, 2, 0)])
        self.assert_relations()
        self.assertFalse(self.locations([first], [2])[2][0])
        self.assertFalse(self.handles([1], [old_row])[2][0])

    def test_reused_identity_changes_generation_and_compaction_preserves_handle(self):
        """Keep handles stable across moves and invalidate them when an identity is reused."""
        self.directory.close(streams=())
        self.directory = worlds.WorldDirectory((3,), id_capacity=1, command_capacity=4, device="cpu")
        self.directory.publish_ready_slots((3,))
        op = worlds.WorldOperation
        self.submit([(op.CREATE, -1, 0, 0)])
        self.assertEqual(int(self.directory.data.slot.numpy()[0]), 2)
        self.directory.plan_compaction()
        self.directory.compaction.copied_count.assign(self.directory.compaction.count.numpy())
        self.directory.publish_compaction()
        p, row, valid = self.locations([0], [1])
        np.testing.assert_array_equal([p[0], row[0], valid[0]], [0, 0, True])
        self.assert_relations()
        self.submit([(op.DESTROY, 0, 1, 0)])
        result, ids = self.submit([(op.CREATE, -1, 0, 0)])
        self.assertEqual((int(result[0]), int(ids[0])), (int(worlds.WorldStatus.OK), 0))
        self.assertEqual(int(self.directory.data.generation.numpy()[0]), 3)
        np.testing.assert_array_equal(self.locations([0, 0], [1, 3])[2], [False, True])
        self.assert_relations()

    def test_slot_rejections_belong_to_consumed_batch_and_survive_replay(self):
        """Keep capacity diagnostics out of lifetime relations and scoped to consumed batches."""
        op = worlds.WorldOperation
        self.submit([(op.CREATE, -1, 0, 0), (op.CREATE, -1, 0, 0)])
        status, _ = self.submit([(op.CREATE, -1, 0, 0)])
        np.testing.assert_array_equal(status, [worlds.WorldStatus.NO_SLOTS])
        batch = self.directory.batch_result
        np.testing.assert_array_equal(batch.slot_rejection_count.numpy(), [1, 0])
        self.assertFalse(hasattr(self.directory.data, "slot_rejection_count"))
        self.commands.count.zero_()
        self.directory.begin(self.commands)
        self.directory.admit(self.commands)
        self.directory.publish(self.commands, self.results)
        self.assertEqual(int(batch.consumed.numpy()[0]), 0)
        np.testing.assert_array_equal(batch.slot_rejection_count.numpy(), [1, 0])
        self.sequence += 1
        self.commands.sequence.fill_(self.sequence)
        self.commands.count.fill_(5)
        self.directory.begin(self.commands)
        self.directory.admit(self.commands)
        self.directory.publish(self.commands, self.results)
        self.assertEqual(int(batch.consumed.numpy()[0]), 0)
        self.assertEqual(int(batch.status.numpy()[0]), worlds.WorldStatus.BAD_COUNT)
        self.assertEqual(int(batch.sequence.numpy()[0]), self.sequence)
        np.testing.assert_array_equal(batch.slot_rejection_count.numpy(), [0, 0])
        self.commands.count.zero_()
        self.directory.begin(self.commands)
        self.directory.admit(self.commands)
        self.directory.publish(self.commands, self.results)
        self.assertEqual(int(batch.consumed.numpy()[0]), 0)
        self.assertEqual(int(batch.status.numpy()[0]), worlds.WorldStatus.BAD_COUNT)
        np.testing.assert_array_equal(batch.slot_rejection_count.numpy(), [0, 0])
        self.submit([])
        np.testing.assert_array_equal(batch.slot_rejection_count.numpy(), [0, 0])
        self.assertEqual(int(batch.advance_allowed.numpy()[0]), 1)
        self.assert_relations()

    def test_named_batch_outcome_preserves_replay_and_rejection_laws(self):
        """Distinguish consumed requests, request failure and batch rejection from old results."""
        op, status, batch = worlds.WorldOperation, worlds.WorldStatus, self.directory.batch_result
        self.submit([(op.CREATE, -1, 0, 0)])
        generation = self.directory.data.generation.numpy().copy()
        original_results = self.results.world_id.numpy().copy()
        self.commands.operation.fill_(op.DESTROY)
        self.directory.begin(self.commands)
        self.directory.admit(self.commands)
        self.directory.publish(self.commands, self.results)
        self.assertEqual(
            (int(batch.consumed.numpy()[0]), int(batch.advance_allowed.numpy()[0]), int(batch.status.numpy()[0])),
            (0, 1, status.OK),
        )
        np.testing.assert_array_equal(self.directory.data.generation.numpy(), generation)
        np.testing.assert_array_equal(self.results.world_id.numpy(), original_results)
        self.commands.sequence.fill_(2)
        self.commands.count.fill_(5)
        self.directory.begin(self.commands)
        self.directory.admit(self.commands)
        self.directory.publish(self.commands, self.results)
        self.assertEqual(
            (int(batch.consumed.numpy()[0]), int(batch.advance_allowed.numpy()[0]), int(batch.status.numpy()[0])),
            (0, 0, status.BAD_COUNT),
        )
        np.testing.assert_array_equal(self.results.world_id.numpy(), original_results)
        self.commands.sequence.fill_(1)
        self.directory.begin(self.commands)
        self.assertEqual(int(batch.status.numpy()[0]), status.STALE_BATCH)
        self.sequence = 2
        self.submit([])
        self.assertEqual(
            (int(batch.consumed.numpy()[0]), int(batch.advance_allowed.numpy()[0]), int(batch.status.numpy()[0])),
            (1, 1, status.OK),
        )
        self.assert_relations()

    def test_unacknowledged_moves_preserve_lifetimes_and_deny_advancement(self):
        """Reject move publication before the domain has copied every retained row."""
        self.submit([(worlds.WorldOperation.CREATE, -1, 0, 0)])
        before = self.assert_relations()
        self.directory.plan_compaction()
        self.directory.publish_compaction()
        self.assertEqual(int(self.directory.batch_result.status.numpy()[0]), worlds.WorldStatus.COMPACTION_INVALID)
        self.assertEqual(int(self.directory.batch_result.advance_allowed.numpy()[0]), 0)
        after = self.assert_relations()
        for name in ("prototype", "slot", "generation", "slot_id"):
            np.testing.assert_array_equal(after[name], before[name])

    def test_capacity_publication_does_not_implicitly_withdraw_rows(self):
        """Keep grow publication separate from validated tail retirement."""
        self.directory.publish_ready_slots((1, 1))
        np.testing.assert_array_equal(self.directory.data.free_slot_count.numpy(), [2, 2])
        self.directory.withdraw_ready_slots((1, 1))
        np.testing.assert_array_equal(self.directory.data.free_slot_count.numpy(), [1, 1])
        self.directory.publish_ready_slots((2, 1))
        np.testing.assert_array_equal(self.directory.data.free_slot_count.numpy(), [2, 1])

    def test_compaction_and_readiness_preserve_free_identity_cache(self):
        """Slot maintenance does not reconstruct the unchanged identity relation."""
        operation, status = worlds.WorldOperation, worlds.WorldStatus
        _, ids = self.submit([(operation.CREATE, -1, 0, 0)])
        identity = int(ids[0])
        scratch = self.directory._scratch
        count = int(scratch.id_count.numpy()[0])
        expected = scratch.free_ids.numpy()[:count][::-1].copy()
        # Any ordering of eligible IDs is valid; reversing detects needless CPU rebuilds.
        permutation = scratch.free_ids.numpy().copy()
        permutation[:count] = expected
        scratch.free_ids.assign(permutation)
        self.directory.plan_compaction()
        self.directory.compaction.copied_count.assign(self.directory.compaction.count.numpy())
        self.directory.publish_compaction()
        self.directory.withdraw_ready_slots((1, 1))
        self.directory.publish_ready_slots((2, 2))
        self.assertEqual(int(scratch.id_count.numpy()[0]), count)
        np.testing.assert_array_equal(scratch.free_ids.numpy()[:count], expected)
        result, _ = self.submit([(operation.DESTROY, identity, 1, 0)])
        self.assertEqual(result[0], status.OK)
        self.assertEqual(int(scratch.id_count.numpy()[0]), 4)
        self.assertEqual(set(scratch.free_ids.numpy()[:4]), {0, 1, 2, 3})
        result, ids = self.submit([(operation.CREATE, -1, 0, 1)])
        self.assertEqual(result[0], status.OK)
        self.assertEqual(int(scratch.id_count.numpy()[0]), 3)
        self.assertEqual(set(scratch.free_ids.numpy()[:3]), {0, 1, 2, 3} - {int(ids[0])})

    def test_retained_graph_prevents_closure_until_borrower_retires(self):
        """Keep graph-borrowed directory metadata alive until the executable is released."""

        class Graph:
            device = self.directory.device

        graph = Graph()
        payload = object()
        self.assertIs(self.directory.retain_graph(graph, payload), graph)
        self.assertEqual(graph._resource_owners, (self.directory, payload))
        with self.assertRaisesRegex(RuntimeError, "Destroy retained graphs"):
            self.directory.close(streams=())
        del graph
        self.directory.close(streams=())
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.directory.begin(self.commands)

    def test_close_joins_only_validated_supplied_streams(self):
        """Validate every consumer before joining precise streams without wrapping raw handles."""

        class Stream:
            def __init__(self, device):
                self.device = device

        original_device = self.directory.device
        device, foreign = SimpleNamespace(is_cuda=True, ordinal=0), SimpleNamespace(is_cuda=True, ordinal=1)
        self.directory.device = device
        first, second = Stream(device), Stream(device)
        try:
            with (
                patch("newton._src.sim.worlds.wp.Stream", Stream),
                patch("newton._src.sim.worlds.wp.synchronize_stream") as join,
                patch(
                    "newton._src.sim.worlds.wp.synchronize_device",
                    side_effect=AssertionError("whole-device synchronization"),
                ),
            ):
                for streams in ((), (0,), (first, Stream(foreign))):
                    with self.subTest(streams=streams), self.assertRaises(ValueError):
                        self.directory.close(streams=streams)
                    join.assert_not_called()
                    self.assertFalse(self.directory._closed)
                self.directory.close(streams=(first, second))
                self.assertEqual([call.args[0] for call in join.call_args_list], [first, second])
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    self.directory.publish_ready_slots((2, 2))
        finally:
            self.directory.device = original_device

    def test_int32_capacity_bounds_reject_before_device_allocation(self):
        """Verify individual and aggregate capacities cannot overflow directory indices."""
        with patch("newton._src.sim.worlds.wp.zeros", side_effect=AssertionError("unexpected allocation")):
            for capacity in (0, -1, True, 1.5, 2**31, 2**31 + 1):
                for factory in (worlds.create_world_commands, worlds.create_world_results):
                    with self.subTest(factory=factory, capacity=capacity), self.assertRaises(ValueError):
                        factory(capacity, device="cpu")
            for limits in ((2**31,), (2**31 - 1, 1), (1, 2**31)):
                with self.subTest(limits=limits), self.assertRaises(ValueError):
                    worlds.WorldDirectory(limits, id_capacity=4, command_capacity=4, device="cpu")
            for key in ("id_capacity", "command_capacity"):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    worlds.WorldDirectory((2,), **{key: 2**31}, device="cpu")

    def test_partial_success_preserves_stale_lifetime_and_gates_physics(self):
        """Verify a stale request preserves its world while an independent reset succeeds."""
        operation, status = worlds.WorldOperation, worlds.WorldStatus
        result, ids = self.submit([(operation.CREATE, -1, 0, 0), (operation.CREATE, -1, 0, 1)])
        np.testing.assert_array_equal(result, [status.OK, status.OK])
        first, second = map(int, ids)
        original_slot = int(self.directory.data.slot.numpy()[first])
        result, _ = self.submit([(operation.RESET, first, 99, 1), (operation.RESET, second, 1, 0)])
        np.testing.assert_array_equal(result, [status.STALE, status.OK])
        generations = self.directory.data.generation.numpy()
        self.assertEqual((generations[first], generations[second]), (1, 2))
        self.assertEqual(self.directory.data.slot.numpy()[first], original_slot)
        np.testing.assert_array_equal(self.directory.data.live_count.numpy(), [2, 0])
        self.assertEqual(self.directory.batch_result.advance_allowed.numpy()[0], 0)
        self.assertEqual(self.directory.batch_result.status.numpy()[0], status.OK)
        self.assertEqual(self.directory.batch_result.consumed.numpy()[0], 1)

    def test_missing_initialization_and_conflicting_requests_preserve_membership(self):
        """Verify publication requires initialization and rejects all conflicting mutations."""
        operation, status = worlds.WorldOperation, worlds.WorldStatus
        result, _ = self.submit([(operation.CREATE, -1, 0, 0)], initialize=False)
        np.testing.assert_array_equal(result, [status.INITIALIZATION_MISSING])
        np.testing.assert_array_equal(self.directory.data.live_count.numpy(), [0, 0])
        result, ids = self.submit([(operation.CREATE, -1, 0, 0)])
        identity = int(ids[0])
        result, _ = self.submit([(operation.RESET, identity, 1, 1), (operation.DESTROY, identity, 1, 0)])
        np.testing.assert_array_equal(result, [status.CONFLICT, status.CONFLICT])
        self.assertEqual(self.directory.data.generation.numpy()[identity], 1)
        np.testing.assert_array_equal(self.directory.data.live_count.numpy(), [1, 0])

    def test_failed_batch_withdrawal_preserves_every_prototype(self):
        """Validate the entire withdrawal before changing any other ready prefix."""
        result, _ = self.submit([(worlds.WorldOperation.CREATE, -1, 0, 1)])
        self.assertEqual(result[0], worlds.WorldStatus.OK)
        before = self.directory.data.slot_id.numpy().copy()
        with self.assertRaisesRegex(RuntimeError, "live worlds"):
            self.directory.withdraw_ready_slots((0, 0))
        np.testing.assert_array_equal(self.directory.data.slot_id.numpy(), before)
        np.testing.assert_array_equal(self.directory.data.free_slot_count.numpy(), [2, 1])
        for ends in ((1,), (2, 3), (True, 1), (-1, 0)):
            with self.subTest(ends=ends), self.assertRaises(ValueError):
                self.directory.publish_ready_slots(ends)
        np.testing.assert_array_equal(self.directory.data.slot_id.numpy(), before)

    def test_admitted_and_published_slots_are_prototype_local(self):
        """Keep local row indices distinct from global slot metadata for later prototypes."""
        result, identities = self.submit([(worlds.WorldOperation.CREATE, -1, 0, 1)])
        self.assertEqual(result[0], worlds.WorldStatus.OK)
        identity = int(identities[0])
        destination = int(self.directory.transaction.destination_slot.numpy()[0])
        self.assertEqual(destination, 1)
        self.assertEqual(int(self.directory.data.slot.numpy()[identity]), destination)
        self.assertEqual(int(self.directory.data.slot_id.numpy()[2 + destination]), identity)
        self.assertEqual(int(self.directory.data.slot_id.numpy()[destination]), -1)

    def test_acknowledged_compaction_preserves_identity_and_allows_tail_retirement(self):
        """Verify acknowledged row moves retain generation and release the unused tail."""
        result, ids = self.submit([(worlds.WorldOperation.CREATE, -1, 0, 0)])
        self.assertEqual(result[0], worlds.WorldStatus.OK)
        identity = int(ids[0])
        self.assertEqual(self.directory.data.slot.numpy()[identity], 1)
        self.directory.plan_compaction()
        np.testing.assert_array_equal(self.directory.compaction.count.numpy(), [1, 0])
        self.directory.compaction.copied_count.assign(np.array([1, 0], np.int32))
        self.directory.publish_compaction()
        self.assertEqual(self.directory.data.slot.numpy()[identity], 0)
        self.assertEqual(self.directory.data.generation.numpy()[identity], 1)
        self.directory.withdraw_ready_slots((1, 2))
        with self.assertRaisesRegex(RuntimeError, "live worlds"):
            self.directory.withdraw_ready_slots((0, 2))


class WorldDirectoryReferenceTests(unittest.TestCase):
    """Compare changing batches with an independent lifetime/payload model under one prepared program."""

    def test_cpu_reference_batches(self):
        self._check_batches("cpu", captured=False)

    def test_captured_cuda_reference_batches(self):
        if not wp.is_cuda_available():
            self.skipTest("CUDA is unavailable")
        self._check_batches("cuda:0", captured=True)

    def test_cpu_terminal_generation(self):
        self._check_terminal_generation("cpu", captured=False)

    def test_captured_cuda_terminal_generation(self):
        if not wp.is_cuda_available():
            self.skipTest("CUDA is unavailable")
        self._check_terminal_generation("cuda:0", captured=True)

    def test_cpu_phase_cancellation(self):
        self._check_phase_cancellation("cpu", captured=False)

    def test_captured_cuda_phase_cancellation(self):
        if not wp.is_cuda_available():
            self.skipTest("CUDA is unavailable")
        self._check_phase_cancellation("cuda:0", captured=True)

    def _check_phase_cancellation(self, device, *, captured):
        for admitted in (False, True):
            with self.subTest(admitted=admitted, captured=captured):
                directory = worlds.WorldDirectory((2,), id_capacity=1, command_capacity=1, device=device)
                directory.publish_ready_slots((2,))
                c = worlds.create_world_commands(1, device=device)
                r = worlds.create_world_results(1, device=device)
                c.sequence.fill_(1)
                c.count.fill_(1)
                c.operation.fill_(int(worlds.WorldOperation.CREATE))
                graph = None

                def record_invalid(directory=directory, c=c, r=r, admitted=admitted):
                    directory.begin(c)
                    if admitted:
                        directory.admit(c)
                    directory.publish_compaction()  # No move phase is open.
                    if not admitted:
                        directory.admit(c)
                    wp.copy(directory.transaction.initialized_sequence, c.sequence)
                    directory.publish(c, r)
                    directory.plan_compaction()
                    wp.copy(directory.compaction.copied_count, directory.compaction.count)
                    directory.publish_compaction()

                try:
                    if captured:
                        wp.load_module(module=worlds.WorldDirectory.__module__, device=device)
                        with wp.ScopedCapture(device=device) as capture:
                            record_invalid()
                        graph = capture.graph
                        del capture
                        directory.retain_graph(graph, c, r)
                        wp.capture_launch(graph)
                    else:
                        record_invalid()
                    self.assertEqual(int(directory.batch_result.status.numpy()[0]), worlds.WorldStatus.PHASE_INVALID)
                    self.assertEqual(int(directory.batch_result.advance_allowed.numpy()[0]), 0)
                    self.assertEqual(int(directory.data.live_count.numpy()[0]), 0)
                    self.assertEqual(int(directory.data.generation.numpy()[0]), 0)
                    np.testing.assert_array_equal(directory.data.slot_id.numpy(), [-1, -1])
                    # Cancellation is recoverable only through a fresh batch, not the aborted transaction.
                    c.sequence.fill_(2)
                    directory.begin(c)
                    directory.admit(c)
                    wp.copy(directory.transaction.initialized_sequence, c.sequence)
                    directory.publish(c, r)
                    self.assertEqual(int(r.status.numpy()[0]), worlds.WorldStatus.OK)
                    self.assertEqual(int(directory.batch_result.advance_allowed.numpy()[0]), 1)
                    self.assertEqual(int(directory.data.live_count.numpy()[0]), 1)
                finally:
                    graph = None
                    directory.close(streams=(wp.get_stream(device),) if wp.get_device(device).is_cuda else ())

    def _check_terminal_generation(self, device, *, captured):
        directory = worlds.WorldDirectory((2,), id_capacity=1, command_capacity=1, device=device)
        directory.publish_ready_slots((2,))
        # Accelerate the sole dead identity to its last reusable generation.
        maximum = 2**64 - 1
        directory.data.generation.fill_(maximum - 1)
        c = worlds.create_world_commands(1, device=device)
        r = worlds.create_world_results(1, device=device)
        payload = wp.full(2, -1, dtype=wp.int64, device=device)
        acknowledge = wp.ones(1, dtype=int, device=device)
        graph = None

        def record():
            directory.begin(c)
            directory.admit(c)
            wp.launch(
                _initialize_reference_payload,
                1,
                [directory.data, c, directory.transaction, acknowledge, payload],
                device=device,
            )
            directory.publish(c, r)
            directory.plan_compaction()
            wp.launch(
                _move_reference_payload,
                1,
                [directory.data, directory.transaction, directory.compaction, payload],
                device=device,
            )
            directory.publish_compaction()

        if captured:
            wp.load_module(module=__name__, device=device)
            wp.load_module(module=worlds.WorldDirectory.__module__, device=device)
            with wp.ScopedCapture(device=device) as capture:
                record()
            graph = capture.graph
            del capture
            directory.retain_graph(graph, c, r, payload, acknowledge)
        operation, status = worlds.WorldOperation, worlds.WorldStatus
        cases = (
            (1, operation.CREATE, status.OK, 1),
            (2, operation.RESET, status.GENERATION_EXHAUSTED, 1),
            (3, operation.DESTROY, status.OK, 0),
            (3, operation.DESTROY, status.OK, 0),  # Equal replay cannot wrap the terminal tombstone.
            (4, operation.CREATE, status.NO_IDS, 0),
            (5, operation.DESTROY, status.NOT_ALIVE, 0),
        )
        try:
            for sequence, op, expected_status, expected_live in cases:
                c.sequence.fill_(sequence)
                c.count.fill_(1)
                c.operation.fill_(int(op))
                c.generation.fill_(maximum)
                wp.capture_launch(graph) if captured else record()
                with self.subTest(operation=op.name, sequence=sequence, captured=captured):
                    self.assertEqual(int(r.status.numpy()[0]), expected_status)
                    self.assertEqual(int(directory.data.generation.numpy()[0]), maximum)
                    self.assertEqual(int(directory.data.live_count.numpy()[0]), expected_live)
                    self.assertEqual(int(directory.data.free_slot_count.numpy()[0]), 2 - expected_live)
                    self.assertEqual(int(directory.data.prototype.numpy()[0]), 0 if expected_live else -1)
                    np.testing.assert_array_equal(
                        directory.data.slot_id.numpy(), [0, -1] if expected_live else [-1, -1]
                    )
            self.assertEqual(int(directory._scratch.id_count.numpy()[0]), 0)
        finally:
            graph = None
            directory.close(streams=(wp.get_stream(device),) if wp.get_device(device).is_cuda else ())

    def _check_batches(self, device, *, captured):
        status = worlds.WorldStatus
        rng = np.random.default_rng(91577)
        capacities, identity_capacity, request_capacity = (5, 7, 3), 9, 8
        directory = worlds.WorldDirectory(
            capacities, id_capacity=identity_capacity, command_capacity=request_capacity, device=device
        )
        directory.publish_ready_slots(capacities)
        c, r = (
            worlds.create_world_commands(request_capacity, device=device),
            worlds.create_world_results(request_capacity, device=device),
        )
        payload = wp.full(sum(capacities), -1, dtype=wp.int64, device=device)
        acknowledge = wp.ones(request_capacity, dtype=int, device=device)
        graph = None

        def record():
            directory.begin(c)
            directory.admit(c)
            wp.launch(
                _initialize_reference_payload,
                request_capacity,
                [directory.data, c, directory.transaction, acknowledge, payload],
                device=device,
            )
            directory.publish(c, r)
            directory.plan_compaction()
            wp.launch(
                _move_reference_payload,
                len(capacities),
                [directory.data, directory.transaction, directory.compaction, payload],
                device=device,
            )
            directory.publish_compaction()

        if captured:
            wp.load_module(module=__name__, device=device)
            wp.load_module(module=worlds.WorldDirectory.__module__, device=device)
            with wp.ScopedCapture(device=device) as capture:
                record()
            graph = capture.graph
            del capture
            directory.retain_graph(graph, payload, acknowledge, c, r)
        # Independent reference knows lifetimes and payloads, never allocator order or chosen slots.
        live, generations, ready = {}, [0] * identity_capacity, list(capacities)
        last_sequence, last_status, last_advance = 0, 0, 0
        rejected_slots = np.zeros(len(capacities), dtype=np.int32)
        seen = set()
        try:
            for step in range(128):
                sequence = last_sequence + 1
                if step % 17 == 5:
                    sequence = last_sequence
                elif step % 17 == 9:
                    sequence = max(0, last_sequence - 1)
                count = int(rng.integers(1, request_capacity + 1))
                if step % 19 == 8:
                    count = -1
                elif step % 19 == 11:
                    count = request_capacity + 1
                requests = np.zeros((request_capacity, 4), dtype=np.int64)
                requests[:, 0] = rng.choice([0, 1, 2, 3, 9], request_capacity, p=[0.12, 0.34, 0.30, 0.20, 0.04])
                for request in requests:
                    identity = (
                        int(rng.choice(list(live)))
                        if live and rng.random() < 0.8
                        else int(rng.integers(-1, identity_capacity + 1))
                    )
                    generation = generations[identity] if 0 <= identity < identity_capacity else 0
                    prototype = int(rng.integers(-1, len(capacities) + 1))
                    request[1:] = identity, generation + int(rng.random() < 0.2), prototype
                if step % 7 == 0:
                    requests[:, 0] = 1
                    requests[:, 3] = rng.integers(0, len(capacities), request_capacity)
                    count = request_capacity
                if step % 5 == 0 and live:
                    identity = next(iter(live))
                    requests[:2] = [[2, identity, generations[identity], 0], [3, identity, generations[identity], 0]]
                    count = max(count, 2)
                ack = rng.integers(0, 5, request_capacity, dtype=np.int32) != 0
                result_fields = ("status", "world_id", "generation")
                before_results = {name: getattr(r, name).numpy().copy() for name in result_fields}
                if step % 11 == 3:
                    count = 0
                pre = dict(live)
                pre_generations = generations.copy()
                pre_slots = directory.data.slot_id.numpy().copy()
                c.sequence.fill_(sequence)
                c.count.fill_(count)
                for column, name in enumerate(("operation", "world_id", "generation", "prototype")):
                    getattr(c, name).assign(requests[:, column].astype(np.uint64 if name == "generation" else np.int32))
                acknowledge.assign(ack.astype(np.int32))
                wp.capture_launch(graph) if captured else record()
                batch = directory.batch_result
                consumed = sequence > last_sequence and 0 <= count <= request_capacity
                if sequence == 0 or sequence < last_sequence:
                    last_status, last_advance = status.STALE_BATCH, 0
                elif sequence > last_sequence:
                    last_sequence = sequence
                    rejected_slots[:] = 0
                    last_status = 0 if consumed else status.BAD_COUNT
                    last_advance = 0
                self.assertEqual(int(batch.consumed.numpy()[0]), int(consumed))
                self.assertEqual(int(batch.sequence.numpy()[0]), last_sequence)
                self.assertEqual(int(batch.status.numpy()[0]), last_status)
                output = {name: getattr(r, name).numpy() for name in before_results}
                destinations = directory.transaction.destination_world_id.numpy()
                destination_slots = directory.transaction.destination_slot.numpy()
                if not consumed:
                    for name, value in before_results.items():
                        np.testing.assert_array_equal(output[name], value)
                else:
                    seen.update(map(int, output["status"][:count]))
                    conflicts = {
                        identity: sum(op in (2, 3) and i == identity for op, i, _, _ in requests[:count])
                        for identity in range(identity_capacity)
                    }
                    ids_allocated = []
                    slots_allocated = [set() for _ in capacities]
                    admitted_requests = [set() for _ in capacities]
                    eligible_slots = [0] * len(capacities)
                    valid_creates = 0
                    for index, (op, identity, generation, prototype) in enumerate(requests[:count]):
                        actual = int(output["status"][index])
                        base = 0
                        if op not in range(4):
                            base = status.INVALID
                        elif op in (2, 3):
                            if not 0 <= identity < identity_capacity:
                                base = status.INVALID
                            elif conflicts[identity] > 1:
                                base = status.CONFLICT
                            elif identity not in pre:
                                base = status.NOT_ALIVE
                            elif generation != pre_generations[identity]:
                                base = status.STALE
                        if base == 0 and op in (1, 2) and not 0 <= prototype < len(capacities):
                            base = status.BAD_PROTOTYPE
                        if base != 0 or op in (0, 3):
                            self.assertEqual(actual, base)
                        else:
                            valid_creates += int(op == 1)
                            allowed = [status.OK, status.NO_SLOTS, status.INITIALIZATION_MISSING]
                            if op == 1:
                                allowed.append(status.NO_IDS)
                            self.assertIn(actual, allowed)
                            if actual != status.NO_IDS:
                                eligible_slots[prototype] += 1
                                destination = int(destinations[index])
                                if op == 1:
                                    self.assertNotIn(destination, pre)
                                    self.assertTrue(0 <= destination < identity_capacity)
                                    ids_allocated.append(destination)
                                else:
                                    self.assertEqual(destination, identity)
                                if actual != status.NO_SLOTS:
                                    row = int(destination_slots[index])
                                    # A granted destination cannot overlap any old live row, including an ending source.
                                    self.assertTrue(0 <= row < ready[prototype])
                                    self.assertNotIn(row, slots_allocated[prototype])
                                    self.assertEqual(pre_slots[sum(capacities[:prototype]) + row], -1)
                                    slots_allocated[prototype].add(row)
                                    admitted_requests[prototype].add(index)
                                    self.assertEqual(actual == 0, bool(ack[index]))
                                else:
                                    rejected_slots[prototype] += 1
                        if actual == 0 and op != 0:
                            result_id = int(output["world_id"][index])
                            result_generation = int(output["generation"][index])
                            self.assertEqual(result_generation, pre_generations[result_id] + 1)
                            self.assertEqual(result_id, int(destinations[index]) if op == 1 else identity)
                            generations[result_id] = result_generation
                            if op == 3:
                                live.pop(result_id)
                            else:
                                tag = sequence * request_capacity + index
                                live[result_id] = (int(prototype), result_generation, tag)
                        else:
                            self.assertEqual(int(output["world_id"][index]), identity)
                            expected_generation = pre_generations[identity] if 0 <= identity < identity_capacity else 0
                            self.assertEqual(int(output["generation"][index]), expected_generation)
                    self.assertEqual(len(ids_allocated), len(set(ids_allocated)))
                    self.assertEqual(len(ids_allocated), min(valid_creates, identity_capacity - len(pre)))
                    starts = directory.transaction.request_starts.numpy()
                    grouped_requests = directory.transaction.admitted_requests.numpy()
                    self.assertEqual(starts[0], 0)
                    for prototype in range(len(capacities)):
                        free = ready[prototype] - sum(p == prototype for p, _, _ in pre.values())
                        self.assertEqual(len(slots_allocated[prototype]), min(free, eligible_slots[prototype]))
                        start, end = starts[prototype : prototype + 2]
                        self.assertEqual(end - start, len(admitted_requests[prototype]))
                        self.assertEqual(set(grouped_requests[start:end]), admitted_requests[prototype])
                    last_advance = int(np.all(output["status"][:count] == 0))
                    for name, value in before_results.items():
                        np.testing.assert_array_equal(output[name][count:], value[count:])
                self.assertEqual(int(batch.advance_allowed.numpy()[0]), last_advance)
                np.testing.assert_array_equal(batch.slot_rejection_count.numpy(), rejected_slots)
                data = {name: getattr(directory.data, name).numpy() for name in worlds.WorldDirectoryData.vars}
                np.testing.assert_array_equal(data["generation"], generations)
                self.assertEqual(set(np.flatnonzero(data["prototype"] >= 0)), set(live))
                self.assertEqual(set(map(int, data["slot_id"][data["slot_id"] >= 0])), set(live))
                self.assertEqual(int((data["slot_id"] >= 0).sum()), len(live))
                physical = payload.numpy()
                for identity, (prototype, generation, tag) in live.items():
                    self.assertEqual(int(data["generation"][identity]), generation)
                    row = int(data["slot"][identity])
                    slot = int(data["slot_starts"][prototype]) + row
                    self.assertEqual(int(data["prototype"][identity]), prototype)
                    self.assertEqual(int(data["slot_id"][slot]), identity)
                    self.assertEqual(int(physical[slot]), tag)
                for prototype in range(len(capacities)):
                    n = sum(p == prototype for p, _, _ in live.values())
                    start = int(data["slot_starts"][prototype])
                    self.assertEqual(int(data["live_count"][prototype]), n)
                    self.assertEqual(int(data["free_slot_count"][prototype]), ready[prototype] - n)
                    np.testing.assert_array_equal(np.sort(data["live_slots"][start : start + n]), np.arange(n))
                    self.assertTrue(np.all(data["slot_id"][start : start + n] >= 0))
                    for rank, row in enumerate(data["live_slots"][start : start + n]):
                        self.assertEqual(int(data["slot_live_rank"][start + row]), rank)
                if step % 13 == 0:
                    ready = [
                        int(rng.integers(int(n), cap + 1))
                        for n, cap in zip(data["live_count"], capacities, strict=True)
                    ]
                    directory.withdraw_ready_slots(tuple(ready))
                    directory.publish_ready_slots(tuple(ready))
            expected = {
                status.OK,
                status.INVALID,
                status.STALE,
                status.NOT_ALIVE,
                status.BAD_PROTOTYPE,
                status.CONFLICT,
                status.NO_IDS,
                status.NO_SLOTS,
                status.INITIALIZATION_MISSING,
            }
            self.assertTrue(expected.issubset(seen), seen)
        finally:
            graph = None
            directory.close(streams=(wp.get_stream(device),) if wp.get_device(device).is_cuda else ())


if __name__ == "__main__":
    unittest.main()
