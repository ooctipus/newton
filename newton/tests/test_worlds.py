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


class WorldArchitectureTests(unittest.TestCase):
    """Guard the intended ownership graph rather than historical module names."""

    root = Path(__file__).resolve().parents[1]
    owners = (
        "_src/utils/cuda_vmm.py",
        "_src/utils/row_storage.py",
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
        row_source = (self.root / "_src/utils/row_storage.py").read_text()
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
            "NativePopulation",
            "make_commands",
            "make_results",
        ):
            self.assertFalse(hasattr(worlds, name), name)
        self.assertEqual([int(value) for value in worlds.WorldOperation], [0, 1, 2, 3])
        self.assertEqual(worlds.WorldStatus.INITIALIZATION_MISSING, 11)

    def test_directory_free_count_is_canonical_without_readiness_alias(self):
        """Keep free admission slots distinct from mechanically backed row prefixes."""
        self.assertIn("free_count", worlds.WorldDirectoryData.vars)
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
                "starts",
                "slot_id",
                "slot_rank",
                "active",
                "active_count",
                "free_count",
                "demand",
            },
        )
        self.assertEqual(set(worlds.WorldBatch.vars), {"sequence", "consumed", "advance", "status"})
        self.assertEqual(
            set(worlds.WorldTransaction.vars),
            {
                "phase",
                "status",
                "initialized",
                "destination_id",
                "destination_slot",
                "accepted_requests",
                "group_starts",
            },
        )
        self.assertEqual(set(worlds.WorldCompaction.vars), {"source", "destination", "count", "copied"})
        self.assertNotIn("_WorldScratch", worlds.__all__)
        self.assertFalse(hasattr(worlds, "_WorldScratch"))
        for record in (worlds.WorldDirectoryData, worlds.WorldBatch, worlds.WorldTransaction, worlds.WorldCompaction):
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
        self.assertEqual([arg.arg for arg in resize.args.args], ["self", "rows"])
        for method in owner.body:
            if not isinstance(method, ast.FunctionDef):
                continue
            for loop in (node for node in ast.walk(method) if isinstance(node, (ast.For, ast.While))):
                for call in (node for node in ast.walk(loop) if isinstance(node, ast.Call)):
                    if isinstance(call.func, ast.Attribute):
                        self.assertNotIn(call.func.attr, ("publish_ready", "withdraw_ready"), method.name)

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
        self.directory.publish_ready((2, 2))

    def tearDown(self):
        self.directory.close(streams=())

    def submit(self, requests, *, initialize=True):
        self.sequence += 1
        self.commands.sequence.fill_(self.sequence)
        self.commands.count.fill_(len(requests))
        for column, name in enumerate(("op", "id", "generation", "prototype")):
            values = np.zeros(4, np.uint64 if name == "generation" else np.int32)
            values[: len(requests)] = [request[column] for request in requests]
            getattr(self.commands, name).assign(values)
        self.directory.begin(self.commands)
        self.directory.admit(self.commands)
        if initialize:
            self.directory.transaction.initialized.fill_(self.sequence)
        self.directory.publish(self.commands, self.results)
        return self.results.status.numpy()[: len(requests)], self.results.id.numpy()[: len(requests)]

    def assert_relations(self):
        """Check independent cardinality and forward/inverse membership laws."""
        d = self.directory.data
        arrays = {name: getattr(d, name).numpy() for name in d._cls.vars}
        live = np.flatnonzero(arrays["prototype"] >= 0)
        self.assertEqual(int(arrays["active_count"].sum()), len(live))
        occupied = np.flatnonzero(arrays["slot_id"] >= 0)
        self.assertEqual(len(occupied), len(live))
        self.assertEqual(len(set(map(int, arrays["slot_id"][occupied]))), len(live))
        for prototype, count in enumerate(arrays["active_count"]):
            start, end = arrays["starts"][prototype : prototype + 2]
            rows = arrays["active"][start : start + count]
            self.assertEqual(len(set(map(int, rows))), count)
            self.assertTrue(np.all((rows >= 0) & (rows < end - start)))
            self.assertLessEqual(count + arrays["free_count"][prototype], end - start)
            for rank, row in enumerate(rows):
                identity = arrays["slot_id"][start + row]
                self.assertGreaterEqual(identity, 0)
                self.assertEqual(arrays["prototype"][identity], prototype)
                self.assertEqual(arrays["slot"][identity], row)
                self.assertEqual(arrays["slot_rank"][start + row], rank)
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
        self.directory.publish_ready((3,))
        op = worlds.WorldOperation
        self.submit([(op.CREATE, -1, 0, 0)])
        self.assertEqual(int(self.directory.data.slot.numpy()[0]), 2)
        self.directory.plan_moves()
        self.directory.compaction.copied.assign(self.directory.compaction.count.numpy())
        self.directory.publish_moves()
        p, row, valid = self.locations([0], [1])
        np.testing.assert_array_equal([p[0], row[0], valid[0]], [0, 0, True])
        self.assert_relations()
        self.submit([(op.DESTROY, 0, 1, 0)])
        result, ids = self.submit([(op.CREATE, -1, 0, 0)])
        self.assertEqual((int(result[0]), int(ids[0])), (int(worlds.WorldStatus.OK), 0))
        self.assertEqual(int(self.directory.data.generation.numpy()[0]), 3)
        np.testing.assert_array_equal(self.locations([0, 0], [1, 3])[2], [False, True])
        self.assert_relations()

    def test_named_batch_outcome_preserves_replay_and_rejection_laws(self):
        """Distinguish consumed requests, request failure and batch rejection from old results."""
        op, status, batch = worlds.WorldOperation, worlds.WorldStatus, self.directory.batch
        self.submit([(op.CREATE, -1, 0, 0)])
        generation = self.directory.data.generation.numpy().copy()
        original_results = self.results.id.numpy().copy()
        self.commands.op.fill_(op.DESTROY)
        self.directory.begin(self.commands)
        self.directory.admit(self.commands)
        self.directory.publish(self.commands, self.results)
        self.assertEqual(
            (int(batch.consumed.numpy()[0]), int(batch.advance.numpy()[0]), int(batch.status.numpy()[0])),
            (0, 1, status.OK),
        )
        np.testing.assert_array_equal(self.directory.data.generation.numpy(), generation)
        np.testing.assert_array_equal(self.results.id.numpy(), original_results)
        self.commands.sequence.fill_(2)
        self.commands.count.fill_(5)
        self.directory.begin(self.commands)
        self.directory.admit(self.commands)
        self.directory.publish(self.commands, self.results)
        self.assertEqual(
            (int(batch.consumed.numpy()[0]), int(batch.advance.numpy()[0]), int(batch.status.numpy()[0])),
            (0, 0, status.BAD_COUNT),
        )
        np.testing.assert_array_equal(self.results.id.numpy(), original_results)
        self.commands.sequence.fill_(1)
        self.directory.begin(self.commands)
        self.assertEqual(int(batch.status.numpy()[0]), status.STALE_BATCH)
        self.sequence = 2
        self.submit([])
        self.assertEqual(
            (int(batch.consumed.numpy()[0]), int(batch.advance.numpy()[0]), int(batch.status.numpy()[0])),
            (1, 1, status.OK),
        )
        self.assert_relations()

    def test_unacknowledged_moves_preserve_lifetimes_and_deny_advancement(self):
        """Reject move publication before the domain has copied every retained row."""
        self.submit([(worlds.WorldOperation.CREATE, -1, 0, 0)])
        before = self.assert_relations()
        self.directory.plan_moves()
        self.directory.publish_moves()
        self.assertEqual(int(self.directory.batch.status.numpy()[0]), worlds.WorldStatus.COMPACTION_INVALID)
        self.assertEqual(int(self.directory.batch.advance.numpy()[0]), 0)
        after = self.assert_relations()
        for name in ("prototype", "slot", "generation", "slot_id"):
            np.testing.assert_array_equal(after[name], before[name])

    def test_capacity_publication_does_not_implicitly_withdraw_rows(self):
        """Keep grow publication separate from validated tail retirement."""
        self.directory.publish_ready((1, 1))
        np.testing.assert_array_equal(self.directory.data.free_count.numpy(), [2, 2])
        self.directory.withdraw_ready((1, 1))
        np.testing.assert_array_equal(self.directory.data.free_count.numpy(), [1, 1])
        self.directory.publish_ready((2, 1))
        np.testing.assert_array_equal(self.directory.data.free_count.numpy(), [2, 1])

    def test_retained_graph_prevents_closure_until_borrower_retires(self):
        """Keep graph-borrowed directory metadata alive until the executable is released."""

        class Graph:
            device = self.directory.device

        graph = Graph()
        payload = object()
        self.assertIs(self.directory.retain_graph(graph, payload), graph)
        self.assertEqual(graph.world_owners, (self.directory, payload))
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
                    self.directory.publish_ready((2, 2))
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
        np.testing.assert_array_equal(self.directory.data.active_count.numpy(), [2, 0])
        self.assertEqual(self.directory.batch.advance.numpy()[0], 0)
        self.assertEqual(self.directory.batch.status.numpy()[0], status.OK)
        self.assertEqual(self.directory.batch.consumed.numpy()[0], 1)

    def test_missing_initialization_and_conflicting_requests_preserve_membership(self):
        """Verify publication requires initialization and rejects all conflicting mutations."""
        operation, status = worlds.WorldOperation, worlds.WorldStatus
        result, _ = self.submit([(operation.CREATE, -1, 0, 0)], initialize=False)
        np.testing.assert_array_equal(result, [status.INITIALIZATION_MISSING])
        np.testing.assert_array_equal(self.directory.data.active_count.numpy(), [0, 0])
        result, ids = self.submit([(operation.CREATE, -1, 0, 0)])
        identity = int(ids[0])
        result, _ = self.submit([(operation.RESET, identity, 1, 1), (operation.DESTROY, identity, 1, 0)])
        np.testing.assert_array_equal(result, [status.CONFLICT, status.CONFLICT])
        self.assertEqual(self.directory.data.generation.numpy()[identity], 1)
        np.testing.assert_array_equal(self.directory.data.active_count.numpy(), [1, 0])

    def test_failed_batch_withdrawal_preserves_every_prototype(self):
        """Validate the entire withdrawal before changing any other ready prefix."""
        result, _ = self.submit([(worlds.WorldOperation.CREATE, -1, 0, 1)])
        self.assertEqual(result[0], worlds.WorldStatus.OK)
        before = self.directory.data.slot_id.numpy().copy()
        with self.assertRaisesRegex(RuntimeError, "live worlds"):
            self.directory.withdraw_ready((0, 0))
        np.testing.assert_array_equal(self.directory.data.slot_id.numpy(), before)
        np.testing.assert_array_equal(self.directory.data.free_count.numpy(), [2, 1])
        for ends in ((1,), (2, 3), (True, 1), (-1, 0)):
            with self.subTest(ends=ends), self.assertRaises(ValueError):
                self.directory.publish_ready(ends)
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
        self.directory.plan_moves()
        np.testing.assert_array_equal(self.directory.compaction.count.numpy(), [1, 0])
        self.directory.compaction.copied.assign(np.array([1, 0], np.int32))
        self.directory.publish_moves()
        self.assertEqual(self.directory.data.slot.numpy()[identity], 0)
        self.assertEqual(self.directory.data.generation.numpy()[identity], 1)
        self.directory.withdraw_ready((1, 2))
        with self.assertRaisesRegex(RuntimeError, "live worlds"):
            self.directory.withdraw_ready((0, 2))


if __name__ == "__main__":
    unittest.main()
