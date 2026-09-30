# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check world-directory ownership and lifecycle semantics independently of physics."""

import ast
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from newton import worlds


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
            self.directory.t.initialized.fill_(self.sequence)
        self.directory.publish(self.commands, self.results)
        return self.results.status.numpy()[: len(requests)], self.results.id.numpy()[: len(requests)]

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
        original_slot = int(self.directory.d.slot.numpy()[first])
        result, _ = self.submit([(operation.RESET, first, 99, 1), (operation.RESET, second, 1, 0)])
        np.testing.assert_array_equal(result, [status.STALE, status.OK])
        generations = self.directory.d.generation.numpy()
        self.assertEqual((generations[first], generations[second]), (1, 2))
        self.assertEqual(self.directory.d.slot.numpy()[first], original_slot)
        np.testing.assert_array_equal(self.directory.d.active_count.numpy(), [2, 0])
        self.assertEqual(self.directory.d.flags.numpy()[1], 0)

    def test_missing_initialization_and_conflicting_requests_preserve_membership(self):
        """Verify publication requires initialization and rejects all conflicting mutations."""
        operation, status = worlds.WorldOperation, worlds.WorldStatus
        result, _ = self.submit([(operation.CREATE, -1, 0, 0)], initialize=False)
        np.testing.assert_array_equal(result, [status.INITIALIZATION_MISSING])
        np.testing.assert_array_equal(self.directory.d.active_count.numpy(), [0, 0])
        result, ids = self.submit([(operation.CREATE, -1, 0, 0)])
        identity = int(ids[0])
        result, _ = self.submit([(operation.RESET, identity, 1, 1), (operation.DESTROY, identity, 1, 0)])
        np.testing.assert_array_equal(result, [status.CONFLICT, status.CONFLICT])
        self.assertEqual(self.directory.d.generation.numpy()[identity], 1)
        np.testing.assert_array_equal(self.directory.d.active_count.numpy(), [1, 0])

    def test_failed_batch_withdrawal_preserves_every_prototype(self):
        """Validate the entire withdrawal before changing any other ready prefix."""
        result, _ = self.submit([(worlds.WorldOperation.CREATE, -1, 0, 1)])
        self.assertEqual(result[0], worlds.WorldStatus.OK)
        before = self.directory.d.slot_state.numpy().copy()
        with self.assertRaisesRegex(RuntimeError, "live worlds"):
            self.directory.withdraw_ready((0, 0))
        np.testing.assert_array_equal(self.directory.d.slot_state.numpy(), before)
        np.testing.assert_array_equal(self.directory.d.ready_count.numpy(), [2, 1])
        for ends in ((1,), (2, 3), (True, 1), (-1, 0)):
            with self.subTest(ends=ends), self.assertRaises(ValueError):
                self.directory.publish_ready(ends)
        np.testing.assert_array_equal(self.directory.d.slot_state.numpy(), before)

    def test_admitted_and_published_slots_are_prototype_local(self):
        """Keep local row indices distinct from global slot metadata for later prototypes."""
        result, identities = self.submit([(worlds.WorldOperation.CREATE, -1, 0, 1)])
        self.assertEqual(result[0], worlds.WorldStatus.OK)
        identity = int(identities[0])
        destination = int(self.directory.t.destination_slot.numpy()[0])
        self.assertEqual(destination, 1)
        self.assertEqual(int(self.directory.d.slot.numpy()[identity]), destination)
        self.assertEqual(int(self.directory.d.slot_id.numpy()[2 + destination]), identity)
        self.assertEqual(int(self.directory.d.slot_id.numpy()[destination]), -1)

    def test_acknowledged_compaction_preserves_identity_and_allows_tail_retirement(self):
        """Verify acknowledged row moves retain generation and release the unused tail."""
        result, ids = self.submit([(worlds.WorldOperation.CREATE, -1, 0, 0)])
        self.assertEqual(result[0], worlds.WorldStatus.OK)
        identity = int(ids[0])
        self.assertEqual(self.directory.d.slot.numpy()[identity], 1)
        self.directory.plan_moves()
        np.testing.assert_array_equal(self.directory.moves.sources.numpy(), [1, 0])
        self.directory.moves.copied.assign(np.array([1, 0], np.int32))
        self.directory.publish_moves()
        self.assertEqual(self.directory.d.slot.numpy()[identity], 0)
        self.assertEqual(self.directory.d.generation.numpy()[identity], 1)
        self.directory.withdraw_ready((1, 2))
        with self.assertRaisesRegex(RuntimeError, "live worlds"):
            self.directory.withdraw_ready((0, 2))


if __name__ == "__main__":
    unittest.main()
