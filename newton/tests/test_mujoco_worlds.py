# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Qualify the public native population against independent dense physics."""

import ast
import gc
import inspect
import json
import os
import subprocess
import sys
import textwrap
import traceback
import unittest
import weakref
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import numpy as np
import warp as wp
from gpu_components import backing as backing_ops
from gpu_components import directory as directory_ops
from gpu_components import fields as field_ops
from gpu_components import graph as graph_ops
from gpu_components.backing_data import VirtualReservation
from gpu_components.directory_data import (
    InstanceBatchResult,
    InstanceCommands,
    InstanceOperation,
    InstancePhase,
    InstanceStatus,
)
from gpu_components.field_data import FieldStorage
from gpu_components.graph_data import GraphUpdateTable
from warp._src import capture_allocation

import newton.solvers
from newton._src.solvers.mujoco import worlds as native
from newton._src.solvers.mujoco.worlds import _MuJoCoWorldPopulation
from newton.solvers import (
    MuJoCoWorldPopulation,
    MuJoCoWorlds,
    mujoco_world_population_ready_capacity,
    mujoco_world_population_validate,
    mujoco_worlds_capture,
    mujoco_worlds_close,
    mujoco_worlds_grow_backing,
    mujoco_worlds_memory_report,
    mujoco_worlds_prepare,
    mujoco_worlds_resize_backing,
    mujoco_worlds_validate,
)

_CREATE, _REPLACE, _DESTROY = (
    int(value) for value in (InstanceOperation.CREATE, InstanceOperation.REPLACE, InstanceOperation.DESTROY)
)
_OK, _INVALID = int(InstanceStatus.OK), int(InstanceStatus.INVALID)
_STATE = (
    "qpos",
    "qvel",
    "qacc",
    "qacc_warmstart",
    "time",
    "ctrl",
    "qfrc_applied",
    "xfrc_applied",
    "mocap_pos",
    "mocap_quat",
    "tree_asleep",
    "tree_awake",
    "body_awake",
)
_POSE = ("xpos", "xquat", "xmat", "geom_xpos", "geom_xmat", "site_xpos", "site_xmat")


def _population_view(group, prototype_index=0):
    """Build the same borrowed descriptor relation as the native composition root."""
    group.prototype_index = prototype_index
    return MuJoCoWorldPopulation(
        prototype_index,
        group.model,
        group.data,
        group.world_storage.capacity,
        group.contact_storage.capacity,
        wp.upper_bound(group.count_parameters[2]),
        group.world_storage.protected_count,
        group.world_storage.ready_count,
        group.contact_count,
        group.ccd_count,
        weakref.ref(group),
    )


class MuJoCoWorldsHostTests(unittest.TestCase):
    """Check joined service admission and failure publication without CUDA."""

    def test_transient_slots_require_canonical_counts_and_native_ordering(self):
        """Reuse compatible W/C/D and fixed slots only with exact operand-completion proofs."""
        counts = tuple(wp.CountParameter(n) for n in (4, 8, 12))
        scalar = wp.zeros(1, dtype=int, device="cpu")

        def owner(capacity):
            return SimpleNamespace(
                capacity=capacity, ready_rows=capacity, device=scalar.device, protected_count=scalar, ready_count=scalar
            )

        group = _MuJoCoWorldPopulation(
            None,
            world_storage=owner(4),
            contact_storage=owner(8),
            contact_count=scalar,
            ccd_count=scalar,
            contact_quota=2,
            ccd_quota=3,
            count_parameters=counts,
        )
        worlds = MuJoCoWorlds(device=scalar.device, _populations=[group])
        requests = tuple(
            SimpleNamespace(index=i, shape=shape, dtype=wp.float32, strides=None, transient_region=region)
            for i, (shape, region) in enumerate(
                (
                    ((counts[0], 2), 0),
                    ((counts[0], 2), 0),
                    ((counts[0], 2), 1),
                    ((counts[0], 2), 1),
                    ((counts[0], 2), 2),
                    ((counts[1], 3), 0),
                    ((counts[2], 2), 0),
                    ((1,), 0),
                    ((1,), 1),
                    ((counts[1], 3), 0),
                )
            )
        )
        allocations = []

        def allocate(capacity, protected_count, *, fields, **kwargs):
            allocations.append(fields)
            arrays = {
                spec.name: wp.empty((capacity, *spec.inner_shape), dtype=spec.dtype, device="cpu") for spec in fields
            }
            return SimpleNamespace(
                capacity=capacity,
                protected_count=protected_count,
                ready_count=scalar,
                device=scalar.device,
                ready_rows=capacity,
                arrays=arrays,
            )

        with (
            patch.object(wp, "capture_get_allocations", return_value=requests),
            patch.object(wp, "capture_get_transient_regions", return_value=(0, 1, 2)),
            patch.object(
                wp,
                "capture_allocation_order",
                side_effect=lambda graph, pairs: tuple(
                    (left.transient_region, right.transient_region) == (0, 1) or (left.index, right.index) == (5, 9)
                    for left, right in pairs
                ),
            ) as order,
            patch.object(field_ops, "allocate", side_effect=allocate),
        ):
            bindings = native._prepare_transient_storage(worlds, object(), [(0, 1, 2)])
            self.assertEqual(order.call_count, 1, "Batch native dependency queries before allocating slots")
            self.assertTrue(all(left.shape == right.shape for left, right in order.call_args.args[1]))
            self.assertEqual(tuple(len(fields) for fields in allocations), (3, 1, 1))
            for first, second in ((0, 2), (1, 3), (5, 9), (7, 8)):
                self.assertEqual(bindings[first][1].ptr, bindings[second][1].ptr)
                self.assertIsNot(bindings[first][1], bindings[second][1])
            self.assertNotIn(bindings[4][1].ptr, (bindings[0][1].ptr, bindings[1][1].ptr))
            self.assertEqual([owner.capacity for owner in group.transient_storages], [4, 8, 12])
            signature = native._binding_signature(group)
            column = group.transient_storages[0].arrays["0"]
            group.transient_storages[0].arrays["0"] = wp.empty_like(column)
            self.assertNotEqual(signature, native._binding_signature(group))
            with self.assertRaisesRegex(ValueError, "Prepared storage or count descriptors changed"):
                native._validate_population(group)
            group.transient_storages[0].arrays["0"] = column
            for shape, strides in (((wp.CountParameter(4), 2), None), ((2, counts[0]), None), ((counts[0], 2), (8, 4))):
                requests[0].shape, requests[0].strides = shape, strides
                with self.assertRaises(ValueError):
                    native._prepare_transient_storage(worlds, object(), [(0, 1, 2)])
            requests[0].shape, requests[0].strides = (counts[0], 2), None
            with self.assertRaisesRegex(ValueError, "exact recorded native region"):
                native._prepare_transient_storage(worlds, object(), [(0, 1)])
            self.assertEqual(len(allocations), 3, "Reject invalid relations before allocating owners")
            group.transient_storages, group.transient_arrays = (), ()
            requests[0].shape, requests[1].shape = (0,), (counts[0], 0)
            bindings = native._prepare_transient_storage(worlds, object(), [(0, 1, 2)])
            self.assertEqual(bindings[0][1].shape, (0,))
            self.assertEqual(bindings[1][1].shape, (4, 0))
            self.assertEqual(bindings[0][1].capacity + bindings[1][1].capacity, 0)
            self.assertIsNot(bindings[0][1], bindings[1][1])

    def test_transient_readiness_limits_execution_without_changing_live_counts(self):
        """Block physics until every scratch domain and published count fits its accessible prefix."""
        directory = directory_ops.allocate((2,), id_capacity=2, command_capacity=2, device="cpu")
        directory.data.live_count.fill_(2)
        directory.batch_result.advance_allowed.fill_(1)
        ready, enabled = (wp.full(1, n, dtype=int, device="cpu") for n in (2, 1))
        scratch = tuple(wp.full(1, 2, dtype=int, device="cpu") for _ in range(3))
        step, poses = (wp.zeros(1, dtype=int, device="cpu") for _ in range(2))
        for domain in range(3):
            for scratch_ready in (0, 1, 2):
                scratch[domain].fill_(scratch_ready)
                wp.launch(
                    native._execution_conditions,
                    1,
                    [
                        directory.data,
                        directory.batch_result,
                        0,
                        ready,
                        ready,
                        *scratch,
                        ready,
                        ready,
                        1,
                        1,
                        enabled,
                        enabled,
                        step,
                        poses,
                    ],
                    device="cpu",
                )
                np.testing.assert_array_equal(step.numpy(), [int(scratch_ready == 2)])
                np.testing.assert_array_equal(poses.numpy(), [int(scratch_ready == 2)])
        np.testing.assert_array_equal(directory.data.live_count.numpy(), [2])

    def test_published_candidate_and_ccd_counts_follow_all_actual_owners(self):
        """Publish the minimum accessible prefix without allocating absent domain owners."""
        group = _MuJoCoWorldPopulation(
            None,
            contact_quota=2,
            ccd_quota=3,
            count_parameters=tuple(wp.CountParameter(n) for n in (4, 8, 12)),
            contact_count=wp.zeros(1, dtype=int, device="cpu"),
            ccd_count=wp.zeros(1, dtype=int, device="cpu"),
            world_storage=SimpleNamespace(ready_rows=4),
            contact_storage=SimpleNamespace(ready_rows=7),
        )
        native._publish_native_counts(group)
        np.testing.assert_array_equal(group.ccd_count.numpy(), [0])
        self.assertEqual(native._world_ready_capacity(group), 0)
        group.transient_storages = (None, SimpleNamespace(ready_rows=5), SimpleNamespace(ready_rows=4))
        native._publish_native_counts(group)
        np.testing.assert_array_equal(group.contact_count.numpy(), [5])
        np.testing.assert_array_equal(group.ccd_count.numpy(), [4])
        self.assertEqual(native._world_ready_capacity(group), 1)
        group.transient_storages = (None, None, None)
        native._publish_native_counts(group)
        np.testing.assert_array_equal(group.contact_count.numpy(), [7])
        np.testing.assert_array_equal(group.ccd_count.numpy(), [12])

    def test_generic_components_are_external_and_native_bindings_have_one_authority(self):
        """Reject retained generic owners, compatibility namespaces and duplicate stage ledgers."""
        root = Path(__file__).resolve().parents[1]
        for name in (
            "worlds.py",
            "_src/sim/worlds.py",
            "_src/utils/field_storage.py",
            "_src/utils/cuda_vmm.py",
            "_src/utils/cuda_graph.py",
            "_src/utils/cuda_graph.cu",
        ):
            self.assertFalse((root / name).exists(), name)
        self.assertNotIn("worlds", newton.__all__)
        self.assertNotIn("_KERNEL_CACHE", inspect.getsource(newton.solvers.SolverMuJoCo._prepare_generated_kernels))
        source = ast.parse(inspect.getsource(native))
        self.assertFalse({"_data_arrays", "_replace_data"} & vars(native).keys())
        generic_imports = [
            node
            for node in ast.walk(source)
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("gpu_components")
        ]
        self.assertTrue(generic_imports)
        for node in generic_imports:
            self.assertFalse(any(part.startswith("_") for part in node.module.split(".")))
            self.assertTrue(all(not item.name.startswith("_") for item in node.names))
        for node in ast.walk(source):
            if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
                receiver = ast.unparse(node.value)
                if receiver in ("worlds._directory", "self._backing", "owner") or receiver.endswith(
                    ("_storage", ".updates")
                ):
                    self.fail(f"Native composition reads private component data: {ast.unparse(node)}")
        for record in (MuJoCoWorlds, MuJoCoWorldPopulation, _MuJoCoWorldPopulation):
            self.assertTrue(is_dataclass(record))
            body = ast.parse(textwrap.dedent(inspect.getsource(record))).body[0].body
            self.assertFalse(any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in body))
        self.assertTrue(
            {"execution_data", "updates", "bindings", "transient_storages", "transient_arrays"}
            <= _MuJoCoWorldPopulation.__dataclass_fields__.keys()
        )
        self.assertFalse(
            {"work", "workspace", "step_bindings", "ccd_storage"} & _MuJoCoWorldPopulation.__dataclass_fields__.keys()
        )
        for rejected in (
            "StepBindings",
            "StepWork",
            "step_work_layout",
            "make_step_work",
            "validate_step_work",
            "make_step_workspace",
            "discover_step_scratch",
            "scratch_array",
        ):
            self.assertNotIn(rejected, inspect.getsource(native))
        planner = inspect.getsource(native._prepare_transient_storage)
        for forbidden in ("qacc", "qLD", "qLDiagInv", "actuator_vel", "mjw."):
            self.assertNotIn(forbidden, planner, "Native transient shapes must have only their producer's authority")
        physics = ast.parse(inspect.getsource(native._record_physics))
        transient = next(
            node
            for node in ast.walk(physics)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "wp.capture_transient"
        )
        calls = {ast.unparse(node.func) for node in ast.walk(transient.args[0]) if isinstance(node, ast.Call)}
        self.assertEqual(calls, {"mjw.step"}, "Application callbacks must stay outside the native transient contract")
        self.assertEqual(
            {keyword.arg: ast.literal_eval(keyword.value) for keyword in transient.keywords},
            {"assume_nonescaping": True, "assume_no_indirect_access": True},
        )
        capture_source = inspect.getsource(mujoco_worlds_capture)
        self.assertIn("record_launches=True", capture_source)
        self.assertIn("record_memory_operations=True", capture_source)
        self.assertIn("_bind_native_program", capture_source)
        self.assertIn("graph_ops.adopt_launches", capture_source)
        self.assertIn("group.application_ranges = None", capture_source)

    def test_base_import_does_not_require_optional_native_components(self):
        """Base Newton must import even when standalone components and physics backends are unavailable."""
        probe = """
import builtins
import sys
original = builtins.__import__
blocked = {'gpu_components', 'mujoco', 'mujoco_warp'}
def independent(name, *args, **kwargs):
    if name.split('.')[0] in blocked:
        raise AssertionError('Base Newton imported optional dependency: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = independent
import newton
assert not blocked.intersection(sys.modules)
assert 'worlds' not in newton.__all__
"""
        subprocess.run([sys.executable, "-c", probe], check=True, capture_output=True, text=True)

    def test_prototype_model_rejects_slot_dependent_parameter_batches_before_allocation(self):
        """Moving a world's Data must never change its immutable mass, geometry or timestep."""
        scalar_batch = wp.array(ndim=1, dtype=wp.float32)
        scalar_batch.shape = ("*",)
        vector_batch = wp.array(ndim=2, dtype=wp.float32)
        vector_batch.shape = ("*", "nbody")
        topology = wp.array(ndim=1, dtype=wp.int32)
        topology.shape = ("nbody",)

        @dataclass
        class Options:
            timestep: scalar_batch

        @dataclass
        class Model:
            qpos0: vector_batch
            body_mass: vector_batch
            body_parentid: topology
            opt: Options

        model = Model(
            wp.zeros((1, 1), device="cpu"),
            wp.ones((1, 3), device="cpu"),
            wp.zeros(3, dtype=int, device="cpu"),
            Options(wp.full(1, 0.005, device="cpu")),
        )
        template = SimpleNamespace(nworld=1, qpos=model.qpos0, naconmax=1, naccdmax=1)
        with (
            patch.object(directory_ops, "allocate", side_effect=AssertionError("Unexpected directory allocation")),
            patch.object(field_ops, "allocate", side_effect=AssertionError("Unexpected field allocation")),
            patch.object(backing_ops, "prepare", side_effect=AssertionError("Unexpected virtual allocation")),
        ):
            for owner, name, shape in (
                (model, "body_mass", (2, 3)),
                (model, "qpos0", (2, 1)),
                (model.opt, "timestep", (2,)),
            ):
                original = getattr(owner, name)
                setattr(owner, name, wp.ones(shape, device="cpu"))
                try:
                    with self.subTest(name=name), self.assertRaisesRegex(ValueError, f"one broadcast row: .*{name}"):
                        mujoco_worlds_prepare(
                            ((model, template),), world_capacities=(4,), id_capacity=2, command_capacity=2
                        )
                finally:
                    setattr(owner, name, original)
            # Three topology entries are not three model parameter rows. Empty optional fields
            # carry no slot-dependent values, so neither case violates prototype uniformity.
            for mass in (model.body_mass, wp.empty((2, 0), device="cpu")):
                model.body_mass = mass
                with self.assertRaisesRegex(ValueError, "CUDA device"):
                    mujoco_worlds_prepare(
                        ((model, template),), world_capacities=(4,), id_capacity=2, command_capacity=2
                    )

    def test_public_relations_exclude_mutators_and_survive_failure_and_close(self):
        """Expose observation records without leaking directory protocol or hiding diagnostics."""
        population = MuJoCoWorlds()
        directory = population._directory = directory_ops.allocate(
            (2,), id_capacity=2, command_capacity=2, device="cpu"
        )
        population.directory, population.batch_result = directory.data, directory.batch_result
        population._closed = population._service_failed = False
        population._backing, population._populations = None, []
        population._healthy = population._always_permit = population._lifecycle_needed = wp.zeros(
            1, dtype=int, device="cpu"
        )
        self.assertFalse(hasattr(population, "backing"))
        self.assertIs(population.directory, directory.data)
        self.assertIs(population.batch_result, directory.batch_result)
        for name in (
            "begin",
            "admit",
            "publish",
            "plan_compaction",
            "publish_compaction",
            "publish_ready_slots",
            "close",
        ):
            self.assertFalse(hasattr(population.directory, name), name)
        for failed, closed in ((True, False), (True, True)):
            population._service_failed, population._closed = failed, closed
            self.assertIs(population.directory, directory.data)
            self.assertIs(population.batch_result, directory.batch_result)
            self.assertGreater(mujoco_worlds_memory_report(population)["directory"]["directory_metadata_bytes"], 0)
        directory_ops.close(directory, streams=())
        self.assertGreater(mujoco_worlds_memory_report(population)["directory"]["directory_metadata_bytes"], 0)

    def test_payload_callbacks_receive_only_their_permitted_output_arrays(self):
        """Keep phase and destination ownership inside the composed population lifecycle."""
        tree = ast.parse(textwrap.dedent(inspect.getsource(mujoco_worlds_capture)))
        calls = {
            node.func.id: node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("validate", "initialize")
        }
        self.assertEqual(set(calls), {"validate", "initialize"})
        self.assertEqual(
            [ast.unparse(arg) for arg in calls["validate"].args],
            ["commands", "worlds._directory.transaction.status", "worlds._directory.batch_result.consumed"],
        )
        self.assertEqual(
            ast.unparse(calls["initialize"].args[-2]), "worlds._directory.transaction.initialized_sequence"
        )
        for kernel in (_validate_payload, _initialize_payload):
            self.assertNotIn("InstanceTransaction", inspect.getsource(kernel.func))
        for call in calls.values():
            self.assertFalse(any(isinstance(arg, ast.Attribute) and arg.attr == "transaction" for arg in call.args))

    def test_partial_retirement_keeps_reports_readable_and_remaining_ownership_retryable(self):
        """Release retired program buffers despite borrowed views and preserve retryable storage reports."""
        population = MuJoCoWorlds()
        population.device = wp.get_device("cpu")
        population._closed = population._service_failed = False
        population._graph = population._backing = None
        population._directory = directory_ops.allocate((2,), id_capacity=2, command_capacity=2, device="cpu")
        population.directory = population._directory.data
        population.batch_result = population._directory.batch_result
        scalar = wp.zeros(1, dtype=int, device="cpu")
        population._healthy = population._always_permit = population._lifecycle_needed = scalar
        group = _MuJoCoWorldPopulation(
            None,
            data=object(),
            contact_quota=1,
            ccd_quota=1,
            contact_count=scalar,
            ccd_count=scalar,
            count_parameters=tuple(wp.CountParameter(2) for _ in range(3)),
        )
        for name in ("world_storage", "contact_storage", "default_storage"):
            setattr(
                group,
                name,
                SimpleNamespace(name=name, capacity=2, ready_rows=2, ready_count=scalar, protected_count=scalar),
            )
        group.transient_storages = tuple(
            SimpleNamespace(name=name, capacity=2, ready_rows=2, ready_count=scalar, protected_count=scalar)
            for name in ("world_scratch", "candidate_scratch", "ccd_scratch")
        )
        group.view = _population_view(group)
        population._populations, population.populations = [group], (group.view,)
        group.updates = GraphUpdateTable(
            enable_count=scalar,
            enable_count_maximum=2,
            device=population.device,
            binding_capacity=2,
            binding_stride_bytes=120,
            bindings=wp.empty(240, dtype=wp.uint8, device="cpu"),
            binding_count=wp.zeros(1, dtype=wp.int32, device="cpu"),
            errors=wp.zeros(2, dtype=wp.int32, device="cpu"),
            _library=None,
            _captured_nodes={},
        )
        retired_program = (weakref.ref(group.updates),)
        retired_buffers = tuple(
            weakref.ref(getattr(group.updates, name)) for name in ("bindings", "binding_count", "errors")
        )
        borrowed_view = group.view
        attempts = []

        def close_storage(owner, *, streams):
            attempts.append(owner)
            if owner is group.contact_storage and attempts.count(owner) == 1:
                raise RuntimeError("retirement failed")

        for name in ("initialization_transfer", "compaction_transfer"):
            setattr(group, name, SimpleNamespace(bindings=group, field_names=("qpos",)))
        for name in (
            "request_indices",
            "source_rows",
            "destination_rows",
            "initialization_count",
            "move_count",
            "step_condition",
            "kinematics_condition",
        ):
            setattr(group, name, scalar)
        relations, batch_result = population.directory, population.batch_result
        stream = Mock(spec=wp.Stream, device=population.device, cuda_stream=11)
        with (
            patch.object(native.wp, "synchronize_stream"),
            patch.object(field_ops, "close", side_effect=close_storage),
            patch.object(field_ops, "memory_report", side_effect=lambda owner: {"owner": owner.name}),
            patch.object(field_ops, "transfer_memory_report", return_value={}),
        ):
            self.assertEqual(mujoco_worlds_memory_report(population)["populations"][0]["retired_subowners"], [])
            with self.assertRaisesRegex(RuntimeError, "retirement failed"):
                mujoco_worlds_close(population, streams=(stream,))
            gc.collect()
            for reference in (*retired_program, *retired_buffers):
                self.assertIsNone(reference(), "An invalid borrowed view must not retain retired program buffers")
            with self.assertRaisesRegex(RuntimeError, "closed"):
                mujoco_world_population_validate(borrowed_view)
            self.assertIs(population.directory, relations)
            self.assertIs(population.batch_result, batch_result)
            report = mujoco_worlds_memory_report(population)["populations"][0]
            self.assertEqual(
                report["retired_subowners"],
                ["initialization_transfer", "compaction_transfer", "updates"],
            )
            for name in ("initialization", "compaction", "compacted_fields", "graph_updates"):
                self.assertIsNone(report[name], name)
            self.assertEqual(report["contact_storage"], {"owner": "contact_storage"})
            self.assertEqual(report["transient_storages"]["ccd"], {"owner": "ccd_scratch"})
            self.assertEqual(report["transient_storages"]["world"], {"owner": "world_scratch"})
            self.assertNotIn(group.transient_storages[2], attempts)
            with self.assertRaisesRegex(RuntimeError, "closed"):
                mujoco_worlds_validate(population)
            mujoco_worlds_close(population, streams=(stream,))
        self.assertEqual(attempts.count(group.contact_storage), 2)
        self.assertEqual(attempts.count(group.transient_storages[2]), 1)
        self.assertEqual(attempts.count(group.transient_storages[0]), 1)
        self.assertEqual(mujoco_worlds_memory_report(population)["populations"], [])
        self.assertIs(population.directory, relations)

    def test_early_capture_failure_releases_unvisited_sibling_callbacks(self):
        """Prepared population owners must not keep a failed task alive through unvisited callbacks."""
        population = MuJoCoWorlds()
        population.device = wp.get_device("cpu")
        population._closed = population._service_failed = population._capture_attempted = False
        population._directory = directory_ops.allocate((2, 2), id_capacity=2, command_capacity=2, device="cpu")
        scalar = wp.ones(1, dtype=int, device="cpu")
        population._healthy = population._always_permit = scalar
        storage = SimpleNamespace(protected_count=scalar, ready_count=scalar, capacity=2, device=population.device)
        population._populations = [
            _MuJoCoWorldPopulation(
                None,
                world_storage=storage,
                transient_storages=(),
                contact_storage=storage,
                count_parameters=tuple(wp.CountParameter(2) for _ in range(3)),
                contact_count=scalar,
                ccd_count=scalar,
            )
            for _ in range(2)
        ]

        class Task:
            def before(self, population):
                raise AssertionError("Physics recording must not be reached")

            def after(self, population):
                raise AssertionError("Physics recording must not be reached")

        task = Task()
        task_reference = weakref.ref(task)
        commands, results = (
            directory_ops.allocate_commands(2, device="cpu"),
            directory_ops.allocate_results(2, device="cpu"),
        )
        with (
            patch.object(graph_ops, "prepare_updates", side_effect=[object(), RuntimeError("preparation failed")]),
            patch.object(native.wp, "get_stream", return_value=object()),
            patch.object(native.wp, "synchronize_stream"),
            self.assertRaisesRegex(RuntimeError, "preparation failed"),
        ):
            mujoco_worlds_capture(population, commands, results, before_step=task.before, after_substep=task.after)
        del task
        gc.collect()
        self.assertIsNone(task_reference())
        for group in population._populations:
            self.assertIsNone(group.before_step)
            self.assertIsNone(group.after_substep)
        self.assertTrue(population._capture_attempted)
        np.testing.assert_array_equal(population._healthy.numpy(), [0])

    def test_capture_preserves_original_failure_when_quarantine_also_fails(self):
        """A failed health publication cannot hide the original preparation error or retain callbacks."""
        population = MuJoCoWorlds()
        population.device = wp.get_device("cpu")
        population._closed = population._service_failed = population._capture_attempted = False
        population._directory = directory_ops.allocate((2,), id_capacity=2, command_capacity=2, device="cpu")
        scalar = population._always_permit = wp.ones(1, dtype=int, device="cpu")
        original, cleanup = RuntimeError("preparation failed"), KeyboardInterrupt("health publication failed")
        population._healthy = SimpleNamespace(fill_=Mock(side_effect=cleanup))
        group = _MuJoCoWorldPopulation(
            None,
            world_storage=SimpleNamespace(protected_count=scalar, capacity=2),
        )
        population._populations = [group]
        with (
            patch.object(graph_ops, "prepare_updates", side_effect=original),
            self.assertRaises(BaseExceptionGroup) as caught,
        ):
            mujoco_worlds_capture(
                population,
                directory_ops.allocate_commands(2, device="cpu"),
                directory_ops.allocate_results(2, device="cpu"),
                before_step=Mock(),
            )
        self.assertEqual(caught.exception.exceptions, (original, cleanup))
        self.assertIsNone(group.before_step)
        frames = []
        traceback_value = original.__traceback__
        while traceback_value is not None:
            if traceback_value.tb_frame.f_code.co_name == "mujoco_worlds_capture":
                frames.append(traceback_value.tb_frame.f_locals)
            traceback_value = traceback_value.tb_next
        self.assertEqual(len(frames), 1)
        self.assertIsNone(frames[0].get("graph"))
        self.assertIsNone(frames[0].get("capture"))

    def test_execution_conditions_gate_physics_and_pose_refresh_independently(self):
        """A reset-only frame refreshes poses while advancement still requires the caller permit."""
        directory = directory_ops.allocate((2,), id_capacity=2, command_capacity=2, device="cpu")
        directory.data.live_count.fill_(2)
        ready, health, permit = (wp.full(1, value, dtype=int, device="cpu") for value in (2, 1, 1))
        step, poses = (wp.zeros(1, dtype=int, device="cpu") for _ in range(2))
        for consumed, advance, stepping in ((1, 1, 0), (1, 0, 1), (1, 1, 1), (0, 1, 0)):
            with self.subTest(consumed=consumed, advance=advance, stepping=stepping):
                directory.batch_result.consumed.fill_(consumed)
                directory.batch_result.advance_allowed.fill_(advance)
                permit.fill_(stepping)
                wp.launch(
                    native._execution_conditions,
                    1,
                    [
                        directory.data,
                        directory.batch_result,
                        0,
                        ready,
                        ready,
                        ready,
                        ready,
                        ready,
                        ready,
                        ready,
                        1,
                        1,
                        health,
                        permit,
                        step,
                        poses,
                    ],
                    device="cpu",
                )
                np.testing.assert_array_equal(step.numpy(), [advance * stepping])
                np.testing.assert_array_equal(poses.numpy(), [advance])

    def test_graph_update_error_latches_health_and_ignores_unused_entries(self):
        """Reject active updater errors before physics and keep the quarantine sticky."""
        healthy = wp.ones(1, dtype=int, device="cpu")
        batch = InstanceBatchResult()
        batch.sequence = wp.zeros(1, dtype=wp.uint64, device="cpu")
        batch.consumed = wp.ones(1, dtype=int, device="cpu")
        batch.advance_allowed = wp.ones(1, dtype=int, device="cpu")
        batch.status = wp.zeros(1, dtype=int, device="cpu")
        count = wp.array([2], dtype=int, device="cpu")
        errors = wp.array([0, 0, -1], dtype=int, device="cpu")
        wp.launch(native._guard_graph_updates, 3, [errors, count, healthy, batch], device="cpu")
        np.testing.assert_array_equal(healthy.numpy(), [1])
        np.testing.assert_array_equal(batch.advance_allowed.numpy(), [1])
        np.testing.assert_array_equal(batch.status.numpy(), [0])
        errors.assign(np.array([0, -1, 0], dtype=np.int32))
        wp.launch(native._guard_graph_updates, 3, [errors, count, healthy, batch], device="cpu")
        np.testing.assert_array_equal(healthy.numpy(), [0])
        np.testing.assert_array_equal(batch.consumed.numpy(), [1])
        np.testing.assert_array_equal(batch.advance_allowed.numpy(), [0])
        np.testing.assert_array_equal(batch.status.numpy(), [int(InstanceStatus.PHASE_INVALID)])
        errors.zero_()
        wp.launch(native._guard_graph_updates, 3, [errors, count, healthy, batch], device="cpu")
        np.testing.assert_array_equal(healthy.numpy(), [0])

    def test_graph_updates_and_global_guards_precede_every_native_condition(self):
        """Keep complete updater and guard phases before every conditional program."""
        prototype = ast.parse(textwrap.dedent(inspect.getsource(native._record_physics)))
        self.assertFalse(
            any(isinstance(node, ast.Attribute) and node.attr == "record_update" for node in ast.walk(prototype))
        )
        capture = ast.parse(textwrap.dedent(inspect.getsource(mujoco_worlds_capture)))
        scopes = [
            node
            for node in ast.walk(capture)
            if isinstance(node, ast.With)
            and any(
                isinstance(item.context_expr, ast.Call)
                and isinstance(item.context_expr.func, ast.Attribute)
                and item.context_expr.func.attr == "ScopedCapture"
                for item in node.items
            )
        ]
        self.assertEqual(len(scopes), 1)
        stages = []
        for name in ("record_update", "_guard_graph_updates", "_execution_conditions", "record_population"):
            matches = [
                index
                for index, statement in enumerate(scopes[0].body)
                if any(
                    (isinstance(node, ast.Attribute) and node.attr == name)
                    or (isinstance(node, ast.Name) and node.id == name)
                    for node in ast.walk(statement)
                )
            ]
            self.assertEqual(len(matches), 1, f"One complete capture phase must own {name}")
            statement = scopes[0].body[matches[0]]
            if name == "record_update":
                self.assertIsInstance(statement, ast.If)
                self.assertEqual(ast.unparse(statement.test), "not discovering")
                statement = statement.body[0]
            if name == "_execution_conditions":
                self.assertIsInstance(statement, ast.For)
            else:
                self.assertIsInstance(statement, ast.Expr)
                self.assertIsInstance(statement.value, ast.Call)
                self.assertIsInstance(statement.value.func, ast.Attribute)
                self.assertEqual(statement.value.func.attr, "capture_parallel")
                self.assertEqual(statement.value.func.value.id, "graph_ops")
            stages.append(matches[0])
        self.assertEqual(stages, sorted(set(stages)), "Execution phases must be distinct and ordered")

    def test_lifecycle_branches_join_before_shared_publication(self):
        """Disjoint initialization and relocation join before changing shared lifetime relations."""
        capture = ast.parse(textwrap.dedent(inspect.getsource(mujoco_worlds_capture)))
        lifecycle = next(
            node for node in ast.walk(capture) if isinstance(node, ast.FunctionDef) and node.name == "record_lifecycle"
        )
        stages = []
        for statement in lifecycle.body:
            if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
                continue
            call = statement.value
            name = ast.unparse(call.func)
            if name == "graph_ops.capture_parallel":
                names = {node.id for node in ast.walk(call) if isinstance(node, ast.Name)}
                stages.append("initialize" if "record_initialization" in names else "compact")
            elif name.startswith("directory_ops."):
                stages.append(name)
        self.assertEqual(
            stages,
            [
                "directory_ops.admit",
                "initialize",
                "directory_ops.publish",
                "directory_ops.plan_compaction",
                "compact",
                "directory_ops.publish_compaction",
            ],
        )
        for branch_name in ("record_initialization", "record_compaction"):
            branch = next(
                node for node in lifecycle.body if isinstance(node, ast.FunctionDef) and node.name == branch_name
            )
            self.assertFalse(
                any(
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "directory_ops"
                    for node in ast.walk(branch)
                ),
                "A parallel prototype branch cannot mutate the shared lifetime directory",
            )

    def test_population_capture_ranges_use_explicit_branches_not_operand_inference(self):
        """A branch owns its exact emitted range even when kernels, shapes and task arrays are shared."""
        capture = ast.parse(textwrap.dedent(inspect.getsource(mujoco_worlds_capture)))
        branch = next(
            node for node in ast.walk(capture) if isinstance(node, ast.FunctionDef) and node.name == "record_population"
        )
        calls = [ast.unparse(node.func) for node in ast.walk(branch) if isinstance(node, ast.Call)]
        self.assertEqual(calls.count("wp.capture_launch_count"), 2)
        self.assertEqual(calls.count("_record_physics"), 1)
        self.assertFalse(
            any(
                isinstance(node, ast.Attribute) and node.attr in ("kernel", "shape", "arrays")
                for node in ast.walk(branch)
            )
        )

    def _growth_fixture(self, *, transient=False):
        """Prepare host-only ownership records; mocked operations track publication ordering."""
        population = MuJoCoWorlds()
        population._closed = population._service_failed = False
        population.device, population._backing = "cpu", object()
        population._healthy = Mock()
        population._directory = object()
        group = _MuJoCoWorldPopulation(
            None,
            contact_quota=2,
            ccd_quota=3,
            contact_count=Mock(),
            ccd_count=Mock(),
            count_parameters=tuple(wp.CountParameter(n) for n in (4, 8, 12)),
            data_defaults={"contact.efc_address": -1},
        )

        def storage(name, quota):
            return SimpleNamespace(
                name=name,
                ready_rows=quota,
                capacity=4 * quota,
                ready_count=object(),
                service_failed=False,
                arrays={"contact.efc_address": object()},
            )

        group.world_storage, group.contact_storage = storage("world_storage", 1), storage("contact_storage", 2)
        group.transient_storages = (
            storage("world_scratch", 1) if transient else None,
            storage("candidate_scratch", 2) if transient else None,
            storage("ccd_scratch", 3),
        )
        population._populations = (group,)
        return population, group

    def test_fresh_growth_maps_all_domains_before_ordered_initialization_and_admission(self):
        """Never publish half a W/C/D mapping plan or expose uninitialized contact scratch."""
        population, group = self._growth_fixture()
        current = Mock(spec=wp.Stream, device="cpu", cuda_stream=11)
        reader = Mock(spec=wp.Stream, device="cpu", cuda_stream=12)
        events = []
        current.wait_stream.side_effect = lambda stream: events.append("wait")

        def ready(owner, target):
            events.append(("ready", owner.name, target))
            owner.ready_rows = target

        with (
            patch.object(field_ops, "can_map_backing_without_join", return_value=True, create=True),
            patch.object(field_ops, "map_backing", side_effect=lambda owner, n: events.append(("map", owner.name, n))),
            patch.object(field_ops, "publish_ready", side_effect=ready),
            patch.object(
                field_ops, "zero", side_effect=lambda owner, **kw: events.append(("zero", owner.name, kw["start"]))
            ),
            patch.object(
                field_ops,
                "fill",
                side_effect=lambda owner, *args, **kw: events.append(("fill", owner.name, kw["start"])),
            ),
            patch.object(directory_ops, "publish_admissible_slots", side_effect=lambda *args: events.append("admit")),
            patch.object(native.wp, "get_stream", return_value=current),
            patch.object(native.wp, "launch", side_effect=lambda *args, **kw: events.append("guard")),
            patch.object(native.wp, "synchronize_stream", side_effect=AssertionError("Fresh growth joined readers")),
            patch.object(backing_ops, "maintenance", side_effect=AssertionError("Fresh growth entered maintenance")),
        ):
            mujoco_worlds_grow_backing(population, (2,), streams=(current, reader))
        self.assertEqual(
            events,
            [
                "wait",
                ("map", "world_storage", 2),
                ("map", "contact_storage", 4),
                ("map", "ccd_scratch", 6),
                ("ready", "world_storage", 2),
                ("ready", "contact_storage", 4),
                ("fill", "contact_storage", 2),
                ("ready", "ccd_scratch", 6),
                "admit",
                "guard",
            ],
        )
        self.assertEqual(native._world_ready_capacity(group), 2)

    def test_transient_growth_budget_failure_never_publishes_partial_readiness(self):
        """Keep scratch in the shared mapping transaction and retry a clean budget rejection."""
        population, group = self._growth_fixture(transient=True)
        stream = Mock(spec=wp.Stream, device="cpu", cuda_stream=11)
        mapped, published = [], []

        def map_backing(owner, target):
            mapped.append(owner.name)
            if owner is group.transient_storages[0] and mapped.count(owner.name) == 1:
                raise MemoryError("transient budget")

        def ready(owner, target):
            published.append(owner.name)
            owner.ready_rows = target

        with (
            patch.object(field_ops, "can_map_backing_without_join", return_value=True),
            patch.object(field_ops, "map_backing", side_effect=map_backing),
            patch.object(field_ops, "publish_ready", side_effect=ready),
            patch.object(field_ops, "fill") as fill,
            patch.object(directory_ops, "publish_admissible_slots") as admit,
            patch.object(native.wp, "get_stream", return_value=stream),
            patch.object(native.wp, "launch"),
            patch.object(native.wp, "synchronize_stream"),
        ):
            with self.assertRaisesRegex(MemoryError, "transient budget"):
                mujoco_worlds_grow_backing(population, (2,), streams=(stream,))
            self.assertEqual(published, [])
            admit.assert_not_called()
            self.assertFalse(population._service_failed)
            self.assertEqual(native._world_ready_capacity(group), 1)
            mujoco_worlds_grow_backing(population, (2,), streams=(stream,))
            self.assertIn("world_scratch", published)
            self.assertEqual(native._world_ready_capacity(group), 2)
            self.assertTrue(all(call.args[0] is not group.transient_storages[0] for call in fill.call_args_list))
            admit.assert_called_once_with(population._directory, (2,))

    def test_transient_resize_joins_and_protects_live_prefix_without_state_initialization(self):
        """Resize scratch with its live source while leaving initialization to the numerical stage."""
        population, group = self._growth_fixture(transient=True)
        population._directory = SimpleNamespace(data=SimpleNamespace(live_count=SimpleNamespace(numpy=lambda: [2])))
        owners = tuple(owner for owner, _ in native._storage_domains(group))
        for owner in owners:
            owner.backing, owner.reservation = population._backing, None
            owner.ready_rows = owner.capacity
        stream = Mock(spec=wp.Stream, device="cpu", cuda_stream=11)
        resized = []

        def resize(owner, target, *, protected_count_host):
            resized.append((owner.name, target, protected_count_host))
            owner.ready_rows = target

        with (
            patch.object(backing_ops, "maintenance", return_value=nullcontext()) as maintenance,
            patch.object(field_ops, "resize_backing", side_effect=resize),
            patch.object(field_ops, "fill") as fill,
            patch.object(directory_ops, "withdraw_admissible_slots") as withdraw,
            patch.object(directory_ops, "publish_admissible_slots") as publish,
            patch.object(native.wp, "get_stream", return_value=stream),
            patch.object(native.wp, "synchronize_stream"),
        ):
            mujoco_worlds_resize_backing(population, (2,), streams=(stream,))
        maintenance.assert_called_once()
        withdraw.assert_called_once_with(population._directory, (2,))
        self.assertIn(("world_scratch", 2, 2), resized)
        fill.assert_not_called()
        publish.assert_called_once_with(population._directory, (2,))

    def test_failed_count_withdrawal_quarantines_before_resizing(self):
        """Quarantine a partial C/D withdrawal and preserve its original publication failure."""
        population, group = self._growth_fixture(transient=True)
        population._directory = SimpleNamespace(data=SimpleNamespace(live_count=SimpleNamespace(numpy=lambda: [1])))
        failure = RuntimeError("CCD withdrawal failed")
        group.ccd_count.zero_.side_effect = failure
        stream = Mock(spec=wp.Stream, device="cpu", cuda_stream=11)
        with (
            patch.object(backing_ops, "maintenance", return_value=nullcontext()),
            patch.object(directory_ops, "withdraw_admissible_slots"),
            patch.object(field_ops, "resize_backing") as resize,
            patch.object(native.wp, "get_stream", return_value=stream),
            patch.object(native.wp, "synchronize_stream"),
            self.assertRaises(RuntimeError) as caught,
        ):
            mujoco_worlds_resize_backing(population, (2,), streams=(stream,))
        self.assertIs(caught.exception, failure)
        group.contact_count.zero_.assert_called_once()
        resize.assert_not_called()
        self.assertTrue(population._service_failed)
        population._healthy.fill_.assert_called_once_with(0)

    def test_growth_during_capture_rejects_before_service_or_quarantine(self):
        """Reject a known unsupported call without poisoning healthy population ownership."""
        population, group = self._growth_fixture()
        current = Mock(spec=wp.Stream, device="cpu", cuda_stream=11)
        reader = Mock(spec=wp.Stream, device="cpu", cuda_stream=12)
        with (
            patch.object(type(wp.get_device("cpu")), "is_capturing", new_callable=PropertyMock, return_value=True),
            patch.object(native.wp, "get_stream", return_value=current),
            patch.object(native.wp, "synchronize_stream") as synchronize,
            patch.object(
                field_ops,
                "can_map_backing_without_join",
                side_effect=RuntimeError("Mapping requires execution outside graph capture"),
            ) as query,
            self.assertRaisesRegex(RuntimeError, "outside graph capture"),
        ):
            mujoco_worlds_grow_backing(population, (2,), streams=(current, reader))
        current.wait_stream.assert_not_called()
        synchronize.assert_not_called()
        query.assert_not_called()
        self.assertFalse(population._service_failed)
        self.assertEqual(native._world_ready_capacity(group), 1)
        population._healthy.fill_.assert_not_called()

    def test_historical_growth_uses_joined_service_before_any_mapping(self):
        """Historical growth adds only mappings and joins publication before returning."""
        population, group = self._growth_fixture()
        streams = (Mock(spec=wp.Stream, device="cpu", cuda_stream=11),)
        events = []

        @contextmanager
        def maintenance(backing, **kwargs):
            self.assertIs(backing, population._backing)
            self.assertEqual(kwargs, {"streams": (11,)})
            events.append("join")
            yield
            events.append("leave")

        def ready(owner, target):
            events.append("ready")
            owner.ready_rows = target

        with (
            patch.object(field_ops, "can_map_backing_without_join", side_effect=(True, False, True), create=True),
            patch.object(field_ops, "map_backing", side_effect=lambda *args: events.append("map")),
            patch.object(field_ops, "publish_ready", side_effect=ready),
            patch.object(field_ops, "fill", side_effect=lambda *args, **kwargs: events.append("initialize")),
            patch.object(backing_ops, "maintenance", side_effect=maintenance),
            patch.object(
                native, "mujoco_worlds_resize_backing", side_effect=AssertionError("Growth cannot release backing")
            ),
            patch.object(directory_ops, "publish_admissible_slots", side_effect=lambda *args: events.append("admit")),
            patch.object(native.wp, "get_stream", return_value=streams[0]),
            patch.object(native.wp, "launch", side_effect=lambda *args, **kwargs: events.append("guard")),
            patch.object(native.wp, "synchronize_stream", side_effect=lambda *args: events.append("publication_join")),
        ):
            mujoco_worlds_grow_backing(population, (2,), streams=streams)
        self.assertEqual(
            events,
            [
                "join",
                "map",
                "map",
                "map",
                "ready",
                "ready",
                "initialize",
                "ready",
                "admit",
                "guard",
                "publication_join",
                "leave",
            ],
        )
        self.assertEqual(native._world_ready_capacity(group), 2)
        for targets in ((0,), (True,), (5,), (), (1, 1)):
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                mujoco_worlds_grow_backing(population, targets, streams=streams)

    def test_historical_growth_completion_failure_quarantines_retained_mappings(self):
        """A failed completion fence cannot leave the populated graph replayable."""
        population, _ = self._growth_fixture()
        current = Mock(spec=wp.Stream, device="cpu", cuda_stream=11)
        with (
            patch.object(field_ops, "can_map_backing_without_join", return_value=False),
            patch.object(field_ops, "map_backing") as mapping,
            patch.object(field_ops, "publish_ready"),
            patch.object(field_ops, "zero"),
            patch.object(field_ops, "fill"),
            patch.object(backing_ops, "maintenance", return_value=nullcontext()),
            patch.object(directory_ops, "publish_admissible_slots"),
            patch.object(native.wp, "get_stream", return_value=current),
            patch.object(native.wp, "launch"),
            patch.object(native.wp, "synchronize_stream", side_effect=[RuntimeError("completion"), None]),
            self.assertRaisesRegex(RuntimeError, "completion"),
        ):
            mujoco_worlds_grow_backing(population, (2,), streams=(current,))
        self.assertEqual(mapping.call_count, 3)
        self.assertTrue(population._service_failed)
        population._healthy.fill_.assert_called_once_with(0)

    def test_growth_budget_rejection_keeps_every_readiness_prefix_unchanged(self):
        """Completed mappings survive budget rejection as headroom, never as premature readiness."""
        population, group = self._growth_fixture()
        current = Mock(spec=wp.Stream, device="cpu", cuda_stream=11)
        mapped = []

        def mapping(owner, target):
            if owner is group.contact_storage:
                raise MemoryError("budget")
            mapped.append((owner.name, target))

        with (
            patch.object(field_ops, "can_map_backing_without_join", return_value=True, create=True),
            patch.object(field_ops, "map_backing", side_effect=mapping),
            patch.object(field_ops, "publish_ready", side_effect=AssertionError("Readiness changed on budget failure")),
            patch.object(directory_ops, "publish_admissible_slots", side_effect=AssertionError("Admission changed")),
            patch.object(native.wp, "get_stream", return_value=current),
            patch.object(native.wp, "synchronize_stream", side_effect=AssertionError("Clean failure joined readers")),
            self.assertRaisesRegex(MemoryError, "budget"),
        ):
            mujoco_worlds_grow_backing(population, (2,), streams=(current,))
        self.assertEqual(mapped, [("world_storage", 2)])
        self.assertEqual(native._world_ready_capacity(group), 1)
        self.assertFalse(population._service_failed)
        population._healthy.fill_.assert_not_called()

    def test_growth_publication_and_mapping_errors_quarantine_existing_graphs(self):
        """A failed ready write or incomplete driver rollback cannot leave retained replay healthy."""
        for stage in ("preflight", "mapping", "ready", "initialization", "admission"):
            with self.subTest(stage=stage):
                population, _ = self._growth_fixture()
                current = Mock(spec=wp.Stream, device="cpu", cuda_stream=11)
                failure = MemoryError("failed service")

                def mapping(owner, target, stage=stage, failure=failure):
                    if stage == "mapping":
                        owner.service_failed = True
                        raise failure

                with (
                    patch.object(
                        field_ops,
                        "can_map_backing_without_join",
                        return_value=True,
                        side_effect=failure if stage == "preflight" else None,
                    ),
                    patch.object(field_ops, "map_backing", side_effect=mapping),
                    patch.object(field_ops, "publish_ready", side_effect=failure if stage == "ready" else None),
                    patch.object(field_ops, "fill", side_effect=failure if stage == "initialization" else None),
                    patch.object(
                        directory_ops, "publish_admissible_slots", side_effect=failure if stage == "admission" else None
                    ),
                    patch.object(native.wp, "get_stream", return_value=current),
                    patch.object(native.wp, "synchronize_stream") as synchronize,
                    self.assertRaises(MemoryError) as caught,
                ):
                    mujoco_worlds_grow_backing(population, (2,), streams=(current,))
                self.assertIs(caught.exception, failure)
                self.assertTrue(population._service_failed)
                population._healthy.fill_.assert_called_once_with(0)
                synchronize.assert_called_once_with(current)

    def test_bad_publication_receipt_latches_replay_health(self):
        """A later valid receipt cannot heal a population whose directory service failed."""
        status = wp.full(1, int(InstanceStatus.PHASE_INVALID), dtype=int, device="cpu")
        healthy = wp.ones(1, dtype=int, device="cpu")
        wp.launch(native._guard_backing_publication, 1, [status, healthy], device="cpu")
        np.testing.assert_array_equal(healthy.numpy(), [0])
        status.zero_()
        wp.launch(native._guard_backing_publication, 1, [status, healthy], device="cpu")
        np.testing.assert_array_equal(healthy.numpy(), [0])

    def test_all_safe_shrinks_precede_growth_with_one_shared_backing_budget(self):
        """Return mapped but unpublished headroom before another group consumes the shared budget."""
        population = MuJoCoWorlds()
        population._closed = population._service_failed = False
        population.device, population._backing = "cpu", object()
        population._healthy = SimpleNamespace(fill_=lambda value: self.fail("healthy service quarantined"))
        population._directory = SimpleNamespace(
            data=SimpleNamespace(live_count=SimpleNamespace(numpy=lambda: np.zeros(2)))
        )
        population._populations = []
        mapped = {}
        for prototype, initial in enumerate((0, 1)):
            group = _MuJoCoWorldPopulation(
                None,
                contact_quota=1,
                ccd_quota=1,
                contact_count=Mock(),
                ccd_count=Mock(),
                count_parameters=tuple(wp.CountParameter(4) for _ in range(3)),
            )
            for name in ("world_storage", "contact_storage", "ccd_scratch"):
                reservation = VirtualReservation(16 * (len(mapped) + 1), 2, 2)
                mapped[reservation] = initial
                owner = SimpleNamespace(
                    prototype=prototype,
                    name=name,
                    capacity=2,
                    ready_rows=0,
                    row_stride_bytes=1,
                    backing=population._backing,
                    reservation=reservation,
                    ready_count=object(),
                    service_failed=False,
                    arrays={"contact.efc_address": object()},
                )
                if name == "ccd_scratch":
                    group.transient_storages = (None, None, owner)
                else:
                    setattr(group, name, owner)
            population._populations.append(group)
        retained, calls = [3], []

        def resize(owner, target, *, protected_count_host):
            retained[0] += target - mapped[owner.reservation]
            self.assertLessEqual(retained[0], 3, "Growth ran before the donor's safe ranges were returned")
            calls.append((owner.prototype, owner.name, target))
            mapped[owner.reservation] = target
            owner.ready_rows = target

        with (
            patch.object(backing_ops, "maintenance", return_value=nullcontext()) as join,
            patch.object(
                backing_ops,
                "mapped_ranges",
                side_effect=lambda backing, reservation: ((0, mapped[reservation]),) if mapped[reservation] else (),
            ),
            patch.object(field_ops, "resize_backing", side_effect=resize),
            patch.object(field_ops, "zero"),
            patch.object(field_ops, "fill"),
            patch.object(directory_ops, "withdraw_admissible_slots") as withdraw,
            patch.object(directory_ops, "publish_admissible_slots") as publish,
            patch.object(native.wp, "get_stream", return_value=SimpleNamespace(cuda_stream=11)),
            patch.object(native.wp, "synchronize_stream"),
        ):
            mujoco_worlds_resize_backing(
                population, (1, 0), streams=(Mock(spec=wp.Stream, device="cpu", cuda_stream=11),)
            )
        self.assertEqual([entry[0] for entry in calls], [1, 1, 1, 0, 0, 0])
        self.assertEqual(retained, [3])
        join.assert_called_once_with(population._backing, streams=(11,))
        withdraw.assert_called_once_with(population._directory, (1, 0))
        publish.assert_called_once_with(population._directory, (1, 0))

    def test_partial_batch_budget_failure_publishes_only_jointly_backed_prefixes(self):
        """Partial batch budget failure publishes only jointly backed prefixes."""
        population = MuJoCoWorlds()
        population._closed = population._service_failed = False
        population.device = "cpu"
        population._backing = object()
        population._directory = object()
        health, publications, withdrawals = [1], [], []
        population._healthy = SimpleNamespace(fill_=lambda value, health=health: health.__setitem__(0, value))
        population._directory = SimpleNamespace(
            data=SimpleNamespace(live_count=SimpleNamespace(numpy=lambda: np.zeros(2, dtype=int))),
        )
        population._populations = []
        mapped = {}
        for p in range(2):
            group = _MuJoCoWorldPopulation(
                None,
                contact_quota=1,
                ccd_quota=1,
                contact_count=Mock(),
                ccd_count=Mock(),
                count_parameters=tuple(wp.CountParameter(4) for _ in range(3)),
            )
            for name in ("world_storage", "contact_storage", "ccd_scratch"):
                reservation = VirtualReservation(16 * (len(mapped) + 1), 4, 4)
                mapped[reservation] = ((0, 1),)
                owner = SimpleNamespace(
                    prototype=p,
                    kind=name,
                    ready_rows=1,
                    capacity=4,
                    service_failed=False,
                    arrays={"contact.efc_address": object()},
                    ready_count=object(),
                    row_stride_bytes=1,
                    backing=population._backing,
                    reservation=reservation,
                )

                if name == "ccd_scratch":
                    group.transient_storages = (None, None, owner)
                else:
                    setattr(group, name, owner)
            population._populations.append(group)

        def resize(owner, target, *, protected_count_host):
            if owner.prototype == 1 and owner.kind == "contact_storage":
                raise MemoryError("budget rejected before mutation")
            owner.ready_rows = target

        with (
            patch.object(backing_ops, "maintenance", return_value=nullcontext()),
            patch.object(backing_ops, "mapped_ranges", side_effect=lambda backing, reservation: mapped[reservation]),
            patch.object(field_ops, "resize_backing", side_effect=resize),
            patch.object(field_ops, "zero"),
            patch.object(field_ops, "fill"),
            patch.object(
                directory_ops, "withdraw_admissible_slots", side_effect=lambda owner, ends: withdrawals.append(ends)
            ),
            patch.object(
                directory_ops, "publish_admissible_slots", side_effect=lambda owner, ends: publications.append(ends)
            ),
            patch("newton._src.solvers.mujoco.worlds.wp.get_stream", return_value=SimpleNamespace(cuda_stream=0)),
            patch("newton._src.solvers.mujoco.worlds.wp.synchronize_stream"),
        ):
            with self.assertRaisesRegex(MemoryError, "budget"):
                mujoco_worlds_resize_backing(
                    population, (2, 2), streams=(Mock(spec=wp.Stream, device=population.device, cuda_stream=0),)
                )
        self.assertEqual(withdrawals, [(2, 2)])
        self.assertEqual(publications, [(2, 1)])
        self.assertEqual(population._populations[1].world_storage.ready_rows, 2)
        self.assertFalse(population._service_failed)
        self.assertEqual(health, [1])

    def test_optional_spare_trim_reuses_service_barrier_and_failure_quarantine(self):
        """Trim runs after ready publication with no second successful-path join."""
        for spare, fail in ((None, False), (0, False), (123, False), (123, True)):
            with self.subTest(spare=spare, fail=fail):
                population = MuJoCoWorlds()
                population._closed = population._service_failed = False
                population.device = "cpu"
                events, health = [], [1]

                @contextmanager
                def maintenance(backing, *, streams, events=events):
                    self.assertEqual(streams, (11,))
                    events.append("join")
                    yield
                    events.append("leave")

                def trim(backing, *, keep_bytes, spare=spare, fail=fail, events=events):
                    self.assertEqual(keep_bytes, spare)
                    events.append("trim")
                    if fail:
                        raise RuntimeError("release failed")

                population._backing = object()
                population._healthy = SimpleNamespace(fill_=lambda value, health=health: health.__setitem__(0, value))
                population._directory = SimpleNamespace(
                    data=SimpleNamespace(live_count=SimpleNamespace(numpy=lambda: np.ones(1, dtype=int))),
                )
                group = _MuJoCoWorldPopulation(
                    None,
                    contact_quota=1,
                    ccd_quota=1,
                    contact_count=Mock(),
                    ccd_count=Mock(),
                    count_parameters=tuple(wp.CountParameter(4) for _ in range(3)),
                )
                mapped = {}
                for name in ("world_storage", "contact_storage", "ccd_scratch"):
                    reservation = VirtualReservation(16 * (len(mapped) + 1), 4, 4)
                    mapped[reservation] = ((0, 1),)
                    owner = SimpleNamespace(
                        ready_rows=1,
                        capacity=4,
                        service_failed=False,
                        row_stride_bytes=1,
                        backing=population._backing,
                        reservation=reservation,
                    )
                    setattr(group, name, owner)
                population._populations = (group,)
                with (
                    patch.object(backing_ops, "maintenance", side_effect=maintenance),
                    patch.object(
                        backing_ops,
                        "mapped_ranges",
                        side_effect=lambda backing, reservation, mapped=mapped: mapped[reservation],
                    ),
                    patch.object(backing_ops, "trim", side_effect=trim),
                    patch.object(
                        field_ops,
                        "resize_backing",
                        side_effect=lambda *args, events=events, **kwargs: events.append("resize"),
                    ),
                    patch.object(
                        directory_ops,
                        "withdraw_admissible_slots",
                        side_effect=lambda *args, events=events: events.append("withdraw"),
                    ),
                    patch.object(
                        directory_ops,
                        "publish_admissible_slots",
                        side_effect=lambda *args, events=events: events.append("publish"),
                    ),
                    patch("newton._src.solvers.mujoco.worlds.wp.get_stream", return_value=object()),
                    patch(
                        "newton._src.solvers.mujoco.worlds.wp.synchronize_stream",
                        side_effect=lambda stream, events=events: events.append("sync"),
                    ),
                ):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "release failed"):
                            mujoco_worlds_resize_backing(
                                population,
                                (1,),
                                streams=(Mock(spec=wp.Stream, device=population.device, cuda_stream=11),),
                                spare_bytes=spare,
                            )
                    else:
                        mujoco_worlds_resize_backing(
                            population,
                            (1,),
                            streams=(Mock(spec=wp.Stream, device=population.device, cuda_stream=11),),
                            spare_bytes=spare,
                        )
                self.assertEqual(events.count("join"), 1)
                self.assertEqual(events.count("sync"), 2 if fail else 1)
                self.assertEqual(events.count("trim"), int(spare is not None))
                if spare is not None:
                    self.assertLess(events.index("publish"), events.index("trim"))
                self.assertEqual(population._service_failed, fail)
                self.assertEqual(health[0], int(not fail))

    def test_invalid_spare_limit_is_rejected_before_maintenance(self):
        """Bad optional reserve arguments cannot withdraw ready populations."""
        population = MuJoCoWorlds()
        population._closed = population._service_failed = False
        population.device = "cpu"
        population._backing = object()
        population._directory = object()
        for value in (-1, True, False, 1.0, "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                mujoco_worlds_resize_backing(
                    population,
                    (),
                    streams=(Mock(spec=wp.Stream, device=population.device, cuda_stream=11),),
                    spare_bytes=value,
                )

    def test_only_healthy_storage_budget_rejection_is_retryable(self):
        """Only healthy storage budget rejection is retryable."""
        for failure in ("budget", "storage_publication", "directory_publication", "scratch_fill"):
            with self.subTest(failure=failure):
                population = MuJoCoWorlds()
                population._closed = population._service_failed = False
                population.device = "cpu"
                population._backing = object()
                health, publications = [1], []
                population._healthy = SimpleNamespace(fill_=lambda value, health=health: health.__setitem__(0, value))

                def publish(directory, world_storage, publications=publications, failure=failure):
                    publications.append(world_storage)
                    if failure == "directory_publication":
                        raise MemoryError("directory publication")

                population._directory = SimpleNamespace(
                    data=SimpleNamespace(live_count=SimpleNamespace(numpy=lambda: np.zeros(1, dtype=int))),
                )
                group = _MuJoCoWorldPopulation(
                    None,
                    contact_quota=2,
                    ccd_quota=1,
                    data_defaults={"contact.efc_address": -1},
                    contact_count=Mock(),
                    ccd_count=Mock(),
                    count_parameters=tuple(wp.CountParameter(n) for n in (4, 8, 4)),
                )
                mapped = {}
                for name, ready, capacity in (
                    ("world_storage", 1, 4),
                    ("contact_storage", 2, 8),
                    ("ccd_scratch", 1, 4),
                ):
                    reservation = VirtualReservation(16 * (len(mapped) + 1), capacity, capacity)
                    mapped[reservation] = ((0, ready),)
                    owner = SimpleNamespace(
                        kind=name,
                        ready_rows=ready,
                        capacity=capacity,
                        service_failed=False,
                        arrays={"contact.efc_address": object()},
                        ready_count=object(),
                        row_stride_bytes=1,
                        backing=population._backing,
                        reservation=reservation,
                    )

                    if name == "ccd_scratch":
                        group.transient_storages = (None, None, owner)
                    else:
                        setattr(group, name, owner)
                population._populations = [group]

                def resize(owner, target, *, protected_count_host, failure=failure):
                    if owner.kind == "world_storage" and failure in ("budget", "storage_publication"):
                        owner.service_failed = failure == "storage_publication"
                        raise MemoryError(failure)
                    owner.ready_rows = target

                def fill(*args, failure=failure, **kwargs):
                    if failure == "scratch_fill":
                        raise MemoryError("scratch initialization")

                with (
                    patch.object(backing_ops, "maintenance", return_value=nullcontext()),
                    patch.object(
                        backing_ops,
                        "mapped_ranges",
                        side_effect=lambda backing, reservation, mapped=mapped: mapped[reservation],
                    ),
                    patch.object(field_ops, "resize_backing", side_effect=resize),
                    patch.object(field_ops, "fill", side_effect=fill),
                    patch.object(field_ops, "zero", side_effect=fill),
                    patch.object(directory_ops, "withdraw_admissible_slots"),
                    patch.object(directory_ops, "publish_admissible_slots", side_effect=publish),
                    patch(
                        "newton._src.solvers.mujoco.worlds.wp.get_stream", return_value=SimpleNamespace(cuda_stream=0)
                    ),
                    patch("newton._src.solvers.mujoco.worlds.wp.synchronize_stream"),
                ):
                    with self.assertRaises(MemoryError):
                        mujoco_worlds_resize_backing(
                            population, (2,), streams=(Mock(spec=wp.Stream, device=population.device, cuda_stream=0),)
                        )
                self.assertEqual(population._service_failed, failure != "budget")
                self.assertEqual(health[0], int(failure == "budget"))
                if failure == "budget":
                    self.assertEqual(publications, [(1,)])
                elif failure != "directory_publication":
                    self.assertEqual(publications, [])

    def test_backing_quarantine_preserves_original_error_and_interruption(self):
        """A failed GPU health publication cannot erase its service failure or reopen the owner."""
        for stage in ("service", "storage_publication", "budget_republication"):
            for interrupted in ("fill", "synchronize"):
                with self.subTest(stage=stage, interrupted=interrupted):
                    population = MuJoCoWorlds()
                    population.device = wp.get_device("cpu")
                    population._closed = population._service_failed = False
                    population._backing = object()
                    original = RuntimeError("service failed") if stage == "service" else MemoryError("storage failed")
                    publication = RuntimeError("readiness publication failed")
                    interruption = KeyboardInterrupt("quarantine interrupted")
                    population._healthy = SimpleNamespace(
                        fill_=Mock(side_effect=interruption if interrupted == "fill" else None)
                    )
                    population._directory = SimpleNamespace(
                        data=SimpleNamespace(live_count=SimpleNamespace(numpy=lambda: np.zeros(1, dtype=int))),
                        batch_result=object(),
                    )
                    population.directory = population._directory.data
                    population.batch_result = population._directory.batch_result
                    owner = SimpleNamespace(
                        ready_rows=1,
                        capacity=2,
                        service_failed=stage == "storage_publication",
                        row_stride_bytes=1,
                        backing=population._backing,
                        reservation=VirtualReservation(16, 2, 2),
                    )
                    population._populations = [
                        _MuJoCoWorldPopulation(
                            None,
                            world_storage=owner,
                            contact_storage=owner,
                            transient_storages=(None, None, owner),
                            contact_count=Mock(),
                            ccd_count=Mock(),
                            count_parameters=tuple(wp.CountParameter(2) for _ in range(3)),
                            contact_quota=1,
                            ccd_quota=1,
                        )
                    ]
                    with (
                        patch.object(backing_ops, "maintenance", return_value=nullcontext()),
                        patch.object(backing_ops, "mapped_ranges", return_value=((0, 1),)),
                        patch.object(field_ops, "resize_backing", side_effect=original),
                        patch.object(directory_ops, "withdraw_admissible_slots"),
                        patch.object(directory_ops, "publish_admissible_slots", side_effect=publication),
                        patch.object(native.wp, "get_stream", return_value=SimpleNamespace(cuda_stream=0)),
                        patch.object(
                            native.wp,
                            "synchronize_stream",
                            side_effect=interruption if interrupted == "synchronize" else None,
                        ),
                        self.assertRaises(BaseExceptionGroup) as raised,
                    ):
                        mujoco_worlds_resize_backing(
                            population, (2,), streams=(Mock(spec=wp.Stream, device=population.device, cuda_stream=0),)
                        )
                    expected = publication if stage == "budget_republication" else original
                    self.assertEqual(raised.exception.exceptions, (expected, interruption))
                    if stage == "budget_republication":
                        self.assertIs(publication.__context__, original)
                    self.assertTrue(population._service_failed)
                    population._healthy.fill_.assert_called_once_with(0)
                    self.assertIs(population.directory, population._directory.data)
                    self.assertIs(population.batch_result, population._directory.batch_result)


class PopulationRecorderTests(unittest.TestCase):
    def setUp(self):
        """Mock emitted Warp records while retaining native count declaration validation."""
        self.stream = Mock(is_capturing=True)
        self.records = []
        self.memory_operations = []

        class Capture:
            pass

        self.graph = Capture()
        self.graph.device = wp.get_device("cpu")
        self.graph.graph_exec = None
        capture = patch.object(graph_ops, "current_capture", return_value=self.graph)
        capture.start()
        self.addCleanup(capture.stop)

        def emit(kernel, dim, *args, **kwargs):
            dimensions = (dim,) if isinstance(dim, int) else dim
            if not all(dimensions):
                return None
            self.records.append(SimpleNamespace(kernel=kernel, dim=dim))

        patcher = patch.object(native.wp, "launch", side_effect=emit)
        self.launch = patcher.start()
        self.addCleanup(patcher.stop)
        tiled = patch.object(native.wp, "launch_tiled", side_effect=emit)
        self.launch_tiled = tiled.start()
        self.addCleanup(tiled.stop)
        count = patch.object(native.wp, "capture_launch_count", side_effect=lambda graph: len(self.records))
        count.start()
        self.addCleanup(count.stop)
        transient = patch.object(native.wp, "capture_transient", side_effect=lambda callback, **kwargs: callback())
        transient.start()
        self.addCleanup(transient.stop)

    def group(self):

        self.native, self.wp = native, wp
        calls = []

        def owner(name):
            return FieldStorage(
                protected_count=wp.zeros(1, dtype=int, device="cpu"),
                ready_count=wp.zeros(1, dtype=int, device="cpu"),
                capacity=17,
                ready_rows=17,
                device=wp.get_device("cpu"),
                backing=None,
                row_stride_bytes=0,
                specifications={},
                arrays={},
                fields={},
                _by_array={},
                _patterns={},
                _graphs=weakref.WeakSet(),
                _transfers=weakref.WeakSet(),
            )

        group = _MuJoCoWorldPopulation(
            model=object(),
            contact_quota=2,
            ccd_quota=4,
            data=SimpleNamespace(qpos=SimpleNamespace(device="cpu"), nacon=None, ncollision=None),
            world_storage=owner("world"),
            contact_storage=owner("candidate"),
            transient_storages=(None, None, owner("ccd")),
            count_parameters=tuple(wp.CountParameter(17) for _ in range(3)),
            application_ranges=[],
        )
        group.contact_count = group.contact_storage.protected_count
        group.ccd_count = group.transient_storages[2].protected_count
        group.execution_data = SimpleNamespace(qpos=group.data.qpos)
        group.view = _population_view(group, 3)
        updates = GraphUpdateTable(
            enable_count=group.world_storage.protected_count,
            enable_count_maximum=17,
            device=wp.get_device("cpu"),
            binding_capacity=16,
            binding_stride_bytes=0,
            bindings=None,
            binding_count=None,
            errors=None,
            _library=None,
            _captured_nodes={},
        )
        group.updates = updates
        kernel = SimpleNamespace(
            key="named_counts",
            func=SimpleNamespace(__module__="test", __qualname__="named_counts"),
            adj=SimpleNamespace(
                kernel_dim=2,
                args=[
                    SimpleNamespace(label="unrelated", type=wp.int32),
                    SimpleNamespace(label="world_live_count", type=wp.int32),
                    SimpleNamespace(label="contact_cap", type=wp.int32),
                    SimpleNamespace(label="ccd_cap", type=wp.int32),
                ],
            ),
        )
        return group, kernel, calls

    def test_public_view_borrows_named_counts_without_exposing_runtime_owners(self):
        """Expose stable numeric prototype identity and borrowed fields without retirement authority."""
        group, _, _ = self.group()
        view = group.view
        self.assertIsInstance(view, MuJoCoWorldPopulation)
        self.assertIs(newton.solvers.MuJoCoWorldPopulation, MuJoCoWorldPopulation)
        self.assertNotIn("_MuJoCoWorldPopulation", newton.solvers.__all__)
        self.assertFalse(hasattr(newton.solvers, "MuJoCoWorldPrototype"))
        self.assertEqual(view.prototype_index, 3)
        self.assertIs(view.model, group.model)
        self.assertIs(view.data, group.data)
        self.assertIs(view.world_live_count, group.world_storage.protected_count)
        self.assertIs(view.world_storage_ready_count, group.world_storage.ready_count)
        self.assertIs(view.contact_storage_ready_count, group.contact_count)
        self.assertIs(view.ccd_storage_ready_count, group.ccd_count)
        self.assertEqual((view.world_capacity, view.contact_capacity, view.ccd_capacity), (17, 17, 17))
        self.assertEqual(mujoco_world_population_ready_capacity(view), 4)
        group.transient_storages[2].ready_rows = 8
        self.assertEqual(mujoco_world_population_ready_capacity(view), 2, "Readiness must come from its existing owner")
        for name in (
            "world_storage",
            "contact_storage",
            "ccd_storage",
            "workspace",
            "updates",
            "initialization_transfer",
            "compaction_transfer",
            "close",
            "resize_backing",
            "grow_backing",
            "bind_launch",
        ):
            self.assertFalse(hasattr(view, name), name)
        population = MuJoCoWorlds()
        population._closed = population._service_failed = False
        population.populations = (view,)
        self.assertIs(population.populations, population.populations)
        group.data = None
        with self.assertRaisesRegex(RuntimeError, "closed"):
            mujoco_world_population_validate(view)

    def test_absent_ccd_scratch_still_freezes_the_published_scalar_descriptor(self):
        """Reject same-object CCD count mutations even when no CCD array owns the scalar."""
        group, _, _ = self.group()
        group.transient_storages = (None, None, None)
        group.storage_signature = native._binding_signature(group)[0:2]
        population = MuJoCoWorlds(
            device=wp.get_device("cpu"),
            _directory=object(),
            _backing=object(),
            _populations=[group],
            populations=(group.view,),
        )
        array = group.ccd_count
        for attribute, changed in (
            ("ptr", array.ptr + 4),
            ("shape", (2,)),
            ("strides", (8,)),
            ("dtype", wp.uint32),
            ("device", object()),
        ):
            previous = getattr(array, attribute)
            try:
                setattr(array, attribute, changed)
                with self.assertRaisesRegex(ValueError, "count descriptors changed"):
                    mujoco_world_population_validate(group.view)
                for public_views in ((group.view,), (), (object(),)):
                    population.populations = public_views
                    for service in (mujoco_worlds_grow_backing, mujoco_worlds_resize_backing):
                        with self.subTest(attribute=attribute, views=public_views, service=service.__name__):
                            with self.assertRaisesRegex(ValueError, "count descriptors changed"):
                                service(population, (1,), streams=())
            finally:
                setattr(array, attribute, previous)
        mujoco_world_population_validate(group.view)

    def test_public_count_validation_does_not_visit_private_scratch(self):
        """Borrowed-count reads stay independent of the private scratch relation checked by service."""
        group, _, _ = self.group()
        group.storage_signature = native._binding_signature(group)[0:2]
        population = MuJoCoWorlds(device=wp.get_device("cpu"), _directory=object(), _populations=[group])
        with patch.object(native, "_binding_signature", side_effect=AssertionError("private storage inspected")):
            mujoco_world_population_validate(group.view)
            for service in (mujoco_worlds_grow_backing, mujoco_worlds_resize_backing):
                with self.subTest(service=service.__name__):
                    with self.assertRaisesRegex(AssertionError, "private storage inspected"):
                        service(population, (1,), streams=())

    def test_borrowed_view_mutations_are_rejected_at_validation(self):
        """Plain records cannot turn replaced counts or prototype metadata into valid submission inputs."""
        group, _, _ = self.group()
        view = group.view
        for name, value in (
            ("prototype_index", 4),
            ("prototype_index", True),
            ("model", object()),
            ("data", object()),
            ("world_capacity", 18),
            ("contact_capacity", 18),
            ("ccd_capacity", 18),
            ("world_live_count", wp.zeros(1, dtype=int, device="cpu")),
            ("world_storage_ready_count", group.world_storage.protected_count),
            ("contact_storage_ready_count", group.world_storage.ready_count),
            ("ccd_storage_ready_count", group.world_storage.ready_count),
        ):
            original = getattr(view, name)
            with self.subTest(field=name):
                setattr(view, name, value)
                with self.assertRaises(ValueError):
                    mujoco_world_population_validate(view)
                setattr(view, name, original)
                mujoco_world_population_validate(view)
        group.storage_signature = native._binding_signature(group)[0:2]
        count = view.world_live_count
        original_shape = count.shape
        try:
            count.shape = (1, 1)
            with self.assertRaisesRegex(ValueError, "count descriptors changed"):
                mujoco_world_population_validate(view)
        finally:
            count.shape = original_shape
        mujoco_world_population_validate(view)

    def _capture(self, group, *, publication=None, extra_groups=(), validate_population=None, **callbacks):
        """Run the real composition root with CPU records and mocked native graph mechanics."""
        import mujoco_warp as mjw

        population = MuJoCoWorlds()
        population.device = wp.get_device("cpu")
        population._closed = population._service_failed = population._capture_attempted = False
        groups = [group, *extra_groups]
        population._directory = directory_ops.allocate(
            (17,) * len(groups), id_capacity=17, command_capacity=17, device="cpu"
        )
        population._healthy = population._always_permit = population._lifecycle_needed = wp.ones(
            1, dtype=int, device="cpu"
        )
        population._populations = groups
        self.population = population
        commands = directory_ops.allocate_commands(17, device="cpu")
        results = directory_ops.allocate_results(17, device="cpu")
        self.records.clear()
        self.native_kernel = object()
        self.native_records = []
        self.adopted = []
        self.invalidations = []
        self.contact_invalidations = []
        condition_stack = []

        def native_bind(owner, graph, records):
            self.assertIn(owner, groups)
            self.native_records.extend(records)

        def application_bind(updates, graph, records, **relations):
            self.adopted.append((records, relations))
            return ()

        def invalidate(graph):
            if graph.graph_exec is not None:
                raise RuntimeError("Cannot invalidate an instantiated graph")
            self.invalidations.append(graph)
            graph._preparation_failed = True

        def capture_if(condition, on_true):
            if on_true.__name__ == "record_lifecycle":
                if callbacks.get("validate") is not None:
                    callbacks["validate"](
                        commands, population._directory.transaction.status, population._directory.batch_result.consumed
                    )
                if callbacks.get("initialize") is not None:
                    callbacks["initialize"](group.view, None, None, None, None, None, None)
            else:
                condition_stack.append(condition)
                try:
                    on_true()
                finally:
                    condition_stack.pop()

        @contextmanager
        def scoped_capture(**kwargs):
            self.records.clear()
            self.memory_operations.clear()
            self.contact_invalidations.clear()
            yield SimpleNamespace(graph=self.graph)

        with ExitStack() as stack:
            for target, name, options in (
                (native.wp, "load_module", {}),
                (native.wp, "get_stream", {"return_value": self.stream}),
                (native.wp, "synchronize_stream", {}),
                (native.wp, "capture_if", {"side_effect": capture_if}),
                (native.wp, "capture_get_launches", {"side_effect": lambda graph: tuple(self.records)}),
                (native.wp, "capture_discovery_memory", {"return_value": {}}),
                (
                    native.wp,
                    "capture_get_memory_operations",
                    {"side_effect": lambda graph: tuple(self.memory_operations)},
                ),
                (graph_ops, "prepare_updates", {"side_effect": [owner.updates for owner in groups]}),
                (graph_ops, "capture_parallel", {"side_effect": lambda branches: [branch() for branch in branches]}),
                (graph_ops, "record_update", {}),
                (graph_ops, "adopt_launches", {"side_effect": application_bind}),
                (graph_ops, "bind", {}),
                (graph_ops, "instantiate_and_upload", {"side_effect": publication}),
                (graph_ops, "invalidate", {"side_effect": invalidate}),
                (directory_ops, "begin", {}),
                (directory_ops, "publish_admissible_slots", {}),
                (mjw, "step", {"side_effect": lambda *args, **kwargs: wp.launch(self.native_kernel, 17)}),
                (mjw, "kinematics", {}),
                (native, "_validate_population", {"side_effect": validate_population}),
                (native, "_retain_graph", {}),
                (native, "_prepare_transient_storage", {"return_value": ()}),
                (
                    mjw,
                    "invalidate_contact_cache",
                    {"side_effect": lambda data: self.contact_invalidations.append((data, condition_stack[-1]))},
                ),
                (native, "_bind_native_program", {"side_effect": native_bind}),
            ):
                stack.enter_context(patch.object(target, name, **options))
            stack.enter_context(patch.object(native.wp, "ScopedCapture", side_effect=scoped_capture))
            return mujoco_worlds_capture(population, commands, results, **callbacks)

    def test_consumed_lifecycle_invalidates_contacts_independently_of_physics_permit(self):
        """Replaced placement invalidates contact indices even for reset-only and failed batches."""
        group, _, _ = self.group()
        self._capture(group, permit=wp.zeros(1, dtype=int, device="cpu"))
        self.assertEqual(len(self.contact_invalidations), 1)
        data, condition = self.contact_invalidations[0]
        self.assertIs(data, group.data)
        self.assertIs(condition, self.population._directory.batch_result.consumed)

    def test_later_application_binding_cannot_mutate_an_already_bound_population(self):
        """Revalidate every population after the final callback compiler and before graph publication."""
        first, kernel, _ = self.group()
        second, _, _ = self.group()
        compiled = []
        publication = Mock()

        def compile_application(view, records):
            compiled.append(view)
            if view is second.view:
                first.view.world_live_count = second.view.world_live_count
            return (), (), tuple(range(len(records)))

        with self.assertRaisesRegex(ValueError, "count source changed"):
            self._capture(
                first,
                extra_groups=(second,),
                before_step=lambda view: wp.launch(kernel, 17),
                application_bindings=compile_application,
                validate_population=lambda owner: mujoco_world_population_validate(owner.view),
                publication=publication,
            )
        self.assertEqual(compiled, [first.view, second.view])
        publication.assert_not_called()
        self.assertEqual(self.invalidations, [self.graph])
        self.assertTrue(self.population._service_failed)

    def test_publication_failure_preserves_created_execution_ownership_and_quarantines(self):
        """Preserve publication errors and quarantine without invalidating a retained executable."""
        for created in (False, True):
            with self.subTest(created=created):
                group, _, _ = self.group()
                failure = RuntimeError("completion failed" if created else "instantiation failed")
                executable = object() if created else None

                def publication(updates, graph, executable=executable, failure=failure):
                    graph.graph_exec = executable
                    graph._preparation_failed = True
                    raise failure

                with self.assertRaises(RuntimeError) as caught:
                    self._capture(group, publication=publication)
                self.assertIs(caught.exception, failure)
                self.assertIs(self.graph.graph_exec, executable)
                self.assertEqual(self.invalidations, [] if created else [self.graph])
                np.testing.assert_array_equal(self.population._healthy.numpy(), [0])
                self.assertTrue(self.population._service_failed)
                with self.assertRaisesRegex(RuntimeError, "closed or its backing service failed"):
                    mujoco_worlds_validate(self.population)
                self.assertIsNone(group.before_step)
                self.assertIsNone(group.after_substep)

    def test_application_compiler_receives_only_exact_callback_records(self):
        """The same kernel can bind two call sites to different numeric populations."""
        group, kernel, _ = self.group()

        def callback(population):
            wp.launch(kernel, 17)
            wp.launch(kernel, 17)

        def compile_application(population, records):
            self.assertIs(population, group.view)
            self.assertEqual(len(records), 2)
            self.assertTrue(all(record.kernel is kernel for record in records))
            return ((0, 0, population.world_live_count), (1, 0, population.contact_storage_ready_count)), (), ()

        self._capture(group, before_step=callback, application_bindings=compile_application)
        records, relations = self.adopted[0]
        self.assertIs(relations["extents"][0][2], group.world_storage.protected_count)
        self.assertIs(relations["extents"][1][2], group.contact_count)
        self.assertTrue(all(record.kernel is self.native_kernel for record in self.native_records))
        self.assertTrue(all(any(record is candidate for candidate in self.records) for record in records))
        self.assertIsNone(group.application_ranges)

    def test_known_native_kernel_in_callback_still_requires_application_compiler(self):
        """A physics catalog identity does not authorize application-owned records."""
        from mujoco_warp._src import sleep

        group, _, _ = self.group()
        with self.assertRaisesRegex(ValueError, "require application_bindings"):
            self._capture(group, before_step=lambda population: wp.launch(sleep._wake_collision_kernel, 17))
        self.assertEqual(self.adopted, [])
        self.assertIsNone(group.application_ranges)

    def test_application_compiler_rejects_wrong_kernel_identity(self):
        """Application count declarations are supplied after checking exact code identity."""
        group, kernel, _ = self.group()

        def compile_application(population, records):
            if any(record.kernel is not kernel for record in records):
                raise ValueError("Unexpected application kernel")
            return (), (), tuple(range(len(records)))

        with self.assertRaisesRegex(ValueError, "Unexpected application kernel"):
            self._capture(
                group, before_step=lambda population: wp.launch(object(), 17), application_bindings=compile_application
            )
        self.assertEqual(self.adopted, [])
        self.assertIsNone(group.application_ranges)

    def test_application_memory_operations_require_storage_admission(self):
        """Count declarations alone cannot authorize full-capacity fills into unbacked storage."""
        group, kernel, _ = self.group()

        def callback(population):
            wp.launch(kernel, 17)
            self.memory_operations.append(SimpleNamespace(launch=self.records[-1]))

        compiler = Mock(return_value=((), (), (0,)))
        with self.assertRaisesRegex(NotImplementedError, "storage admission"):
            self._capture(group, before_step=callback, application_bindings=compiler)
        compiler.assert_not_called()
        self.assertEqual(self.adopted, [])

    def test_lifecycle_memory_operations_require_storage_admission(self):
        """Initializer/validator arrays do not inherit native population storage permissions."""
        for phase in ("initialize", "validate"):
            with self.subTest(phase=phase):
                group, kernel, _ = self.group()
                self.memory_operations.clear()

                def callback(*args, kernel=kernel):
                    wp.launch(kernel, 17)
                    self.memory_operations.append(SimpleNamespace(launch=self.records[-1]))

                with self.assertRaisesRegex(NotImplementedError, "Lifecycle fills/copies"):
                    self._capture(group, **{phase: callback})
                self.assertEqual(self.adopted, [])

    def test_plain_callback_launches_keep_exact_intervals_and_numeric_sources(self):
        """Repeated kernels and equal capacities do not merge application ownership."""
        import mujoco_warp as mjw

        first, kernel, _ = self.group()
        second, _, _ = self.group()
        for group in (first, second):
            group.before_step = lambda view: wp.launch(kernel, (0, 17))
            group.after_substep = lambda view: wp.launch_tiled(kernel, (17, 17), block_dim=32)
            group.substeps = 2
            with (
                patch.object(mjw, "step", side_effect=lambda *a, **k: wp.launch(object(), 17)),
                patch.object(mjw, "kinematics"),
                patch.object(native, "_validate_population"),
                patch.object(native.wp, "capture_if", side_effect=lambda condition, on_true: on_true()),
            ):
                native._record_physics(group, [])
        self.assertEqual(first.application_ranges, [(1, 2), (3, 4)])
        self.assertEqual(second.application_ranges, [(5, 6), (7, 8)])
        for group in (first, second):
            self.assertEqual(group.bindings, [])
            self.assertIsNotNone(group.before_step)
            self.assertIsNotNone(group.after_substep)
        self.assertTrue(all(self.records[index].kernel is kernel for index in (1, 3, 5, 7)))

    def test_physics_and_pose_refresh_share_one_parent_child_scope(self):
        """Contain nested physics conditionals so pose recording cannot depend on earlier siblings."""
        group, _, calls = self.group()
        depth, condition_depths = [0], []

        def conditional(step_condition, on_true):
            condition_depths.append(depth[0])
            depth[0] += 1
            try:
                on_true()
            finally:
                depth[0] -= 1

        def before(view):
            self.assertIs(view, group.view)
            self.assertEqual(depth[0], 2)
            calls.append("control")

        def poses(model, data):
            self.assertIs(model, group.model)
            self.assertIs(data, group.execution_data)
            self.assertEqual(depth[0], 1)
            calls.append("poses")

        def step(model, data):
            self.assertIs(model, group.model)
            self.assertIs(data, group.execution_data)
            calls.append("step")

        def validate(owner):
            self.assertIs(owner, group)
            self.assertIn(depth[0], (0, 2))
            calls.append("validate")

        group.before_step = before
        group.after_substep = lambda view: calls.append("after")
        with (
            patch.dict(
                sys.modules,
                {"mujoco_warp": SimpleNamespace(step=step, kinematics=poses)},
            ),
            patch.object(native, "_validate_population", side_effect=validate),
            patch.object(native.wp, "capture_if", side_effect=conditional),
        ):
            native._record_physics(group, [])
        self.assertEqual(condition_depths, [0, 1])
        self.assertEqual(calls, ["validate", "control", "validate", "step", "after", "validate", "poses"])

    def test_final_callback_validation_failure_rejects_before_pose_recording(self):
        """Reject changed prepared descriptors after the last callback, with or without pose refresh."""
        for refresh in (True, False):
            with self.subTest(refresh=refresh):
                group, _, calls = self.group()
                group.refresh_kinematics = refresh
                group.after_substep = lambda view, calls=calls: calls.append("after")
                validate = Mock(side_effect=[None, None, ValueError("prepared descriptors changed")])
                poses = Mock()
                with (
                    patch.dict(
                        sys.modules,
                        {"mujoco_warp": SimpleNamespace(step=Mock(), kinematics=poses)},
                    ),
                    patch.object(native, "_validate_population", validate),
                    patch.object(native.wp, "capture_if", side_effect=lambda condition, on_true: on_true()),
                ):
                    with self.assertRaisesRegex(ValueError, "prepared descriptors changed"):
                        native._record_physics(group, [])
                self.assertEqual(calls, ["after"])
                self.assertEqual(validate.call_count, 3)
                validate.assert_called_with(group)
                poses.assert_not_called()
                self.assertTrue(group.recording_failed)

    def test_unhandled_callback_error_rejects_recording_and_releases_callbacks(self):
        """A callback failure cannot leave a retryable or partially usable population program."""
        import mujoco_warp as mjw

        group, kernel, _ = self.group()

        def callback(view):
            wp.launch(kernel, 17)
            raise ValueError("callback failed")

        group.before_step = callback
        with (
            patch.object(mjw, "step"),
            patch.object(mjw, "kinematics"),
            patch.object(native, "_validate_population"),
            patch.object(native.wp, "capture_if", side_effect=lambda condition, on_true: on_true()),
            self.assertRaisesRegex(ValueError, "callback failed"),
        ):
            native._record_physics(group, [])
        self.assertTrue(group.recording_failed)
        self.assertIsNone(group.before_step)
        with self.assertRaisesRegex(RuntimeError, "failed recording"):
            native._record_physics(group, [])

    def test_model_array_walk_includes_tuple_arrays_and_nested_tile_descriptors(self):
        """Account tuple-held model arrays without interpreting their shapes as world domains."""
        import mujoco_warp as mjw

        @dataclass
        class Tile:
            elements: object

        @dataclass
        class Model:
            body_tree: tuple
            M_tiles: tuple

        first, second = wp.zeros(3, dtype=int, device="cpu"), wp.zeros(5, dtype=float, device="cpu")
        model = Model((first,), (Tile(second),))
        fields = list(mjw.array_fields(model))
        self.assertEqual([name for name, _, _ in fields], ["body_tree[0]", "M_tiles[0].elements"])
        self.assertIs(fields[0][1], first)
        self.assertIs(fields[1][1], second)

        scalar = wp.zeros(1, dtype=int, device="cpu")
        storage = SimpleNamespace(protected_count=scalar, field_names=(), ready_rows=0, capacity=1)
        group = SimpleNamespace(
            model=model,
            world_storage=storage,
            transient_storages=(None, None, storage),
            transient_arrays=(),
            contact_count=scalar,
            ccd_count=scalar,
            contact_storage=storage,
            default_storage=storage,
            initialization_transfer=storage,
            compaction_transfer=storage,
            updates=None,
            global_arrays={},
            empty_fields=[],
            contact_quota=1,
            ccd_quota=1,
            **dict.fromkeys(
                (
                    "request_indices",
                    "source_rows",
                    "destination_rows",
                    "initialization_count",
                    "move_count",
                    "step_condition",
                    "kinematics_condition",
                ),
                scalar,
            ),
        )
        population = MuJoCoWorlds()
        population._populations = [group, group]
        population._directory, population._backing = storage, None
        population._healthy = population._always_permit = population._lifecycle_needed = scalar
        with (
            patch.object(directory_ops, "memory_report", return_value={}),
            patch.object(field_ops, "memory_report", return_value={}),
            patch.object(field_ops, "transfer_memory_report", return_value={}),
        ):
            report = mujoco_worlds_memory_report(population)
        self.assertFalse({"groups", "health_metadata_bytes"} & report.keys())
        self.assertFalse(
            {"relocation", "relocated_fields", "absent_fields", "index_bytes", "globals", "global_bytes"}
            & report["populations"][0].keys()
        )
        self.assertEqual(report["populations"][0]["immutable_model_array_bytes"], first.capacity + second.capacity)
        self.assertEqual(
            report["populations"][1]["immutable_model_array_bytes"],
            0,
            "Shared immutable model allocations must be counted once across populations",
        )

    def test_recording_scope_clears_recorder_on_failure_and_keeps_controls_inside_if(self):
        """Verify recording scope clears recorder on failure and keeps controls inside if."""
        group, _, calls = self.group()
        group.substeps = 2
        group.before_step = lambda group: calls.append("control")

        def step(*args, **kwargs):
            calls.append("step")

        with (
            patch.object(native, "_validate_population"),
            patch.dict(
                sys.modules,
                {
                    "mujoco_warp": SimpleNamespace(
                        step=step,
                        kinematics=lambda *args, **kwargs: calls.append("poses"),
                    )
                },
            ),
        ):
            with patch.object(self.native.wp, "capture_if", side_effect=lambda step_condition, on_true: on_true()):
                native._record_physics(group, [])
            self.assertEqual(calls, ["control", "step", "step", "poses"])
            self.assertIsNotNone(group.before_step)
            calls.clear()
            group.before_step = lambda group: calls.append("control")
            with patch.object(self.native.wp, "capture_if", return_value=None):
                native._record_physics(group, [])
            self.assertEqual(calls, [], "The disabled native IF must also suppress control writes")
            with patch.object(self.native.wp, "capture_if", side_effect=RuntimeError("capture failed")):
                with self.assertRaisesRegex(RuntimeError, "capture failed"):
                    native._record_physics(group, [])
            self.assertIsNone(group.before_step)


@wp.kernel
def _inject_graph_update_error(errors: wp.array[int], enabled: wp.array[int]):
    if enabled[0] != 0:
        errors[0] = -777


@wp.kernel
def _validate_payload(
    commands: InstanceCommands, request_status: wp.array[int], consumed: wp.array[int], valid: wp.array[int]
):
    if consumed[0] == 0:
        return
    request = wp.tid()
    if request < commands.count[0] and request_status[request] == _OK:
        if (commands.operation[request] == _CREATE or commands.operation[request] == _REPLACE) and valid[request] == 0:
            request_status[request] = _INVALID


@wp.kernel
def _initialize_payload(
    requests: wp.array[int],
    destinations: wp.array[int],
    count: wp.array[int],
    status: wp.array[int],
    initialized_sequence: wp.array[wp.uint64],
    sequence: wp.array[wp.uint64],
    offset: wp.array[float],
    qpos: wp.array2d[float],
    mocap: wp.array2d[wp.vec3],
):
    ordinal = wp.tid()
    if ordinal < count[0] and status[0] == 0:
        request, row = requests[ordinal], destinations[ordinal]
        qpos[row, 0] += offset[request]
        mocap[row, 0] += wp.vec3(offset[request], 0.0, 0.0)
        initialized_sequence[request] = sequence[0]


@wp.kernel
def _record_convex_work(nccd: wp.array[int], nacon: wp.array[int], observed: wp.array[int]):
    for pair in range(nccd.shape[0]):
        if nccd[pair] > 0:
            observed[0] = 1
    if nacon[0] > 0:
        observed[1] = 1


@wp.kernel
def _record_control(calls: wp.array[int]):
    calls[0] += 1


@wp.kernel
def _record_substep(overflow: wp.array[int], sticky: wp.array[int], calls: wp.array[int]):
    world = wp.tid()
    wp.atomic_or(sticky, 0, overflow[world])
    if world == 0:
        calls[1] += 1


def _prefix(source, count):
    """Build independent dense-oracle views using declared native world axes."""
    changes = {}
    for field in fields(source):
        array = getattr(source, field.name)
        if isinstance(array, wp.array):
            axes = getattr(field.type, "shape", ())
            if axes and axes[0] == "nworld" and array.shape[0] != 0:
                changes[field.name] = wp.array(
                    ptr=array.ptr,
                    dtype=array.dtype,
                    device=array.device,
                    shape=(count, *array.shape[1:]),
                    strides=array.strides,
                )
        elif is_dataclass(array):
            changes[field.name] = _prefix(array, count)
    if hasattr(source, "nworld"):
        changes["nworld"] = count
    return replace(source, **changes)


def _world_fields(source, prefix=""):
    """Enumerate oracle fields from the engine schema without importing runtime layout code."""
    for field in fields(source):
        value = getattr(source, field.name)
        name = prefix + field.name
        if isinstance(value, wp.array):
            axes = getattr(field.type, "shape", ())
            if axes and axes[0] == "nworld" and value.size:
                yield name, value
        elif is_dataclass(value):
            yield from _world_fields(value, name + ".")


def _array(data, path):
    """Resolve one oracle-owned native field."""
    for name in path.split("."):
        data = getattr(data, name)
    return data


def _poison_transients(population):
    """Poison every backed scratch domain and fixed slot before testing numerical first writes."""

    def value(array):
        scalar = getattr(array.dtype, "_wp_scalar_type_", array.dtype)
        fill = float("nan") if scalar in (wp.float32, wp.float64) else 127
        return array.dtype(fill) if scalar is not array.dtype else fill

    for group in population._populations:
        for storage in group.transient_storages:
            if storage is not None:
                for array in storage.arrays.values():
                    field_ops.fill(storage, array, value(array), count=storage.ready_count)
        for array in group.transient_arrays:
            if array.size:
                array.fill_(value(array))


def _prototype(keys, mujoco, mjw, *, convex=False):
    """Prepare slider keys with primitive or convex floor contacts, mocap and sites."""
    key_geom = (
        '<geom type="box" size="0.009 0.009 0.009" mass="0.01"/>'
        if convex
        else '<geom type="sphere" size="0.009" mass="0.01"/>'
    )
    floor_geom = '<geom type="box" size="2 2 0.1" pos="0 0 -0.1"/>' if convex else '<geom type="plane" size="2 2 0.1"/>'
    height = 0.008 if convex else 0.011
    bodies = "".join(
        f'<body name="key_{i}" pos="{(i % 6) * 0.025} {(i // 6) * 0.025} {height}">'
        '<joint type="slide" axis="0 0 1" limited="true" range="-0.006 0.015" damping="0.1"/>'
        f'{key_geom}<site size="0.001" pos="0 0 0.009"/></body>'
        for i in range(keys)
    )
    xml = (
        '<mujoco><option timestep="0.005" solver="Newton" integrator="implicitfast" cone="pyramidal" '
        'jacobian="sparse" iterations="30" ls_iterations="10"><flag sleep="enable" multiccd="disable"/></option>'
        f"<worldbody>{floor_geom}"
        + bodies
        + '<body name="cursor" mocap="true" pos="1 0 1"><geom type="sphere" size="0.01" contype="0" '
        'conaffinity="0"/><site name="cursor_tip" size="0.001"/></body></worldbody></mujoco>'
    )
    cpu = mujoco.MjModel.from_xml_string(xml)
    model = mjw.put_model(cpu)
    model.opt.broadphase = mjw.BroadphaseType.NXN
    model.opt.graph_conditional = True
    data = mjw.make_data(cpu, nworld=1, nconmax=512, nccdmax=128, njmax=1024)
    mjw.forward(model, data)
    for name in _STATE:
        if not np.isfinite(getattr(data, name).numpy()).all():
            raise AssertionError(f"Nonfinite prepared template state: {keys} keys, {name}")
    warm = mjw.replicate_data(data, 1)
    mjw.step(model, warm)
    mjw.kinematics(model, warm)
    wp.synchronize_stream(wp.get_stream())
    return model, data


class TestMuJoCoWorlds(unittest.TestCase):
    """Exercise a caller-controlled CUDA device; default CPU suites skip this gate."""

    @unittest.skipUnless(
        os.environ.get("NEWTON_TEST_CUDA_UUID"), "Set an explicit CUDA UUID for native population tests"
    )
    def test_failed_discovery_releases_internal_borrows_without_clearing_caller_exceptions(self):
        """Close a failed discovery with its traceback alive, preserving real external borrowers."""
        import mujoco
        import mujoco_warp as mjw

        wp.init()
        device = wp.get_device("cuda:0")
        self.assertEqual(device.uuid, os.environ["NEWTON_TEST_CUDA_UUID"])
        with wp.ScopedDevice(device):
            prepared = [_prototype(6, mujoco, mjw, convex=True)]
            original = capture_allocation._discovery_empty
            for retain_external in (False, True):
                with self.subTest(retain_external=retain_external):
                    population = mujoco_worlds_prepare(
                        prepared,
                        world_capacities=(4,),
                        id_capacity=1,
                        command_capacity=1,
                        memory_budget_bytes=64 * 1024**2,
                        initial_world_ready_capacities=(1,),
                    )
                    external = population.populations[0].data.qpos if retain_external else None
                    commands = directory_ops.allocate_commands(1, device=device)
                    results = directory_ops.allocate_results(1, device=device)
                    allocations, graph_reference = 0, None

                    def fail(graph, *args, **kwargs):
                        nonlocal allocations, graph_reference
                        allocations += 1
                        graph_reference = weakref.ref(graph)
                        if allocations == 20:
                            raise RuntimeError("injected native discovery allocation failure")
                        return original(graph, *args, **kwargs)

                    try:
                        with patch.object(capture_allocation, "_discovery_empty", new=fail):
                            mujoco_worlds_capture(population, commands, results)
                    except RuntimeError as failure:
                        self.assertEqual(str(failure), "injected native discovery allocation failure")
                        self.assertIsNotNone(failure.__traceback__)
                        self.assertEqual(allocations, 20)
                        self.assertFalse(device.is_capturing)
                        self.assertIsNone(graph_reference())
                        if external is not None:
                            with self.assertRaisesRegex(RuntimeError, "retained by external views"):
                                mujoco_worlds_close(population, streams=(wp.get_stream(device),))
                            external = None
                        mujoco_worlds_close(population, streams=(wp.get_stream(device),))
                        report = backing_ops.memory_report(population._backing)
                        self.assertEqual(report["references"], 0)
                        self.assertEqual(report["physical_retained_bytes"], 0)
                        self.assertEqual(report["virtual_reserved_bytes"], 0)
                    else:
                        self.fail("Discovery did not reach the injected allocation failure")

    @unittest.skipUnless(
        os.environ.get("NEWTON_TEST_CUDA_UUID"), "Set an explicit CUDA UUID for native population tests"
    )
    def test_growth_rejects_capture_on_another_stream_without_poisoning(self):
        """Preserve a live capture and population when growth is attempted from another stream."""
        import mujoco
        import mujoco_warp as mjw

        wp.init()
        device = wp.get_device("cuda:0")
        self.assertEqual(wp.get_cuda_device_count(), 1)
        self.assertEqual(device.uuid, os.environ["NEWTON_TEST_CUDA_UUID"])
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), device.uuid)
        with wp.ScopedDevice(device):
            prepared = [_prototype(6, mujoco, mjw)]
            population = mujoco_worlds_prepare(
                prepared,
                world_capacities=(4,),
                id_capacity=1,
                command_capacity=1,
                memory_budget_bytes=64 * 1024**2,
                initial_world_ready_capacities=(0,),
            )
            counter = wp.zeros(1, dtype=int, device=device)
            capture_stream, service = wp.Stream(device), wp.Stream(device)
            wp.load_module(module=__name__, device=device)
            try:
                with wp.ScopedCapture(stream=capture_stream, capture_mode=wp.CaptureMode.THREAD_LOCAL) as capture:
                    wp.launch(_record_control, 1, [counter], stream=capture_stream)
                    with wp.ScopedStream(service, sync_enter=False, sync_exit=False):
                        self.assertFalse(service.is_capturing)
                        self.assertTrue(device.is_capturing)
                        with self.assertRaisesRegex(RuntimeError, "outside graph capture"):
                            mujoco_worlds_grow_backing(population, (1,), streams=(service,))
                    wp.launch(_record_control, 1, [counter], stream=capture_stream)
                wp.capture_launch(capture.graph)
                np.testing.assert_array_equal(counter.numpy(), [2])
                self.assertFalse(population._service_failed)
                np.testing.assert_array_equal(population._healthy.numpy(), [1])
                self.assertEqual(mujoco_world_population_ready_capacity(population.populations[0]), 0)
                mujoco_worlds_grow_backing(population, (1,), streams=(wp.get_stream(device),))
                self.assertEqual(mujoco_world_population_ready_capacity(population.populations[0]), 0)
                commands = directory_ops.allocate_commands(1, device=device)
                results = directory_ops.allocate_results(1, device=device)
                graph = mujoco_worlds_capture(population, commands, results)
                self.assertGreaterEqual(mujoco_world_population_ready_capacity(population.populations[0]), 1)
                del graph
                gc.collect()
            finally:
                mujoco_worlds_close(population, streams=(wp.get_stream(device),))

    @unittest.skipUnless(
        os.environ.get("NEWTON_TEST_CUDA_UUID"), "Set an explicit CUDA UUID for native population tests"
    )
    def test_discovered_scratch_uses_actual_owners_and_rejects_descriptor_replacement(self):
        """Discover ordinary scratch into actual owners and freeze every substituted descriptor."""
        import mujoco
        import mujoco_warp as mjw

        wp.init()
        device = wp.get_device("cuda:0")
        self.assertEqual(device.uuid, os.environ["NEWTON_TEST_CUDA_UUID"])
        with wp.ScopedDevice(device):
            prepared = [_prototype(6, mujoco, mjw)]
            wp.load_module(module=__name__, device=device)
            for storage in ("fixed", "vmm"):
                with self.subTest(storage=storage):
                    with (
                        patch.object(
                            native.wp, "ScopedCapture", side_effect=AssertionError("Preparation recorded a graph")
                        ),
                        patch.object(mjw, "step", side_effect=AssertionError("Preparation executed physics")),
                    ):
                        population = mujoco_worlds_prepare(
                            prepared,
                            world_capacities=(4,),
                            id_capacity=2,
                            command_capacity=2,
                            memory_budget_bytes=64 * 1024**2 if storage == "vmm" else None,
                            initial_world_ready_capacities=(0,) if storage == "vmm" else None,
                        )
                    commands = directory_ops.allocate_commands(2, device=device)
                    results = directory_ops.allocate_results(2, device=device)
                    group = population._populations[0]
                    graph = original = owner = None
                    try:
                        self.assertEqual(group.transient_storages, ())
                        np.testing.assert_array_equal(group.ccd_count.numpy(), [0])
                        self.assertEqual(mujoco_world_population_ready_capacity(group.view), 0)
                        graph = mujoco_worlds_capture(population, commands, results)
                        native._validate_population(group)
                        for domain, owner in enumerate(group.transient_storages):
                            if owner is not None:
                                self.assertEqual(owner.capacity, wp.upper_bound(group.count_parameters[domain]))
                                self.assertIs(
                                    owner.protected_count,
                                    (group.world_storage.protected_count, group.contact_count, group.ccd_count)[domain],
                                )
                        for transfer in (group.initialization_transfer, group.compaction_transfer):
                            self.assertFalse(any(name.startswith("work.") for name in transfer.field_names))
                        owner = next(owner for owner in group.transient_storages if owner is not None and owner.arrays)
                        name = next(iter(owner.arrays))
                        original = owner.arrays[name]
                        owner.arrays[name] = wp.empty_like(original)
                        with self.assertRaises(ValueError):
                            native._validate_population(group)
                        owner.arrays[name] = original
                        native._validate_population(group)
                        if storage == "vmm":
                            self.assertEqual(mujoco_world_population_ready_capacity(population.populations[0]), 0)
                            mujoco_worlds_grow_backing(population, (2,), streams=(wp.get_stream(device),))
                            self.assertEqual(mujoco_world_population_ready_capacity(population.populations[0]), 2)
                        commands.sequence.fill_(1)
                        commands.count.fill_(1)
                        commands.operation.fill_(_CREATE)
                        commands.prototype.zero_()
                        wp.capture_launch(graph)
                        commands.sequence.fill_(2)
                        commands.count.zero_()
                        wp.capture_launch(graph)
                        wp.synchronize_stream(wp.get_stream(device))
                        np.testing.assert_array_equal(results.status.numpy()[:1], [_OK])
                        self.assertEqual(population._healthy.numpy()[0], 1)
                        self.assertGreater(float(_prefix(group.data, 1).time.numpy()[0]), 0.0)
                    finally:
                        graph = original = owner = transfer = None
                        gc.collect()
                        mujoco_worlds_close(population, streams=(wp.get_stream(device),))

    @unittest.skipUnless(
        os.environ.get("NEWTON_TEST_CUDA_UUID"), "Set an explicit CUDA UUID for native population tests"
    )
    def test_transient_last_mapping_budget_rejection_recovers_without_publication(self):
        """Exhaust real shared backing at scratch after mapping state, then reclaim and replay."""
        import mujoco
        import mujoco_warp as mjw

        device = wp.get_device("cuda:0")
        self.assertEqual(device.uuid, os.environ["NEWTON_TEST_CUDA_UUID"])
        with wp.ScopedDevice(device):
            prepared = [_prototype(6, mujoco, mjw, convex=True)]
            settings = {
                "world_capacities": (128,),
                "id_capacity": 1,
                "command_capacity": 1,
                "initial_world_ready_capacities": (0,),
            }
            probe = mujoco_worlds_prepare(prepared, memory_budget_bytes=64 * 1024**2, **settings)
            probe_commands = directory_ops.allocate_commands(1, device=device)
            probe_results = directory_ops.allocate_results(1, device=device)
            probe_graph = mujoco_worlds_capture(probe, probe_commands, probe_results, substeps=8)
            group = probe._populations[0]
            self.assertTrue(all(owner is not None for owner in group.transient_storages))
            ccd_requests = tuple(
                request
                for request in wp.capture_get_allocations(probe_graph)
                if request.shape
                and request.shape[0] is group.count_parameters[2]
                and all(wp.upper_bound(axis) for axis in request.shape)
            )
            for region in wp.capture_get_transient_regions(probe_graph):
                self.assertLess(
                    len(group.transient_storages[2].arrays),
                    sum(request.transient_region == region.index for request in ccd_requests),
                    "Native CCD scratch must reuse storage within each step, not only between steps",
                )
            granule = probe._backing.granularity_bytes
            widths = tuple(
                owner.row_stride_bytes * (1, group.contact_quota, group.ccd_quota)[domain]
                for owner, domain in native._storage_domains(group)
                if owner.backing is not None
            )

            def required(rows, widths=widths):
                return sum((rows * width + granule - 1) // granule * granule for width in widths)

            target = next(n for n in range(2, 129) if required(n, widths[:-1]) >= required(1))
            budget = required(target, widths[:-1])
            del probe_graph
            gc.collect()
            mujoco_worlds_close(probe, streams=(wp.get_stream(device),))
            population = mujoco_worlds_prepare(prepared, memory_budget_bytes=budget, **settings)
            graph = None
            try:
                commands = directory_ops.allocate_commands(1, device=device)
                results = directory_ops.allocate_results(1, device=device)
                graph = mujoco_worlds_capture(population, commands, results, substeps=8)
                group = population._populations[0]
                original_map = field_ops.map_backing
                with patch.object(field_ops, "map_backing", wraps=original_map) as mapping:
                    with self.assertRaises(MemoryError):
                        mujoco_worlds_grow_backing(population, (target,), streams=(wp.get_stream(device),))
                self.assertIs(mapping.call_args.args[0], group.transient_storages[2])
                self.assertEqual(backing_ops.memory_report(population._backing)["mapped_bytes"], budget)
                self.assertEqual(mujoco_world_population_ready_capacity(group.view), 0)
                np.testing.assert_array_equal(population.directory.free_slot_count.numpy(), [0])
                self.assertFalse(population._service_failed)
                commands.sequence.fill_(1)
                commands.count.zero_()
                wp.capture_launch(graph)
                np.testing.assert_array_equal(population.directory.live_count.numpy(), [0])
                mujoco_worlds_resize_backing(population, (0,), streams=(wp.get_stream(device),))
                mujoco_worlds_grow_backing(population, (1,), streams=(wp.get_stream(device),))
                _poison_transients(population)
                commands.sequence.fill_(2)
                commands.count.fill_(1)
                commands.operation.fill_(_CREATE)
                commands.prototype.zero_()
                wp.capture_launch(graph)
                oracle = mjw.replicate_data(prepared[0][1], 1)
                for _ in range(8):
                    mjw.step(group.model, oracle)
                mjw.kinematics(group.model, oracle)
                np.testing.assert_array_equal(results.status.numpy(), [_OK])
                for name in (*_STATE, *_POSE):
                    actual, expected = getattr(_prefix(group.data, 1), name).numpy(), getattr(oracle, name).numpy()
                    self.assertTrue(np.isfinite(actual).all(), name)
                    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6, equal_nan=False, err_msg=name)
            finally:
                graph = None
                gc.collect()
                mujoco_worlds_close(population, streams=(wp.get_stream(device),))

    @unittest.skipUnless(
        os.environ.get("NEWTON_TEST_CUDA_UUID"), "Set an explicit CUDA UUID for native population tests"
    )
    def test_partial_backing_budget_rejection_preserves_replay_and_recovers(self):
        """Reject real mapping after one domain grows, retaining coherent readiness and dense physics parity."""
        import mujoco
        import mujoco_warp as mjw

        wp.init()
        device = wp.get_device("cuda:0")
        self.assertEqual(device.uuid, os.environ["NEWTON_TEST_CUDA_UUID"])
        with wp.ScopedDevice(device):
            prepared = [_prototype(6, mujoco, mjw, convex=True)]
            budget, capacity = 32 * 1024**2, 128
            population = mujoco_worlds_prepare(
                prepared,
                world_capacities=(capacity,),
                id_capacity=2,
                command_capacity=2,
                memory_budget_bytes=budget,
                initial_world_ready_capacities=(1,),
            )
            graph = None
            try:
                group = population._populations[0]
                commands = directory_ops.allocate_commands(2, device=device)
                results = directory_ops.allocate_results(2, device=device)
                graph = mujoco_worlds_capture(population, commands, results)
                domains = native._storage_domains(group)
                owners = tuple(owner for owner, _ in domains)
                quotas = tuple((1, group.contact_quota, group.ccd_quota)[domain] for _, domain in domains)
                granule = population._backing.granularity_bytes

                def mapped_bytes(n):
                    return tuple(
                        (n * quota * owner.row_stride_bytes + granule - 1) // granule * granule
                        for owner, quota in zip(owners, quotas, strict=True)
                    )

                initial = mapped_bytes(1)
                target = next(
                    (
                        n
                        for n in range(2, capacity + 1)
                        if initial[0] < mapped_bytes(n)[0] <= budget - sum(initial[1:])
                        and sum(mapped_bytes(n)) > budget
                    ),
                    None,
                )
                self.assertIsNotNone(target, "Fixture must fit world growth but reject a later domain on real budget")
                oracle = mjw.replicate_data(prepared[0][1], 1)

                def replay(sequence, *, create=False):
                    commands.sequence.fill_(sequence)
                    commands.count.fill_(int(create))
                    commands.operation.fill_(_CREATE)
                    commands.prototype.zero_()
                    wp.capture_launch(graph)
                    mjw.step(group.model, oracle)
                    mjw.kinematics(group.model, oracle)
                    for name in (*_STATE, *_POSE):
                        np.testing.assert_allclose(
                            getattr(_prefix(group.data, 1), name).numpy(),
                            getattr(oracle, name).numpy(),
                            rtol=2e-5,
                            atol=2e-6,
                            equal_nan=False,
                            err_msg=f"sequence {sequence}: {name}",
                        )
                    np.testing.assert_array_equal(population._healthy.numpy(), [1])

                replay(1, create=True)
                ready_before = tuple(int(owner.ready_count.numpy()[0]) for owner in owners)
                free_before = population.directory.free_slot_count.numpy().copy()
                bytes_before = backing_ops.memory_report(population._backing)["mapped_bytes"]
                with self.assertRaises(MemoryError):
                    mujoco_worlds_grow_backing(population, (target,), streams=(wp.get_stream(device),))
                self.assertGreater(backing_ops.memory_report(population._backing)["mapped_bytes"], bytes_before)
                self.assertEqual(tuple(int(owner.ready_count.numpy()[0]) for owner in owners), ready_before)
                np.testing.assert_array_equal(population.directory.free_slot_count.numpy(), free_before)
                self.assertFalse(population._service_failed)
                self.assertFalse(any(owner.service_failed for owner in owners))
                replay(2)

                # Return unpublished headroom to the same pool, then admit a feasible retry.
                mujoco_worlds_resize_backing(population, (1,), streams=(wp.get_stream(device),))
                retry = max(2, native._world_ready_capacity(group) + 1)
                self.assertLessEqual(retry, capacity)
                self.assertLessEqual(sum(mapped_bytes(retry)), budget)
                mujoco_worlds_grow_backing(population, (retry,), streams=(wp.get_stream(device),))
                self.assertEqual(native._world_ready_capacity(group), retry)
                np.testing.assert_array_equal(population.directory.free_slot_count.numpy(), [retry - 1])
                replay(3)
            finally:
                graph = None
                gc.collect()
                mujoco_worlds_close(population, streams=(wp.get_stream(device),))

    @unittest.skipUnless(
        os.environ.get("NEWTON_TEST_CUDA_UUID"), "Set an explicit CUDA UUID for native population tests"
    )
    def test_native_lifetimes_pose_refresh_and_callbacks(self):
        """Match fixed/VMM native dynamics, reset-only poses and per-substep diagnostics under one graph."""
        import mujoco
        import mujoco_warp as mjw

        wp.init()
        device = wp.get_device("cuda:0")
        self.assertEqual(wp.get_cuda_device_count(), 1)
        self.assertEqual(device.uuid, os.environ["NEWTON_TEST_CUDA_UUID"])
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), device.uuid)
        reports = []
        with wp.ScopedDevice(device):
            prepared = [_prototype(keys, mujoco, mjw) for keys in (6, 108)]
            default_storage = [
                {name: array.numpy()[0].copy() for name, array in _world_fields(data)} for _, data in prepared
            ]
            wp.load_module(module=__name__, device=device)
            for storage in os.environ.get("NEWTON_TEST_STORAGE", "fixed,vmm").split(","):
                with self.subTest(storage=storage):
                    population = mujoco_worlds_prepare(
                        prepared,
                        world_capacities=(8, 8),
                        id_capacity=4,
                        command_capacity=4,
                        memory_budget_bytes=64 * 1024**2 if storage == "vmm" else None,
                        initial_world_ready_capacities=(0, 0) if storage == "vmm" else None,
                    )
                    graph = None
                    try:
                        report, graph = self._trace(population, prepared, default_storage, mjw)
                        report["storage"] = storage
                        reports.append(report)
                    except BaseException as error:
                        traceback.clear_frames(error.__traceback__)
                        raise
                    finally:
                        graph = None  # noqa: F841 - Drop executable ownership before explicit storage close.
                        gc.collect()
                        mujoco_worlds_close(population, streams=(wp.get_stream(device),))
                    if population._backing is not None:
                        ledger = backing_ops.memory_report(population._backing)
                        self.assertEqual(ledger["mapped_bytes"], 0)
                        self.assertEqual(ledger["physical_retained_bytes"], 0)
                        self.assertEqual(ledger["virtual_reserved_bytes"], 0)
                        report["final_backing"] = ledger
            output = os.environ.get("NEWTON_TEST_OUTPUT")
            if output:
                Path(output).write_text(
                    json.dumps({"passed": True, "gpu_uuid": device.uuid, "cases": reports}, indent=2) + "\n"
                )

    @unittest.skipUnless(
        os.environ.get("NEWTON_TEST_CUDA_UUID"), "Set an explicit CUDA UUID for native population tests"
    )
    def test_convex_scratch_survives_switch_compaction_and_regrowth(self):
        """Exercise real box-box contacts through the borrowed scratch under fixed and VMM ownership."""
        import mujoco
        import mujoco_warp as mjw

        wp.init()
        device = wp.get_device("cuda:0")
        self.assertEqual(wp.get_cuda_device_count(), 1)
        self.assertEqual(device.uuid, os.environ["NEWTON_TEST_CUDA_UUID"])
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), device.uuid)
        with wp.ScopedDevice(device):
            prepared = [_prototype(keys, mujoco, mjw, convex=True) for keys in (2, 3)]
            for model, data in prepared:
                self.assertFalse(model.opt.disableflags & mjw.DisableBit.NATIVECCD)
                self.assertGreater(int(data.nacon.numpy()[0]), 0, "The fixture must generate real convex contacts")
            default_storage = [
                {name: array.numpy()[0].copy() for name, array in _world_fields(data)} for _, data in prepared
            ]
            wp.load_module(module=__name__, device=device)
            for storage in os.environ.get("NEWTON_TEST_STORAGE", "fixed,vmm").split(","):
                with self.subTest(storage=storage):
                    population = mujoco_worlds_prepare(
                        prepared,
                        world_capacities=(8, 8),
                        id_capacity=4,
                        command_capacity=4,
                        memory_budget_bytes=64 * 1024**2 if storage == "vmm" else None,
                        initial_world_ready_capacities=(0, 0) if storage == "vmm" else None,
                    )
                    graph = None
                    try:
                        report, graph = self._trace(population, prepared, default_storage, mjw, require_convex=True)
                        self.assertEqual((report["graph_captures"], report["graph_recaptures"]), (1, 0))
                    except BaseException as error:
                        traceback.clear_frames(error.__traceback__)
                        raise
                    finally:
                        graph = None  # noqa: F841 - Drop executable ownership before explicit storage close.
                        gc.collect()
                        mujoco_worlds_close(population, streams=(wp.get_stream(device),))
                    if population._backing is not None:
                        ledger = backing_ops.memory_report(population._backing)
                        self.assertEqual((ledger["mapped_bytes"], ledger["physical_retained_bytes"]), (0, 0))

    @unittest.skipUnless(
        os.environ.get("NEWTON_TEST_CUDA_UUID"), "Set an explicit CUDA UUID for native population tests"
    )
    def test_graph_update_error_quarantines_all_native_domains(self):
        """A bounded updater error blocks every prototype and every later raw replay."""
        import mujoco
        import mujoco_warp as mjw

        wp.init()
        device = wp.get_device("cuda:0")
        self.assertEqual(wp.get_cuda_device_count(), 1)
        self.assertEqual(device.uuid, os.environ["NEWTON_TEST_CUDA_UUID"])
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), device.uuid)
        with wp.ScopedDevice(device):
            prepared = [_prototype(keys, mujoco, mjw) for keys in (6, 108)]
            population = mujoco_worlds_prepare(prepared, world_capacities=(4, 4), id_capacity=4, command_capacity=4)
            commands, results = (
                directory_ops.allocate_commands(4, device=device),
                directory_ops.allocate_results(4, device=device),
            )
            permit, injected = wp.zeros(1, dtype=int, device=device), wp.zeros(1, dtype=int, device=device)
            calls = [wp.zeros(2, dtype=int, device=device) for _ in prepared]
            original, recorded = graph_ops.record_update, []

            def inject(updater, stream=None):
                original(updater, stream=stream)
                recorded.append(updater)
                if len(recorded) == 2:
                    wp.launch(_inject_graph_update_error, 1, [updater.errors, injected], device=device)

            class Task:
                def __init__(self, buffers):
                    self.buffers = buffers

                def before_step(self, group):
                    wp.launch(_record_control, 1, inputs=[self.buffers[group.prototype_index]])

                def application_bindings(self, group, launches):
                    if any(record.kernel is not _record_control for record in launches):
                        raise ValueError("Unexpected control kernel")
                    return (), (), tuple(range(len(launches)))

            task = Task(calls)
            task_reference, array_references = weakref.ref(task), [weakref.ref(array) for array in calls]

            graph = None
            try:
                wp.load_module(module=__name__, device=device)
                with patch.object(graph_ops, "record_update", inject):
                    graph = mujoco_worlds_capture(
                        population,
                        commands,
                        results,
                        permit=permit,
                        before_step=task.before_step,
                        application_bindings=task.application_bindings,
                        retain=(injected, *calls),
                    )
                del task, calls
                gc.collect()
                self.assertIsNone(task_reference(), "Graph preparation must not retain the application task")
                calls = [reference() for reference in array_references]
                for array in calls:
                    self.assertIsNotNone(array)
                    self.assertTrue(any(owner is array for owner in graph._resource_owners))
                for group in population._populations:
                    self.assertIsNone(group.before_step)
                    self.assertIsNone(group.after_substep)
                graph_id = int(graph.graph_exec.value)
                commands.sequence.fill_(1)
                commands.count.fill_(4)
                commands.operation.fill_(_CREATE)
                commands.prototype.assign(np.array([0, 0, 1, 1], dtype=np.int32))
                wp.capture_launch(graph)
                wp.synchronize_stream(wp.get_stream(device))
                np.testing.assert_array_equal(results.status.numpy(), [0, 0, 0, 0])
                saved = [
                    {name: getattr(_prefix(group.data, 2), name).numpy() for name in ("qpos", "qvel", "time")}
                    for group in population._populations
                ]
                permit.fill_(1)
                commands.count.zero_()
                for sequence, injection_enabled in ((2, 1), (3, 0)):
                    commands.sequence.fill_(sequence)
                    injected.fill_(injection_enabled)
                    wp.capture_launch(graph)
                    wp.synchronize_stream(wp.get_stream(device))
                    self.assertEqual(population._healthy.numpy()[0], 0)
                    self.assertEqual(population.batch_result.status.numpy()[0], int(InstanceStatus.PHASE_INVALID))
                    self.assertEqual(int(graph.graph_exec.value), graph_id)
                    for p, group in enumerate(population._populations):
                        np.testing.assert_array_equal(calls[p].numpy(), [0, 0])
                        self.assertEqual(group.step_condition.numpy()[0], 0)
                        self.assertEqual(group.kinematics_condition.numpy()[0], 0)
                        self.assertEqual(group.move_count.numpy()[0], 0)
                        errors = group.updates.errors.numpy()[: len(group.bindings)]
                        self.assertEqual(np.count_nonzero(errors), int(p == 1 and injection_enabled))
                        for name, value in saved[p].items():
                            np.testing.assert_array_equal(getattr(_prefix(group.data, 2), name).numpy(), value)
            except BaseException as error:
                traceback.clear_frames(error.__traceback__)
                raise
            finally:
                graph = None
                gc.collect()
                mujoco_worlds_close(population, streams=(wp.get_stream(device),))

    @unittest.skipUnless(
        os.environ.get("NEWTON_TEST_CUDA_UUID"), "Set an explicit CUDA UUID for native population tests"
    )
    def test_equal_sequence_skips_lifecycle_after_compaction(self):
        """Skip transaction scratch on repeats while preserving advancement and every error path."""
        import mujoco
        import mujoco_warp as mjw

        wp.init()
        device = wp.get_device("cuda:0")
        assert wp.get_cuda_device_count() == 1
        self.assertEqual(device.uuid, os.environ["NEWTON_TEST_CUDA_UUID"])
        graph = None
        with wp.ScopedDevice(device):
            prepared = [_prototype(keys, mujoco, mjw) for keys in (6, 108)]
            population = mujoco_worlds_prepare(prepared, world_capacities=(8, 8), id_capacity=6, command_capacity=6)
            commands, results = (
                directory_ops.allocate_commands(6, device=device),
                directory_ops.allocate_results(6, device=device),
            )
            permit, calls = wp.zeros(1, dtype=int, device=device), wp.zeros(2, dtype=int, device=device)
            initialized_calls = calls[1:]
            offset = wp.zeros(6, dtype=float, device=device)

            def validate(command, request_status, consumed):
                wp.launch(_record_control, 1, [calls], device=device)

            def payload(group, requests, destinations, count, status, initialized_sequence, sequence):
                wp.launch(
                    _initialize_payload,
                    6,
                    [
                        requests,
                        destinations,
                        count,
                        status,
                        initialized_sequence,
                        sequence,
                        offset,
                        group.data.qpos,
                        group.data.mocap_pos,
                    ],
                    device=device,
                )
                wp.launch(_record_control, 1, [initialized_calls], device=device)

            def replay():
                wp.capture_launch(graph)
                wp.synchronize_stream(wp.get_stream(device))

            try:
                wp.load_module(module=__name__, device=device)
                graph = mujoco_worlds_capture(
                    population,
                    commands,
                    results,
                    permit=permit,
                    validate=validate,
                    initialize=payload,
                    retain=(calls, offset, initialized_calls),
                )
                executable = int(graph.graph_exec.value)
                commands.sequence.fill_(1)
                commands.count.fill_(6)
                commands.operation.fill_(int(InstanceOperation.CREATE))
                commands.prototype.assign(np.array([0, 0, 0, 1, 1, 1], dtype=np.int32))
                replay()
                np.testing.assert_array_equal(results.status.numpy(), np.zeros(6, dtype=np.int32))
                d = population.directory
                populations, slots, generations = d.prototype.numpy(), d.slot.numpy(), d.generation.numpy()
                victim = min(np.flatnonzero(populations == 0), key=lambda identity: slots[identity])
                commands.sequence.fill_(2)
                commands.count.fill_(1)
                commands.operation.fill_(int(InstanceOperation.DESTROY))
                commands.instance_id.fill_(int(victim))
                commands.generation.fill_(int(generations[victim]))
                replay()
                assert results.status.numpy()[0] == 0
                assert population._populations[0].move_count.numpy()[0] > 0
                assert population._directory.transaction.phase.numpy()[0] == int(InstancePhase.IDLE)
                np.testing.assert_array_equal(calls.numpy(), [2, 4])
                counts = d.live_count.numpy()
                saved = [
                    {name: getattr(_prefix(group.data, int(n)), name).numpy() for name in ("qpos", "qvel", "time")}
                    for group, n in zip(population._populations, counts, strict=True)
                ]
                handles = {name: getattr(d, name).numpy() for name in ("prototype", "slot", "generation")}
                # Deliberately stale transfer metadata must never be consumed when the branch is skipped.
                for group in population._populations:
                    group.initialization_count.fill_(3)
                    group.move_count.fill_(3)
                    group.source_rows.fill_(-1)
                    group.destination_rows.fill_(-1)
                    group.move_source_rows.fill_(-1)
                    group.move_destination_rows.fill_(-1)
                commands.operation.fill_(int(InstanceOperation.REPLACE))
                commands.instance_id.fill_(-1)
                for _ in range(3):
                    replay()
                    np.testing.assert_array_equal(calls.numpy(), [2, 4])
                    assert population._lifecycle_needed.numpy()[0] == 0
                    for name, value in handles.items():
                        np.testing.assert_array_equal(getattr(d, name).numpy(), value)
                    for group, n, old in zip(population._populations, counts, saved, strict=True):
                        for name, value in old.items():
                            np.testing.assert_array_equal(getattr(_prefix(group.data, int(n)), name).numpy(), value)
                permit.fill_(1)
                replay()
                np.testing.assert_array_equal(calls.numpy(), [2, 4])
                for group, n, old in zip(population._populations, counts, saved, strict=True):
                    assert np.all(_prefix(group.data, int(n)).time.numpy() > old["time"])
                    assert not group.updates.errors.numpy()[: len(group.bindings)].any()
                permit.zero_()
                commands.sequence.fill_(3)
                commands.count.zero_()
                replay()
                np.testing.assert_array_equal(calls.numpy(), [3, 6])
                for group in population._populations:
                    assert group.initialization_count.numpy()[0] == 0
                    assert group.move_count.numpy()[0] == 0
                current_prototypes, current_generations = d.prototype.numpy(), d.generation.numpy()
                survivor = int(np.flatnonzero(current_prototypes == 0)[0])
                commands.sequence.fill_(4)
                commands.count.fill_(1)
                commands.operation.fill_(int(InstanceOperation.REPLACE))
                commands.instance_id.fill_(survivor)
                commands.generation.fill_(int(current_generations[survivor]))
                commands.prototype.fill_(1)
                replay()
                assert results.status.numpy()[0] == 0
                assert d.prototype.numpy()[survivor] == 1
                assert d.generation.numpy()[survivor] == current_generations[survivor] + 1
                np.testing.assert_array_equal(calls.numpy(), [4, 8])
                # A failed individual request has no batch error, but its permit must remain blocked.
                commands.sequence.fill_(5)
                commands.generation.fill_(int(d.generation.numpy()[survivor]) - 1)
                replay()
                assert results.status.numpy()[0] != 0
                assert (
                    population.batch_result.status.numpy()[0] == 0
                    and population.batch_result.advance_allowed.numpy()[0] == 0
                )
                np.testing.assert_array_equal(calls.numpy(), [5, 10])
                failed_times = [group.data.time.numpy().copy() for group in population._populations]
                permit.fill_(1)
                for _ in range(2):
                    replay()
                    np.testing.assert_array_equal(calls.numpy(), [5, 10])
                    assert population._lifecycle_needed.numpy()[0] == 0
                    for group, value in zip(population._populations, failed_times, strict=True):
                        np.testing.assert_array_equal(group.data.time.numpy(), value)
                permit.zero_()
                # Stale-sequence and repeated-error frames retain the previous complete error path.
                commands.sequence.fill_(1)
                replay()
                np.testing.assert_array_equal(calls.numpy(), [6, 12])
                assert population._lifecycle_needed.numpy()[0] == 1
                assert population.batch_result.advance_allowed.numpy()[0] == 0
                replay()
                np.testing.assert_array_equal(calls.numpy(), [7, 14])
                assert int(graph.graph_exec.value) == executable
                self.assertEqual(mujoco_worlds_memory_report(population)["control_metadata_bytes"], 12)
            except BaseException as error:
                traceback.clear_frames(error.__traceback__)
                raise
            finally:
                graph = None
                gc.collect()
                mujoco_worlds_close(population, streams=(wp.get_stream(device),))

    @unittest.skipUnless(
        os.environ.get("NEWTON_TEST_CUDA_UUID"), "Set an explicit CUDA UUID for native population tests"
    )
    def test_reset_invalidates_contacts_and_retains_partial_sleep_through_compaction(self):
        """A continuing sleeping world matches dense dynamics after another world switches topology."""
        import mujoco
        import mujoco_warp as mjw
        from mujoco_warp._src import sleep

        wp.init()
        device = wp.get_device("cuda:0")
        self.assertEqual(wp.get_cuda_device_count(), 1)
        self.assertEqual(device.uuid, os.environ["NEWTON_TEST_CUDA_UUID"])
        graph = None
        with wp.ScopedDevice(device):
            prepared = [_prototype(keys, mujoco, mjw) for keys in (6, 108)]
            population = mujoco_worlds_prepare(prepared, world_capacities=(4, 4), id_capacity=2, command_capacity=2)
            commands, results = (
                directory_ops.allocate_commands(2, device=device),
                directory_ops.allocate_results(2, device=device),
            )
            permit = wp.ones(1, dtype=int, device=device)
            try:
                graph = mujoco_worlds_capture(population, commands, results, permit=permit)
                commands.sequence.fill_(1)
                commands.count.fill_(2)
                commands.operation.fill_(_CREATE)
                commands.prototype.zero_()
                for _ in range(40):
                    wp.capture_launch(graph)
                    self.assertTrue(np.isfinite(population.populations[0].data.qpos.numpy()[:2]).all())
                    if population.populations[0].data.nacon.numpy()[0] > 0:
                        break
                d = population.directory
                slots = d.slot.numpy()
                keeper, victim = int(np.argmax(slots)), int(np.argmin(slots))
                original_slot = int(slots[keeper])
                group = population.populations[0]
                self.assertGreater(group.data.nacon.numpy()[0], 0, "Fixture needs current contacts")
                asleep = group.data.tree_asleep.numpy()
                asleep[original_slot, :3] = np.arange(3)
                asleep[original_slot, 3:] = -11
                group.data.tree_asleep.assign(asleep)
                for name in ("qvel", "qacc_warmstart", "qfrc_applied"):
                    values = getattr(group.data, name).numpy()
                    values[original_slot, :3] = 0
                    getattr(group.data, name).assign(values)
                sleep.update_sleep(group.model, _prefix(group.data, 2))
                oracle = mjw.replicate_data(prepared[0][1], 1)
                for name, array in _world_fields(group.data):
                    _array(oracle, name).assign(array.numpy()[original_slot : original_slot + 1])
                saved = {name: getattr(group.data, name).numpy()[original_slot].copy() for name in _STATE}
                generation = int(d.generation.numpy()[keeper])
                permit.zero_()
                commands.sequence.fill_(2)
                commands.count.fill_(1)
                commands.operation.fill_(_REPLACE)
                commands.instance_id.fill_(victim)
                commands.generation.fill_(int(d.generation.numpy()[victim]))
                commands.prototype.fill_(1)
                wp.capture_launch(graph)
                self.assertEqual(results.status.numpy()[0], _OK)
                self.assertEqual(d.slot.numpy()[keeper], 0)
                self.assertEqual(d.generation.numpy()[keeper], generation)
                for view in population.populations:
                    np.testing.assert_array_equal(view.data.nacon.numpy(), [0])
                    np.testing.assert_array_equal(view.data.ncollision.numpy(), [0])
                for name, values in saved.items():
                    np.testing.assert_array_equal(getattr(group.data, name).numpy()[0], values, err_msg=name)
                permit.fill_(1)
                for wake in (False, True):
                    if wake:
                        force = group.data.qfrc_applied.numpy()
                        force[0, 0] = 0.01
                        group.data.qfrc_applied.assign(force)
                        oracle.qfrc_applied.assign(force[:1])
                    wp.capture_launch(graph)
                    mjw.step(group.model, oracle)
                    mjw.kinematics(group.model, oracle)
                    for name in (*_STATE, *_POSE):
                        actual, expected = getattr(group.data, name).numpy()[0], getattr(oracle, name).numpy()[0]
                        self.assertTrue(np.isfinite(actual).all() and np.isfinite(expected).all(), name)
                        np.testing.assert_allclose(
                            actual,
                            expected,
                            rtol=3e-5,
                            atol=2e-6,
                            equal_nan=False,
                            err_msg=f"wake={wake}, field={name}",
                        )
                    if wake:
                        self.assertLess(group.data.tree_asleep.numpy()[0, 0], 0)
                    else:
                        np.testing.assert_array_equal(group.data.tree_asleep.numpy()[0, :3], np.arange(3))
            except BaseException as error:
                traceback.clear_frames(error.__traceback__)
                raise
            finally:
                graph = None
                gc.collect()
                mujoco_worlds_close(population, streams=(wp.get_stream(device),))

    def _trace(self, population, prepared, default_storage, mjw, *, require_convex=False):
        """Run independent lifetime-aware oracle rows and compare physical/derived fields."""
        device = population.device
        commands = directory_ops.allocate_commands(4, device=device)
        results = directory_ops.allocate_results(4, device=device)
        valid, offset = wp.ones(4, dtype=int, device=device), wp.zeros(4, dtype=float, device=device)
        permit = wp.zeros(1, dtype=int, device=device)
        counters = [(wp.zeros(2, dtype=int, device=device), wp.zeros(1, dtype=int, device=device)) for _ in range(2)]
        convex_observed = [wp.zeros(2, dtype=int, device=device) for _ in prepared] if require_convex else []
        oracle = [mjw.replicate_data(data, 8) for _, data in prepared]

        def validate(c, request_status, consumed):
            wp.launch(_validate_payload, 4, [c, request_status, consumed, valid], device=device)

        def initialize(group, requests, destinations, count, status, initialized_sequence, sequence):
            wp.launch(
                _initialize_payload,
                4,
                [
                    requests,
                    destinations,
                    count,
                    status,
                    initialized_sequence,
                    sequence,
                    offset,
                    group.data.qpos,
                    group.data.mocap_pos,
                ],
                device=device,
            )

        def before_step(group):
            wp.launch(_record_control, 1, inputs=[counters[group.prototype_index][0]])

        def after_substep(group):
            calls, sticky = counters[group.prototype_index]
            wp.launch(_record_substep, group.world_capacity, inputs=[group.data.overflow, sticky, calls])

        def application_bindings(group, launches):
            extents, fixed = [], []
            for index, record in enumerate(launches):
                if record.kernel is _record_control:
                    fixed.append(index)
                elif record.kernel is _record_substep:
                    extents.append((index, 0, group.world_live_count))
                else:
                    raise ValueError("Unexpected dynamics-test callback kernel")
            return tuple(extents), (), tuple(fixed)

        with ExitStack() as probe:
            if require_convex:
                from mujoco_warp._src import collision_driver

                narrowphase = collision_driver.convex_narrowphase
                groups = {id(group.execution_data): group.view for group in population._populations}

                def record_convex_pass(model, data, *args, **kwargs):
                    group = groups[id(data)]
                    launch = wp.launch

                    def observe_kernel(kernel, *launch_args, **launch_kwargs):
                        launch(kernel, *launch_args, **launch_kwargs)
                        parameters = [arg.label for arg in kernel.adj.args]
                        if "nccd_in" in parameters:
                            operands = (*launch_kwargs.get("inputs", ()), *launch_kwargs.get("outputs", ()))
                            launch(
                                _record_convex_work,
                                1,
                                inputs=[
                                    operands[parameters.index("nccd_in")],
                                    group.data.nacon,
                                    convex_observed[group.prototype_index],
                                ],
                            )

                    # Observe the exact local counter while its native activation is still live.
                    with patch.object(wp, "launch", new=observe_kernel):
                        narrowphase(model, data, *args, **kwargs)

                probe.enter_context(
                    patch.object(collision_driver, "convex_narrowphase", side_effect=record_convex_pass)
                )
            graph = mujoco_worlds_capture(
                population,
                commands,
                results,
                permit=permit,
                validate=validate,
                initialize=initialize,
                substeps=2,
                before_step=before_step,
                after_substep=after_substep,
                application_bindings=application_bindings,
                retain=(valid, offset, permit, *(array for pair in counters for array in pair), *convex_observed),
            )
        graph_id = int(graph.graph_exec.value)
        addresses = [(group.data.qpos.ptr, group.data.site_xpos.ptr) for group in population.populations]
        for group in population.populations:
            self.assertGreater(group.data.site_xpos.shape[1], 0)
            for name in ("flex", "elem", "vert"):
                field = getattr(group.data.contact, name)
                self.assertEqual(field.shape[0], 0)
                self.assertNotIn(
                    "contact." + name, population._populations[group.prototype_index].contact_storage.fields
                )
            owner = population._populations[group.prototype_index]
            self.assertIsNone(owner.before_step)
            self.assertIsNone(owner.after_substep)
            self.assertIsNotNone(owner.updates)
            self.assertTrue(owner.bindings)
            self.assertFalse(hasattr(group, "world_storage"))
            self.assertFalse(hasattr(group, "updates"))
        labels, previous, expected_calls = {}, {}, np.zeros((2, 2), dtype=np.int32)
        cases = [
            "create",
            "step",
            "switch",
            "step",
            "delete",
            "refill",
            "invalid",
            "delete_all",
            "empty",
            "create",
            "step",
        ]
        maximum = 0.0
        for sequence, case in enumerate(cases, start=1):
            if os.environ.get("NEWTON_TEST_VERBOSE"):
                print(f"native frame {sequence}: {case}, VMM={population._backing is not None}", flush=True)
            requests = []
            if case == "create":
                requests = [(i, _CREATE, -1, 0, i % 2) for i in range(4)]
            elif case == "switch":
                requests = [(i, _REPLACE, *labels[i], 1 - previous[labels[i][0]]["prototype"]) for i in (0, 1)]
            elif case in ("delete", "delete_all"):
                selected = [2] if case == "delete" else list(labels)
                requests = [(i, _DESTROY, *labels[i], 0) for i in selected]
            elif case == "refill":
                requests = [(2, _CREATE, -1, 0, 0)]
            elif case == "invalid":
                requests = [(0, _REPLACE, *labels[0], 0)]
            if population._backing is not None:
                targets = tuple(
                    int(count) + sum(op in (_CREATE, _REPLACE) and target == p for _, op, _, _, target in requests)
                    for p, count in enumerate(population.directory.live_count.numpy())
                )
                consumer = wp.get_stream(device)
                if all(
                    target >= mujoco_world_population_ready_capacity(group)
                    for group, target in zip(population.populations, targets, strict=True)
                ):
                    service = wp.Stream(device)
                    with wp.ScopedStream(service):
                        mujoco_worlds_grow_backing(population, targets, streams=(consumer,))
                    consumer.wait_stream(service)
                else:
                    mujoco_worlds_resize_backing(population, targets, streams=(consumer,))
            do_step = case in ("step", "delete", "invalid", "empty")
            _poison_transients(population)
            permit.fill_(int(do_step))
            commands.sequence.fill_(sequence)
            commands.count.fill_(len(requests))
            valid.assign(np.array([0, 1, 1, 1] if case == "invalid" else [1, 1, 1, 1], dtype=np.int32))
            deltas = np.arange(1, 5, dtype=np.float32) * np.float32(0.0001)
            offset.assign(deltas)
            for name, column, dtype in (
                ("operation", 1, np.int32),
                ("instance_id", 2, np.int32),
                ("generation", 3, np.uint64),
                ("prototype", 4, np.int32),
            ):
                values = np.zeros(4, dtype=dtype)
                values[: len(requests)] = [request[column] for request in requests]
                getattr(commands, name).assign(values)
            wp.capture_launch(graph)
            status, assigned, generations = (
                results.status.numpy(),
                results.instance_id.numpy(),
                results.generation.numpy(),
            )
            wanted_status = [_INVALID] if case == "invalid" else [_OK] * len(requests)
            np.testing.assert_array_equal(status[: len(requests)], wanted_status)
            fresh_offsets = {}
            for request, (label, op, _, _, _) in enumerate(requests):
                if status[request] != _OK:
                    continue
                if op == _DESTROY:
                    del labels[label]
                else:
                    labels[label] = int(assigned[request]), int(generations[request])
                    fresh_offsets[int(assigned[request])] = deltas[request]
            directory = population.directory
            populations, slots, generations = (
                directory.prototype.numpy(),
                directory.slot.numpy(),
                directory.generation.numpy(),
            )
            counts, advance_allowed = (
                directory.live_count.numpy(),
                bool(population.batch_result.advance_allowed.numpy()[0]),
            )
            self.assertEqual(advance_allowed, case != "invalid")
            after = {}
            for p, group in enumerate(population.populations):
                count = int(counts[p])
                ids = sorted(np.flatnonzero(populations == p), key=lambda identity: slots[identity])
                np.testing.assert_array_equal(slots[ids], np.arange(count))
                for name, default in default_storage[p].items():
                    array = _array(oracle[p], name)
                    values = array.numpy()
                    for identity in ids:
                        old = previous.get(int(identity))
                        continuing = old is not None and old["generation"] == int(generations[identity])
                        row = int(slots[identity])
                        values[row] = old["state"][name] if continuing and name in _STATE else default
                        if not continuing and name == "qpos":
                            values[row, 0] += fresh_offsets[int(identity)]
                        if not continuing and name == "mocap_pos":
                            values[row, 0, 0] += fresh_offsets[int(identity)]
                    array.assign(values)
                oracle_view = _prefix(oracle[p], count)
                if count and advance_allowed:
                    if do_step:
                        for _ in range(2):
                            mjw.step(group.model, oracle_view)
                        expected_calls[p] += (1, 2)
                    mjw.kinematics(group.model, oracle_view)
                actual_view = _prefix(group.data, count)
                saved = {}
                for name in (*_STATE, *_POSE):
                    actual, expected = getattr(actual_view, name).numpy(), getattr(oracle_view, name).numpy()
                    if not advance_allowed and name in _POSE:
                        for identity in ids:
                            expected[int(slots[identity])] = previous[int(identity)]["pose"][name]
                    self.assertTrue(np.isfinite(actual).all() and np.isfinite(expected).all(), f"{case}/{p}/{name}")
                    np.testing.assert_allclose(
                        actual, expected, rtol=2e-5, atol=2e-6, equal_nan=False, err_msg=f"{case}/{p}/{name}"
                    )
                    maximum = max(
                        maximum, float(np.max(np.abs(actual.astype(float) - expected.astype(float)), initial=0))
                    )
                    saved[name] = expected
                for identity in ids:
                    row = int(slots[identity])
                    after[int(identity)] = {
                        "prototype": p,
                        "generation": int(generations[identity]),
                        "state": {name: saved[name][row].copy() for name in _STATE},
                        "pose": {name: saved[name][row].copy() for name in _POSE},
                    }
                np.testing.assert_array_equal(counters[p][0].numpy(), expected_calls[p])
                self.assertEqual(counters[p][1].numpy()[0], 0)
            self.assertEqual(population.batch_result.status.numpy()[0], int(InstanceStatus.OK))
            previous = after
            self.assertEqual(graph_id, int(graph.graph_exec.value))
            self.assertEqual(addresses, [(g.data.qpos.ptr, g.data.site_xpos.ptr) for g in population.populations])
        for prototype, observed in enumerate(convex_observed):
            ccd, contacts = observed.numpy()
            self.assertTrue(ccd, f"Prototype {prototype} must execute convex CCD during a collision pass")
            self.assertTrue(contacts, f"Prototype {prototype} must produce contacts during a collision pass")
        # Rejected counts never expose their prefix to payload callbacks or mutate live state.
        before_results = results.status.numpy().copy()
        before_directory = {
            name: getattr(population.directory, name).numpy().copy() for name in ("prototype", "slot", "generation")
        }
        before_state = [
            (_prefix(group.data, int(count)).qpos.numpy().copy(), _prefix(group.data, int(count)).time.numpy().copy())
            for group, count in zip(population.populations, population.directory.live_count.numpy(), strict=True)
        ]
        for sequence, invalid_count in enumerate((-1, 2**31 - 1), start=len(cases) + 1):
            commands.sequence.fill_(sequence)
            commands.count.fill_(invalid_count)
            wp.capture_launch(graph)
            self.assertEqual(int(population.batch_result.status.numpy()[0]), int(InstanceStatus.BAD_COUNT))
            self.assertEqual(int(population.batch_result.consumed.numpy()[0]), 0)
            np.testing.assert_array_equal(results.status.numpy(), before_results)
            for name, expected in before_directory.items():
                np.testing.assert_array_equal(getattr(population.directory, name).numpy(), expected)
            for group, expected, count in zip(
                population.populations, before_state, population.directory.live_count.numpy(), strict=True
            ):
                view = _prefix(group.data, int(count))
                np.testing.assert_array_equal(view.qpos.numpy(), expected[0])
                np.testing.assert_array_equal(view.time.numpy(), expected[1])
            self.assertEqual(graph_id, int(graph.graph_exec.value))
        return {
            "passed": True,
            "invalid_count_batches": 2,
            "frames": len(cases),
            "graph_captures": 1,
            "graph_recaptures": 0,
            "max_error": maximum,
            "substeps": 2,
            "site_count": [g.model.nsite for g in population.populations],
            "per_substep_overflow": [int(pair[1].numpy()[0]) for pair in counters],
            "callback_counts": expected_calls.tolist(),
            "memory": mujoco_worlds_memory_report(population),
        }, graph


if __name__ == "__main__":
    unittest.main()
