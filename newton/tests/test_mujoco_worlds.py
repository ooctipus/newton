# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Qualify the public native population against independent dense physics."""

import gc
import json
import os
import traceback
import unittest
from contextlib import nullcontext
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.mujoco.worlds import _MuJoCoPrototype
from newton._src.utils import row_storage as storage
from newton._src.utils.cuda_vmm import CudaBacking
from newton.solvers import MuJoCoWorlds
from newton.tests.test_cuda_vmm import FakeDriver
from newton.tests.test_row_storage import FakeArray, FakeWarp
from newton.worlds import (
    WorldCommands,
    WorldOperation,
    WorldPhase,
    WorldStatus,
    WorldTransaction,
    create_world_commands,
    create_world_results,
)

_CREATE, _RESET, _DESTROY = (
    int(value) for value in (WorldOperation.CREATE, WorldOperation.RESET, WorldOperation.DESTROY)
)
_VALIDATED, _ADMITTED = int(WorldPhase.VALIDATED), int(WorldPhase.ADMITTED)
_OK, _INVALID = int(WorldStatus.OK), int(WorldStatus.INVALID)
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


class MuJoCoWorldsHostTests(unittest.TestCase):
    """Check joined service admission and failure publication without CUDA."""

    def test_all_safe_shrinks_precede_growth_with_one_shared_backing_budget(self):
        """Reuse a later prototype's backing without exceeding a full physical budget."""

        fake, driver = FakeWarp(), FakeDriver()
        backing = CudaBacking(3 * driver.granularity, driver=driver)
        population = object.__new__(MuJoCoWorlds)
        population._closed = population._service_failed = False
        population.device, population.backing = "cpu", backing
        population._healthy = SimpleNamespace(fill_=lambda value: self.fail("healthy batch quarantined"))
        publications, withdrawals = [], []
        population.directory = SimpleNamespace(
            withdraw_ready=withdrawals.append,
            publish_ready=publications.append,
            d=SimpleNamespace(active_count=SimpleNamespace(numpy=lambda: np.zeros(2, dtype=int))),
        )
        population.prototypes = []
        with (
            patch.object(storage, "wp", fake),
            patch("newton._src.solvers.mujoco.worlds.wp.get_stream", return_value=SimpleNamespace(cuda_stream=11)),
            patch("newton._src.solvers.mujoco.worlds.wp.synchronize_stream"),
        ):
            try:
                for initial in (0, 1):
                    group = _MuJoCoPrototype(None, contact_quota=1, ccd_quota=1)
                    population.prototypes.append(group)
                    for name in ("rows", "contacts", "ccd"):
                        fields = (storage.FieldSpec("contact.efc_address" if name == "contacts" else "value", (), int),)
                        owner = storage.RowStorage(
                            257, FakeArray([0], dtype=int), fields=fields, backing=backing, initial_rows=initial
                        )
                        setattr(group, name, owner)
                self.assertEqual(backing.memory_report()["physical_retained_bytes"], backing.budget_bytes)
                creates = driver.calls["cuMemCreate"]
                joins = driver.calls["cuStreamSynchronize"]
                population.resize_backing((1, 0), streams=(11,))
                self.assertEqual(driver.calls["cuMemCreate"], creates)
                self.assertEqual(driver.calls["cuStreamSynchronize"] - joins, 1)
                self.assertEqual(withdrawals, [(1, 0)])
                self.assertEqual(publications, [(256, 0)])
                self.assertEqual(backing.memory_report()["mapped_bytes"], backing.budget_bytes)
            finally:
                for group in population.prototypes:
                    for name in ("rows", "contacts", "ccd"):
                        owner = getattr(group, name, None)
                        if owner is not None:
                            owner.close(streams=(11,))
                with backing.maintenance(streams=(11,)):
                    backing.close()
        self.assertEqual(backing.memory_report()["physical_retained_bytes"], 0)

    def test_partial_batch_budget_failure_publishes_only_jointly_backed_prefixes(self):
        """Partial batch budget failure publishes only jointly backed prefixes."""
        population = object.__new__(MuJoCoWorlds)
        population._closed = population._service_failed = False
        population.device = "cpu"
        population.backing = SimpleNamespace(maintenance=lambda **kwargs: nullcontext())
        health, publications, withdrawals = [1], [], []
        population._healthy = SimpleNamespace(fill_=lambda value, health=health: health.__setitem__(0, value))
        population.directory = SimpleNamespace(
            withdraw_ready=withdrawals.append,
            publish_ready=publications.append,
            d=SimpleNamespace(active_count=SimpleNamespace(numpy=lambda: np.zeros(2, dtype=int))),
        )
        population.prototypes = []
        for p in range(2):
            group = _MuJoCoPrototype(None, contact_quota=1, ccd_quota=1)
            for name in ("rows", "contacts", "ccd"):
                owner = SimpleNamespace(
                    ready_rows=1,
                    capacity=4,
                    service_failed=False,
                    arrays={"contact.efc_address": object()},
                    ready_count=object(),
                    zero=lambda **kwargs: None,
                    fill=lambda *args, **kwargs: None,
                )

                def resize(target, *, live_count, owner=owner, p=p, name=name):
                    if p == 1 and name == "contacts":
                        raise MemoryError("budget rejected before mutation")
                    owner.ready_rows = target

                owner.resize_backing = resize
                setattr(group, name, owner)
            population.prototypes.append(group)
        with (
            patch("newton._src.solvers.mujoco.worlds.wp.get_stream", return_value=SimpleNamespace(cuda_stream=0)),
            patch("newton._src.solvers.mujoco.worlds.wp.synchronize_stream"),
        ):
            with self.assertRaisesRegex(MemoryError, "budget"):
                population.resize_backing((2, 2), streams=(0,))
        self.assertEqual(withdrawals, [(2, 2)])
        self.assertEqual(publications, [(2, 1)])
        self.assertEqual(population.prototypes[1].rows.ready_rows, 2)
        self.assertFalse(population._service_failed)
        self.assertEqual(health, [1])

    def test_only_healthy_storage_budget_rejection_is_retryable(self):
        """Only healthy storage budget rejection is retryable."""
        for failure in ("budget", "storage_publication", "directory_publication", "scratch_fill"):
            with self.subTest(failure=failure):
                population = object.__new__(MuJoCoWorlds)
                population._closed = population._service_failed = False
                population.device = "cpu"
                population.backing = SimpleNamespace(maintenance=lambda **kwargs: nullcontext())
                health, publications = [1], []
                population._healthy = SimpleNamespace(fill_=lambda value, health=health: health.__setitem__(0, value))

                def publish(rows, publications=publications, failure=failure):
                    publications.append(rows)
                    if failure == "directory_publication":
                        raise MemoryError("directory publication")

                population.directory = SimpleNamespace(
                    withdraw_ready=lambda *args: None,
                    publish_ready=publish,
                    d=SimpleNamespace(active_count=SimpleNamespace(numpy=lambda: np.zeros(1, dtype=int))),
                )
                group = _MuJoCoPrototype(None, contact_quota=2, ccd_quota=1)
                for name, ready, capacity in (("rows", 1, 4), ("contacts", 2, 8), ("ccd", 1, 4)):
                    owner = SimpleNamespace(
                        ready_rows=ready,
                        capacity=capacity,
                        service_failed=False,
                        arrays={"contact.efc_address": object()},
                        ready_count=object(),
                    )

                    def resize(target, *, live_count, owner=owner, name=name, failure=failure):
                        if name == "rows" and failure in ("budget", "storage_publication"):
                            owner.service_failed = failure == "storage_publication"
                            raise MemoryError(failure)
                        owner.ready_rows = target

                    def fill(*args, failure=failure, **kwargs):
                        if failure == "scratch_fill":
                            raise MemoryError("scratch initialization")

                    owner.resize_backing, owner.fill, owner.zero = resize, fill, fill
                    setattr(group, name, owner)
                population.prototypes = [group]
                with (
                    patch(
                        "newton._src.solvers.mujoco.worlds.wp.get_stream", return_value=SimpleNamespace(cuda_stream=0)
                    ),
                    patch("newton._src.solvers.mujoco.worlds.wp.synchronize_stream"),
                ):
                    with self.assertRaises(MemoryError):
                        population.resize_backing((2,), streams=(0,))
                self.assertEqual(population._service_failed, failure != "budget")
                self.assertEqual(health[0], int(failure == "budget"))
                if failure == "budget":
                    self.assertEqual(publications, [(1,)])
                elif failure != "directory_publication":
                    self.assertEqual(publications, [])


@wp.kernel
def _validate_payload(commands: WorldCommands, transaction: WorldTransaction, valid: wp.array[int]):
    request = wp.tid()
    if transaction.phase[0] == _VALIDATED and request < commands.count[0] and transaction.status[request] == _OK:
        if (commands.op[request] == _CREATE or commands.op[request] == _RESET) and valid[request] == 0:
            transaction.status[request] = _INVALID


@wp.kernel
def _initialize_payload(
    requests: wp.array[int],
    destinations: wp.array[int],
    count: wp.array[int],
    status: wp.array[int],
    transaction: WorldTransaction,
    sequence: wp.array[wp.uint64],
    offset: wp.array[float],
    qpos: wp.array2d[float],
    mocap: wp.array2d[wp.vec3],
):
    ordinal = wp.tid()
    if transaction.phase[0] == _ADMITTED and ordinal < count[0] and status[0] == 0:
        request, row = requests[ordinal], destinations[ordinal]
        qpos[row, 0] += offset[request]
        mocap[row, 0] += wp.vec3(offset[request], 0.0, 0.0)
        transaction.initialized[request] = sequence[0]


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


def _prototype(keys, mujoco, mjw):
    """Prepare an asset-independent slider keyboard, mocap body and nonempty sites."""
    bodies = "".join(
        f'<body name="key_{i}" pos="{(i % 6) * 0.025} {(i // 6) * 0.025} 0.011">'
        '<joint type="slide" axis="0 0 1" limited="true" range="-0.006 0.015" damping="0.1"/>'
        '<geom type="sphere" size="0.009" mass="0.01"/><site size="0.001" pos="0 0 0.009"/></body>'
        for i in range(keys)
    )
    xml = (
        '<mujoco><option timestep="0.005" solver="Newton" integrator="implicitfast" cone="pyramidal" '
        'jacobian="sparse" iterations="30" ls_iterations="10"><flag sleep="enable" multiccd="disable"/></option>'
        '<worldbody><geom type="plane" size="2 2 0.1"/>'
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
    warm = mjw.replicate_data(data, 1)
    workspace = mjw.make_step_workspace(model, warm)
    mjw.step(model, warm, workspace=workspace)
    mjw.kinematics(model, warm, workspace=workspace)
    wp.synchronize_stream(wp.get_stream())
    return model, data


class TestMuJoCoWorlds(unittest.TestCase):
    """Exercise a caller-controlled CUDA device; default CPU suites skip this gate."""

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
            defaults = [{name: array.numpy()[0].copy() for name, array in _world_fields(data)} for _, data in prepared]
            wp.load_module(module=__name__, device=device)
            for storage in os.environ.get("NEWTON_TEST_STORAGE", "fixed,vmm").split(","):
                with self.subTest(storage=storage):
                    population = MuJoCoWorlds(
                        prepared,
                        capacities=(8, 8),
                        id_capacity=4,
                        command_capacity=4,
                        memory_budget_bytes=64 * 1024**2 if storage == "vmm" else None,
                        initial_rows=(0, 0) if storage == "vmm" else None,
                    )
                    graph = None
                    try:
                        report, graph = self._trace(population, prepared, defaults, mjw)
                        report["storage"] = storage
                        reports.append(report)
                    except BaseException as error:
                        traceback.clear_frames(error.__traceback__)
                        raise
                    finally:
                        graph = None  # noqa: F841 - Drop executable ownership before explicit storage close.
                        gc.collect()
                        population.close(streams=(wp.get_stream(device).cuda_stream,))
                    if population.backing is not None:
                        ledger = population.backing.memory_report()
                        self.assertEqual(ledger["mapped_bytes"], 0)
                        self.assertEqual(ledger["physical_retained_bytes"], 0)
                        self.assertEqual(ledger["virtual_reserved_bytes"], 0)
                        report["final_backing"] = ledger
            output = os.environ.get("NEWTON_TEST_OUTPUT")
            if output:
                Path(output).write_text(
                    json.dumps({"passed": True, "gpu_uuid": device.uuid, "cases": reports}, indent=2) + "\n"
                )

    def _trace(self, population, prepared, defaults, mjw):
        """Run independent lifetime-aware oracle rows and compare physical/derived fields."""
        device = population.device
        commands = create_world_commands(4, device=device)
        results = create_world_results(4, device=device)
        valid, offset = wp.ones(4, dtype=int, device=device), wp.zeros(4, dtype=float, device=device)
        permit = wp.zeros(1, dtype=int, device=device)
        counters = [(wp.zeros(2, dtype=int, device=device), wp.zeros(1, dtype=int, device=device)) for _ in range(2)]
        by_group = {id(group): counters[p] for p, group in enumerate(population.prototypes)}
        oracle = [mjw.replicate_data(data, 8) for _, data in prepared]

        def validate(c, t):
            wp.launch(_validate_payload, 4, [c, t, valid], device=device)

        def initialize(group, requests, destinations, count, status, transaction, sequence):
            wp.launch(
                _initialize_payload,
                4,
                [
                    requests,
                    destinations,
                    count,
                    status,
                    transaction,
                    sequence,
                    offset,
                    group.data.qpos,
                    group.data.mocap_pos,
                ],
                device=device,
            )

        def before_step(group):
            wp.launch(_record_control, 1, [by_group[id(group)][0]], device=device)

        def after_substep(group):
            calls, sticky = by_group[id(group)]
            wp.launch(_record_substep, group.rows.capacity, [group.data.overflow, sticky, calls], device=device)
            group.observe_launch(_record_substep, group.rows.capacity, "world")

        graph = population.capture(
            commands,
            results,
            permit=permit,
            validate=validate,
            initialize=initialize,
            substeps=2,
            before_step=before_step,
            after_substep=after_substep,
            retain=(valid, offset, permit, *(array for pair in counters for array in pair)),
        )
        graph_id = int(graph.graph_exec.value)
        addresses = [(group.data.qpos.ptr, group.data.site_xpos.ptr) for group in population.prototypes]
        for group in population.prototypes:
            self.assertGreater(group.data.site_xpos.shape[1], 0)
            for name in ("flex", "elem", "vert"):
                field = getattr(group.data.contact, name)
                self.assertEqual(field.shape[0], 0)
                self.assertNotIn("contact." + name, group.contacts.fields)
            self.assertIsNone(group.before_step)
            self.assertIsNone(group.after_substep)
            self.assertIsNone(group.workspace.observer)
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
                print(f"native frame {sequence}: {case}, VMM={population.backing is not None}", flush=True)
            requests = []
            if case == "create":
                requests = [(i, _CREATE, -1, 0, i % 2) for i in range(4)]
            elif case == "switch":
                requests = [(i, _RESET, *labels[i], 1 - previous[labels[i][0]]["prototype"]) for i in (0, 1)]
            elif case in ("delete", "delete_all"):
                selected = [2] if case == "delete" else list(labels)
                requests = [(i, _DESTROY, *labels[i], 0) for i in selected]
            elif case == "refill":
                requests = [(2, _CREATE, -1, 0, 0)]
            elif case == "invalid":
                requests = [(0, _RESET, *labels[0], 0)]
            if population.backing is not None:
                targets = tuple(
                    int(count) + sum(op in (_CREATE, _RESET) and target == p for _, op, _, _, target in requests)
                    for p, count in enumerate(population.directory.d.active_count.numpy())
                )
                population.resize_backing(targets, streams=(wp.get_stream(device).cuda_stream,))
            do_step = case in ("step", "delete", "invalid", "empty")
            permit.fill_(int(do_step))
            commands.sequence.fill_(sequence)
            commands.count.fill_(len(requests))
            valid.assign(np.array([0, 1, 1, 1] if case == "invalid" else [1, 1, 1, 1], dtype=np.int32))
            deltas = np.arange(1, 5, dtype=np.float32) * np.float32(0.0001)
            offset.assign(deltas)
            for name, column, dtype in (
                ("op", 1, np.int32),
                ("id", 2, np.int32),
                ("generation", 3, np.uint64),
                ("prototype", 4, np.int32),
            ):
                values = np.zeros(4, dtype=dtype)
                values[: len(requests)] = [request[column] for request in requests]
                getattr(commands, name).assign(values)
            wp.capture_launch(graph)
            status, assigned, generations = results.status.numpy(), results.id.numpy(), results.generation.numpy()
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
            directory = population.directory.d
            prototypes, slots, generations = (
                directory.prototype.numpy(),
                directory.slot.numpy(),
                directory.generation.numpy(),
            )
            counts, advance = directory.active_count.numpy(), bool(directory.flags.numpy()[1])
            self.assertEqual(advance, case != "invalid")
            after = {}
            for p, group in enumerate(population.prototypes):
                count = int(counts[p])
                ids = sorted(np.flatnonzero(prototypes == p), key=lambda identity: slots[identity])
                np.testing.assert_array_equal(slots[ids], np.arange(count))
                for name, default in defaults[p].items():
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
                if count and advance:
                    if do_step:
                        for _ in range(2):
                            mjw.step(group.model, oracle_view)
                        expected_calls[p] += (1, 2)
                    mjw.kinematics(group.model, oracle_view)
                actual_view = _prefix(group.data, count)
                saved = {}
                for name in (*_STATE, *_POSE):
                    actual, expected = getattr(actual_view, name).numpy(), getattr(oracle_view, name).numpy()
                    if not advance and name in _POSE:
                        for identity in ids:
                            expected[int(slots[identity])] = previous[int(identity)]["pose"][name]
                    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6, err_msg=f"{case}/{p}/{name}")
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
                self.assertFalse(group.updates.errors.numpy()[: len(group.bindings)].any())
            previous = after
            self.assertEqual(graph_id, int(graph.graph_exec.value))
            self.assertEqual(addresses, [(g.data.qpos.ptr, g.data.site_xpos.ptr) for g in population.prototypes])
        return {
            "passed": True,
            "frames": len(cases),
            "graph_captures": 1,
            "graph_recaptures": 0,
            "max_error": maximum,
            "substeps": 2,
            "site_count": [g.model.nsite for g in population.prototypes],
            "per_substep_overflow": [int(pair[1].numpy()[0]) for pair in counters],
            "callback_counts": expected_calls.tolist(),
            "memory": population.memory_report(),
        }, graph


if __name__ == "__main__":
    unittest.main()
