# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Connectedness controls for experimental contact-island sleeping."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.sleep_islands import SleepIslands
from newton._src.solvers.feather_pgs.sleep_topology import SleepTopology


def _array(values, device, dtype=wp.int32):
    return wp.array(values, dtype=dtype, device=device)


def _fixture(worlds, device="cpu", *, topology_eligible=None):
    worlds = np.asarray(worlds, dtype=np.int32)
    count = len(worlds)
    if topology_eligible is None:
        topology_eligible = np.ones(count, np.int32)
    body = np.r_[np.arange(count, dtype=np.int32), np.int32(-1)]
    empty = np.empty(0, np.int32)
    offsets = np.arange(count + 1, dtype=np.int32)
    topology = SleepTopology(
        body_component_host=body,
        joint_component_host=empty,
        component_world_host=worlds,
        component_eligible_host=np.asarray(topology_eligible, np.int32),
        component_body_start_host=offsets,
        component_bodies_host=offsets[:-1],
        component_joint_start_host=np.zeros(count + 1, np.int32),
        component_joints_host=empty,
        body_component=_array(body, device),
        joint_component=_array(empty, device),
        component_world=_array(worlds, device),
        component_eligible=_array(topology_eligible, device),
    )
    return SimpleNamespace(
        topology=topology,
        # A body-less global ground and a world-zero fixed skeleton are both
        # static, but only the former may contact components in every world.
        shape_body=_array(np.r_[np.arange(count, dtype=np.int32), [-1, count]], device),
        shape_world=_array(np.r_[worlds, [-1, 0]], device),
    )


def _reference(worlds, edges, wake, eligible, old_roots, leased):
    count = len(worlds)
    parent = np.arange(count, dtype=np.int32)

    def root(node):
        while parent[node] != node:
            node = parent[node]
        return node

    def union(a, b):
        a, b = root(a), root(b)
        parent[max(a, b)] = min(a, b)

    for component in range(count):
        if leased[component]:
            union(component, int(old_roots[component]))
    for a, b in edges:
        if a < count and b < count:
            union(a, b)
    roots = np.array([root(i) for i in range(count)], np.int32)
    out_wake = np.zeros(count, np.int32)
    out_eligible = np.ones(count, np.int32)
    for component in range(count):
        members = roots == roots[component]
        out_eligible[component] = int(np.all(eligible[members] != 0))
        out_wake[component] = int(np.any(wake[members] != 0) or not out_eligible[component])
    return roots, out_wake, out_eligible


def _launch(owner, fixture, pairs, count, old_roots, leased, wake, eligible, *, contacts=False):
    owner.reset()
    owner.union_previous(old_roots, leased)
    if contacts:
        owner.union_contacts(count, pairs[0], pairs[1], fixture.shape_body, fixture.shape_world)
    else:
        owner.union_pairs(count, pairs, fixture.shape_body, fixture.shape_world)
    owner.finalize(wake, eligible)


def _check_graph(test, worlds, edges, *, device="cpu", old_roots=None, leased=None, wake=None, eligible=None):
    count = len(worlds)
    old_roots = np.arange(count, dtype=np.int32) if old_roots is None else np.asarray(old_roots, np.int32)
    leased = np.zeros(count, np.int32) if leased is None else np.asarray(leased, np.int32)
    wake = np.zeros(count, np.int32) if wake is None else np.asarray(wake, np.int32)
    eligible = np.ones(count, np.int32) if eligible is None else np.asarray(eligible, np.int32)
    edge_array = np.asarray(edges, np.int32).reshape(-1, 2)
    expected = _reference(worlds, edge_array, wake, eligible, old_roots, leased)
    fixture = _fixture(worlds, device)
    owner = SleepIslands(fixture.topology)
    inputs = [_array(x, device) for x in (old_roots, leased, wake, eligible)]
    device_count = _array([len(edges)], device)
    for contacts in (False, True):
        pairs = (
            (_array(edge_array[:, 0], device), _array(edge_array[:, 1], device))
            if contacts
            else _array(edge_array, device, wp.vec2i)
        )
        _launch(owner, fixture, pairs, device_count, *inputs, contacts=contacts)
        test.assertEqual(int(owner.status.numpy()[0]), 0)
        for actual, wanted in zip(
            (owner.roots, owner.component_awake, owner.component_eligible), expected, strict=True
        ):
            np.testing.assert_array_equal(actual.numpy(), wanted)
    return owner, fixture


class TestSleepIslandsCPU(unittest.TestCase):
    def test_factory_api(self):
        """Require the bounded device connectedness primitive."""
        self.assertTrue(callable(SleepIslands))

    def test_static_random_graphs_and_old_island_closure(self):
        """Match a serial union oracle without coupling through static ground."""
        rng = np.random.default_rng(60216)
        for count in (0, 1, 2, 7, 31, 65, 257):
            worlds = np.arange(count, dtype=np.int32) % 3
            edges = [(i, count) for i in range(count)]
            for _ in range(4 * count):
                a = int(rng.integers(count))
                same_world = np.flatnonzero(worlds == worlds[a])
                b = int(rng.choice(same_world))
                edges.extend(((a, b), (b, a), (a, a)))
            rng.shuffle(edges)
            _check_graph(
                self,
                worlds,
                edges,
                wake=rng.integers(0, 2, count, dtype=np.int32),
                eligible=rng.integers(0, 2, count, dtype=np.int32),
            )
        # Current edges join two old islands. A single seed must wake all four,
        # while a separate island sharing ground remains asleep.
        owner, fixture = _check_graph(
            self,
            [0] * 6,
            [(1, 2), (4, 6), (5, 6)],
            old_roots=[0, 0, 2, 2, 4, 5],
            leased=[1] * 6,
            wake=[1, 0, 0, 0, 0, 0],
        )
        np.testing.assert_array_equal(owner.component_awake.numpy(), [1, 1, 1, 1, 0, 0])
        # Reuse the prior output roots directly: reset intentionally retains
        # them until union_previous has consumed them, without another bank.
        pointers = [array.ptr for array in (owner.parent, owner.roots, owner.root_wake, owner.root_eligible)]
        owner.reset()
        owner.union_previous(owner.roots, _array([1] * 6, "cpu"))
        owner.finalize(_array([0, 0, 0, 1, 0, 0], "cpu"), _array([1] * 6, "cpu"))
        np.testing.assert_array_equal(owner.component_awake.numpy(), [1, 1, 1, 1, 0, 0])
        self.assertEqual(pointers, [a.ptr for a in (owner.parent, owner.roots, owner.root_wake, owner.root_eligible)])
        self.assertEqual(fixture.topology.component_count, 6)

    def test_invalid_inputs_fail_awake_and_readiness_is_separate(self):
        """Fail awake on malformed edges and aggregate instantaneous eligibility."""
        fixture = _fixture([0, 0, 1])
        owner = SleepIslands(fixture.topology)
        quiet = _array([0, 0, 0], "cpu")
        eligible = _array([1, 1, 1], "cpu")
        for edges, total in (([(0, 2)], 1), ([(-1, 0)], 1), ([(5, 0)], 1), ([(0, 1)], -1), ([(0, 1)], 2)):
            owner.reset()
            owner.union_pairs(
                _array([total], "cpu"), _array(edges, "cpu", wp.vec2i), fixture.shape_body, fixture.shape_world
            )
            owner.finalize(quiet, eligible)
            self.assertNotEqual(int(owner.status.numpy()[0]), 0)
            np.testing.assert_array_equal(owner.component_awake.numpy(), [1, 1, 1])
            np.testing.assert_array_equal(owner.component_eligible.numpy(), [0, 0, 0])
        # Local static world0 is not a legal endpoint for dynamic world1.
        owner.reset()
        owner.union_pairs(
            _array([1], "cpu"), _array([(2, 4)], "cpu", wp.vec2i), fixture.shape_body, fixture.shape_world
        )
        owner.finalize(quiet, eligible)
        self.assertEqual(int(owner.status.numpy()[0]), 4)
        for roots, leased in (([0, 8, 2], [1, 1, 1]), ([0, 0, 2], [0, 1, 0]), ([2, 1, 2], [1, 0, 1])):
            owner.reset()
            owner.union_previous(_array(roots, "cpu"), _array(leased, "cpu"))
            owner.finalize(quiet, eligible)
            self.assertEqual(int(owner.status.numpy()[0]), 8)
            np.testing.assert_array_equal(owner.component_awake.numpy(), [1, 1, 1])
        # Readiness veto and explicit wake are distinct root aggregates.
        _check_graph(self, [0, 0, 1], [(0, 1)], eligible=[1, 0, 1])
        fixture.topology.component_eligible.assign(np.array([0, 1, 1], np.int32))
        owner.reset()
        owner.union_pairs(
            _array([1], "cpu"), _array([(0, 1)], "cpu", wp.vec2i), fixture.shape_body, fixture.shape_world
        )
        owner.finalize(quiet, eligible)
        np.testing.assert_array_equal(owner.component_eligible.numpy(), [0, 0, 1])
        with self.assertRaises(ValueError):
            owner.finalize(_array([0], "cpu"), eligible)
        # Body IDs need not equal mechanical component IDs.
        permuted = _fixture([0, 0, 0])
        permuted.topology.body_component.assign(np.array([2, 0, 1, -1], np.int32))
        mapped = SleepIslands(permuted.topology)
        mapped.reset()
        mapped.union_contacts(
            _array([1], "cpu"),
            _array([0], "cpu"),
            _array([1], "cpu"),
            permuted.shape_body,
            permuted.shape_world,
        )
        mapped.finalize(_array([0, 0, 1], "cpu"), eligible)
        np.testing.assert_array_equal(mapped.roots.numpy(), [0, 1, 0])
        np.testing.assert_array_equal(mapped.component_awake.numpy(), [1, 0, 1])


@unittest.skipUnless(wp.is_cuda_available(), "CUDA is required")
class TestSleepIslandsCUDA(unittest.TestCase):
    def test_contended_union_and_captured_count_changes(self):
        """Exercise contested links and repeated graph input changes natively."""
        device = wp.get_device("cuda:0")
        count = 257
        rng = np.random.default_rng(16)
        edges = [(i, i + 1) for i in range(count - 1)]
        edges += [(int(rng.integers(count)), int(rng.integers(count))) for _ in range(4096)]
        rng.shuffle(edges)
        _check_graph(self, [0] * count, edges, device=device, wake=[0] * (count - 1) + [1])
        fixture = _fixture([0] * 6, device)
        owner = SleepIslands(fixture.topology)
        pairs = _array([(1, 2), (4, 6)], device, wp.vec2i)
        edge_count = _array([2], device)
        old_roots = _array([0, 0, 2, 2, 4, 5], device)
        leased = _array([1] * 6, device)
        wake = _array([1, 0, 0, 0, 0, 0], device)
        eligible = _array([1] * 6, device)
        _launch(owner, fixture, pairs, edge_count, old_roots, leased, wake, eligible)
        with wp.ScopedCapture(device=device) as capture:
            _launch(owner, fixture, pairs, edge_count, old_roots, leased, wake, eligible)
        for active in (2, 0, 2):
            edge_count.assign(np.array([active], np.int32))
            wp.capture_launch(capture.graph)
            expected = [1, 1, 1, 1, 0, 0] if active else [1, 1, 0, 0, 0, 0]
            np.testing.assert_array_equal(owner.component_awake.numpy(), expected)
            self.assertEqual(int(owner.status.numpy()[0]), 0)
        edge_count.assign(np.array([3], np.int32))
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(owner.component_awake.numpy(), [1] * 6)
        self.assertEqual(int(owner.status.numpy()[0]), 1)


if __name__ == "__main__":
    unittest.main()
