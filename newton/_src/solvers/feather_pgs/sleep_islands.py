# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Linear-storage device connectedness for experimental sleeping islands.

This module owns no quiet policy, contact cache, or solver dispatch. Each pass
resets parents, unions current edges and optional old leased membership, then
aggregates caller-provided wake/eligibility flags. Kernel boundaries provide
ordering; no device-to-host read or per-pass allocation is required.
"""

import warp as wp

from .sleep_topology import SleepTopology


@wp.func
def _root(parent: wp.array[int], node: int) -> int:
    # Union writes are atomic. Atomic reads avoid relying on cached ordinary
    # loads while another edge links a root. Parents only decrease.
    next_node = wp.atomic_add(parent, node, 0)
    while next_node != node:
        node = next_node
        next_node = wp.atomic_add(parent, node, 0)
    return node


@wp.func
def _union(parent: wp.array[int], a: int, b: int):
    done = bool(False)
    while not done:
        a = _root(parent, a)
        b = _root(parent, b)
        if a == b:
            done = True
        else:
            high = wp.max(a, b)
            low = wp.min(a, b)
            previous = wp.atomic_cas(parent, high, high, low)
            if previous == high:
                done = True
            else:
                # The competing link made progress. Retry its component
                # against the other endpoint, never overwrite that link.
                a = previous
                b = low


@wp.kernel
def _reset(
    parent: wp.array[int],
    root_wake: wp.array[int],
    root_eligible: wp.array[int],
    component_awake: wp.array[int],
    component_eligible: wp.array[int],
    status: wp.array[int],
):
    component = wp.tid()
    if component == 0:
        status[0] = 0
    if component < parent.shape[0]:
        parent[component] = component
        root_wake[component] = 0
        root_eligible[component] = 1
        component_awake[component] = 1
        component_eligible[component] = 0


@wp.func
def _shape_component(
    shape: int,
    shape_body: wp.array[int],
    body_component: wp.array[int],
    component_count: int,
) -> int:
    # -1 is an actual static endpoint; -2 denotes malformed input.
    component = int(-2)
    if shape >= 0 and shape < shape_body.shape[0]:
        body = shape_body[shape]
        if body == -1:
            component = -1
        elif body >= 0 and body < body_component.shape[0]:
            value = body_component[body]
            if value >= -1 and value < component_count:
                component = value
    return component


@wp.func
def _edge(
    shape_a: int,
    shape_b: int,
    shape_body: wp.array[int],
    shape_world: wp.array[int],
    body_component: wp.array[int],
    component_world: wp.array[int],
    parent: wp.array[int],
    status: wp.array[int],
):
    a = _shape_component(shape_a, shape_body, body_component, parent.shape[0])
    b = _shape_component(shape_b, shape_body, body_component, parent.shape[0])
    if a == -2 or b == -2:
        wp.atomic_or(status, 0, 2)
        return
    wa = shape_world[shape_a]
    wb = shape_world[shape_b]
    valid = wa >= -1 and wb >= -1
    if wa >= 0 and wb >= 0 and wa != wb:
        valid = False
    if a >= 0:
        valid = valid and component_world[a] >= 0 and wa == component_world[a]
    if b >= 0:
        valid = valid and component_world[b] >= 0 and wb == component_world[b]
    if not valid:
        wp.atomic_or(status, 0, 4)
        return
    # A common static ground/root must never join otherwise independent trees.
    if a >= 0 and b >= 0:
        _union(parent, a, b)


@wp.kernel
def _union_contacts(
    count: wp.array[int],
    shape0: wp.array[int],
    shape1: wp.array[int],
    shape_body: wp.array[int],
    shape_world: wp.array[int],
    body_component: wp.array[int],
    component_world: wp.array[int],
    parent: wp.array[int],
    status: wp.array[int],
):
    edge = wp.tid()
    total = count[0]
    if total < 0 or total > shape0.shape[0] or total > shape1.shape[0]:
        if edge == 0:
            wp.atomic_or(status, 0, 1)
        return
    if edge < total:
        _edge(shape0[edge], shape1[edge], shape_body, shape_world, body_component, component_world, parent, status)


@wp.kernel
def _union_pairs(
    count: wp.array[int],
    pairs: wp.array[wp.vec2i],
    shape_body: wp.array[int],
    shape_world: wp.array[int],
    body_component: wp.array[int],
    component_world: wp.array[int],
    parent: wp.array[int],
    status: wp.array[int],
):
    edge = wp.tid()
    total = count[0]
    if total < 0 or total > pairs.shape[0]:
        if edge == 0:
            wp.atomic_or(status, 0, 1)
        return
    if edge < total:
        pair = pairs[edge]
        _edge(pair[0], pair[1], shape_body, shape_world, body_component, component_world, parent, status)


@wp.kernel
def _union_previous(
    previous_roots: wp.array[int],
    previous_leased: wp.array[int],
    component_world: wp.array[int],
    parent: wp.array[int],
    status: wp.array[int],
):
    component = wp.tid()
    if component >= parent.shape[0]:
        return
    if previous_leased[component] == 0:
        return
    root = previous_roots[component]
    if root < 0 or root >= parent.shape[0]:
        wp.atomic_or(status, 0, 8)
        return
    if previous_roots[root] != root or previous_leased[root] == 0:
        wp.atomic_or(status, 0, 8)
        return
    if component_world[component] < 0 or component_world[component] != component_world[root]:
        wp.atomic_or(status, 0, 8)
        return
    _union(parent, component, root)


@wp.kernel
def _finalize_roots(parent: wp.array[int], roots: wp.array[int]):
    component = wp.tid()
    roots[component] = _root(parent, component)


@wp.kernel
def _aggregate(
    roots: wp.array[int],
    wake: wp.array[int],
    eligible: wp.array[int],
    topology_eligible: wp.array[int],
    component_world: wp.array[int],
    root_wake: wp.array[int],
    root_eligible: wp.array[int],
):
    component = wp.tid()
    root = roots[component]
    if wake[component] != 0:
        wp.atomic_max(root_wake, root, 1)
    if eligible[component] == 0 or topology_eligible[component] == 0 or component_world[component] < 0:
        wp.atomic_min(root_eligible, root, 0)


@wp.kernel
def _propagate(
    roots: wp.array[int],
    root_wake: wp.array[int],
    root_eligible: wp.array[int],
    status: wp.array[int],
    component_awake: wp.array[int],
    component_eligible: wp.array[int],
):
    component = wp.tid()
    root = roots[component]
    eligible = int(status[0] == 0 and root_eligible[root] != 0)
    component_eligible[component] = eligible
    component_awake[component] = int(eligible == 0 or root_wake[root] != 0)


class SleepIslands:
    """Own linear scratch for one ordered connectedness pass.

    Call ``reset``, any number of union methods, then ``finalize`` on one
    ordered stream. Inputs must stay live and immutable until those launches
    finish. ``roots`` survives reset, so it may be passed directly to
    ``union_previous`` before the next finalize; leased flags belong to the
    caller and must describe the entire old island, including its root.

    Status bits are count overflow/negative (1), invalid endpoint (2),
    incompatible world (4), and invalid old membership (8). Any error makes
    every component awake and ineligible, never silently truncates edges.
    Non-root entries in the root aggregates are not meaningful.
    """

    def __init__(self, topology: SleepTopology):
        self.topology = topology
        self.component_count = topology.component_count
        self.device = topology.body_component.device
        self.parent = wp.empty(self.component_count, dtype=wp.int32, device=self.device)
        self.roots = wp.empty_like(self.parent)
        self.root_wake = wp.empty_like(self.parent)
        self.root_eligible = wp.empty_like(self.parent)
        self.component_awake = wp.empty_like(self.parent)
        self.component_eligible = wp.empty_like(self.parent)
        self.status = wp.zeros(1, dtype=wp.int32, device=self.device)

    def reset(self) -> None:
        """Reset union scratch and fail-awake outputs, retaining prior roots."""
        wp.launch(
            _reset,
            dim=max(1, self.component_count),
            inputs=[
                self.parent,
                self.root_wake,
                self.root_eligible,
                self.component_awake,
                self.component_eligible,
                self.status,
            ],
            device=self.device,
        )

    def _check_component_array(self, array: wp.array) -> None:
        if array.ndim != 1 or array.shape[0] != self.component_count:
            raise ValueError("component arrays must match SleepTopology.component_count")

    def _edge_inputs(self, count, shape_body, shape_world):
        if count.ndim != 1 or count.shape[0] != 1:
            raise ValueError("edge count must have shape (1,)")
        if shape_body.ndim != 1 or shape_world.ndim != 1 or shape_body.shape[0] != shape_world.shape[0]:
            raise ValueError("shape_body and shape_world must be equal-length one-dimensional arrays")
        return [
            shape_body,
            shape_world,
            self.topology.body_component,
            self.topology.component_world,
            self.parent,
            self.status,
        ]

    def union_contacts(self, count, shape0, shape1, shape_body, shape_world) -> None:
        """Union all actual raw contacts, with unchanged device count bounds."""
        inputs = self._edge_inputs(count, shape_body, shape_world)
        wp.launch(
            _union_contacts,
            dim=max(1, shape0.shape[0]),
            inputs=[count, shape0, shape1, *inputs],
            device=self.device,
        )

    def union_pairs(self, count, pairs, shape_body, shape_world) -> None:
        """Union all current broad candidates without treating static as a node."""
        inputs = self._edge_inputs(count, shape_body, shape_world)
        wp.launch(
            _union_pairs,
            dim=max(1, pairs.shape[0]),
            inputs=[count, pairs, *inputs],
            device=self.device,
        )

    def union_previous(self, previous_roots, previous_leased) -> None:
        """Keep old leased islands connected while computing wake closure."""
        self._check_component_array(previous_roots)
        self._check_component_array(previous_leased)
        wp.launch(
            _union_previous,
            dim=max(1, self.component_count),
            inputs=[previous_roots, previous_leased, self.topology.component_world, self.parent, self.status],
            device=self.device,
        )

    def finalize(self, component_wake, component_eligible) -> None:
        """Aggregate current wake and instantaneous readiness over complete islands."""
        self._check_component_array(component_wake)
        self._check_component_array(component_eligible)
        if self.component_count == 0:
            return
        wp.launch(_finalize_roots, dim=self.component_count, inputs=[self.parent, self.roots], device=self.device)
        wp.launch(
            _aggregate,
            dim=self.component_count,
            inputs=[
                self.roots,
                component_wake,
                component_eligible,
                self.topology.component_eligible,
                self.topology.component_world,
                self.root_wake,
                self.root_eligible,
            ],
            device=self.device,
        )
        wp.launch(
            _propagate,
            dim=self.component_count,
            inputs=[
                self.roots,
                self.root_wake,
                self.root_eligible,
                self.status,
                self.component_awake,
                self.component_eligible,
            ],
            device=self.device,
        )
