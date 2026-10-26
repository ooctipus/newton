# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""One scan skeleton for every tree pass: ``tree_scan(plan, combine, direction, schedule)``.

Every dynamics pass walks a joint forest the same way: root to leaf
composing a parent's value with a node's own (poses, twists), leaf to root
summing a subtree's values (wrenches, composite inertia), or from a node up
its root path. The passes differ in three things, which are this module's
arguments; the plan (:class:`.tree_plan.TreePlan`) supplies the rest:

``combine`` (:class:`Combine`)
    The associative operator and the node value it acts on:
    :data:`TRANSFORM` (pose, ``parent * node``), :data:`TWIST` (twist and bias
    acceleration, :func:`.spatial_laws.twist_compose`), and :func:`add` of
    ``k`` channels in named groups (wrenches, the 13 inertia channels
    :data:`INERTIA`, a dense 36-channel matrix), a group optionally active
    under a runtime condition.
``direction`` (:class:`Direction`)
    ``DOWN``: every node's value becomes ``combine(parent's result, node's
    own)``, an inclusive prefix along its root path. ``UP`` (additive
    combines): every node's value becomes its own plus its children's results,
    the subtree sum. ``PATH``: a loop over a node's root path, the node first,
    whose body the pass writes (Jacobian columns, inertia entries).
``schedule`` (:class:`Schedule`)
    How the lanes of one tree's group share the work:

    ============  ======================================================  ==========
    schedule      mapping                                                 direction
    ============  ======================================================  ==========
    SERIAL        one thread, parent-first order (reversed for UP)        all
    LEVEL         lanes stride each depth level, a barrier per level      DOWN, UP
    JUMP_SMEM     a lane per node, pointer jumping through memory         DOWN
    JUMP_SHFL     a lane per node, pointer jumping through shuffles       DOWN
    HEAVY_CHAIN   shuffle scans along heavy chains, anchors per level     DOWN, UP
    DFS_INTERVAL  a lane per depth-first position, window sums            UP
    ============  ======================================================  ==========

    :func:`choose_schedule` picks one from the plan's data (sizes, depth,
    chain packing), never from a timing. The CPU path of every schedule is the
    serial one: a native scan's ``#else`` branch, and :func:`serial_scan` for Warp
    kernels (differentiable, the general route's fallback).

SERIAL and LEVEL compute every node's result from its parent's final one, so
they agree bit for bit; the jump, chain and interval schedules regroup the same
combines (an associative change of order).

A scan works in place on node values held in memory (:class:`Storage`: one
slot expression per channel group), or, on register schedules, from a source
fragment that forms a node's own value into a sink fragment that consumes its
result. Plan arrays are read through :class:`PlanAccess` expressions: the
device plan (:func:`device_plan`, :class:`ScanPlan`) or another struct with
the same arrays. Sizes are literals from the plan, or names a caller's
template fills.

Generation is text substitution only (``str.format`` on the fragments below).
"""

import enum
import functools
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from .device_preludes import DevicePrelude, dual_native, lane_group
from .law_source import law_preludes
from .spatial_laws import SPATIAL, transform_compose, twist_compose
from .tree_plan import TreePlan

__all__ = [
    "INERTIA",
    "TRANSFORM",
    "TWIST",
    "Combine",
    "Direction",
    "Group",
    "PlanAccess",
    "ScanPlan",
    "ScanSource",
    "Schedule",
    "Storage",
    "add",
    "choose_schedule",
    "device_plan",
    "serial_scan",
    "tree_scan",
]


class Direction(enum.Enum):
    """Which way a scan runs over the forest."""

    DOWN = "down"
    """Root to leaf: each node's result composes its parent's result with its own value."""
    UP = "up"
    """Leaf to root: each node's result adds its children's results to its own value."""
    PATH = "path"
    """A node's root path, the node first: a loop whose body the pass writes."""


class Schedule(enum.Enum):
    """How a tree's lanes share a scan (see the module docstring)."""

    SERIAL = "serial"
    LEVEL = "level"
    JUMP_SMEM = "jump_smem"
    JUMP_SHFL = "jump_shfl"
    HEAVY_CHAIN = "heavy_chain"
    DFS_INTERVAL = "dfs_interval"


_DIRECTIONS = {
    Direction.DOWN: (Schedule.SERIAL, Schedule.LEVEL, Schedule.JUMP_SMEM, Schedule.JUMP_SHFL, Schedule.HEAVY_CHAIN),
    Direction.UP: (Schedule.SERIAL, Schedule.LEVEL, Schedule.HEAVY_CHAIN, Schedule.DFS_INTERVAL),
    Direction.PATH: (Schedule.SERIAL,),
}


# ---------------------------------------------------------------------------------------------------------------------
# Combines
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Group:
    """A named block of a node value's channels, stored together; ``active`` a C++ condition it is summed under."""

    name: str
    channels: int
    active: str | None = None


@dataclass(frozen=True)
class Combine:
    """An associative operator on node values made of channel groups."""

    name: str
    groups: tuple[Group, ...]
    additive: bool = False
    """Channel-wise addition: commutative, the only kind UP, HEAVY_CHAIN's UP and DFS_INTERVAL accept."""
    zero_acceleration: bool = False
    """TWIST only: a node's own bias acceleration is zero (the velocity-product pass), so it is never read."""

    @property
    def channels(self) -> int:
        return sum(group.channels for group in self.groups)

    def with_zero_acceleration(self) -> "Combine":
        """This twist combine with nodes whose own bias acceleration is zero: only velocities are read."""
        if self.name != "twist":
            raise ValueError("Only the twist combine has a bias acceleration")
        return Combine(self.name, self.groups, self.additive, True)

    # --- register values ----------------------------------------------------------------------------------------
    def declare(self, var: str, identity: bool = False) -> str:
        """Declare register value ``var``; ``identity`` initializes it to the combine's identity."""
        if self.name == "transform":
            return f"wp::transform {var}" + ("=wp::transform_identity<float>();" if identity else ";")
        if self.name == "twist":
            return f"wp::spatial_vector {var}_v,{var}_a;"
        return f"float {var}[{self.channels}]" + ("={};" if identity else ";")

    def channel(self, var: str, k: int | str, group: Group | None = None) -> str:
        """The register lvalue of channel ``k`` (of ``group`` when given, else of the whole value)."""
        if self.name == "transform":
            return f"{var}[{k}]"
        if self.name == "twist":
            suffix = "v" if group is None or group.name == self.groups[0].name else "a"
            return f"{var}_{suffix}[{k}]"
        if group is None:
            return f"{var}[{k}]"
        base = self._offset(group)
        return f"{var}[{base}+{k}]" if base else f"{var}[{k}]"

    def _offset(self, group: Group) -> int:
        offset = 0
        for g in self.groups:
            if g is group or g.name == group.name:
                return offset
            offset += g.channels
        raise KeyError(group.name)

    def compose(self, var: str, parent: str) -> str:
        """C++ statement: ``var = parent combine var``."""
        if self.name == "transform":
            return f"{var}=Spatial::transform_compose({parent},{var});"
        if self.name == "twist":
            return (
                f"{{const auto composed=Spatial::twist_compose({parent}_v,{parent}_a,{var}_v,{var}_a);"
                f"{var}_v=composed.v0;{var}_a=composed.v1;}}"
            )
        return f"for(int k=0;k<{self.channels};++k){var}[k]+={parent}[k];"

    def preludes(self) -> tuple[DevicePrelude, ...]:
        """Device preludes the combine's statements call."""
        return () if self.additive else law_preludes(SPATIAL)


TRANSFORM = Combine("transform", (Group("pose", 7),))
"""Poses: ``parent * node``, 7 channels (translation, then the quaternion)."""
TWIST = Combine("twist", (Group("velocity", 6), Group("acceleration", 6)))
"""Twists with their bias accelerations: :func:`.spatial_laws.twist_compose`."""


def add(*groups: Group | tuple) -> Combine:
    """Channel-wise addition over ``groups`` (each a :class:`Group` or ``(name, channels[, active])``)."""
    built = tuple(group if isinstance(group, Group) else Group(*group) for group in groups)
    return Combine("add", built, additive=True)


INERTIA = add(Group("inertia", 13))
"""Composite inertia: mass, first moment and rotational inertia about the origin (:mod:`.spatial_laws`)."""


# ---------------------------------------------------------------------------------------------------------------------
# Storage and plan access
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Storage:
    """Where node values live during a scan: per channel group, a C++ lvalue with ``{node}`` and ``{k}``."""

    slots: dict[str, str]

    def slot(self, group: Group, node: str, k: int | str) -> str:
        return self.slots[group.name].format(node=node, k=k)


@dataclass(frozen=True)
class PlanAccess:
    """C++ expressions that read the plan's arrays; ``{...}`` fields are the indices."""

    parent: str = "plan.parent.data[{node}]"
    jump: str = "plan.jump.data[{round}*{nodes}+{node}]"
    depth: str = "plan.depth.data[{node}]"
    children_begin: str = "plan.children_offsets.data[{node}]"
    children_end: str = "plan.children_offsets.data[{node}+1]"
    child: str = "plan.children.data[{entry}]"
    level_begin: str | None = "plan.level_offsets.data[{level}]"
    level_end: str | None = "plan.level_offsets.data[{level}+1]"
    level_node: str | None = "plan.level_nodes.data[{entry}]"
    dfs_node: str = "plan.dfs_order.data[{position}]"
    dfs_end: str = "plan.dfs_end.data[{position}]"
    chain_meta: str = "plan.chain_meta.data[5*{lane}+{field}]"
    light_begin: str = "plan.light_offsets.data[{node}]"
    light_end: str = "plan.light_offsets.data[{node}+1]"
    light_child: str = "plan.light_children.data[{entry}]"
    stage_width: str = "plan.stage_width.data[{stage}]"
    chain_levels: str = "plan.chain_levels"
    chain_width: str = "plan.chain_width"

    def renamed(self, struct: str) -> "PlanAccess":
        """The same expressions on another struct variable (``plan`` -> ``struct``)."""
        values = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            values[name] = None if value is None else value.replace("plan.", f"{struct}.")
        return PlanAccess(**values)


@wp.struct
class ScanPlan:
    """A :class:`.tree_plan.TreePlan` on the device, read by every generated scan (one tree, local indices)."""

    parent: wp.array[int]
    depth: wp.array[int]
    dof_start: wp.array[int]
    jump: wp.array[int]
    children_offsets: wp.array[int]
    children: wp.array[int]
    level_offsets: wp.array[int]
    level_nodes: wp.array[int]
    dfs_order: wp.array[int]
    dfs_end: wp.array[int]
    chain_meta: wp.array[int]
    light_offsets: wp.array[int]
    light_children: wp.array[int]
    stage_width: wp.array[int]
    chain_levels: int
    chain_width: int


def device_plan(plan: TreePlan, device, *, warps: int = 2) -> ScanPlan:
    """Upload ``plan`` for the generated scans; the chain packing is empty when the chains do not fit ``warps``."""
    packing = plan.chain_packing(warps)
    result = ScanPlan()
    arrays = {
        "parent": plan.parent,
        "depth": plan.depth,
        "dof_start": plan.dof_start,
        "jump": plan.jump.reshape(-1),
        "children_offsets": plan.children_offsets,
        "children": plan.children if plan.children.size else np.zeros(1, np.int32),
        "level_offsets": plan.level_offsets,
        "level_nodes": plan.level_nodes,
        "dfs_order": plan.dfs_order,
        "dfs_end": plan.dfs_end,
        "chain_meta": (packing.meta if packing is not None else np.full((32, 5), -1, np.int32)).reshape(-1),
        "light_offsets": plan.light_offsets,
        "light_children": plan.light_children if plan.light_children.size else np.zeros(1, np.int32),
        "stage_width": packing.stage_width if packing is not None else np.zeros(1, np.int32),
    }
    for name, values in arrays.items():
        setattr(result, name, wp.array(np.ascontiguousarray(values, dtype=np.int32), dtype=int, device=device))
    result.chain_levels = packing.levels if packing is not None else 0
    result.chain_width = packing.width if packing is not None else 1
    return result


# ---------------------------------------------------------------------------------------------------------------------
# Schedule choice from the plan's data
# ---------------------------------------------------------------------------------------------------------------------


def choose_schedule(
    plan: TreePlan, direction: Direction, combine: Combine, *, cuda: bool, lanes: int, chains: bool = True
) -> Schedule:
    """Pick the schedule of a tree group from its plan's data alone (deterministic, never timed).

    Args:
        plan: The group's template tree.
        direction: The scan's direction.
        combine: The scan's combine.
        cuda: Whether the group runs on a CUDA device; the CPU runs SERIAL.
        lanes: Lanes per tree of the launch (a power of two up to 32 within a warp, or 64 for two warps).
        chains: Whether the launch gives each tree its own block, as the heavy-chain schedule addresses lanes
            by thread index.

    The rule, from the shapes measured in ``tests/test_feather_pgs_tree_scan.py``:

    * nothing to scan (no edge), a path, or the CPU: SERIAL;
    * a tree that fits one lane per node in a warp: JUMP_SHFL down (``ceil(log2(depth + 1))`` shuffle rounds,
      no barrier) and, for additive combines, DFS_INTERVAL up (window sums, no barrier);
    * otherwise, heavy chains packed into two warps (with a block per tree): HEAVY_CHAIN, whose shuffle rounds
      cover a chain and whose barriers count light levels, not depth;
    * otherwise a lane per node within the launch: JUMP_SMEM down, LEVEL up;
    * otherwise LEVEL.
    """
    if direction not in _DIRECTIONS:
        raise ValueError(f"Unknown direction {direction}")
    if direction is Direction.UP and not combine.additive:
        raise ValueError("A leaf-to-root scan sums its values: it takes an additive combine")
    if not cuda or direction is Direction.PATH or plan.max_depth == 0:
        return Schedule.SERIAL
    if plan.count <= min(lanes, 32):
        return Schedule.JUMP_SHFL if direction is Direction.DOWN else Schedule.DFS_INTERVAL
    if chains and plan.chain_packing(2) is not None:
        return Schedule.HEAVY_CHAIN
    if plan.count <= lanes:
        return Schedule.JUMP_SMEM if direction is Direction.DOWN else Schedule.LEVEL
    return Schedule.LEVEL


# ---------------------------------------------------------------------------------------------------------------------
# Native skeletons
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ScanSource:
    """A generated scan: its CUDA schedule, its CPU (serial) twin and the preludes they call."""

    cuda: str
    cpu: str
    preludes: tuple[DevicePrelude, ...] = ()
    schedule: Schedule = Schedule.SERIAL

    @property
    def text(self) -> str:
        """The scan for both devices: the schedule under the architecture guard, the serial twin otherwise."""
        if self.cuda == self.cpu:
            return self.cuda
        return "#if defined(__CUDA_ARCH__)\n" + self.cuda + "#else\n" + self.cpu + "#endif\n"


@dataclass(frozen=True)
class _Context:
    combine: Combine
    storage: Storage | None
    access: PlanAccess
    nodes: str
    rounds: str
    max_depth: str
    lanes: str
    windows: str
    order: str | None
    level_csr: bool
    source: str | None
    sink: str | None
    channel_lanes: bool = False

    def slot(self, group: Group, node: str, k: int | str) -> str:
        if self.storage is None:
            raise ValueError("This schedule reads node values from memory: give the scan a storage")
        return self.storage.slot(group, node, k)

    def load(self, var: str, node: str, *, own: bool = False) -> str:
        """Load node ``node``'s value into ``var``; ``own`` skips a zero own acceleration."""
        statements = []
        for group in self.combine.groups:
            if own and self.combine.zero_acceleration and group is self.combine.groups[1]:
                continue
            statements.append(
                f"for(int k=0;k<{group.channels};++k){self.combine.channel(var, 'k', group)}={self.slot(group, node, 'k')};"
            )
        return "".join(statements)

    def store(self, var: str, node: str) -> str:
        return "".join(
            f"for(int k=0;k<{group.channels};++k){self.slot(group, node, 'k')}={self.combine.channel(var, 'k', group)};"
            for group in self.combine.groups
        )

    def own(self, var: str, node: str) -> str:
        """Form node ``node``'s own value in ``var``: the source fragment, or a load from storage."""
        if self.source is not None:
            return self.source.format(value=var, node=node)
        return self.load(var, node, own=True)

    def result(self, var: str, node: str) -> str:
        """Consume node ``node``'s result: the sink fragment, or a store to storage."""
        if self.sink is not None:
            return self.sink.format(value=var, node=node)
        return self.store(var, node)

    def publish_own(self, node: str) -> str:
        """Memory a memory schedule must hold before its first round: a zero own acceleration."""
        if self.combine.zero_acceleration:
            group = self.combine.groups[1]
            return f"for(int k=0;k<{group.channels};++k){self.slot(group, node, 'k')}=0.0f;"
        return ""

    def shuffle(self, target: str, var: str, source: str, mask: str = "MASK", width: str = "WIDTH") -> str:
        return "".join(
            f"for(int k=0;k<{group.channels};++k){self.combine.channel(target, 'k', group)}="
            f"__shfl_sync({mask},{self.combine.channel(var, 'k', group)},{source},{width});"
            for group in self.combine.groups
        )

    def parent(self, node: str) -> str:
        return self.access.parent.format(node=node)

    def level_loop(self, level: str, body: str) -> str:
        """Lanes stride the nodes of depth ``level``: the level's CSR, or a depth test over every node."""
        if self.level_csr:
            begin = self.access.level_begin.format(level=level)
            end = self.access.level_end.format(level=level)
            node = self.access.level_node.format(entry="entry")
            return f"for(int entry={begin}+LANE;entry<{end};entry+=STRIDE){{const int node={node};\n{body}\n}}"
        depth = self.access.depth.format(node="node")
        return f"for(int node=LANE;node<{self.nodes};node+=STRIDE)if({depth}=={level}){{\n{body}\n}}"


def _serial_down(c: _Context) -> str:
    value, up = "value", "up"
    order = "index" if c.order is None else c.order.format(index="index")
    return f"""    for(int index=0;index<{c.nodes};++index) {{
        const int node={order};
        const int parent={c.parent("node")};
        if(parent<0){{{c.publish_own("node")}continue;}}
        {c.combine.declare(value)}{c.combine.declare(up)}
        {c.load(up, "parent")}
        {c.own(value, "node")}
        {c.combine.compose(value, up)}
        {c.result(value, "node")}
    }}
"""


def _serial_up(c: _Context) -> str:
    order = "index" if c.order is None else c.order.format(index="index")
    # With channel lanes, each lane owns the same channels of every node: no lane waits for another.
    first, step = ("LANE", "k+=STRIDE") if c.channel_lanes else ("0", "++k")
    adds = "".join(
        _guarded(
            group,
            f"for(int k={first};k<{group.channels};{step}){c.slot(group, 'parent', 'k')}+={c.slot(group, 'node', 'k')};",
        )
        for group in c.combine.groups
    )
    return f"""    for(int index={c.nodes}-1;index>=0;--index) {{
        const int node={order};
        const int parent={c.parent("node")};
        if(parent>=0){{{adds}}}
    }}
"""


def _guarded(group: Group, statement: str) -> str:
    return statement if group.active is None else f"if({group.active}){statement}"


def _level_down(c: _Context) -> str:
    value, up = "value", "up"
    publish = c.publish_own("node")
    prepare = f"    for(int node=LANE;node<{c.nodes};node+=STRIDE){{{publish}}}\n    SYNC();\n" if publish else ""
    body = f"""        const int parent={c.parent("node")};
        {c.combine.declare(value)}{c.combine.declare(up)}
        {c.load(up, "parent")}
        {c.own(value, "node")}
        {c.combine.compose(value, up)}
        {c.result(value, "node")}"""
    return (
        prepare
        + f"""    for(int level=1;level<={c.max_depth};++level) {{
    {c.level_loop("level", body)}
        SYNC();
    }}
"""
    )


def _level_up(c: _Context) -> str:
    begin = c.access.children_begin.format(node="node")
    end = c.access.children_end.format(node="node")
    child = c.access.child.format(entry="position")
    adds = "".join(
        _guarded(
            group, f"for(int k=0;k<{group.channels};++k){c.slot(group, 'node', 'k')}+={c.slot(group, 'child', 'k')};"
        )
        for group in c.combine.groups
    )
    body = f"""        for(int position={begin};position<{end};++position) {{
            const int child={child};
            {adds}
        }}"""
    return f"""    for(int level={c.max_depth}-1;level>=0;--level) {{
    {c.level_loop("level", body)}
        SYNC();
    }}
"""


def _jump_smem(c: _Context) -> str:
    value, up = "value", "up"
    publish = c.publish_own("node")
    jump = c.access.jump.format(round="round", nodes=c.nodes, node="node")
    prepare = f"if(node<{c.nodes}){{{c.own(value, 'node')}{publish}}}"
    return f"""    const int node=LANE;
    {c.combine.declare(value)}
    {prepare}
{"    SYNC();" + chr(10) if publish else ""}    for(int round=0;round<{c.rounds};++round) {{
        const int parent=node<{c.nodes}?{jump}:-1;
        {c.combine.declare(up)}
        if(parent>=0){{{c.load(up, "parent")}}}
        SYNC();
        if(parent>=0){c.combine.compose(value, up)}
        if(node<{c.nodes}){{{c.store(value, "node")}}}
        SYNC();
    }}
"""


def _jump_shfl(c: _Context) -> str:
    value, up = "value", "up"
    sink = c.result(value, "node")
    return f"""    const int node=LANE;
    {c.combine.declare(value, identity=True)}
    if(node<{c.nodes}){{{c.own(value, "node")}}}
    int parent=node<{c.nodes}?{c.parent("node")}:-1;
    for(int round=0;round<{c.rounds};++round) {{
        const int source=parent>=0?parent:LANE;
        {c.combine.declare(up)}
        {c.shuffle(up, value, "source")}
        const int next=__shfl_sync(MASK,parent,source,WIDTH);
        if(parent>=0){c.combine.compose(value, up)}
        parent=parent>=0?next:-1;
    }}
    if(node<{c.nodes}){{{sink}}}
{"" if c.sink is not None else "    SYNC();" + chr(10)}"""


def _chain_down(c: _Context) -> str:
    value, up = "value", "up"
    meta = c.access.chain_meta
    return f"""    const int lane=threadIdx.x,node={meta.format(lane="lane", field=0)};
    const int start={meta.format(lane="lane", field=1)},position=lane-start;
    const int level={meta.format(lane="lane", field=3)},anchor={meta.format(lane="lane", field=4)};
    {c.combine.declare(value, identity=True)}
    if(node>=0){{{c.own(value, "node")}}}
    for(int offset=1;offset<{c.access.chain_width};offset*=2) {{
        const bool combine=node>=0&&position>=offset;
        const int source=combine?((lane-offset)&31):(lane&31);
        {c.combine.declare(up)}
        {c.shuffle(up, value, "source", "0xffffffffu", "32")}
        if(combine){c.combine.compose(value, up)}
    }}
    for(int stage=0;stage<{c.access.chain_levels};++stage) {{
        if(node>=0&&level==stage) {{
            if(anchor>=0) {{
                {c.combine.declare(up)}
                {c.load(up, "anchor")}
                {c.combine.compose(value, up)}
            }}
            {c.store(value, "node")}
        }}
        __syncthreads();
    }}
"""


def _chain_up(c: _Context) -> str:
    meta = c.access.chain_meta
    light_begin = c.access.light_begin.format(node="node")
    light_end = c.access.light_end.format(node="node")
    child = c.access.light_child.format(entry="edge")
    stage_width = c.access.stage_width.format(stage="stage")

    def vector(groups) -> str:
        count = sum(group.channels for group in groups)
        loads, adds, stores = [], [], []
        offset = 0
        for group in groups:
            loads.append(
                f"#pragma unroll\n        for(int k=0;k<{group.channels};++k)values[{offset}+k]=active?{c.slot(group, 'node', 'k')}:0.0f;"
            )
            adds.append(
                f"#pragma unroll\n            for(int k=0;k<{group.channels};++k)values[{offset}+k]+={c.slot(group, 'child', 'k')};"
            )
            stores.append(
                f"#pragma unroll\n            for(int k=0;k<{group.channels};++k){c.slot(group, 'node', 'k')}=values[{offset}+k];"
            )
            offset += group.channels
        return f"""    {{
    float values[{count}];
    for(int stage={c.access.chain_levels}-1;stage>=0;--stage) {{
        const bool active=node>=0&&level==stage;
        const int width={stage_width};
        {chr(10).join(loads)}
        if(active)for(int edge=first;edge<last;++edge) {{
            const int child={child};
            {chr(10).join(adds)}
        }}
        for(int offset=1;offset<width;offset*=2) {{
            const bool combine=active&&position+offset<length;
            const int source=combine?((thread+offset)&31):(thread&31);
            #pragma unroll
            for(int k=0;k<{count};++k) {{
                const float next=__shfl_sync(0xffffffffu,values[k],source,32);
                if(combine)values[k]+=next;
            }}
        }}
        if(active) {{
            {chr(10).join(stores)}
        }}
        __syncthreads();
    }}
    }}
"""

    groups = c.combine.groups
    always = tuple(group for group in groups if group.active is None)
    guarded = [group for group in groups if group.active is not None]
    if not guarded:
        reduce = vector(groups)
    else:
        condition = "&&".join(sorted({group.active for group in guarded}))
        reduce = f"if({condition}) " + vector(groups) + " else " + vector(always)
    return f"""    {{
    const int thread=threadIdx.x,node={meta.format(lane="thread", field=0)};
    const int position=thread-{meta.format(lane="thread", field=1)};
    const int length={meta.format(lane="thread", field=2)},level={meta.format(lane="thread", field=3)};
    const int first=node>=0?{light_begin}:0;
    const int last=node>=0?{light_end}:0;
{reduce}    }}
"""


def _interval_up(c: _Context) -> str:
    dfs_node = c.access.dfs_node.format(position="LANE")
    dfs_end = c.access.dfs_end.format(position="LANE")
    groups = c.combine.groups
    total = c.combine.channels
    always = sum(group.channels for group in groups if group.active is None)
    guarded = [group for group in groups if group.active is not None]
    if guarded:
        condition = "&&".join(sorted({group.active for group in guarded}))
        count = f"({condition})?{total}:{always}"
    else:
        count = str(total)
    # Channel c's slot: the groups in order.
    slot, offset = None, 0
    for group in reversed(groups):
        offset = sum(g.channels for g in groups[: groups.index(group)])
        here = c.slot(group, "dfs_node", f"c-{offset}" if offset else "c")
        slot = here if slot is None else f"c<{offset + group.channels}?{here}:{slot}"
    first_slot = slot
    return f"""    {{
    const int dfs_node=LANE<{c.nodes}?{dfs_node}:-1;
    const int dfs_length=LANE<{c.nodes}?{dfs_end}-LANE:0;
    const int dfs_count={count};
    for(int c=0;c<{total};++c) {{
        if(c>=dfs_count)break;
        float w[{c.windows}];
        w[0]=dfs_node>=0?({first_slot}):0.0f;
        #pragma unroll
        for(int b=1;b<{c.windows};++b) {{
            const float t=__shfl_down_sync(MASK,w[b-1],1<<(b-1),WIDTH);
            w[b]=w[b-1]+(LANE+(1<<(b-1))<{c.lanes}?t:0.0f);
        }}
        float sum=0.0f;
        int offset=LANE;
        #pragma unroll
        for(int b={c.windows}-1;b>=0;--b) {{
            const float t=__shfl_sync(MASK,w[b],offset,WIDTH);
            if((dfs_length>>b)&1){{sum+=t;offset+=1<<b;}}
        }}
        if(dfs_node>=0)({first_slot})=sum;
    }}
    SYNC();
    }}
"""


def _path(c: _Context, visit: str) -> str:
    return f"""    for(int ancestor=node;ancestor>=0;ancestor={c.parent("ancestor")}) {{
{visit}
    }}
"""


_SCHEDULES = {
    (Direction.DOWN, Schedule.SERIAL): _serial_down,
    (Direction.UP, Schedule.SERIAL): _serial_up,
    (Direction.DOWN, Schedule.LEVEL): _level_down,
    (Direction.UP, Schedule.LEVEL): _level_up,
    (Direction.DOWN, Schedule.JUMP_SMEM): _jump_smem,
    (Direction.DOWN, Schedule.JUMP_SHFL): _jump_shfl,
    (Direction.DOWN, Schedule.HEAVY_CHAIN): _chain_down,
    (Direction.UP, Schedule.HEAVY_CHAIN): _chain_up,
    (Direction.UP, Schedule.DFS_INTERVAL): _interval_up,
}


def tree_scan(
    plan: TreePlan,
    combine: Combine,
    direction: Direction,
    schedule: Schedule,
    *,
    storage: Storage | None = None,
    access: PlanAccess | None = None,
    lanes: int | str = 32,
    sizes: dict[str, str] | None = None,
    level_csr: bool = True,
    channel_lanes: bool = False,
    source: str | None = None,
    sink: str | None = None,
    visit: str | None = None,
    cpu: bool = True,
) -> ScanSource:
    """Generate the native scan of ``combine`` over ``plan`` in ``direction`` on ``schedule``.

    Args:
        plan: The tree; its sizes become literals unless ``sizes`` names them.
        combine: The operator and its value's channel groups.
        direction: ``DOWN``, ``UP`` or ``PATH``.
        schedule: The CUDA schedule; the CPU twin is always SERIAL.
        storage: Slots of the node values (every schedule but a register one with a source and a sink).
        access: Plan array expressions (default: a :class:`ScanPlan` named ``plan``).
        lanes: Lanes per tree (the lane group's ``LANES``), a number or a name; the interval windows use it.
        sizes: Replacements of the plan's sizes by names: ``nodes``, ``rounds``, ``max_depth``, ``windows``.
        level_csr: LEVEL enumerates a depth through the plan's level CSR, else by testing every node's depth.
        channel_lanes: SERIAL UP only: the lane group strides an additive value's channels (one tree per group,
            nodes in order, no barrier: a lane owns the same channels of every node).
        source: A register schedule's own-value fragment (``{value}``, ``{node}``) in place of a load.
        sink: A register schedule's result fragment (``{value}``, ``{node}``) in place of a store.
        visit: PATH only: the loop body, with ``ancestor`` the visited node and ``node`` the start.
        cpu: Emit the serial CPU twin; ``False`` leaves the CPU branch empty (a CUDA-only scan).

    Returns:
        The CUDA schedule, its serial CPU twin and the preludes they call. The text uses the lane group's
        ``LANE``, ``STRIDE``, ``MASK``, ``WIDTH`` and ``SYNC`` (:func:`.device_preludes.lane_group`).
    """
    if schedule not in _DIRECTIONS[direction]:
        raise ValueError(f"A {direction.value} scan has no {schedule.value} schedule")
    if direction is Direction.UP and not combine.additive:
        raise ValueError("A leaf-to-root scan sums its values: it takes an additive combine")
    if combine.additive and (combine.zero_acceleration):
        raise ValueError("An additive combine has no bias acceleration")
    if channel_lanes and not (combine.additive and schedule is Schedule.SERIAL and direction is Direction.UP):
        raise ValueError("Lanes over channels serve the serial sum of an additive combine")
    sizes = dict(sizes or {})
    nodes = sizes.get("nodes", str(plan.count))
    if isinstance(lanes, int):
        windows = str(lanes.bit_length() if lanes <= 32 else 6)
    else:
        windows = sizes.get("windows", "WINDOWS")
    context = _Context(
        combine=combine,
        storage=storage,
        access=access or PlanAccess(),
        nodes=nodes,
        rounds=sizes.get("rounds", str(plan.rounds)),
        max_depth=sizes.get("max_depth", str(plan.max_depth)),
        lanes=str(lanes),
        windows=sizes.get("windows", windows),
        order=None if plan.parent_first else (access or PlanAccess()).dfs_node.replace("{position}", "{index}"),
        level_csr=level_csr,
        source=source,
        sink=sink,
        channel_lanes=channel_lanes,
    )
    if direction is Direction.PATH:
        if visit is None:
            raise ValueError("A path scan takes the visit of each ancestor")
        text = "{\n" + _path(context, visit) + "}\n"
        return ScanSource(text, text, (), schedule)
    cuda = "{\n" + _SCHEDULES[(direction, schedule)](context) + "}\n"
    serial = "{\n" + (_serial_down if direction is Direction.DOWN else _serial_up)(context) + "}\n" if cpu else ""
    preludes = combine.preludes()
    return ScanSource(cuda, serial, preludes, schedule)


# ---------------------------------------------------------------------------------------------------------------------
# Warp serial scans (the general route's fallback: differentiable, any size)
# ---------------------------------------------------------------------------------------------------------------------


@functools.cache
def serial_scan(combine: Combine, direction: Direction):
    """Return the Warp serial scan of ``combine`` in ``direction``: :data:`TRANSFORM`, :data:`TWIST` or an :func:`add`.

    The scan visits nodes ``start`` to ``end - 1`` (parent-first) of one tree or forest in place:
    ``parent[i]`` is the slot of node ``i``'s parent value (``-1`` at a root) and ``slot[i]`` its own, so a
    joint range scans body-indexed arrays through ``joint_parent`` and ``joint_child``. ``DOWN``: TRANSFORM
    takes ``values: array[transform]``; TWIST ``velocity`` and ``acceleration`` arrays; ADD any array whose
    values add. ``UP`` (ADD only) visits the range backwards, adding each node's value to its parent's.
    """
    if direction is Direction.UP:
        if not combine.additive:
            raise ValueError("A leaf-to-root scan sums its values: it takes an additive combine")

        @wp.func
        def scan_up(start: int, end: int, parent: wp.array[int], slot: wp.array[int], values: wp.array[Any]):
            for offset in range(end - start):
                node = end - 1 - offset
                up = parent[node]
                if up >= 0:
                    values[up] = values[up] + values[slot[node]]

        return scan_up
    if direction is not Direction.DOWN:
        raise ValueError("Warp scans run DOWN or UP; a path is the caller's loop")
    if combine.name == "transform":

        @wp.func
        def scan_transform(
            start: int, end: int, parent: wp.array[int], slot: wp.array[int], values: wp.array[wp.transform]
        ):
            for node in range(start, end):
                up = parent[node]
                if up >= 0:
                    values[slot[node]] = transform_compose(values[up], values[slot[node]])

        return scan_transform
    if combine.name == "twist":

        @wp.func
        def scan_twist(
            start: int,
            end: int,
            parent: wp.array[int],
            slot: wp.array[int],
            velocity: wp.array[wp.spatial_vector],
            acceleration: wp.array[wp.spatial_vector],
        ):
            for node in range(start, end):
                up = parent[node]
                if up >= 0:
                    own = slot[node]
                    v, a = twist_compose(velocity[up], acceleration[up], velocity[own], acceleration[own])
                    velocity[own] = v
                    acceleration[own] = a

        return scan_twist
    if combine.additive:

        @wp.func
        def scan_add(start: int, end: int, parent: wp.array[int], slot: wp.array[int], values: wp.array[Any]):
            for node in range(start, end):
                up = parent[node]
                if up >= 0:
                    values[slot[node]] = values[up] + values[slot[node]]

        return scan_add
    raise ValueError(f"Unknown combine {combine.name!r}")


def native_scan(text: ScanSource, lanes: int | str, extra: tuple[DevicePrelude, ...] = ()) -> str:
    """A ``func_native`` snippet of a scan for both devices: the lane group's and the combine's preludes, then it."""
    return dual_native(text.text, preludes=(lane_group(lanes), *text.preludes, *extra))
