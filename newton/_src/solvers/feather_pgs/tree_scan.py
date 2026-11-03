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
    the subtree sum. ``PATH``: a loop over a node's root path, root first,
    whose body the pass writes (Jacobian columns, inertia entries); ancestry
    is a depth-first interval test, not a parent-pointer walk.
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
    serial one: a native scan's ``#else`` branch.

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

The factor's walk (TREE_SCAN T8) is the same scan on the factor's records: a
coordinate's step and a joint's edge, root first or leaf first. Its arithmetic
is written once here (:func:`edge`, on a schedule's names for the record,
:class:`Record`), and every walk's schedule writes its steps through it.
"""

import enum
from dataclasses import dataclass

import numpy as np
import warp as wp

from . import field_group
from .device_preludes import DevicePrelude, device_branches
from .field_group import Field
from .law_source import law_preludes
from .spatial_laws import SPATIAL
from .tree_plan import TreePlan

__all__ = [
    "FETCHED",
    "INERTIA",
    "LOADED",
    "SCAN_PLAN",
    "TRANSFORM",
    "TWIST",
    "Combine",
    "Direction",
    "Group",
    "PlanAccess",
    "Record",
    "ScanPlan",
    "ScanSource",
    "Schedule",
    "Storage",
    "add",
    "braced",
    "choose_schedule",
    "cross",
    "device_plan",
    "edge",
    "scan_arrays",
    "schedule_costs",
    "schedule_features",
    "statements",
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
    dfs_position: str = "plan.dfs_position.data[{node}]"
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


SCAN_PLAN = (
    Field("parent", wp.int32, ("node",), owner="plan"),
    Field("depth", wp.int32, ("node",), owner="plan"),
    Field("dof_start", wp.int32, ("node_bound",), owner="plan"),
    Field("jump", wp.int32, ("jump",), owner="plan"),
    Field("children_offsets", wp.int32, ("node_bound",), owner="plan"),
    Field("children", wp.int32, ("edge",), owner="plan"),
    Field("level_offsets", wp.int32, ("level_bound",), owner="plan"),
    Field("level_nodes", wp.int32, ("node",), owner="plan"),
    Field("dfs_order", wp.int32, ("node",), owner="plan"),
    Field("dfs_end", wp.int32, ("node",), owner="plan"),
    Field("dfs_position", wp.int32, ("node",), owner="plan"),
    Field("chain_meta", wp.int32, ("chain_lane",), owner="plan"),
    Field("light_offsets", wp.int32, ("node_bound",), owner="plan"),
    Field("light_children", wp.int32, ("light_edge",), owner="plan"),
    Field("stage_width", wp.int32, ("chain_level",), owner="plan"),
    Field("chain_levels", wp.int32, owner="plan"),
    Field("chain_width", wp.int32, owner="plan"),
)
"""A :class:`.tree_plan.TreePlan` on the device, as every generated scan reads it (:class:`PlanAccess`): one
tree in local indices, its jump table flat by rounds and its heavy chains packed into two warps."""

ScanPlan = field_group.struct("ScanPlan", SCAN_PLAN, __name__)
"""The device plan's struct; owners that keep more per tree extend :data:`SCAN_PLAN`."""


def scan_arrays(plan: TreePlan, *, warps: int = 2) -> dict:
    """The host values of :data:`SCAN_PLAN` for ``plan``; the chain packing is empty when the chains do not fit
    ``warps``. An empty edge list holds one unused entry."""
    packing = plan.chain_packing(warps)

    def nonempty(values):
        return values if values.size else np.zeros(1, np.int32)

    return {
        "parent": plan.parent,
        "depth": plan.depth,
        "dof_start": plan.dof_start,
        "jump": plan.jump.reshape(-1),
        "children_offsets": plan.children_offsets,
        "children": nonempty(plan.children),
        "level_offsets": plan.level_offsets,
        "level_nodes": plan.level_nodes,
        "dfs_order": plan.dfs_order,
        "dfs_end": plan.dfs_end,
        "dfs_position": plan.dfs_position,
        "chain_meta": (packing.meta if packing is not None else np.full((32, 5), -1, np.int32)).reshape(-1),
        "light_offsets": plan.light_offsets,
        "light_children": nonempty(plan.light_children),
        "stage_width": packing.stage_width if packing is not None else np.zeros(1, np.int32),
        "chain_levels": packing.levels if packing is not None else 0,
        "chain_width": packing.width if packing is not None else 1,
    }


def device_plan(plan: TreePlan, device, *, warps: int = 2) -> ScanPlan:
    """Upload ``plan`` for the generated scans (:func:`scan_arrays`)."""
    return field_group.upload(ScanPlan, device, **scan_arrays(plan, warps=warps))


# ---------------------------------------------------------------------------------------------------------------------
# Schedule choice from the plan's data
# ---------------------------------------------------------------------------------------------------------------------


# Unit costs of each schedule's critical-path counts (``schedule_features``), fitted by non-negative least
# squares to tools/fpgs_bench/tree_scan_bench.py ``schedules`` on the RTX PRO 6000 (16384 trees per launch, every
# shape of the tree-scan fixture, pose, motion and both reduction widths). Only their ratios matter.
_UNIT_COSTS = {
    (Direction.DOWN, Schedule.JUMP_SHFL): {"const": 34.86, "rounds": 7.70, "channel_rounds": 0.15},
    (Direction.DOWN, Schedule.JUMP_SMEM): {"const": 13.70, "rounds": 21.97, "channel_rounds": 0.07},
    (Direction.DOWN, Schedule.HEAVY_CHAIN): {
        "const": 24.94,
        "stages": 4.91,
        "chain_rounds": 11.41,
        "channel_stage_rounds": 0.11,
        "channel_fan": 0.14,
    },
    (Direction.DOWN, Schedule.LEVEL): {"const": 74.82, "levels": 9.60, "channel_fan": 0.01},
    (Direction.UP, Schedule.DFS_INTERVAL): {"const": 43.53, "channels": 9.12},
    (Direction.UP, Schedule.HEAVY_CHAIN): {
        "const": 37.95,
        "stages": 5.88,
        "channel_stage_rounds": 0.88,
        "channel_fan": 0.60,
    },
    (Direction.UP, Schedule.LEVEL): {"const": 131.81, "levels": 8.33, "channel_fan": 0.95},
}


def schedule_features(plan: TreePlan, schedule: Schedule, channels: int) -> dict[str, int]:
    """The counts on one tree's critical path under ``schedule``, which the unit costs weigh.

    Rounds of pointer jumping, depth levels, light stages and chain rounds of the heavy chains, the widest
    fan-out a lane walks (children per level, light children per stage), each times the value's channels where
    they move it, and doubled for a tree that spans two warps.
    """
    warps = 2 if plan.count > 32 else 1
    if schedule is Schedule.JUMP_SHFL:
        return {"rounds": plan.rounds, "channel_rounds": channels * plan.rounds}
    if schedule is Schedule.JUMP_SMEM:
        return {"rounds": plan.rounds * warps, "channel_rounds": channels * plan.rounds * warps}
    if schedule is Schedule.DFS_INTERVAL:
        return {"channels": channels}
    if schedule is Schedule.LEVEL:
        fan = np.diff(plan.children_offsets)
        widest = 0
        for level in range(plan.max_depth):
            nodes = plan.level_nodes[plan.level_offsets[level] : plan.level_offsets[level + 1]]
            widest += int(fan[nodes].max()) if nodes.size else 0
        return {"levels": plan.max_depth * warps, "channel_fan": channels * widest * warps}
    if schedule is Schedule.HEAVY_CHAIN:
        packing = plan.chain_packing(2)
        warps = packing.lanes // 32
        light = np.diff(plan.light_offsets)
        stage_of = plan.chain_levels[plan.chain_of]
        fan = sum(int(light[stage_of == stage].max()) for stage in range(packing.levels))
        stage_rounds = sum(int(packing.stage_width[stage]).bit_length() - 1 for stage in range(packing.levels))
        return {
            "stages": packing.levels * warps,
            "chain_rounds": (packing.width.bit_length() - 1) * warps,
            "channel_stage_rounds": channels * stage_rounds * warps,
            "channel_fan": channels * fan * warps,
        }
    raise ValueError(f"No cost of the {schedule.value} schedule")


def schedule_costs(
    plan: TreePlan, direction: Direction, combine: Combine, *, lanes: int, chains: bool = True
) -> dict[Schedule, float]:
    """The estimated cost of each schedule the plan and the launch admit (see :func:`choose_schedule`)."""
    feasible = []
    if direction is Direction.DOWN:
        if plan.count <= min(lanes, 32):
            feasible.append(Schedule.JUMP_SHFL)
        if plan.count <= lanes:
            feasible.append(Schedule.JUMP_SMEM)
    elif plan.count <= min(lanes, 32):
        feasible.append(Schedule.DFS_INTERVAL)
    packing = plan.chain_packing(2) if chains else None
    if packing is not None and packing.lanes <= lanes:
        feasible.append(Schedule.HEAVY_CHAIN)
    feasible.append(Schedule.LEVEL)
    costs = {}
    for schedule in feasible:
        units = _UNIT_COSTS[(direction, schedule)]
        counts = schedule_features(plan, schedule, combine.channels)
        costs[schedule] = units["const"] + sum(units.get(name, 0.0) * count for name, count in counts.items())
    return costs


def choose_schedule(
    plan: TreePlan, direction: Direction, combine: Combine, *, cuda: bool, lanes: int, chains: bool = True
) -> Schedule:
    """Pick the schedule of a tree group from its plan's data alone: deterministic, never timed.

    The CPU, a path and a tree without edges run SERIAL. Otherwise each schedule the launch admits is
    costed from the plan's counts (:func:`schedule_features`) at unit costs measured once per schedule
    (``_UNIT_COSTS``), and the cheapest wins. The launch admits JUMP_SHFL and DFS_INTERVAL for a tree a warp's
    lanes cover, JUMP_SMEM for one the group's lanes cover, HEAVY_CHAIN when the heavy chains pack into the
    group's lanes (one warp or two) and the launch gives each tree its own block (``chains``: the chain scans
    address the block's threads), and LEVEL always.

    Args:
        plan: The group's template tree.
        direction: The scan's direction.
        combine: The scan's combine; its channels weigh the counts.
        cuda: Whether the group runs on a CUDA device.
        lanes: Lanes per tree of the launch (a power of two up to 32 within a warp, or 64 for two warps).
        chains: Whether the launch gives each tree its own block.
    """
    if direction not in _DIRECTIONS:
        raise ValueError(f"Unknown direction {direction}")
    if direction is Direction.UP and not combine.additive:
        raise ValueError("A leaf-to-root scan sums its values: it takes an additive combine")
    if not cuda or direction is Direction.PATH or plan.max_depth == 0:
        return Schedule.SERIAL
    costs = schedule_costs(plan, direction, combine, lanes=lanes, chains=chains)
    return min(costs, key=costs.get)


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
        return self.cuda if self.cuda == self.cpu else device_branches(self.cuda, self.cpu)


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
        {c.shuffle(up, value, "source")}
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
                const float next=__shfl_sync(MASK,values[k],source,WIDTH);
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
    # Channel c's slot, the groups in order; an idle lane addresses node 0's and never reads or writes it.
    slot = None
    for group in reversed(groups):
        offset = sum(g.channels for g in groups[: groups.index(group)])
        here = c.slot(group, "dfs_slot", f"c-{offset}" if offset else "c")
        slot = here if slot is None else f"c<{offset + group.channels}?{here}:{slot}"
    return f"""    {{
    const int dfs_node=LANE<{c.nodes}?{dfs_node}:-1;
    const int dfs_length=LANE<{c.nodes}?{dfs_end}-LANE:0;
    const int dfs_count={count};
    const int dfs_slot=dfs_node>=0?dfs_node:0;
    for(int c=0;c<{total};++c) {{
        if(c>=dfs_count)break;
        float* const cell=&({slot});
        float w[{c.windows}];
        w[0]=dfs_node>=0?*cell:0.0f;
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
        if(dfs_node>=0)*cell=sum;
    }}
    SYNC();
    }}
"""


def _path(c: _Context, visit: str) -> str:
    # The depth-first positions up to the node's, root first: an ancestor's subtree interval holds the node's
    # position. Independent loads where a parent-pointer walk waits on each.
    return f"""    const int here={c.access.dfs_position.format(node="node")};
    for(int position=0;position<=here;++position) {{
        if(here>={c.access.dfs_end.format(position="position")})continue;
        const int ancestor={c.access.dfs_node.format(position="position")};
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
    level_csr: bool | None = None,
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
        level_csr: LEVEL enumerates a depth through the plan's level CSR, or by testing every node's depth. By
            default the CSR serves a tree of more nodes than lanes; a tree the lanes cover tests depths, one
            load per lane and level instead of three dependent ones: measured 1-24% faster on the 13 trees of
            up to 32 nodes and 3-8% on 40- and 64-joint chains, where the CSR measured up to 13% faster on
            bushy trees of 44-63 nodes over two warps (RTX PRO 6000).
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
    if level_csr is None:
        level_csr = isinstance(lanes, int) and "nodes" not in sizes and plan.count > lanes
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
# The factor's walk: its arithmetic, written once
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Record:
    """How a walk's schedule names a coordinate's record and a body's lever: the column ``s`` and the row ``ut`` (six
    floats each), the inverse pivot ``1 / ell``, and the lever ``d``, the body's offset to its parent's frame."""

    s: tuple[str, ...]
    ut: tuple[str, ...]
    inverse: str
    lever: tuple[str, str, str]


LOADED = Record(
    s=tuple(f"d[{k}]" for k in range(6)),
    ut=tuple(f"d[{6 + k}]" for k in range(6)),
    inverse="d[12]",
    lever=("l[0]", "l[1]", "l[2]"),
)
"""A record loaded into registers (``WALK_LOAD_RECORD``'s ``d``, a held word's) and a link's lever ``l``."""

FETCHED = Record(
    s=("$X.s.x", "$X.s.y", "$X.s.z", "$X.s.w", "$X.t.x", "$X.t.y"),
    ut=("$X.t.z", "$X.t.w", "$X.w.x", "$X.w.y", "$X.w.z", "$X.w.w"),
    inverse="$X.e",
    lever=("$L.x", "$L.y", "$L.z"),
)
"""A record and a link the class emitter fetches ahead of the chain (``WalkIn`` as ``$X``, ``WALK_LINK_IN`` as ``$L``)."""

_SKEW = ((1, 2), (2, 0), (0, 1))
"""``(a x b)_k = a_i b_j - a_j b_i`` for ``(i, j) = _SKEW[k]``."""


def cross(a, b, k: int) -> str:
    """Component ``k`` of ``a x b``, two names of three each."""
    i, j = _SKEW[k]
    return f"{a[i]} * {b[j]} - {a[j]} * {b[i]}"


def _channels(state: str, start: int = 0) -> tuple[str, str, str]:
    return tuple(f"{state}[{start + k}]" for k in range(3))


def edge(direction: Direction, lever, state: str, *, into: str | None = None) -> list[str]:
    """The statements that pass a joint of lever ``d`` (``lever``, three names).

    ``UP``: a wrench ``(f, t)`` passes to the parent's frame as ``(f, t - d x f)``: in place on ``state``, its force
    read first, or added into the parent's ``into``. ``DOWN``: a twist ``(v, w)`` passes to the child's frame as
    ``(v + d x w, w)``: in place on ``state``, or from the parent's ``state`` into the child's ``into``.
    """
    if direction is Direction.UP:
        if into is None:
            force = ("fx", "fy", "fz")
            head = f"const float fx = {state}[0], fy = {state}[1], fz = {state}[2];"
            return [head] + [f"{state}[{3 + k}] -= {cross(lever, force, k)};" for k in range(3)]
        force = _channels(state)
        return [f"{into}[{k}] += {state}[{k}];" for k in range(3)] + [
            f"{into}[{3 + k}] += {state}[{3 + k}] - ({cross(lever, force, k)});" for k in range(3)
        ]
    if direction is Direction.DOWN:
        turn = _channels(state, 3)
        if into is None:
            return [f"{state}[{k}] += {cross(lever, turn, k)};" for k in range(3)]
        return [f"{into}[{k}] = {state}[{k}] + ({cross(lever, turn, k)});" for k in range(3)] + [
            f"{into}[{3 + k}] = {state}[{3 + k}];" for k in range(3)
        ]
    raise ValueError(f"a joint's edge runs up or down, not {direction.value}")


def statements(lines, pad: int = 0) -> str:
    """Native statements one a line, every line after the first indented by ``pad`` (the first takes its slot's)."""
    return ("\n" + " " * pad).join(lines)


def braced(lines) -> str:
    """Native statements in a block of their own, indented by four."""
    return "{\n" + "\n".join("    " + line for line in lines) + "\n}"
