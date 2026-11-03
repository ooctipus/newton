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
import functools
import hashlib
from dataclasses import dataclass
from string import Template

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
    "WALK_AHEAD",
    "WALK_ANGULAR",
    "WALK_BOTH",
    "WALK_LINEAR",
    "WALK_REGISTERS",
    "WALK_STAGED",
    "WALK_TOUCH",
    "WALK_UNROLL",
    "Combine",
    "Direction",
    "Group",
    "PlanAccess",
    "Record",
    "ScanPlan",
    "ScanSource",
    "Schedule",
    "Storage",
    "WalkClass",
    "add",
    "braced",
    "by_class",
    "channels",
    "choose_schedule",
    "cross",
    "device_plan",
    "dot",
    "edge",
    "enter",
    "held_fits",
    "indent_text",
    "inward_edge",
    "inward_step",
    "root_path",
    "scan_arrays",
    "schedule_costs",
    "schedule_features",
    "slip",
    "statements",
    "step",
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
    indexed: tuple[str, str] | None = None
    """``s`` and ``ut`` at a loop's index ``{k}``, for a record a loop reads (a state's update is then that loop)."""


LOADED = Record(
    s=tuple(f"d[{k}]" for k in range(6)),
    ut=tuple(f"d[{6 + k}]" for k in range(6)),
    inverse="d[12]",
    lever=("l[0]", "l[1]", "l[2]"),
    indexed=("d[{k}]", "d[6 + {k}]"),
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


def dot(a, b) -> str:
    """``a . b`` of six names each, summed as every walk sums it: the two halves, then their sum."""
    terms = [f"{x} * {y}" for x, y in zip(a, b, strict=True)]
    return f"(({' + '.join(terms[:3])}) + ({' + '.join(terms[3:])}))"


def channels(state: str) -> tuple[str, ...]:
    """A six-float state's names."""
    return tuple(f"{state}[{k}]" for k in range(6))


def step(
    direction: Direction,
    record: Record,
    state: str,
    *,
    source: str = "",
    value: str | None = None,
    out: str = "const float z",
    into: str | None = None,
    index: str = "k",
) -> list[str]:
    """One coordinate's step on a six-float ``state``: its output statement, then its state's update.

    ``UP`` (the whitening, leaf first): ``z = (x + s . p) / ell``, then ``p -= ut z``. ``DOWN`` (root first):
    ``q = (u - ut . V) / ell``, then ``V += s q``. The output is rounded on its own (``WALK_MUL``), so its operations do
    not depend on the block its consumer sits in. ``source`` is the step's input ``x`` or ``u`` (none: the state's
    alone); ``value`` instead names a value that already holds ``source`` and the dot, for a source written as
    statements. The output is assigned to ``out``; the update is in place, or into ``into`` (``into = state ...``), a
    loop over ``index`` for a record a loop reads (:attr:`Record.indexed`), six statements for a named one. The
    caller places its sink: after the output (before the update) or after the update.
    """
    up = direction is Direction.UP
    if direction not in (Direction.UP, Direction.DOWN):
        raise ValueError(f"a coordinate's step runs up or down, not {direction.value}")
    if value is None:
        head = dot(record.s if up else record.ut, channels(state))
        value = f"{source} {'+' if up else '-'} {head}" if source else head
    output = f"{out} = WALK_MUL(\n    {value},\n    {record.inverse});"
    return [output, *_update(up, record, state, out.rsplit(maxsplit=1)[-1], into, index)]


def enter(record: Record, state: str, velocity: str, out: str, *, index: str = "k") -> list[str]:
    """One coordinate's step of ``ENTER``, the walk down inverted (``u = L^T P qd``, root first): ``u = qd / (1 / ell)
    + ut . V`` into ``out``, then ``V += s qd``, on the coordinate's known ``velocity``."""
    head = f"{out} = {velocity} / {record.inverse} + {dot(record.ut, channels(state))};"
    return [head, *_update(False, record, state, velocity, None, index)]


def _update(up: bool, record: Record, state: str, name: str, into: str | None, index: str) -> list[str]:
    """A step's state update by its output ``name``: ``p -= ut z`` up, ``V += s q`` down (:func:`step`)."""
    sign, terms = ("-", record.ut) if up else ("+", record.s)
    if record.indexed is not None:
        term = (record.indexed[1] if up else record.indexed[0]).format(k=index)
        target = f"{into}[{index}] = {state}[{index}] {sign}" if into else f"{state}[{index}] {sign}="
        return [f"for (int {index} = 0; {index} < 6; ++{index}) {target} {term} * {name};"]
    if into:
        return [f"{into}[{k}] = {state}[{k}] {sign} {terms[k]} * {name};" for k in range(6)]
    return [f"{state}[{k}] {sign}= {terms[k]} * {name};" for k in range(6)]


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
        force = channels(state)[:3]
        return [f"{into}[{k}] += {state}[{k}];" for k in range(3)] + [
            f"{into}[{3 + k}] += {state}[{3 + k}] - ({cross(lever, force, k)});" for k in range(3)
        ]
    if direction is Direction.DOWN:
        turn = channels(state)[3:]
        if into is None:
            return [f"{state}[{k}] += {cross(lever, turn, k)};" for k in range(3)]
        return [f"{into}[{k}] = {state}[{k}] + ({cross(lever, turn, k)});" for k in range(3)] + [
            f"{into}[{3 + k}] = {state}[{3 + k}];" for k in range(3)
        ]
    raise ValueError(f"a joint's edge runs up or down, not {direction.value}")


def statements(lines, pad: int = 0) -> str:
    """Native statements one a line, every line after the first indented by ``pad`` (the first takes its slot's)."""
    return ("\n" + " " * pad).join("\n".join(lines).split("\n"))


def braced(lines) -> str:
    """Native statements in a block of their own, indented by four."""
    return "{\n" + "\n".join("    " + line for line in "\n".join(lines).split("\n")) + "\n}"


def root_path(body: str, c: str, *, count: str = "", joint=()) -> tuple[str, str]:
    """The path table's walk of a root path (``WALK_PARENT_OF``, ``WALK_FIRST_OF``), leaf first: coordinate ``c`` of
    ``body`` down to its body's first, then each ancestor's coordinates, last first. Returns the statements before and
    after a coordinate's visit (``c`` and ``body`` in scope), both loops rolled: ``joint`` passes a body to its parent
    ``up`` (the edge); ``count`` advances with ``c``."""
    head = [
        "#pragma unroll 1",
        f"for (int body = {body}, c = {c}; body >= 0;) {{",
        "    #pragma unroll 1",
        f"    for (; c >= WALK_FIRST_OF(body); --c{', ' + count if count else ''}) {{",
    ]
    tail = [
        "    }",
        "    const int up = WALK_PARENT_OF(body);",
        "    if (up >= 0) {",
        *_indented(joint, 8),
        "        c = WALK_FIRST_OF(up + 1) - 1;",
        "    }",
        "    body = up;",
        "}",
    ]
    return "\n".join(head), "\n".join(tail)


def by_class(texts: list[str]) -> str:
    """A kernel's text for each of the walking classes it serves, switched on ``cls``; the one class's text alone."""
    if len(texts) == 1:
        return texts[0]
    cases = (f"case {k}: {{\n{indent_text(text, 4)}\n    break;\n}}" for k, text in enumerate(texts))
    return "switch (cls) {\n" + "\n".join(cases) + "\n}"


def _indented(lines, pad: int) -> list[str]:
    return [" " * pad + line for line in "\n".join(lines).split("\n")] if lines else []


def slip(kind: str, value: str, carry: str, b: str, a: str, g: str, out: str) -> list[str]:
    """One step of a slip's release along its root path (``R = L_F^-1 L_K``, a rank-one recurrence on the walk's
    coordinates, :class:`.tree_factor.WalkRelease`): ``"R"`` leaf first, ``x = value - b s``, ``s += a x``, out
    ``g x``; ``"RT"`` its transpose root first, ``x = g value + a t``, ``t -= b x``, out ``x``; ``"RIT"`` the
    transpose's inverse, ``x = (value - a t) / g``, ``t -= b value``, out ``x``. ``carry`` is the recurrence's ``s``
    or ``t``; ``out`` a statement with ``{}`` for the step's output."""
    if kind == "R":
        return [f"const float x = {value} - {b} * {carry};", f"{carry} += {a} * x;", out.format(f"{g} * x")]
    if kind == "RT":
        return [f"const float x = {g} * {value} + {a} * {carry};", f"{carry} -= {b} * x;", out.format("x")]
    if kind == "RIT":
        return [f"const float x = ({value} - {a} * {carry}) / {g};", f"{carry} -= {b} * {value};", out.format("x")]
    raise ValueError(f"a slip's step is R, RT or RIT, not {kind}")


# ``INWARD``, the refresh: the same walk leaf first on a body's articulated inertia ``h``, 21 floats, symmetric
# (``A`` 00 01 02 11 12 22, ``B`` row-major 3x3, ``D`` 00 01 02 11 12 22; ``SYM(a, e)`` a symmetric block's index).

_INWARD_STEP = Template("""const float* A = $h;
const float* B = $h + 6;
const float* D = $h + 15;
float Is[6];
for (int a = 0; a < 3; ++a) {
    Is[a] = (A[SYM(a, 0)] * $s[0] + A[SYM(a, 1)] * $s[1] + A[SYM(a, 2)] * $s[2])
            + (B[3 * a] * $s[3] + B[3 * a + 1] * $s[4] + B[3 * a + 2] * $s[5]);
    Is[3 + a] = (B[a] * $s[0] + B[3 + a] * $s[1] + B[6 + a] * $s[2])
                + (D[SYM(a, 0)] * $s[3] + D[SYM(a, 1)] * $s[4] + D[SYM(a, 2)] * $s[5]);
}
const float own = $own;
#if defined(__CUDA_ARCH__)
const float inv = rsqrtf(own + $s[7]);
#else
const float inv = 1.0f / sqrtf(own + $s[7]);
#endif
float ut[6];
for (int k = 0; k < 6; ++k) ut[k] = Is[k] * inv;""")

_INWARD_UPDATE = Template("""// $h -= ut ut^T
for (int a = 0; a < 3; ++a)
    for (int e = a; e < 3; ++e) {
        $h[SYM(a, e)] -= ut[a] * ut[e];
        $h[15 + SYM(a, e)] -= ut[3 + a] * ut[3 + e];
    }
for (int a = 0; a < 3; ++a)
    for (int e = 0; e < 3; ++e) $h[6 + 3 * a + e] -= ut[a] * ut[3 + e];""")

_INWARD_EDGE = Template("""// B' = B + A X, D' = D - X B' + B^T X (its symmetric part).
const float X[3][3] = $skew;
float Bn[9];
for (int a = 0; a < 3; ++a)
    for (int e = 0; e < 3; ++e)
        Bn[3 * a + e] = $h[6 + 3 * a + e] + ($h[SYM(a, 0)] * X[0][e] + $h[SYM(a, 1)] * X[1][e] + $h[SYM(a, 2)] * X[2][e]);
for (int a = 0; a < 3; ++a)
    for (int e = a; e < 3; ++e) {
        const float xb = X[a][0] * Bn[e] + X[a][1] * Bn[3 + e] + X[a][2] * Bn[6 + e];
        const float bx = $h[6 + a] * X[0][e] + $h[9 + a] * X[1][e] + $h[12 + a] * X[2][e];
        $h[15 + SYM(a, e)] = ($h[15 + SYM(a, e)] - xb) + bx;
    }
for (int k = 0; k < 9; ++k) $h[6 + k] = Bn[k];""")


def inward_step(state: str, column: str) -> tuple[str, str]:
    """``INWARD``'s step on one coordinate: the inertia ``state`` eliminates the coordinate's ``column`` (``s`` 6, then
    ``R`` and ``R + K``): ``Is = h s``, ``own = s . Is`` (:func:`dot`), ``1 / ell = (own + R + K)^-1/2`` (``inv``),
    ``ut = Is / ell``; then ``h -= ut ut^T``. Returns the statements before the caller's sink (the record, from
    ``own``, ``inv`` and ``ut``) and the update after it."""
    own = dot(tuple(f"Is[{k}]" for k in range(6)), channels(column))
    return _INWARD_STEP.substitute(h=state, s=column, own=own), _INWARD_UPDATE.substitute(h=state)


def inward_edge(lever, state: str) -> str:
    """``INWARD``'s edge: the inertia ``state`` passes a joint of lever ``d`` (``lever``, three names) to the parent's
    frame, ``[A, A X + B; ., D - X (A X + B) + B^T X]``, ``X`` the skew of ``d`` (:func:`cross`'s pattern), in place."""
    rows = []
    for i, j in _SKEW:
        row = ["0.0f"] * 3
        row[j], row[i] = lever[i], f"-{lever[j]}"
        rows.append("{" + ", ".join(row) + "}")
    return _INWARD_EDGE.substitute(h=state, skew="{" + ", ".join(rows) + "}")


WALK_STAGED = 13
"""The floats of a coordinate's record the walk's functions read (``s``, ``ut``, ``1 / ell``), unpacked
(:data:`.tree_factor._WALK_LOADS`): what a kernel stages of a record in shared memory (:class:`.response_operator.LocalBlock`)."""

WALK_LINEAR, WALK_ANGULAR, WALK_BOTH = 0, 1, 2
"""A column's half at its body's frame: the linear one, the angular one, or both (the linear in the record's side)."""


# ---------------------------------------------------------------------------------------------------------------------
# The class emitter (ZB, T8.8): a walking class's schedules, compiled (straight-line or the loop form) and held
# ---------------------------------------------------------------------------------------------------------------------

WALK_TOUCH = 16
"""Items a walk asks L1 for ahead of its chain (:meth:`WalkClass.touch`): a hint without a register or a scoreboard,
so the loads :data:`WALK_AHEAD` items ahead find their lines in L1."""

WALK_REGISTERS = 255
"""Registers a thread holds at most (the ISA's line): what a kernel without a launch bound may spend holding a walk's
items for its two passes (:meth:`WalkClass.walk_twice`)."""

WALK_UNROLL = WALK_REGISTERS // 4
"""Coordinates of a class up to which :meth:`WalkClass.walk` emits its walk as straight-line code; a larger class takes
the loop form. From the registers: a straight-line walk holds a class's states and values as named registers, about
three a coordinate with its loads ahead (g1's 43-coordinate walk solve 141 registers, no stack), so up to a quarter of
:data:`WALK_REGISTERS` coordinates fit with room for the chain; past it (chain_106's 112) the loop form's ring keeps
:data:`WALK_AHEAD` items in flight. The loop form indexes its states on the stack (g1's walk solve: 272 B, 41 local
loads and 43 stores): straight-line, g1's walk solve runs 78.0 -> 58.1 us a launch and its leave 47.1 -> 40.0 (RTX PRO
6000, 16,384 worlds; ZP5, ZV). An earlier bench (ZE: the loop form faster on g1, 41.6 against 51.9 us) predates the
record's vector loads and the touches."""


def held_fits(words: int, registers: int) -> bool:
    """Whether a thread may hold ``words`` floats of a walk's data across the walk: they fit ``registers``, the kernel's
    budget a thread (:data:`WALK_REGISTERS` without a launch bound). The one rule by which a walk solve holds its
    class's items for both walks (:meth:`WalkClass.walk_twice`) and the support writer holds a unit's side for the
    slips' release (:data:`.walk_rows._RELEASE_HELD`)."""
    return words <= registers


WALK_AHEAD = 4
"""Items (a body's joint input, a coordinate's step input) a walk keeps in flight ahead of its serial chain
(:meth:`WalkClass.ahead`): each step of the chain then waits on no load, and a lane holds this many items' inputs in
registers."""


@functools.cache
def _walk_slots(parent: tuple[int, ...]) -> tuple[tuple[int, ...], int]:
    """Each body's slot for its state where a child other than the next body meets it (a join), else -1, and the slot
    count. A join's slot is live over its bodies from the join to its last child, in either direction of the walk, so
    joins whose spans do not overlap share a slot."""
    last = {}
    for body, up in enumerate(parent):
        if up >= 0:
            last[up] = body
    joins = {up for body, up in enumerate(parent) if up >= 0 and up != body - 1}
    slot, free, held, count = [-1] * len(parent), [], {}, 0
    for body in range(len(parent)):
        for join in [j for j, child in held.items() if child == body]:
            free.append(slot[join])
            del held[join]
        if body in joins:
            if free:
                slot[body] = free.pop()
            else:
                slot[body], count = count, count + 1
            held[body] = last[body]
    return tuple(slot), count


def _fill(text: str, **values) -> str:
    """``text`` with its ``$`` slots filled from ``values``: a walk text names only the slots it reads."""
    template = Template(text)
    return template.substitute({name: str(values[name]) for name in template.get_identifiers()})


def indent_text(text: str, spaces: int) -> str:
    """``text`` with every non-empty line indented by ``spaces``."""
    pad = " " * spaces
    return "\n".join(pad + line if line.strip() else line for line in text.splitlines())


@dataclass(frozen=True)
class WalkClass:
    """A walking class: the topology its trees share, each body's parent (-1: the world, every parent before its
    children) and first coordinate, with the count after the last (:func:`.tree_factor.walk_prelude`'s table).

    :meth:`walk` is the class emitter: it turns the table into a walk's native statements, root first (the leave, a
    predictor's solve down) or leaf first (an inward pass, a solve up), so a kernel built for the class walks its tree
    with no table in front of its steps. The walk's own arithmetic is the caller's text."""

    parent: tuple[int, ...]
    first: tuple[int, ...]
    half: tuple[int, ...] = ()
    """Each coordinate's column half (:data:`WALK_LINEAR`, :data:`WALK_ANGULAR`, or :data:`WALK_BOTH`, read from the
    record), compiled into the class's walk; empty: every coordinate's read from its record."""

    def __post_init__(self):
        if len(self.first) != len(self.parent) + 1 or any(p >= b for b, p in enumerate(self.parent)):
            raise ValueError("a walk's class needs every parent before its children and a first coordinate per body")
        if self.half and len(self.half) != self.first[-1]:
            raise ValueError("a walk's class names a half for every coordinate or for none")

    def root_path(self, c: int) -> list[int]:
        """Coordinate ``c``'s root path, leaf first: ``c`` down to its body's first, then each ancestor's coordinates
        last first (the release's path, :class:`.tree_factor.WalkRelease`): :func:`root_path` with the table compiled."""
        body = max(b for b in range(len(self.parent)) if self.first[b] <= c < self.first[b + 1])
        path = []
        while body >= 0:
            path += list(range(c, self.first[body] - 1, -1))
            body = self.parent[body]
            c = self.first[body + 1] - 1 if body >= 0 else -1
        return path

    def half_of(self, c: int) -> int:
        """Coordinate ``c``'s column half as the class's walk compiles it (:data:`WALK_BOTH`: read from the record)."""
        return self.half[c] if self.half else WALK_BOTH

    @property
    def coordinates(self) -> int:
        return int(self.first[-1])

    @property
    def slots(self) -> tuple[tuple[int, ...], int]:
        """Each body's join slot and the slot count (:func:`_walk_slots`): the loop form's and the refresh's."""
        return _walk_slots(self.parent)

    @property
    def tag(self) -> str:
        """A short name of the table, for the class's emitted names and its kernels' names."""
        return hashlib.sha256(repr((self.parent, self.first, self.half)).encode()).hexdigest()[:8]

    def walk(
        self,
        joint: str,
        step: str,
        *,
        begin: str = "",
        end: str = "",
        root_first: bool = True,
        width: int = 6,
        cuda: bool = True,
        unroll: int = WALK_UNROLL,
        fetch_joint: str = "",
        fetch_step: str = "",
        touch_joint: str = "",
        touch_step: str = "",
        ahead: int | None = None,
    ) -> str:
        """Native statements that walk the class's tree on a state of ``width`` floats a body, in either direction.

        Root first, a body's state ``$V`` takes its parent's ``$P`` through ``joint`` (a root's starts at zero), then
        ``begin``, its coordinates in order (``step``) and ``end``. Leaf first, a body's state starts at zero and
        gathers what its children pass through ``joint`` (written into ``$P``, the parent's), then ``begin``, its
        coordinates last first and ``end``; then it passes to its parent.

        **Loads ahead of the chain.** A walk's inputs do not depend on its chain: ``fetch_joint`` (body ``$b``'s, an
        expression) and ``fetch_step`` (coordinate ``$c``'s) are evaluated :meth:`ahead` items before the joint (as
        ``$L``) and the step (as ``$X``) that read them, in the walk's order, so each load is in flight while the chain
        computes the items before it, and no store of the chain stands between a load and its issue.

        Up to ``unroll`` coordinates the walk is straight-line code: each body's state an array of constant indices
        (registers), each fetched value a named register. Past it, the loop form over the class's table in constant
        memory: a chain's state in registers, a join's in a slot (:func:`_walk_slots`), the fetched values in two
        register rings (bodies and coordinates), each :meth:`ahead` deep.

        Args:
            joint: Body ``$b``'s joint to its parent: ``$V`` from ``$P`` root first, ``$V`` into ``$P`` leaf first.
            step: Coordinate ``$c`` of body ``$b`` on its state ``$V``.
            begin: Body ``$b``'s statements before its coordinates, on ``$V`` (``$up`` its parent, -1 at a root).
            end: Body ``$b``'s statements after its coordinates, on ``$V``.
            root_first: The walk's direction.
            width: Floats of a body's state.
            cuda: The CUDA text (its tables in constant memory) or its CPU twin.
            unroll: Coordinates up to which the walk is straight-line code (:data:`WALK_UNROLL`).
            fetch_joint: Body ``$b``'s joint inputs (an expression; ``$up`` its parent), read by ``joint`` as ``$L``; none
                if empty.
            fetch_step: Coordinate ``$c``'s step inputs (an expression; ``$half`` its column half, :meth:`half_of`, a
                constant in straight-line code, :data:`WALK_BOTH` in the loop form), read by ``step`` as ``$X``; none
                if empty.
            touch_joint: Body ``$b``'s statement asking L1 for its joint inputs' lines (a hint, no register).
            touch_step: Coordinate ``$c``'s statement asking L1 for its step inputs' lines.
            ahead: Items whose inputs a lane holds in registers ahead of the chain (:data:`WALK_AHEAD` by default; a
                kernel whose launch bound has no room for them takes 0 and keeps the touches).
        """
        bodies, coordinates = len(self.parent), self.coordinates
        zero = "{" + ", ".join(["0.0f"] * width) + "}"
        bodies_order = self.bodies_in(root_first)

        def coordinates_of(b):
            return self.coordinates_in(b, root_first)

        if coordinates <= unroll:
            # The fetched items in the walk's order: a body's joint input where it has a parent (root first before its
            # coordinates, leaf first after them), each coordinate's step input.
            order = []
            for b in bodies_order:
                if root_first and (fetch_joint or touch_joint) and self.parent[b] >= 0:
                    order.append(("l", b))
                if fetch_step or touch_step:
                    order += [("x", c) for c in coordinates_of(b)]
                if not root_first and (fetch_joint or touch_joint) and self.parent[b] >= 0:
                    order.append(("l", b))
            depth = self.ahead(len(order), ahead) if fetch_joint or fetch_step else 0
            early = self.touch(len(order)) if touch_joint or touch_step else 0
            position = {item: k for k, item in enumerate(order)}

            def fetch(item):
                kind, index = item
                text = (
                    _fill(fetch_joint, b=index, up=self.parent[index])
                    if kind == "l"
                    else _fill(fetch_step, c=index, half=self.half_of(index))
                )
                return [f"const auto walk_{kind}{index} = {text};"] if text else []

            def touched(item):
                kind, index = item
                text = _fill(touch_joint, b=index, up=self.parent[index]) if kind == "l" else _fill(touch_step, c=index)
                return [text] if text else []

            def take(item):
                k = position.get(item, -1)
                if k < 0:
                    return []
                out = touched(order[k + early]) if k + early < len(order) else []
                return out + (fetch(order[k + depth]) if k + depth < len(order) else [])

            def body_text(b, state):
                out = [_fill(begin, b=b, V=state, up=self.parent[b])] if begin else []
                for c in coordinates_of(b):
                    out += take(("x", c))
                    out.append(_fill(step, b=b, c=c, V=state, X=f"walk_x{c}"))
                out += [_fill(end, b=b, V=state)] if end else []
                return out

            lines = [line for item in order[:early] for line in touched(item)]
            lines += [line for item in order[:depth] for line in fetch(item)]
            if root_first:
                for b, up in enumerate(self.parent):
                    state = f"walk{b}"
                    if up < 0:
                        lines.append(f"float {state}[{width}] = {zero};")
                    else:
                        lines += [f"float {state}[{width}];", *take(("l", b))]
                        lines.append(_fill(joint, b=b, V=state, P=f"walk{up}", L=f"walk_l{b}"))
                    lines += body_text(b, state)
            else:
                lines.append(f"float {', '.join(f'walk{b}[{width}] = {zero}' for b in range(bodies))};")
                for b in bodies_order:
                    up = self.parent[b]
                    lines += body_text(b, f"walk{b}")
                    if up >= 0:
                        lines += take(("l", b))
                        lines.append(_fill(joint, b=b, V=f"walk{b}", P=f"walk{up}", L=f"walk_l{b}"))
            return "\n".join(lines)

        # The loop form: the walk's steps as a sequence of slots in the walk's order, a slot per coordinate and one
        # for a body without coordinates, each naming its body, its coordinate (-1 for none) and whether it is the
        # first (the body's joint before it root first, its gathering leaf first) or the last of its body's (the
        # body's end after it, and leaf first its joint). The fetched inputs sit in a register ring of `ahead`
        # slots: the loop body is that many copies of the slot's text, so copy j reads ring slot j and refills it
        # with the inputs `ahead` slots on, which no later copy reads before the next pass: each load has the whole
        # pass to arrive, and no copy of an arriving value stands in its way.
        slots, count = _walk_slots(self.parent)
        sequence = []
        for b in bodies_order:
            span = list(coordinates_of(b)) or [-1]
            sequence += [(b, c, k == 0, k == len(span) - 1) for k, c in enumerate(span)]
        steps = len(sequence)
        depth = self.ahead(steps, ahead) if fetch_joint or fetch_step else 0
        memory = "__constant__" if cuda else "const"
        name = f"WALK_CLASS_{self.tag}"
        flags = [b | (int(first) << 16) | (int(last) << 17) for b, _, first, last in sequence]

        def table(suffix, values):
            return f"static {memory} int {name}_{suffix}[{len(values)}] = {{{', '.join(map(str, values))}}};"

        tables = [
            table("PARENT", self.parent),
            table("STEP", flags),
            table("COORDINATE", [c for _, c, _, _ in sequence]),
        ]
        # The class's column halves (half_of): a table where none has both, so the walk's fetch takes no side; else
        # each record's own.
        one_half = bool(self.half) and max(self.half) < WALK_BOTH
        if one_half:
            tables.append(table("HALF", list(self.half)))
        if count:
            tables.append(table("SLOT", slots))
        each = "for (int k = 0; k < $width; ++k)"
        slot = f"S[{name}_SLOT[$at]]"

        def fetched(j, at):
            out = []
            if fetch_joint:
                body = f"({name}_STEP[{at}] & 0xffff)"
                out.append(f"walk_l{j} = {_fill(fetch_joint, b=body, up=f'{name}_PARENT[{body}]')};")
            if fetch_step:
                coordinate = f"(({name}_COORDINATE[{at}]) < 0 ? 0 : {name}_COORDINATE[{at}])"
                out.append(
                    f"walk_x{j} = {_fill(fetch_step, c=coordinate, half=f'({name}_HALF[{coordinate}] & 1)' if one_half else WALK_BOTH)};"
                )
            return out

        ring = [f"auto {line}" for j in range(depth) for line in fetched(j, min(j, steps - 1))]
        early = self.touch(steps) if touch_joint or touch_step else 0

        def touched(at):
            out = []
            if touch_joint:
                body = f"({name}_STEP[{at}] & 0xffff)"
                out.append(_fill(touch_joint, b=body, up=f"{name}_PARENT[{body}]"))
            if touch_step:
                out.append(_fill(touch_step, c=f"(({name}_COORDINATE[{at}]) < 0 ? 0 : {name}_COORDINATE[{at}])"))
            return out

        if early:
            ring.append(f"for (int at = 0; at < {early}; ++at) {{\n{indent_text(chr(10).join(touched('at')), 4)}\n}}")
        if root_first:
            parent_state = f"{each} P[k] = V[k];"
            if count:
                parent_state = (
                    f"if (up == b - 1) {{\n    {each} P[k] = V[k];\n}} else {{\n"
                    f"    {each} P[k] = {_fill(slot, at='up')}[k];\n}}"
                )
            state = f"float V[$width], P[$width];{f' float S[{count}][$width];' if count else ''}"
        else:
            gather = f"{each} {{\n    V[k] = N[k];\n    N[k] = 0.0f;\n}}"
            if count:
                gather = (
                    f"if ({name}_SLOT[b] >= 0) {{\n    {each} {{\n        V[k] = {_fill(slot, at='b')}[k];\n"
                    f"        {_fill(slot, at='b')}[k] = 0.0f;\n    }}\n}} else {{\n{indent_text(gather, 4)}\n}}"
                )
            state = f"float V[$width], N[$width] = {zero};{f' float S[{count}][$width] = {{}};' if count else ''}"

        def copy(j):
            """The text of slot `at` with ring slot j (or the inputs loaded in place, without a ring)."""
            lines = [f"const int b = {name}_STEP[at] & 0xffff, c = {name}_COORDINATE[at];"]
            names = {"L": f"walk_l{j}", "X": f"walk_x{j}"}
            if not depth:
                names = {"L": "walk_l", "X": "walk_x"}
                if fetch_joint:
                    lines.append(f"const auto walk_l = {_fill(fetch_joint, b='b', up=f'{name}_PARENT[b]')};")
                if fetch_step:
                    half = f"({name}_HALF[c < 0 ? 0 : c] & 1)" if one_half else WALK_BOTH
                    lines.append(f"const auto walk_x = {_fill(fetch_step, c='(c < 0 ? 0 : c)', half=half)};")
            if root_first:
                head = [f"const int up = {name}_PARENT[b];", f"if (up < 0) {{\n    {each} V[k] = 0.0f;\n}} else {{"]
                head += [indent_text(parent_state, 4), indent_text(_fill(joint, b="b", V="V", P="P", **names), 4), "}"]
            else:
                head = [gather]
            if begin:
                head.append(_fill(begin, b="b", V="V", up=f"{name}_PARENT[b]"))
            lines.append(f"if (({name}_STEP[at] >> 16) & 1) {{\n{indent_text(chr(10).join(head), 4)}\n}}")
            lines.append(f"if (c >= 0) {{\n{indent_text(_fill(step, b='b', c='c', V='V', **names), 4)}\n}}")
            tail = [_fill(end, b="b", V="V")] if end else []
            if root_first and count:
                tail.append(f"if ({name}_SLOT[b] >= 0)\n    {each} {_fill(slot, at='b')}[k] = V[k];")
            if not root_first:
                give = _fill(joint, b="b", V="V", P="N", **names)
                if count:
                    into = _fill(joint, b="b", V="V", P=_fill(slot, at="up"), **names)
                    give = (
                        f"if ({name}_SLOT[up] >= 0) {{\n{indent_text(into, 4)}\n}} else {{\n{indent_text(give, 4)}\n}}"
                    )
                tail.append(f"const int up = {name}_PARENT[b];\nif (up >= 0) {{\n{indent_text(give, 4)}\n}}")
            if tail:
                lines.append(f"if (({name}_STEP[at] >> 17) & 1) {{\n{indent_text(chr(10).join(tail), 4)}\n}}")
            if depth:
                lines += fetched(j, f"(at + {depth} < {steps} ? at + {depth} : {steps - 1})")
            if early:
                lines.append(
                    f"if (at + {early} < {steps}) {{\n{indent_text(chr(10).join(touched(f'at + {early}')), 4)}\n}}"
                )
            return "\n".join(lines)

        width_of_pass = max(depth, 1)
        copies = "\n".join(
            f"if (at0 + {j} < {steps}) {{\n    const int at = at0 + {j};\n{indent_text(copy(j), 4)}\n}}"
            for j in range(width_of_pass)
        )
        loop = f"""{state}
{chr(10).join(ring)}
for (int at0 = 0; at0 < {steps}; at0 += {width_of_pass}) {{
{indent_text(copies, 4)}
}}"""
        return "\n".join(tables) + "\n" + Template(loop).substitute(width=width)

    def bodies_in(self, root_first: bool) -> range:
        """The class's bodies in a walk's order: parents first (root first), or children first (leaf first)."""
        bodies = len(self.parent)
        return range(bodies) if root_first else range(bodies - 1, -1, -1)

    def coordinates_in(self, b: int, root_first: bool):
        """Body ``b``'s coordinates in a walk's order."""
        span = range(self.first[b], self.first[b + 1])
        return span if root_first else reversed(span)

    def held(self, x: str, u: str, state: str, words: int) -> tuple[str, str]:
        """The ``HELD`` schedule: native statements of the decided step's two held walks (CUDA, the border's held
        ``h`` in scope, :data:`.tree_kinetic_state._HELD`): the walk up of ``x`` into ``u`` (``u = L^-1 P x``) and the
        walk down of ``u`` in place (``P^T L^-T u``), on every lane of the segment in lockstep, straight-line in the
        class's order, no loop and no pick of a parent or a first coordinate on the chain. Each coordinate's step runs
        on every lane from the lane's own held word of the records (the coordinate's lane: its own record), and the
        walk's six-float state is then read from the coordinate's lane (six shuffles, where reading the record took
        thirteen); each body's link by shuffle from the lane that holds it (``HOLD_LINK`` at a constant). A body's
        state passes to a parent that is not the body before it through ``state`` (six words a body, lane 0 writes),
        the walk down's twists through the same words (every lane writes them, as each body's twist out); a body no
        other child joins adds the zero state as ``+ 0.0f``, the walk up's addition of an untouched slot. Lane 0 writes
        ``u`` (both walks). ``words``: the words a lane holds (:func:`.tree_kinetic_state._held_words`)."""
        parent = self.parent
        edge_up = "".join(edge(Direction.UP, LOADED.lever, "p"))
        edge_down = "".join(edge(Direction.DOWN, LOADED.lever, "V"))
        up, joined = [], set()
        up.append("{\n    float p[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};")
        for b in self.bodies_in(False):
            if b in joined:
                up.append(f"    for (int k = 0; k < 6; ++k) p[k] += {state}[{6 * b} + k];")
            else:
                up.append("    for (int k = 0; k < 6; ++k) p[k] += 0.0f;")
            for c in self.coordinates_in(b, False):
                output, update = step(Direction.UP, LOADED, "p", source=f"{x}[{c}]", into="q")
                up.append(
                    f"    {{const float* d=h.r[{c}/WIDTH];{output}float q[6];{update}if(lane=={c}%WIDTH){u}[{c}]=z;"
                    f"for(int k=0;k<6;++k)p[k]=READ(q[k],{c}%WIDTH);}}"
                )
            parent_b = parent[b]
            if parent_b >= 0:
                up.append(f"    {{float l[3];HOLD_LINK(h,{b},l);{edge_up}}}")
            if parent_b != b - 1:
                if parent_b >= 0:
                    if parent_b in joined:
                        up.append(f"    if(lane==0)for(int k=0;k<6;++k){state}[{6 * parent_b}+k]+=p[k];")
                    else:
                        up.append(f"    if(lane==0)for(int k=0;k<6;++k){state}[{6 * parent_b}+k]=0.0f+p[k];")
                        joined.add(parent_b)
                    up.append("    WARP_SYNC();")
                if b > 0:
                    up.append("    for (int k = 0; k < 6; ++k) p[k] = 0.0f;")
        up.append("    WARP_SYNC();\n}")
        down = [
            f"{{\n    float uh[{words}];",
            "    #pragma unroll",
            f"    for (int k = 0; k < {words}; ++k) uh[k] = lane + k * WIDTH < {self.coordinates} ? {u}[lane + k * WIDTH] : 0.0f;",
            "    WARP_SYNC();",
            "    float V[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};",
        ]
        for b in self.bodies_in(True):
            parent_b = parent[b]
            if parent_b < 0:
                down.append("    for (int k = 0; k < 6; ++k) V[k] = 0.0f;")
            else:
                if parent_b != b - 1:
                    down.append(f"    for (int k = 0; k < 6; ++k) V[k] = {state}[{6 * parent_b} + k];")
                down.append(f"    {{float l[3];HOLD_LINK(h,{b},l);{edge_down}}}")
            for c in self.coordinates_in(b, True):
                output, update = step(
                    Direction.DOWN, LOADED, "V", source=f"uh[{c}/WIDTH]", out="const float q", into="W"
                )
                down.append(
                    f"    {{const float* d=h.r[{c}/WIDTH];{output}float W[6];{update}if(lane=={c}%WIDTH){u}[{c}]=q;"
                    f"for(int k=0;k<6;++k)V[k]=READ(W[k],{c}%WIDTH);}}"
                )
            down.append(f"    for (int k = 0; k < 6; ++k) {state}[{6 * b} + k] = V[k];")
        down.append("    WARP_SYNC();\n}")
        return "\n".join(up), "\n".join(down)

    def touch(self, items: int) -> int:
        """How many items ahead of its chain a walk asks L1 for an item's lines: :data:`WALK_TOUCH`, at most the
        walk's ``items``."""
        return max(0, min(WALK_TOUCH, items))

    def ahead(self, items: int, ahead: int | None = None) -> int:
        """How many fetched items a walk keeps in flight ahead of its chain: ``ahead`` (:data:`WALK_AHEAD` by default),
        at most the walk's ``items`` (a short tree's every load issues before its first step)."""
        return max(0, min(WALK_AHEAD if ahead is None else ahead, items))

    @property
    def held_words(self) -> int:
        """Registers a lane spends holding every item of the class for two walks over it: a coordinate's record
        (:data:`WALK_STAGED` floats) and its value, a body's link where it has a parent (one ``float4``)."""
        return (WALK_STAGED + 1) * self.coordinates + 4 * sum(1 for up in self.parent if up >= 0)

    def walk_twice(self, first: dict, second: dict, *, hand: str, registers: int, cuda: bool = True) -> str:
        """Native statements of two walks over one tree of the class (a solve's walk up, then its walk down), each
        :meth:`walk`'s keywords (``first``, ``second``), their records and links the same items.

        **Each item fetched once, when the class's items fit the kernel's registers.** A straight-line class whose
        :attr:`held_words` fit ``registers`` (the kernel's budget, :data:`WALK_REGISTERS` without a launch bound) fetches
        every item once, with the first walk's inputs (``first``'s ``fetch_joint``, ``fetch_step``), before either walk,
        and hands each to both: the second walk's step input is ``hand``, built from the held item ``$X`` of
        coordinate ``$c`` (its own values in place of the first walk's); each item's line is touched before the
        fetches. Past that, each walk fetches its own items ahead of its chain and touches them (:meth:`walk`).
        """
        if self.coordinates > first.get("unroll", WALK_UNROLL) or not held_fits(self.held_words, registers):
            return "{\n" + self.walk(**first, cuda=cuda) + "\n}\n{\n" + self.walk(**second, cuda=cuda) + "\n}"
        bare = {"fetch_joint": "", "fetch_step": "", "touch_joint": "", "touch_step": ""}
        # Every held item's line asked of L1 first, then the fetches: on the RTX PRO 6000 the bare fetches of a free
        # cube's records ran 9.2 -> 12.6 us; touched first, 8.3 (hold p2).
        items = [_fill(first["touch_step"], c=c) for c in range(self.coordinates) if first.get("touch_step")]
        items += [
            _fill(first["touch_joint"], b=b) for b, up in enumerate(self.parent) if up >= 0 and first.get("touch_joint")
        ]
        items += [
            f"const auto walk_l{b} = {_fill(first['fetch_joint'], b=b, up=up)};"
            for b, up in enumerate(self.parent)
            if up >= 0
        ]
        items += [
            f"const auto walk_x{c} = {_fill(first['fetch_step'], c=c, half=self.half_of(c))};"
            for c in range(self.coordinates)
        ]
        step = Template(second["step"]).safe_substitute(X="walk_h")
        handed = f"{{\nconst auto walk_h = {hand};\n{step}\n}}"
        return (
            "\n".join(items)
            + "\n{\n"
            + self.walk(**{**first, **bare}, cuda=cuda)
            + "\n}\n{\n"
            + self.walk(**{**second, **bare, "step": handed}, cuda=cuda)
            + "\n}"
        )
