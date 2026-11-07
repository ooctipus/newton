# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""One scan skeleton for every tree pass: ``tree_scan(plan, combine, direction, schedule)``.

Every dynamics pass walks a joint forest the same way: root to leaf
composing a parent's value with a node's own (poses, twists), or leaf to
root summing a subtree's values (wrenches, composite inertia). The passes
differ in three things, which are this module's arguments; the plan
(:class:`.tree_plan.TreePlan`) supplies the rest:

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
    the subtree sum.
``schedule`` (:class:`Schedule`)
    How the lanes of one tree's group share the work:

    ============  ======================================================  ==========
    schedule      mapping                                                 direction
    ============  ======================================================  ==========
    SERIAL        one thread, parent-first order (reversed for UP)        DOWN, UP
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
is written once here (:class:`WalkCombine`, on a schedule's names for the record,
:class:`Record`), and every walk's schedule writes its steps through it.
"""

import enum
import functools
import hashlib
from dataclasses import dataclass
from string import Template

import numpy as np
import warp as wp

from ...core.memory import REGISTERS_PER_THREAD
from . import field_group
from .device_preludes import DevicePrelude, device_branches
from .field_group import Field
from .law_source import law_preludes
from .spatial_laws import SPATIAL
from .tree_plan import UP_FIRST, UP_SUMS, TreePlan

__all__ = [
    "ENTER",
    "FETCHED",
    "INERTIA",
    "LEAVE",
    "LOADED",
    "SCAN_PLAN",
    "TRANSFORM",
    "TWIST",
    "WALK_AHEAD",
    "WALK_ANGULAR",
    "WALK_BOTH",
    "WALK_EITHER",
    "WALK_LINEAR",
    "WALK_STAGED",
    "WALK_TOUCH",
    "WALK_UNROLL",
    "WHITEN",
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
    "WalkCombine",
    "add",
    "braced",
    "by_class",
    "channels",
    "choose_schedule",
    "classes_half",
    "constant_table",
    "cross",
    "device_plan",
    "dot",
    "held_fits",
    "indent_text",
    "inward_edge",
    "inward_step",
    "record_half",
    "root_path",
    "scan_arrays",
    "schedule_costs",
    "schedule_features",
    "slip",
    "statements",
    "tree_scan",
    "walk_texts",
]


class Direction(enum.Enum):
    """Which way a scan runs over the forest."""

    DOWN = "down"
    """Root to leaf: each node's result composes its parent's result with its own value."""
    UP = "up"
    """Leaf to root: each node's result adds its children's results to its own value."""


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
    if not cuda or plan.max_depth == 0:
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

    def shuffle(self, target: str, var: str, source: str) -> str:
        return "".join(
            f"for(int k=0;k<{group.channels};++k){self.combine.channel(target, 'k', group)}="
            f"READ({self.combine.channel(var, 'k', group)},{source});"
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


def _compose_down(c: _Context) -> str:
    return f"""        {c.combine.declare("value")}{c.combine.declare("up")}
        {c.load("up", "parent")}
        {c.own("value", "node")}
        {c.combine.compose("value", "up")}
        {c.result("value", "node")}"""


def _serial(c: _Context, root_first: bool) -> str:
    """SERIAL: the walk's one step text (:func:`_visit`) on the plan's nodes in the walk's order, a node's state in its
    own words (:class:`_Block`)."""
    order = f"int index=0;index<{c.nodes};++index" if root_first else f"int index={c.nodes}-1;index>=0;--index"
    step = _Step("node", -1, "parent", True, True, "begin<end", False)
    return f"    for({order}) {{\n{indent_text(chr(10).join(_visit(step, _Block(c, root_first), '')), 8)}\n    }}\n"


def _guarded(group: Group, statement: str) -> str:
    return statement if group.active is None else f"if({group.active}){statement}"


def _level_down(c: _Context) -> str:
    publish = c.publish_own("node")
    prepare = f"    for(int node=LANE;node<{c.nodes};node+=STRIDE){{{publish}}}\n    SYNC();\n" if publish else ""
    body = f"""        const int parent={c.parent("node")};
{_compose_down(c)}"""
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
        const int next=READ(parent,source);
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
                const float next=READ(values[k],source);
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
            const float t=READ(w[b],offset);
            if((dfs_length>>b)&1){{sum+=t;offset+=1<<b;}}
        }}
        if(dfs_node>=0)*cell=sum;
    }}
    SYNC();
    }}
"""


_SCHEDULES = {
    (Direction.DOWN, Schedule.SERIAL): functools.partial(_serial, root_first=True),
    (Direction.UP, Schedule.SERIAL): functools.partial(_serial, root_first=False),
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
    source: str | None = None,
    sink: str | None = None,
    cpu: bool = True,
) -> ScanSource:
    """Generate the native scan of ``combine`` over ``plan`` in ``direction`` on ``schedule``.

    Args:
        plan: The tree; its sizes become literals unless ``sizes`` names them.
        combine: The operator and its value's channel groups.
        direction: ``DOWN`` or ``UP``.
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
        source: A register schedule's own-value fragment (``{value}``, ``{node}``) in place of a load.
        sink: A register schedule's result fragment (``{value}``, ``{node}``) in place of a store.
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
    )
    cuda = "{\n" + _SCHEDULES[(direction, schedule)](context) + "}\n"
    serial = "{\n" + _serial(context, direction is Direction.DOWN) + "}\n" if cpu else ""
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


def _unpacked(record: str, link: str, indexed: bool) -> Record:
    """The one record vocabulary: a record unpacked as the walk's one access reads it (``WALK_READ``'s floats ``s``
    0..5, ``ut`` 6..11, ``1 / ell`` 12) under the name ``record``, and a link's lever (``WALK_READ_LINK``'s offset) under
    ``link``; ``indexed`` where a loop reads it."""
    return Record(
        s=tuple(f"{record}[{k}]" for k in range(6)),
        ut=tuple(f"{record}[{6 + k}]" for k in range(6)),
        inverse=f"{record}[12]",
        lever=(f"{link}[0]", f"{link}[1]", f"{link}[2]"),
        indexed=(f"{record}[{{k}}]", f"{record}[6 + {{k}}]") if indexed else None,
    )


LOADED = _unpacked("d", "l", indexed=True)
"""A record read into registers (``WALK_READ``'s ``d``, a held word's) and a link's lever ``l``."""

FETCHED = _unpacked("$X.d", "$L.l", indexed=False)
"""A record and a link the class emitter fetches ahead of the chain (``WalkIn`` as ``$X``, ``WalkLinkIn`` as ``$L``): the
same names, its update written as six statements."""

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


@dataclass(frozen=True)
class WalkCombine:
    """A walk's arithmetic, the one skeleton's combine (``walk(tree, direction, combine, schedule, entries)``, U14): a
    coordinate's step and a joint's edge, in the walk's direction. Every walk of the factor names one (:data:`WHITEN`,
    :data:`LEAVE`, :data:`ENTER`); the schedules (the class emitter :meth:`WalkClass.walk`, the held lanes
    :meth:`WalkClass.held`, the path table's :func:`root_path`) place its statements and hold no arithmetic of their
    own."""

    direction: Direction
    inverted: bool = False

    def step(
        self,
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
        ``q = (u - ut . V) / ell``, then ``V += s q``. Inverted (:data:`ENTER`, root first): ``u = qd / (1 / ell) + ut .
        V`` into ``out``, then ``V += s qd``, ``source`` the coordinate's known velocity ``qd``. The output is rounded on
        its own (``WALK_MUL``), so its operations do not depend on the block its consumer sits in. ``source`` is the
        step's input ``x`` or ``u`` (none: the state's alone); ``value`` instead names a value that already holds
        ``source`` and the dot, for a source written as statements. The output is assigned to ``out``; the update is in
        place, or into ``into`` (``into = state ...``), a loop over ``index`` for a record a loop reads
        (:attr:`Record.indexed`), six statements for a named one. The caller places its sink: after the output (before
        the update) or after the update.
        """
        up = self.direction is Direction.UP
        if self.direction not in (Direction.UP, Direction.DOWN):
            raise ValueError(f"a coordinate's step runs up or down, not {self.direction.value}")
        if self.inverted:
            head = f"{out} = {source} / {record.inverse} + {dot(record.ut, channels(state))};"
            return [head, *self._update(record, state, source, None, index)]
        if value is None:
            head = dot(record.s if up else record.ut, channels(state))
            value = f"{source} {'+' if up else '-'} {head}" if source else head
        output = f"{out} = WALK_MUL(\n    {value},\n    {record.inverse});"
        return [output, *self._update(record, state, out.rsplit(maxsplit=1)[-1], into, index)]

    def _update(self, record: Record, state: str, name: str, into: str | None, index: str) -> list[str]:
        """A step's state update by its output ``name``: ``p -= ut z`` up, ``V += s q`` down."""
        up = self.direction is Direction.UP
        sign, terms = ("-", record.ut) if up else ("+", record.s)
        if record.indexed is not None:
            term = (record.indexed[1] if up else record.indexed[0]).format(k=index)
            target = f"{into}[{index}] = {state}[{index}] {sign}" if into else f"{state}[{index}] {sign}="
            return [f"for (int {index} = 0; {index} < 6; ++{index}) {target} {term} * {name};"]
        if into:
            return [f"{into}[{k}] = {state}[{k}] {sign} {terms[k]} * {name};" for k in range(6)]
        return [f"{state}[{k}] {sign}= {terms[k]} * {name};" for k in range(6)]

    def joint(self, lever, state: str, *, into: str | None = None, first: bool = False) -> list[str]:
        """The statements that pass a joint of lever ``d`` (``lever``, three names).

        ``UP``: a wrench ``(f, t)`` passes to the parent's frame as ``(f, t - d x f)``: in place on ``state``, its force
        read first, or added into the parent's ``into`` (``first``: assigned, a leaf-first walk's first child of its
        parent). ``DOWN``: a twist ``(v, w)`` passes to the child's frame as ``(v + d x w, w)``: in place on ``state``,
        or from the parent's ``state`` into the child's ``into``.
        """
        if self.direction is Direction.UP:
            if into is None:
                force = ("fx", "fy", "fz")
                head = f"const float fx = {state}[0], fy = {state}[1], fz = {state}[2];"
                return [head] + [f"{state}[{3 + k}] -= {cross(lever, force, k)};" for k in range(3)]
            force, op = channels(state)[:3], "=" if first else "+="
            return [f"{into}[{k}] {op} {state}[{k}];" for k in range(3)] + [
                f"{into}[{3 + k}] {op} {state}[{3 + k}] - ({cross(lever, force, k)});" for k in range(3)
            ]
        if self.direction is Direction.DOWN:
            turn = channels(state)[3:]
            if into is None:
                return [f"{state}[{k}] += {cross(lever, turn, k)};" for k in range(3)]
            return [f"{into}[{k}] = {state}[{k}] + ({cross(lever, turn, k)});" for k in range(3)] + [
                f"{into}[{3 + k}] = {state}[{3 + k}];" for k in range(3)
            ]
        raise ValueError(f"a joint's edge runs up or down, not {self.direction.value}")


WHITEN = WalkCombine(Direction.UP)
"""The whitening, leaf first, ``u = L^-1 P x`` (a row's footprint, a solve's walk up)."""

LEAVE = WalkCombine(Direction.DOWN)
"""The leave, root first, ``P^T L^-T u`` (a velocity's walk down, a solve's walk down)."""

ENTER = WalkCombine(Direction.DOWN, inverted=True)
"""The leave inverted, root first, ``u = L^T P qd`` (the audit's coordinates, a free body's change pushed back)."""


def statements(lines, pad: int = 0) -> str:
    """Native statements one a line, every line after the first indented by ``pad`` (the first takes its slot's)."""
    return ("\n" + " " * pad).join("\n".join(lines).split("\n"))


def braced(lines) -> str:
    """Native statements in a block of their own, indented by four."""
    return "{\n" + "\n".join("    " + line for line in "\n".join(lines).split("\n")) + "\n}"


def root_path(body: str, c: str, *, begin=(), joint=(), count: str = "") -> tuple[str, str]:
    """The path table's walk of a root path (``WALK_PARENT_OF``, ``WALK_FIRST_OF``), leaf first: coordinate ``c`` of
    ``body`` down to its body's first, then each ancestor's coordinates, last first. Returns the statements before and
    after a coordinate's visit (``c`` and the body ``b`` in scope), both loops rolled: ``begin`` opens a body (before
    its coordinates), ``joint`` passes it to its parent ``up`` (the edge); ``count`` advances with ``c``."""
    head = [
        "#pragma unroll 1",
        f"for (int b = {body}, c = {c}; b >= 0;) {{",
        *_indented(begin, 4),
        "    #pragma unroll 1",
        f"    for (; c >= WALK_FIRST_OF(b); --c{', ' + count if count else ''}) {{",
    ]
    tail = [
        "    }",
        "    const int up = WALK_PARENT_OF(b);",
        "    if (up >= 0) {",
        *_indented(joint, 8),
        "        c = WALK_FIRST_OF(up + 1) - 1;",
        "    }",
        "    b = up;",
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
    coordinates, :class:`.factor.walk.WalkRelease`): ``"R"`` leaf first, ``x = value - b s``, ``s += a x``, out
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


def walk_texts(
    combine: WalkCombine,
    *,
    source: str,
    sink: str,
    value: str,
    prior: str,
    coordinate: int,
    link: int,
    end: str = "",
) -> dict[str, str]:
    """A whole-tree walk of ``combine`` on fetched items (:data:`FETCHED`), as :meth:`WalkClass.walk`'s keywords:
    :data:`WHITEN` leaf first (``u = L^-1 P x``), :data:`LEAVE` root first (``P^T L^-T u``), :data:`ENTER` root first
    (``u = L^T P qd``). Its joint is the combine's (into the parent leaf first, from it root first); its step the
    combine's on ``$V`` from ``source`` (the enter's: ``sink`` its output's name, the coordinate's known velocity
    ``source``), the caller's ``sink`` after the output and before the state's update; ``end`` a body's
    statements after its coordinates. Each step's record and input (``value``, and ``prior``, its output's prior value)
    and each joint's link are fetched ahead of the chain (``WALK_IN``, ``WALK_LINK_IN``) and asked of L1 before that
    (``WALK_TOUCH``); ``coordinate`` and ``link`` are the floats of a record and a link."""
    up = combine.direction is Direction.UP
    # The enter's output is its sink (the coordinate's u); the others' a named value the sink reads.
    out = sink if combine.inverted else "const float z" if up else "const float q"
    output, *update = combine.step(FETCHED, "$V", source=source, out=out)
    sink = "" if combine.inverted else sink
    texts = {
        "joint": braced(combine.joint(FETCHED.lever, "$V" if up else "$P", into="$P" if up else "$V")),
        **({"joint_first": braced(combine.joint(FETCHED.lever, "$V", into="$P", first=True))} if up else {}),
        "step": braced([output, *([sink] if sink else []), *update]),
        "end": end,
        "root_first": not up,
        "fetch_joint": "WALK_LINK_IN(link, $b, $up)",
        "fetch_step": f"WALK_IN(data + {coordinate} * ($c), {value}, {prior}, $half)",
        "touch_joint": f"WALK_TOUCH(link + {link} * ($b));",
        "touch_step": f"WALK_TOUCH(data + {coordinate} * ($c));",
    }
    return texts


WALK_STAGED = 13
"""The floats of a coordinate's record the walk's functions read (``s``, ``ut``, ``1 / ell``), unpacked
(:data:`.factor.walk._WALK_LOADS`): what a kernel stages of a record in shared memory."""

WALK_LINEAR, WALK_ANGULAR, WALK_BOTH, WALK_EITHER = 0, 1, 2, 3
"""A column's half at its body's frame as a reader of its record knows it (``WALK_READ``'s ``half``): the linear one,
the angular one (a class's data, compiled), or named by the record, with the side of a column of both halves
(``WALK_BOTH``, the linear in the record's side) or in a model without one (``WALK_EITHER``)."""


def record_half(side: bool) -> int:
    """The half a reader of any tree's records passes the one access: named by the record, with the side where a
    column can have two halves (``side``, :attr:`.tree_factor.TreeFactor.walk_side`)."""
    return WALK_BOTH if side else WALK_EITHER


def classes_half(classes) -> int:
    """The half a walk that reads the records of the walking classes ``classes`` by one text passes the one access
    (``WALK_READ``): the half every coordinate of theirs shares, compiled; else named by the record
    (:func:`record_half`)."""
    halves = {half for walk in classes for half in (walk.half or (WALK_BOTH,))}
    return halves.pop() if len(halves) == 1 else record_half(WALK_BOTH in halves)


# ---------------------------------------------------------------------------------------------------------------------
# The class emitter (ZB, T8.8): a walking class's schedules, compiled (straight-line or the loop form) and held
# ---------------------------------------------------------------------------------------------------------------------

WALK_TOUCH = 16
"""Items a walk asks L1 for ahead of its chain (:meth:`WalkClass.touch`): a hint without a register or a scoreboard,
so the loads :data:`WALK_AHEAD` items ahead find their lines in L1."""

WALK_UNROLL = REGISTERS_PER_THREAD // 4
"""Coordinates of a class up to which :meth:`WalkClass.walk` places its states in named registers (straight-line code);
a larger class takes the chain (the loop form). A straight-line walk holds about three registers a coordinate with its
loads ahead (g1's 43-coordinate walk solve 141, no stack), so a quarter of
:data:`~newton._src.core.memory.REGISTERS_PER_THREAD` coordinates fit with room for the chain (chain_106's 112 do not).
The chain indexes its states on the stack (g1's walk solve: 272 B, 41 local loads and 43 stores): straight-line, g1's
walk solve runs 78.0 -> 58.1 us a launch and its leave 47.1 -> 40.0 (RTX PRO 6000, 16,384 worlds; ZP5, ZV; RETWALK2's
P1: 12 of the walk family's 41 kernels lose blocks or gain stack on the chain)."""


def held_fits(words: int, registers: int) -> bool:
    """Whether a thread may hold ``words`` floats of a walk's data across the walk: they fit ``registers``, the kernel's
    budget a thread (the shape's ``registers``: :data:`~newton._src.core.memory.REGISTERS_PER_THREAD` without a
    launch bound). The one rule by which a walk solve holds its
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


def constant_table(name: str, values) -> str:
    """A native table of ints in the walk's constant memory (``WALK_CONSTANT``: CUDA's ``__constant__``, the CPU's
    ``const``), at least one entry long: a class's path, its emitter's steps, a placement's nodes."""
    values = [str(int(v)) for v in values]
    return f"static WALK_CONSTANT int {name}[{max(len(values), 1)}] = {{{', '.join(values)}}};"


def _fill(text: str, **values) -> str:
    """``text`` with its ``$`` slots filled from ``values``: a walk text names only the slots it reads."""
    template = Template(text)
    return template.substitute({name: str(values[name]) for name in template.get_identifiers()})


def indent_text(text: str, spaces: int) -> str:
    """``text`` with every non-empty line indented by ``spaces``."""
    pad = " " * spaces
    return "\n".join(pad + line if line.strip() else line for line in text.splitlines())


def _when(condition, then, otherwise=()) -> list[str]:
    """Statements under a condition known as the walk is emitted (a ``bool``) or read at run time (an ``if``)."""
    then, otherwise = list(then), list(otherwise)
    if isinstance(condition, bool):
        return then if condition else otherwise
    if not (then or otherwise):
        return []
    text = f"if ({condition}) {{\n{indent_text(chr(10).join(then), 4)}\n}}"
    return [text + (f" else {{\n{indent_text(chr(10).join(otherwise), 4)}\n}}" if otherwise else "")]


@dataclass(frozen=True)
class _Step:
    """A walk's step: coordinate ``c`` (-1: none) of body ``b``, its parent, the body's first and last step, its join bits
    (:attr:`.tree_plan.TreePlan.up_flags`), each a constant or a table read; the loop's copy of it, its ring slot."""

    b: int | str
    c: int | str
    up: int | str
    first: bool | str
    last: bool | str
    sums: bool | str
    leading: bool | str
    ring: int = 0

    @property
    def coordinate(self) -> bool | str:
        return self.c >= 0 if isinstance(self.c, int) else f"{self.c} >= 0"


class _Placement:
    """Where a walk's states and items live (:meth:`WalkClass.walk`): a body's ``enter`` takes its state (from its
    parent's root first, its children's leaf first), its ``leave`` keeps it for its children or joins it into its
    parent's; an item (``("l", b)`` body ``b``'s joint input, ``("x", c)`` coordinate ``c``'s step input) is fetched and
    touched ahead of the step that reads it (``take`` before it, or ``after`` the step before)."""

    def __init__(self, root_first: bool, width: int, joint: str, joint_first: str, texts: dict):
        self.root_first = root_first
        self.width = width
        self.joint = joint
        self.joint_first = joint_first
        self.texts = texts

    def items(self, entries, named=None) -> list[str]:
        out = []
        for kind, index in entries:
            if kind == "l":
                text = _fill(self.texts["fetch_joint" if named else "touch_joint"], b=index, up=self.up(index))
            else:
                text = _fill(self.texts["fetch_step" if named else "touch_step"], c=index, half=self.half(index))
            out += [f"{named(kind, index)} = {text};" if named else text] if text else []
        return out

    def edge(self, s: _Step, parent: str, first: bool = False) -> str:
        return _fill(self.joint_first if first else self.joint, b=s.b, V=self.state(s), P=parent, **self.names(s))

    def take(self, item) -> list[str]:
        return []

    def after(self, s: _Step) -> list[str]:
        return []


class _Registers(_Placement):
    """Each body's state a named register array ``walk{b}`` and each item a named register in the walk's order, fetched
    ``depth`` and touched ``early`` items ahead, every index a constant: straight-line code."""

    def __init__(
        self, walk: "WalkClass", root_first: bool, width: int, joint: str, joint_first: str, texts, depth, early
    ):
        super().__init__(root_first, width, joint, joint_first, texts)
        self.bodies, self.up, self.half = len(walk.parent), walk.parent.__getitem__, walk.half_of
        self.zero = "{" + ", ".join(["0.0f"] * width) + "}"
        # A body's joint input where it has a parent (root first before its coordinates, leaf first after them), each
        # coordinate's step input.
        joints, steps = texts["fetch_joint"] or texts["touch_joint"], texts["fetch_step"] or texts["touch_step"]
        self.order = []
        for b in walk.bodies_in(root_first):
            if root_first and joints and walk.parent[b] >= 0:
                self.order.append(("l", b))
            if steps:
                self.order += [("x", c) for c in walk.coordinates_in(b, root_first)]
            if not root_first and joints and walk.parent[b] >= 0:
                self.order.append(("l", b))
        self.depth, self.early = depth(len(self.order)), early(len(self.order))
        self.position = {item: k for k, item in enumerate(self.order)}

    def fetched(self, item) -> list[str]:
        return self.items([item], "const auto walk_{}{}".format)

    def prologue(self) -> list[str]:
        out = [line for item in self.order[: self.early] for line in self.items([item])]
        out += [line for item in self.order[: self.depth] for line in self.fetched(item)]
        if self.root_first:
            return out
        return [*out, f"float {', '.join(f'walk{b}[{self.width}] = {self.zero}' for b in range(self.bodies))};"]

    def bind(self, s: _Step) -> list[str]:
        return []

    def state(self, s: _Step) -> str:
        return f"walk{s.b}"

    def names(self, s: _Step) -> dict:
        return {"L": f"walk_l{s.b}", "X": f"walk_x{s.c}"}

    def take(self, item) -> list[str]:
        k = self.position.get(item, -1)
        if k < 0:
            return []
        out = self.items([self.order[k + self.early]]) if k + self.early < len(self.order) else []
        return out + (self.fetched(self.order[k + self.depth]) if k + self.depth < len(self.order) else [])

    def enter(self, s: _Step) -> list[str]:
        if not self.root_first:
            return []
        if s.up < 0:
            return [f"float walk{s.b}[{self.width}] = {self.zero};"]
        return [f"float walk{s.b}[{self.width}];", *self.take(("l", s.b)), self.edge(s, f"walk{s.up}")]

    def leave(self, s: _Step) -> list[str]:
        if self.root_first or s.up < 0:
            return []
        return [*self.take(("l", s.b)), self.edge(s, f"walk{s.up}", s.leading)]


class _Chain(_Placement):
    """The loop form's chain: the body walked in ``V``, its parent's ``P`` (or children's ``N``) beside it, a join's in a
    slot ``S`` indexed at run time (:func:`_walk_slots`); a step's two items in a ring of ``depth`` registers, the loop
    body a copy of the step a slot (copy ``j``, :attr:`_Step.ring`, reads slot ``j`` and refills it ``depth`` steps on), or
    loaded in place without a ring; each step's lines touched ``early`` steps ahead."""

    def __init__(
        self, walk: "WalkClass", root_first: bool, width: int, joint: str, joint_first: str, texts, name, half, ring
    ):
        super().__init__(root_first, width, joint, joint_first, texts)
        self.name, self.half, (self.steps, self.depth, self.early) = name, half, ring
        self.up = f"{name}_PARENT[{{}}]".format
        self.count = walk.slots[1]
        self.each = f"for (int k = 0; k < {width}; ++k)"

    def slot(self, body: str) -> str:
        return f"S[{self.name}_SLOT[{body}]]"

    def entries(self, at) -> list:
        coordinate = f"{self.name}_COORDINATE[{at}]"
        return [("l", f"({self.name}_STEP[{at}] & 0xffff)"), ("x", f"(({coordinate}) < 0 ? 0 : {coordinate})")]

    def fetched(self, j: int, at) -> list[str]:
        return self.items(self.entries(at), lambda kind, _: f"walk_{kind}{j}")

    def prologue(self) -> list[str]:
        out = [f"auto {line}" for j in range(self.depth) for line in self.fetched(j, min(j, self.steps - 1))]
        if self.early:
            touched = chr(10).join(self.items(self.entries("at")))
            out.append(f"for (int at = 0; at < {self.early}; ++at) {{\n{indent_text(touched, 4)}\n}}")
        slots = f" float S[{self.count}][{self.width}];" if self.count else ""
        return [*out, f"float V[{self.width}], {'P' if self.root_first else 'N'}[{self.width}];{slots}"]

    def bind(self, s: _Step) -> list[str]:
        out = [f"const int b = {self.name}_STEP[at] & 0xffff, c = {self.name}_COORDINATE[at];"]
        if self.depth:
            return out
        return out + self.items([("l", "b"), ("x", "(c < 0 ? 0 : c)")], lambda kind, _: f"const auto walk_{kind}")

    def state(self, s: _Step) -> str:
        return "V"

    def names(self, s: _Step) -> dict:
        return {"L": f"walk_l{s.ring}", "X": f"walk_x{s.ring}"} if self.depth else {"L": "walk_l", "X": "walk_x"}

    def after(self, s: _Step) -> list[str]:
        at = f"(at + {self.depth} < {self.steps} ? at + {self.depth} : {self.steps - 1})"
        out = self.fetched(s.ring, at) if self.depth else []
        early = f"at + {self.early}"
        return out + (_when(f"{early} < {self.steps}", self.items(self.entries(early))) if self.early else [])

    def enter(self, s: _Step) -> list[str]:
        each = self.each
        if self.root_first:
            parent = [f"{each} P[k] = V[k];"]
            if self.count:
                parent = _when("up == b - 1", parent, [f"{each} P[k] = {self.slot('up')}[k];"])
            zero = [f"{each} V[k] = 0.0f;"]
            return [f"const int up = {s.up};", *_when("up < 0", zero, [*parent, self.edge(s, "P")])]
        # A parent's state is its first child's, the others added (no slot cleared); a leaf's zero.
        gather = [f"{each} V[k] = N[k];"]
        if self.count:
            gather = _when(f"{self.name}_SLOT[b] >= 0", [f"{each} V[k] = {self.slot('b')}[k];"], gather)
        return _when(s.sums, gather, [f"{each} V[k] = 0.0f;"])

    def leave(self, s: _Step) -> list[str]:
        if self.root_first:
            keep = [f"{self.each} {self.slot('b')}[k] = V[k];"]
            return _when(f"{self.name}_SLOT[b] >= 0", keep) if self.count else []
        # A parent without a slot has one child, the next body: its first.
        give = [self.edge(s, "N", True)]
        if self.count:
            slot = self.slot("up")
            give = _when(
                f"{self.name}_SLOT[up] >= 0", _when(s.leading, [self.edge(s, slot, True)], [self.edge(s, slot)]), give
            )
        return [f"const int up = {s.up};", *_when("up >= 0", give)]


def _visit(s: _Step, place: _Placement, step: str, begin: str = "", end: str = "") -> list[str]:
    """The one text of a walk's step on its placement's states and items (:meth:`WalkClass.walk`, :func:`_serial`): a
    body's first step takes its state and runs ``begin``, every step its coordinate's ``step``, the last runs ``end`` and
    passes the state on."""
    state = place.state(s)
    head = place.enter(s) + ([_fill(begin, b=s.b, V=state, up=s.up)] if begin else [])
    tail = ([_fill(end, b=s.b, V=state)] if end else []) + place.leave(s)
    coordinate = [*place.take(("x", s.c)), _fill(step, b=s.b, c=s.c, V=state, **place.names(s))]
    return [
        *place.bind(s),
        *_when(s.first, head),
        *_when(s.coordinate, coordinate),
        *_when(s.last, tail),
        *place.after(s),
    ]


class _Block(_Placement):
    """A scan's placement (:func:`_serial`): each node's state in its own words (the scan's storage), the steps the
    plan's nodes read at run time. Root first, a node composes its parent's words with its own value; leaf first, it
    gathers its children's (the first in the walk's order assigns, the others add, a leaf's zero), then adds its own."""

    def __init__(self, c: "_Context", root_first: bool):
        self.c, self.root_first = c, root_first

    def bind(self, s: _Step) -> list[str]:
        order = "index" if self.c.order is None else self.c.order.format(index="index")
        return [
            f"const int node={order};",
            f"const int parent={self.c.parent('node')};",
            self.c.combine.declare("value"),
        ]

    def state(self, s: _Step) -> str:
        return "value"

    def names(self, s: _Step) -> dict:
        return {}

    def _groups(self, statement) -> list[str]:
        return [_guarded(g, f"for(int k=0;k<{g.channels};++k){statement(g)}") for g in self.c.combine.groups]

    def enter(self, s: _Step) -> list[str]:
        c = self.c
        if self.root_first:
            compose = [
                c.combine.declare("up"),
                c.load("up", "parent"),
                c.own("value", "node"),
                c.combine.compose("value", "up"),
            ]
            return _when("parent<0", [c.publish_own("node")], compose)
        access, value = c.access, lambda g: c.combine.channel("value", "k", g)
        child = access.child.format(entry="e")
        gather = _when(
            s.sums,
            [
                f"int e=begin;const int first={child};",
                *self._groups(lambda g: f"{value(g)}={c.slot(g, 'first', 'k')};"),
                f"for(++e;e<end;++e){{const int child={child};",
                *self._groups(lambda g: f"{value(g)}+={c.slot(g, 'child', 'k')};"),
                "}",
            ],
            self._groups(lambda g: f"{value(g)}=0.0f;"),
        )
        bounds = f"const int begin={access.children_begin.format(node='node')},end={access.children_end.format(node='node')};"
        return [bounds, *gather, *self._groups(lambda g: f"{value(g)}+={c.slot(g, 'node', 'k')};")]

    def leave(self, s: _Step) -> list[str]:
        c = self.c
        if self.root_first:
            return _when("parent>=0", [c.result("value", "node")])
        return self._groups(lambda g: f"{c.slot(g, 'node', 'k')}={c.combine.channel('value', 'k', g)};")


@dataclass(frozen=True)
class WalkClass:
    """A walking class: the topology its trees share, each body's parent (-1: the world, every parent before its
    children) and first coordinate, with the count after the last (:func:`.factor.walk.walk_prelude`'s table).

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
        last first (the release's path, :class:`.factor.walk.WalkRelease`): :func:`root_path` with the table compiled."""
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
        unroll: int = WALK_UNROLL,
        fetch_joint: str = "",
        fetch_step: str = "",
        touch_joint: str = "",
        touch_step: str = "",
        ahead: int | None = None,
        hold: bool = False,
        joint_first: str = "",
    ) -> str:
        """Native statements that walk the class's tree on a state of ``width`` floats a body, in either direction.

        Root first, a body's state ``$V`` takes its parent's ``$P`` through ``joint`` (a root's zero), then ``begin``, its
        coordinates in order (``step``) and ``end``. Leaf first, a body's state is its children's (the first assigns by
        ``joint_first``, the others add; a leaf's zero), then ``begin``, its coordinates last first, ``end``, its joint.

        One text of the walk's steps (``visit``) on a placement of its states and items, chosen by the class's size
        against the registers: up to ``unroll`` coordinates :class:`_Registers` and :class:`_NamedItems` (straight-line
        code), past it :class:`_Chain` and :class:`_RingItems` (a loop over the class's table in constant memory). A
        walk's inputs do not depend on its chain: ``fetch_joint`` (body ``$b``'s) and ``fetch_step`` (coordinate
        ``$c``'s) are evaluated :meth:`ahead` items before the joint (as ``$L``) and the step (as ``$X``) that read them.

        Args:
            joint: Body ``$b``'s joint to its parent: ``$V`` from ``$P`` root first, ``$V`` into ``$P`` leaf first.
            step: Coordinate ``$c`` of body ``$b`` on its state ``$V``.
            begin: Body ``$b``'s statements before its coordinates, on ``$V`` (``$up`` its parent, -1 at a root).
            end: Body ``$b``'s statements after its coordinates, on ``$V``.
            root_first: The walk's direction.
            width: Floats of a body's state.
            unroll: Coordinates up to which the walk is straight-line code (:data:`WALK_UNROLL`).
            fetch_joint: Body ``$b``'s joint inputs (an expression; ``$up`` its parent); none if empty.
            fetch_step: Coordinate ``$c``'s step inputs (an expression; ``$half`` its column half, :meth:`half_of`, a
                constant in straight-line code, :data:`WALK_BOTH` in the loop form); none if empty.
            touch_joint: Body ``$b``'s statement asking L1 for its joint inputs' lines (a hint, no register).
            touch_step: Coordinate ``$c``'s statement asking L1 for its step inputs' lines.
            ahead: Items whose inputs a lane holds in registers ahead of the chain (:data:`WALK_AHEAD` by default).
            hold: Every item touched, then fetched, before the chain (straight-line only), held for a second walk
                (:meth:`walk_twice`; touched first, a free cube's fetches 12.6 -> 8.3 us on the RTX PRO 6000).
            joint_first: Leaf first: the joint of a parent's first child in the walk's order (it assigns).
        """
        if not root_first and not joint_first:
            raise ValueError("a leaf-first walk assigns each parent's first child: give joint_first")
        texts = {
            "fetch_joint": fetch_joint,
            "fetch_step": fetch_step,
            "touch_joint": touch_joint,
            "touch_step": touch_step,
        }
        fetches, touches = bool(fetch_joint or fetch_step), bool(touch_joint or touch_step)
        compiled = self.coordinates <= unroll
        if hold and not compiled:
            raise ValueError("a walk holds its items in straight-line code only")

        # The steps in the walk's order: a coordinate's, or a body's without one, with the body's join bits leaf first
        # (the plan's up flags: a parent's first child in the walk's order, a parent).
        up_flags = TreePlan.build(self.parent).up_flags if not root_first else np.zeros(len(self.parent), np.int32)
        sequence = []
        for b in self.bodies_in(root_first):
            span = list(self.coordinates_in(b, root_first)) or [-1]
            sequence += [(b, c, k == 0, k == len(span) - 1) for k, c in enumerate(span)]
        if compiled:
            place = _Registers(
                self,
                root_first,
                width,
                joint,
                joint_first,
                texts,
                depth=lambda n: n if hold else self.ahead(n, ahead) if fetches else 0,
                early=lambda n: n if hold else self.touch(n) if touches else 0,
            )
            steps = [
                _Step(b, c, self.parent[b], first, last, bool(up_flags[b] & UP_SUMS), bool(up_flags[b] & UP_FIRST))
                for b, c, first, last in sequence
            ]
            return "\n".join([*place.prologue(), *(line for s in steps for line in _visit(s, place, step, begin, end))])

        # The loop form: the class's steps in constant memory, each naming its body (16 bits), whether it is the body's
        # first (16) and last (17) step and its body's join bits (18, 19), and its coordinate (-1 for none).
        name = f"WALK_CLASS_{self.tag}"
        join = [bool(f & UP_FIRST) << 18 | bool(f & UP_SUMS) << 19 for f in up_flags]
        slots, count = self.slots
        tables = [
            constant_table(f"{name}_PARENT", self.parent),
            constant_table(f"{name}_STEP", [b | first << 16 | last << 17 | join[b] for b, _, first, last in sequence]),
            constant_table(f"{name}_COORDINATE", [c for _, c, _, _ in sequence]),
        ]
        # The class's column halves (half_of): a table where none has both, so the walk's fetch takes no side; else
        # each record's own.
        one_half = bool(self.half) and max(self.half) < WALK_BOTH
        if one_half:
            tables.append(constant_table(f"{name}_HALF", self.half))
        if count:
            tables.append(constant_table(f"{name}_SLOT", slots))
        n = len(sequence)
        half = f"({name}_HALF[{{}}] & 1)".format if one_half else lambda c: WALK_BOTH
        ring = (n, self.ahead(n, ahead) if fetches else 0, self.touch(n) if touches else 0)
        place = _Chain(self, root_first, width, joint, joint_first, texts, name, half, ring)
        flag = f"({name}_STEP[at] >> {{}}) & 1".format
        copies = max(place.depth, 1)
        steps = [_Step("b", "c", f"{name}_PARENT[b]", flag(16), flag(17), flag(19), flag(18), j) for j in range(copies)]
        loop = "\n".join(
            f"if (at0 + {j} < {n}) {{\n    const int at = at0 + {j};\n{indent_text(chr(10).join(_visit(s, place, step, begin, end)), 4)}\n}}"
            for j, s in enumerate(steps)
        )
        head = f"for (int at0 = 0; at0 < {n}; at0 += {copies}) {{"
        return "\n".join([*tables, *place.prologue(), head, indent_text(loop, 4), "}"])

    def bodies_in(self, root_first: bool) -> range:
        """The class's bodies in a walk's order: parents first (root first), or children first (leaf first)."""
        bodies = len(self.parent)
        return range(bodies) if root_first else range(bodies - 1, -1, -1)

    def coordinates_in(self, b: int, root_first: bool):
        """Body ``b``'s coordinates in a walk's order."""
        span = range(self.first[b], self.first[b + 1])
        return span if root_first else reversed(span)

    def held(self, x: str, u: str, state: str) -> tuple[str, str]:
        """The ``HELD`` schedule: native statements of the decided step's two held walks (both devices, the border's
        held ``h`` in scope, :data:`.tree_kinetic_state._HELD`): the walk up of ``x`` into ``u`` (``u = L^-1 P x``) and the
        walk down of ``u`` in place (``P^T L^-T u``), on every lane of the segment in lockstep, straight-line in the
        class's order, no loop and no pick of a parent or a first coordinate on the chain. Each coordinate's step runs
        on every lane from the lane's own held word of the records (the coordinate's lane: its own record), and the
        walk's six-float state is then read from the coordinate's lane (six shuffles, where reading the record took
        thirteen); each body's link by shuffle from the lane that holds it (``HELD_ROW`` at a constant). Both walks are
        the class emitter's straight-line form (:meth:`walk`, the held texts its step and joint): each body's state a
        register array, a parent's first child assigning it (``joint_first``), the walk down's twists out in ``state``
        (six words a body, every lane writes them). Lane 0 writes ``u`` (both walks). A lane holds ``HOLD_WORDS``
        words of each held array (on the CPU's one lane, every item)."""
        # The held texts name the border's words with the layout's $ slots: escaped for the emitter's own slots.
        x, u, state = (name.replace("$", "$$") for name in (x, u, state))
        up_out, up_update = WHITEN.step(LOADED, "$V", source=f"{x}[$c]", into="q")
        down_out, down_update = LEAVE.step(LOADED, "$V", source="uh[$c/WIDTH]", out="const float q", into="W")
        held = "{const float* d=h.r[$c/WIDTH];%s%sfor(int k=0;k<6;++k)$V[k]=READ(%s[k],$c%%WIDTH);}"
        link = "{float l[3];HELD_ROW(h.l,$b,l);%s}"
        up = self.walk(
            link % "".join(WHITEN.joint(LOADED.lever, "$V", into="$P")),
            held % (up_out, f"float q[6];{up_update}if(lane==$c%WIDTH){u}[$c]=z;", "q"),
            root_first=False,
            unroll=self.coordinates,
            joint_first=link % "".join(WHITEN.joint(LOADED.lever, "$V", into="$P", first=True)),
        )
        down = self.walk(
            link % "".join(LEAVE.joint(LOADED.lever, "$P", into="$V")),
            held % (down_out, f"float W[6];{down_update}if(lane==$c%WIDTH){u}[$c]=q;", "W"),
            end=f"for (int k = 0; k < 6; ++k) {state}[6 * $b + k] = $V[k];",
            unroll=self.coordinates,
        )
        uh = (
            f"float uh[HOLD_WORDS];\n#pragma unroll\nfor (int k = 0; k < HOLD_WORDS; ++k) uh[k] = lane + k * WIDTH < "
            f"{self.coordinates} ? {u.replace('$$', '$')}[lane + k * WIDTH] : 0.0f;\nWARP_SYNC();"
        )
        return "{\n" + up + "\nWARP_SYNC();\n}", "{\n" + uh + "\n" + down + "\nWARP_SYNC();\n}"

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

    def walk_twice(self, first: dict, second: dict, *, hand: str, registers: int) -> str:
        """Native statements of two walks over one tree of the class (a solve's walk up, then its walk down), each
        :meth:`walk`'s keywords (``first``, ``second``), their records and links the same items.

        **Each item fetched once, when the class's items fit the kernel's registers.** A straight-line class whose
        :attr:`held_words` fit ``registers`` (the kernel's budget, its shape's ``registers``) fetches
        every item once, with the first walk's inputs (``first``'s ``fetch_joint``, ``fetch_step``), before either walk,
        and hands each to both: the second walk's step input is ``hand``, built from the held item ``$X`` of
        coordinate ``$c`` (its own values in place of the first walk's); each item's line is touched before the
        fetches. Past that, each walk fetches its own items ahead of its chain and touches them (:meth:`walk`).
        """
        if self.coordinates > first.get("unroll", WALK_UNROLL) or not held_fits(self.held_words, registers):
            return "{\n" + self.walk(**first) + "\n}\n{\n" + self.walk(**second) + "\n}"
        bare = {"fetch_joint": "", "fetch_step": "", "touch_joint": "", "touch_step": ""}
        step = Template(second["step"]).safe_substitute(X="walk_h")
        handed = f"{{\nconst auto walk_h = {hand};\n{step}\n}}"
        # The second walk in the first's scope, where the held items are.
        return (
            "{\n" + self.walk(**first, hold=True) + "\n{\n" + self.walk(**{**second, **bare, "step": handed}) + "\n}\n}"
        )
