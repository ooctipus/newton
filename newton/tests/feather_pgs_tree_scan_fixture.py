# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared machinery of the tree-scan and tree-pass tests.

Tree shapes, every held-tree scan configuration compiled as a stand-alone
native on a held tree's scratch layout (to compare with a float64 reference or
time with one launch) and articulated models of the shapes. It defines no
tests.
"""

import functools
from dataclasses import dataclass

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import tree_kinetic_state
from newton._src.solvers.feather_pgs import tree_scan as ts
from newton._src.solvers.feather_pgs.device_preludes import CUDA_ONLY, device_branches
from newton._src.solvers.feather_pgs.tree_kinetic_layout import TreeLayout
from newton._src.solvers.feather_pgs.tree_plan import TreePlan

# ---------------------------------------------------------------------------------------------------------------------
# Shapes: parent arrays of articulations the solver runs, and synthetic extremes
# ---------------------------------------------------------------------------------------------------------------------


def _chain(n):
    return [-1, *range(n - 1)]


def _legs(base, legs, segments):
    parent = [-1]
    for _ in range(legs):
        previous = base
        for _ in range(segments):
            parent.append(previous)
            previous = len(parent) - 1
    return parent


def _append_chain(parent, root, length):
    previous = root
    for _ in range(length):
        parent.append(previous)
        previous = len(parent) - 1
    return previous


def _humanoid13():
    # MuJoCo humanoid: torso, waist, pelvis, two legs of three, two arms of two.
    return [-1, 0, 1, 2, 3, 4, 2, 6, 7, 0, 9, 0, 11]


def _g1():
    # Pelvis, two legs of six, a three-joint waist to the torso, two arms of seven.
    parent = [-1]
    _append_chain(parent, 0, 6)
    _append_chain(parent, 0, 6)
    torso = _append_chain(parent, 0, 3)
    _append_chain(parent, torso, 7)
    _append_chain(parent, torso, 7)
    return parent


def _h1():
    parent = [-1]
    _append_chain(parent, 0, 5)
    _append_chain(parent, 0, 5)
    torso = _append_chain(parent, 0, 1)
    _append_chain(parent, torso, 4)
    _append_chain(parent, torso, 4)
    return parent


def _humanoid44():
    # A 44-body humanoid with three-fingered hands, depth 10 (the reference layout's size).
    parent = [-1]
    _append_chain(parent, 0, 6)
    _append_chain(parent, 0, 6)
    torso = _append_chain(parent, 0, 2)
    _append_chain(parent, torso, 1)
    for _ in range(2):
        wrist = _append_chain(parent, torso, 4)
        for _ in range(3):
            _append_chain(parent, wrist, 2)
    _append_chain(parent, torso, 4)
    _append_chain(parent, 0, 4)
    return parent


def _binary(n):
    return [-1] + [(i - 1) // 2 for i in range(1, n)]


def _random(n, seed):
    rng = np.random.default_rng(seed)
    return [-1] + [int(rng.integers(max(0, i - 4), i)) for i in range(1, n)]


SHAPES = {
    "pendulum2": [-1, 0],
    "kuka7": _chain(7),
    "ant9": _legs(0, 4, 2),
    "franka11": [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8],
    "anymal13": _legs(0, 4, 3),
    "humanoid13": _humanoid13(),
    "allegro17": _legs(0, 4, 4),
    "h1_20": _h1(),
    "g1_30": _g1(),
    "star32": [-1] + [0] * 31,
    "binary31": _binary(31),
    "chain32": _chain(32),
    "random24": _random(24, 7),
    "humanoid44": _humanoid44(),
    "chain40": _chain(40),
    "binary63": _binary(63),
    "chain64": _chain(64),
    "random60": _random(60, 11),
}
"""Template trees, parent-first, up to 64 bodies."""


def plan_of(name: str) -> TreePlan:
    return TreePlan.build(np.asarray(SHAPES[name]))


def lanes_of(plan: TreePlan, packed: int = 0) -> int:
    """A held tree's lanes for ``plan``: 32 (or ``packed``) up to 32 bodies, 64 beyond."""
    if plan.count > 32:
        return 64
    if packed:
        return max(packed, 4, 1 << (plan.count - 1).bit_length())
    return 32


def layout_of(plan: TreePlan, lanes: int) -> TreeLayout:
    """A held-tree layout of ``plan`` with one coordinate per body (the scans read only its sizes)."""
    n = plan.count
    return TreeLayout(n, n, n, 1, plan.max_depth, 0, tuple(range(n - 1, -1, -1)), lanes)


# ---------------------------------------------------------------------------------------------------------------------
# Passes: each held-tree scan configuration, on one scratch layout
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Pass:
    """A held-tree pass: the ``tree_scan`` configuration a held kernel runs."""

    name: str
    combine: ts.Combine
    direction: ts.Direction
    schedule: ts.Schedule


POSE, MOTION = ts.TRANSFORM, ts.TWIST.with_zero_acceleration()
WRENCH_MOMENTS = ts.add(ts.Group("wrench", 6), ts.Group("moment", 13, "refresh"))
TWIST_SUM = ts.add(ts.Group("twist", 6))

PASSES = {
    item.name: item
    for item in (
        Pass("pose_smem", POSE, ts.Direction.DOWN, ts.Schedule.JUMP_SMEM),
        Pass("motion_smem", MOTION, ts.Direction.DOWN, ts.Schedule.JUMP_SMEM),
        Pass("pose_shfl", POSE, ts.Direction.DOWN, ts.Schedule.JUMP_SHFL),
        Pass("motion_shfl", MOTION, ts.Direction.DOWN, ts.Schedule.JUMP_SHFL),
        Pass("pose_chain", POSE, ts.Direction.DOWN, ts.Schedule.HEAVY_CHAIN),
        Pass("motion_chain", MOTION, ts.Direction.DOWN, ts.Schedule.HEAVY_CHAIN),
        Pass("pose_level", POSE, ts.Direction.DOWN, ts.Schedule.LEVEL),
        Pass("reduce_level", WRENCH_MOMENTS, ts.Direction.UP, ts.Schedule.LEVEL),
        Pass("reduce_interval", WRENCH_MOMENTS, ts.Direction.UP, ts.Schedule.DFS_INTERVAL),
        Pass("reduce_chain", WRENCH_MOMENTS, ts.Direction.UP, ts.Schedule.HEAVY_CHAIN),
        Pass("twist_shfl", TWIST_SUM, ts.Direction.DOWN, ts.Schedule.JUMP_SHFL),
    )
}
"""Every held-tree scan configuration (the schedules the held kernels choose from)."""


def chain_lanes(plan: TreePlan) -> int:
    """Lanes of a heavy-chain launch of ``plan``: one tree per block of the warps its chains pack into."""
    packing = plan.chain_packing(2)
    return max(lanes_of(plan), packing.lanes if packing is not None else 32)


def applies(item: Pass, plan: TreePlan, lanes: int) -> bool:
    """Whether the pass runs on a tree of this plan at ``lanes`` lanes (one tree per block for a chain)."""
    if item.schedule is ts.Schedule.HEAVY_CHAIN:
        packing = plan.chain_packing(2)
        return packing is not None and lanes >= 32 and packing.lanes <= lanes
    if item.schedule in (ts.Schedule.JUMP_SHFL, ts.Schedule.DFS_INTERVAL):
        return plan.count <= min(lanes, 32)
    return plan.count <= lanes


def storage(layout: TreeLayout, item: Pass) -> ts.Storage:
    """The pass's slots in the held tree's scratch (``s`` at the state block's start)."""
    bodies = layout.bodies
    if item.combine is POSE:
        return ts.Storage({"pose": "s[7*({node})+({k})]"})
    if item.combine is MOTION:
        motion = 7 * bodies
        return ts.Storage(
            {
                "velocity": f"s[{motion}+6*({{node}})+({{k}})]",
                "acceleration": f"s[{motion + 6 * bodies}+6*({{node}})+({{k}})]",
            }
        )
    return ts.Storage(
        {
            "wrench": f"s[{layout.wrench_offset}+6*({{node}})+({{k}})]",
            "moment": f"s[{layout.moment_offset}+13*({{node}})+({{k}})]",
        }
    )


# One native per pass with one signature, ``(address, plan, flag)``; ``flag`` is ``refresh`` for the subtree
# reductions and the twist publication switch for the twist sum.
_REDUCE_SCOPE = """    float* s=reinterpret_cast<float*>(address);
#if defined(__CUDA_ARCH__)
    const int lane=LANE,stride=STRIDE;
#else
    const int lane=0,stride=1;
#endif
    const int refresh=flag;
"""

_TWIST_SCOPE = """    const int lane=LANE;
    const int ds=group*$BODIES,js=ds;
"""

_TWIST_SOURCE = (
    "{{const int first=ds+plan.dof_start.data[{node}],count=plan.dof_start.data[{node}+1]-plan.dof_start.data[{node}];"
    "for(int d=0;d<count;++d){{const wp::spatial_vector axis=f.S.data[first+d];const float speed=f.vhat.data[first+d];"
    "for(int k=0;k<6;++k){value}[k]+=axis[k]*speed;}}}}"
)
_TWIST_SINK = (
    "{{wp::spatial_vector twist;for(int k=0;k<6;++k)twist[k]={value}[k];"
    "f.predicted_twist.data[p.joint_child.data[js+{node}]]=twist;}}"
)


def generated_text(item: Pass, plan: TreePlan, layout: TreeLayout, *, cpu: bool = True) -> str:
    """The pass's ``tree_scan`` configuration on a :class:`~newton._src.solvers.feather_pgs.tree_scan.ScanPlan`
    named ``plan`` (``p`` in the down scans), in its wrapper scope."""
    reduction = item.direction is ts.Direction.UP or item.name == "twist_shfl"
    access = ts.PlanAccess() if reduction else ts.PlanAccess().renamed("p")
    source = sink = None
    if item.name == "twist_shfl":
        source, sink = _TWIST_SOURCE, _TWIST_SINK
    scan = ts.tree_scan(
        plan,
        item.combine,
        item.direction,
        item.schedule,
        storage=None if source is not None else storage(layout, item),
        access=access,
        lanes=layout.lanes,
        source=source,
        sink=sink,
        cpu=item.name != "twist_shfl" and cpu,
    )
    if item.name == "twist_shfl":
        # The twist publication is CUDA only, as in the predictor's twist block.
        return _TWIST_SCOPE + device_branches(
            "    if(f.publish_twists!=0) {\n        SYNC();\n" + scan.cuda + "    }\n", CUDA_ONLY
        )
    if reduction:
        return _REDUCE_SCOPE + scan.text
    return "    float* s=reinterpret_cast<float*>(address);\n" + scan.text


@functools.cache
def pass_kernel(item: Pass, layout: TreeLayout, plan: TreePlan, repeat: int = 1, text: str | None = None):
    """A kernel that runs one pass (or the native ``text`` in its scope), ``repeat`` times, on each group's
    scratch.

    Launch ``launch_tiled(dim=[groups // layout.pack], block_dim=layout.block)`` with ``(base, total, sparse,
    plan, f, flag)``; ``base`` is the scratch's address and ``total`` its floats per group.
    """
    text = generated_text(item, plan, layout) if text is None else text
    snippet = layout.native(text, preludes=item.combine.preludes())

    if item.direction is ts.Direction.UP or item.name == "twist_shfl":

        @wp.func_native(snippet)
        def run_pass(
            address: wp.uint64,
            p: tree_kinetic_state.GroupTrees,
            plan: ts.ScanPlan,
            f: tree_kinetic_state.ForceInput,
            flag: int,
            group: int,
        ): ...

    else:

        @wp.func_native(snippet)
        def run_pass(
            address: wp.uint64,
            p_unused: tree_kinetic_state.GroupTrees,
            p: ts.ScanPlan,
            f: tree_kinetic_state.ForceInput,
            flag: int,
            group: int,
        ): ...

    _lanes, _sync, _sub, _ = tree_kinetic_state._thread_helpers(layout)
    pack = layout.pack

    def run(
        base: wp.uint64,
        total: int,
        sparse: tree_kinetic_state.GroupTrees,
        plan: ts.ScanPlan,
        f: tree_kinetic_state.ForceInput,
        flag: int,
    ):
        block, logical = wp.tid()
        lanes = _lanes()
        if lanes[1] == 1 and logical != 0:
            return
        group = block * wp.static(pack) + lanes[2]
        address = base + wp.uint64(4) * wp.uint64(total) * wp.uint64(group)
        for _ in range(wp.static(repeat)):
            run_pass(address, sparse, plan, f, flag, group)

    run.__name__ = run.__qualname__ = f"tree_scan_{item.name}_{layout.bodies}_{layout.lanes}"
    return wp.kernel(module="unique", enable_backward=False)(run)


# ---------------------------------------------------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------------------------------------------------


def random_scratch(layout: TreeLayout, groups: int, seed: int) -> np.ndarray:
    """Random state blocks: unit-quaternion poses, then motion, wrenches and moments of order one."""
    rng = np.random.default_rng(seed)
    total = layout.sizes()["TOTAL"]
    values = rng.standard_normal((groups, total)).astype(np.float32)
    poses = values[:, : 7 * layout.bodies].reshape(groups, layout.bodies, 7)
    quaternions = rng.standard_normal((groups, layout.bodies, 4))
    poses[..., 3:] = (quaternions / np.linalg.norm(quaternions, axis=-1, keepdims=True)).astype(np.float32)
    return values


@dataclass
class Inputs:
    """Arrays of a run: the scratch, the plan structs and the twist publication's force input."""

    scratch: wp.array
    total: int
    sparse: tree_kinetic_state.GroupTrees
    scan: ts.ScanPlan
    force: tree_kinetic_state.ForceInput
    twists: wp.array


def inputs(plan: TreePlan, layout: TreeLayout, groups: int, seed: int, device) -> Inputs:
    total = layout.sizes()["TOTAL"]
    scratch = wp.array(random_scratch(layout, groups, seed).reshape(-1), dtype=float, device=device)
    rng = np.random.default_rng(seed + 1)
    sparse = tree_kinetic_state.GroupTrees()
    sparse.joint_child = wp.array(np.arange(groups * plan.count, dtype=np.int32), dtype=int, device=device)
    force = tree_kinetic_state.ForceInput()
    force.publish_twists = 1
    force.S = wp.array(
        rng.standard_normal((groups * plan.count, 6)).astype(np.float32), dtype=wp.spatial_vector, device=device
    )
    force.vhat = wp.array(rng.standard_normal(groups * plan.count).astype(np.float32), dtype=float, device=device)
    twists = wp.zeros(groups * plan.count, dtype=wp.spatial_vector, device=device)
    force.predicted_twist = twists
    return Inputs(scratch, total, sparse, ts.device_plan(plan, device), force, twists)


def launch(item: Pass, plan: TreePlan, layout: TreeLayout, data: Inputs, groups: int, flag: int, device, repeat=1):
    """Run one pass over ``groups`` trees in place on ``data``."""
    wp.launch_tiled(
        pass_kernel(item, layout, plan, repeat),
        dim=[groups // layout.pack],
        inputs=[wp.uint64(data.scratch.ptr), data.total, data.sparse, data.scan, data.force, flag],
        block_dim=layout.block,
        device=device,
    )


def run(item: Pass, plan: TreePlan, layout: TreeLayout, groups: int, flag: int, seed: int, device):
    """Run the pass on random scratch; return the scratch and the published twists."""
    data = inputs(plan, layout, groups, seed, device)
    launch(item, plan, layout, data, groups, flag, device)
    return data.scratch.numpy().copy(), data.twists.numpy().copy()
