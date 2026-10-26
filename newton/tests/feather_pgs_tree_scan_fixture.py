# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared machinery of the tree-scan tests and ``tools/fpgs_bench/tree_scan_bench.py``.

Tree shapes, the existing held-tree passes compiled as stand-alone natives
(their module texts, unchanged), the ``tree_scan`` configuration of each, and
kernels that run either on the same scratch so that the two can be compared
bit for bit or timed with the same launch. It defines no tests.
"""

import functools
from dataclasses import dataclass

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import sparse_factor, tree_chain_scan, tree_kinetic_state
from newton._src.solvers.feather_pgs import tree_scan as ts
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
    """The held tree's lanes for ``plan``: 32 (or ``packed``) up to 32 bodies, 64 beyond."""
    if plan.count > 32:
        return 64
    if packed:
        return max(packed, 4, 1 << (plan.count - 1).bit_length())
    return 32


def layout_of(plan: TreePlan, lanes: int) -> TreeLayout:
    """A held-tree layout of ``plan`` with one coordinate per body (the scans read only its sizes)."""
    n = plan.count
    return TreeLayout(n, n, n, 1, plan.max_depth, 0, tuple(range(n - 1, -1, -1)), lanes)


def kinetic_plan(plan: TreePlan, device) -> tree_kinetic_state.KineticPlan:
    """The held tree's kinetic plan struct, filled from ``plan`` (the arrays the scans and reductions read)."""
    result = tree_kinetic_state.KineticPlan()
    one = np.zeros(1, np.int32)
    arrays = {
        "parent": plan.parent,
        "depth": plan.depth,
        "jump": plan.jump,
        "children_offsets": plan.children_offsets,
        "children": plan.children if plan.children.size else one,
        "q_index": one,
        "root_slot": one,
        "dfs_order": plan.dfs_order,
        "dfs_end": plan.dfs_end,
        "body_dof_start": np.arange(plan.count, dtype=np.int32),
        "body_dof_count": np.ones(plan.count, np.int32),
        "arts": one.reshape(1, 1),
        "root_of": plan.root_of,
        "root_local": plan.roots,
        "free_root_local": one,
        "free_root_dof": one,
        "dof_passive": np.ones(plan.count, np.int32),
    }
    for name, values in arrays.items():
        setattr(result, name, wp.array(np.ascontiguousarray(values, dtype=np.int32), dtype=int, device=device))
    return result


def chain_plan(plan: TreePlan, device):
    """The held tree's chain plan struct, through the existing builder; ``None`` where it rejects the tree."""
    try:
        host = tree_chain_scan.make_plan(plan.parent)
    except ValueError:
        return None
    result = tree_chain_scan.ChainPlan()
    one = np.zeros(1, np.int32)
    for name, values in (
        ("parent", plan.parent),
        ("depth", plan.depth),
        ("jump", plan.jump),
        ("children_offsets", plan.children_offsets),
        ("children", plan.children if plan.children.size else one),
        ("q_index", one),
        ("root_slot", one),
        ("chain_meta", host["meta"].reshape(-1)),
        ("light_offsets", host["light_offsets"]),
        ("light_children", host["light_children"] if host["light_children"].size else one),
        ("stage_width", host["stage_width"]),
    ):
        setattr(result, name, wp.array(np.ascontiguousarray(values, dtype=np.int32), dtype=int, device=device))
    result.chain_levels = int(host["levels"])
    result.chain_width = int(host["widths"].max())
    return result


# ---------------------------------------------------------------------------------------------------------------------
# Passes: each existing text and its generated configuration, on one scratch layout
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Pass:
    """A held-tree pass: its existing native text and the ``tree_scan`` configuration that replaces it."""

    name: str
    combine: ts.Combine
    direction: ts.Direction
    schedule: ts.Schedule
    site: str
    """Where the existing text lives."""
    plan_struct: str
    """``kinetic`` or ``chain``: the struct the existing text reads."""


POSE, MOTION = ts.TRANSFORM, ts.TWIST.with_zero_acceleration()
WRENCH_MOMENTS = ts.add(ts.Group("wrench", 6), ts.Group("moment", 13, "refresh"))
EXTERNAL = ts.add(ts.Group("wrench", 6))
TWIST_SUM = ts.add(ts.Group("twist", 6))

PASSES = {
    "pose_smem": Pass(
        "pose_smem", POSE, ts.Direction.DOWN, ts.Schedule.JUMP_SMEM, "tree_kinetic_state._POSE", "kinetic"
    ),
    "motion_smem": Pass(
        "motion_smem", MOTION, ts.Direction.DOWN, ts.Schedule.JUMP_SMEM, "tree_kinetic_state._MOTION", "kinetic"
    ),
    "pose_shfl": Pass(
        "pose_shfl", POSE, ts.Direction.DOWN, ts.Schedule.JUMP_SHFL, "tree_kinetic_state._POSE_SHFL", "kinetic"
    ),
    "motion_shfl": Pass(
        "motion_shfl", MOTION, ts.Direction.DOWN, ts.Schedule.JUMP_SHFL, "tree_kinetic_state._MOTION_SHFL", "kinetic"
    ),
    "pose_chain": Pass(
        "pose_chain", POSE, ts.Direction.DOWN, ts.Schedule.HEAVY_CHAIN, "tree_chain_scan._POSE", "chain"
    ),
    "motion_chain": Pass(
        "motion_chain", MOTION, ts.Direction.DOWN, ts.Schedule.HEAVY_CHAIN, "tree_chain_scan._MOTION", "chain"
    ),
    "reduce_level": Pass(
        "reduce_level",
        WRENCH_MOMENTS,
        ts.Direction.UP,
        ts.Schedule.LEVEL,
        "tree_kinetic_state._LEVEL_REDUCE",
        "kinetic",
    ),
    "reduce_interval": Pass(
        "reduce_interval",
        WRENCH_MOMENTS,
        ts.Direction.UP,
        ts.Schedule.DFS_INTERVAL,
        "tree_kinetic_state._INTERVAL_REDUCE",
        "kinetic",
    ),
    "reduce_chain": Pass(
        "reduce_chain",
        WRENCH_MOMENTS,
        ts.Direction.UP,
        ts.Schedule.HEAVY_CHAIN,
        "tree_chain_scan.reduction_source(moments=True)",
        "chain",
    ),
    "external_level": Pass(
        "external_level",
        EXTERNAL,
        ts.Direction.UP,
        ts.Schedule.LEVEL,
        "tree_kinetic_state._EXTERNAL_REDUCE",
        "kinetic",
    ),
    "external_chain": Pass(
        "external_chain",
        EXTERNAL,
        ts.Direction.UP,
        ts.Schedule.HEAVY_CHAIN,
        "tree_chain_scan.reduction_source(moments=False)",
        "chain",
    ),
    "twist_shfl": Pass(
        "twist_shfl", TWIST_SUM, ts.Direction.DOWN, ts.Schedule.JUMP_SHFL, "tree_kinetic_state._TWIST_BLOCK", "kinetic"
    ),
}
"""Every held-tree pass with its generated configuration."""


def applies(item: Pass, plan: TreePlan, lanes: int) -> bool:
    """Whether the existing pass runs on a tree of this plan at ``lanes`` lanes."""
    if item.plan_struct == "chain":
        return plan.chain_packing(2) is not None and plan.count <= 64
    if item.schedule in (ts.Schedule.JUMP_SHFL, ts.Schedule.DFS_INTERVAL):
        return lanes <= 32 and plan.count <= lanes
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
    if item.combine is EXTERNAL:
        return ts.Storage({"wrench": "s[6*({node})+({k})]"})
    return ts.Storage(
        {
            "wrench": f"s[{layout.wrench_offset}+6*({{node}})+({{k}})]",
            "moment": f"s[{layout.moment_offset}+13*({{node}})+({{k}})]",
        }
    )


# Wrappers: one native per pass with one signature, ``(address, plan, flag)``; ``flag`` is ``refresh`` for the
# subtree reductions, ``nonzero`` for the external ones and the twist publication switch for the twist sum.
_REDUCE_SCOPE = """    float* s=reinterpret_cast<float*>(address);
#if defined(__CUDA_ARCH__)
    const int lane=LANE,stride=STRIDE;
#else
    const int lane=0,stride=1;
#endif
    const int refresh=flag,nonzero=flag;
"""

_TWIST_SCOPE = """    const int lane=LANE;
    const int ds=group*$BODIES,js=ds;
"""

_TWIST_SOURCE = (
    "{{const int first=ds+$FIRST,count=$COUNT;"
    "for(int d=0;d<count;++d){{const wp::spatial_vector axis=f.S.data[first+d];const float speed=f.vhat.data[first+d];"
    "for(int k=0;k<6;++k){value}[k]+=axis[k]*speed;}}}}"
)
_TWIST_SINK = (
    "{{wp::spatial_vector twist;for(int k=0;k<6;++k)twist[k]={value}[k];"
    "f.predicted_twist.data[p.joint_child.data[js+{node}]]=twist;}}"
)


def existing_text(item: Pass, layout: TreeLayout) -> str:
    """The pass's existing text, unchanged, in its wrapper scope."""
    if item.name == "pose_smem":
        return tree_kinetic_state._POSE
    if item.name == "motion_smem":
        return tree_kinetic_state._MOTION
    if item.name == "pose_shfl":
        return tree_kinetic_state._POSE_SHFL
    if item.name == "motion_shfl":
        return tree_kinetic_state._MOTION_SHFL
    if item.name == "pose_chain":
        return tree_chain_scan._POSE
    if item.name == "motion_chain":
        return tree_chain_scan._MOTION
    if item.name == "reduce_level":
        return _REDUCE_SCOPE + tree_kinetic_state._LEVEL_REDUCE
    if item.name == "reduce_interval":
        return _REDUCE_SCOPE + tree_kinetic_state._INTERVAL_REDUCE
    if item.name == "reduce_chain":
        return _REDUCE_SCOPE + tree_chain_scan.reduction_source(moments=True, layout=layout)
    if item.name == "external_level":
        return _REDUCE_SCOPE + tree_kinetic_state._EXTERNAL_REDUCE
    if item.name == "external_chain":
        return (
            _REDUCE_SCOPE + "if(nonzero) {\n" + tree_chain_scan.reduction_source(moments=False, layout=layout) + "}\n"
        )
    if item.name == "twist_shfl":
        return _TWIST_SCOPE + tree_kinetic_state._TWIST_BLOCK
    raise KeyError(item.name)


def generated_text(item: Pass, plan: TreePlan, layout: TreeLayout, *, dialect: str, cpu: bool = True) -> str:
    """The pass's ``tree_scan`` configuration in its wrapper scope.

    ``dialect`` ``"held"`` reads the existing plan structs (``p`` in scans, ``plan`` in reductions, depth tests
    for levels); ``"scan"`` reads a :class:`~newton._src.solvers.feather_pgs.tree_scan.ScanPlan` named ``plan``
    with its level CSR, as the scans will once the owners hold a tree plan.
    """
    reduction = item.direction is ts.Direction.UP or item.name == "twist_shfl"
    access = ts.PlanAccess() if reduction else ts.PlanAccess().renamed("p")
    level_csr = dialect != "held"
    source = sink = None
    if item.name == "twist_shfl":
        if dialect == "held":
            first, count = "plan.body_dof_start.data[{node}]", "plan.body_dof_count.data[{node}]"
        else:
            first, count = "plan.dof_start.data[{node}]", "plan.dof_start.data[{node}+1]-plan.dof_start.data[{node}]"
        source = _TWIST_SOURCE.replace("$FIRST", first).replace("$COUNT", count)
        sink = _TWIST_SINK
        access = ts.PlanAccess()
    scan = ts.tree_scan(
        plan,
        item.combine,
        item.direction,
        item.schedule,
        storage=None if source is not None else storage(layout, item),
        access=access,
        lanes=layout.lanes,
        level_csr=level_csr,
        source=source,
        sink=sink,
        cpu=item.name != "twist_shfl" and cpu,
    )
    if item.name == "twist_shfl":
        return (
            _TWIST_SCOPE
            + "#if defined(__CUDA_ARCH__)\n    if(f.publish_twists!=0) {\n        SYNC();\n"
            + scan.cuda
            + "    }\n#endif\n"
        )
    if reduction:
        if item.combine is EXTERNAL:
            return _REDUCE_SCOPE + "if(nonzero) {\n" + scan.text + "}\n"
        return _REDUCE_SCOPE + scan.text
    return "    float* s=reinterpret_cast<float*>(address);\n" + scan.text


def _plan_type(item: Pass, dialect: str):
    if dialect == "scan":
        return ts.ScanPlan
    return tree_chain_scan.ChainPlan if item.plan_struct == "chain" else tree_kinetic_state.KineticPlan


@functools.cache
def pass_kernel(item: Pass, layout: TreeLayout, arm: str, plan=None, repeat: int = 1):
    """A kernel that runs one pass, ``repeat`` times, on each group's scratch: the ``existing`` text or a
    ``generated`` configuration (``generated-held`` or ``generated-scan`` dialect).

    Launch ``launch_tiled(dim=[groups // layout.pack], block_dim=layout.block)`` with ``(base, total, plan,
    flag, f, sparse)``; ``base`` is the scratch's address and ``total`` its floats per group. ``staged``
    copies each group's block into shared memory first, as the held-tree kernels hold it.
    """
    if arm == "existing":
        text, dialect = existing_text(item, layout), "held"
    else:
        dialect = arm.split("-", 1)[1]
        text = generated_text(item, plan, layout, dialect=dialect)
    preludes = item.combine.preludes() if arm != "existing" else ()
    snippet = layout.native(text) if not preludes else _with_preludes(layout, text, preludes)
    plan_type = _plan_type(item, dialect)

    if item.name == "twist_shfl":

        @wp.func_native(snippet)
        def run_pass(
            address: wp.uint64,
            p: sparse_factor.SparsePlan,
            plan: plan_type,
            f: tree_kinetic_state.ForceInput,
            flag: int,
            group: int,
        ): ...

    elif item.direction is ts.Direction.UP:

        @wp.func_native(snippet)
        def run_pass(
            address: wp.uint64,
            p: sparse_factor.SparsePlan,
            plan: plan_type,
            f: tree_kinetic_state.ForceInput,
            flag: int,
            group: int,
        ): ...

    else:

        @wp.func_native(snippet)
        def run_pass(
            address: wp.uint64,
            p_unused: sparse_factor.SparsePlan,
            p: plan_type,
            f: tree_kinetic_state.ForceInput,
            flag: int,
            group: int,
        ): ...

    _lanes, _sync, _sub, _ = tree_kinetic_state._thread_helpers(layout)
    # A chain scan addresses its block's threads directly: one tree per block, whatever the lane group says.
    pack = 0 if item.plan_struct == "chain" else layout.pack

    def run(
        base: wp.uint64,
        total: int,
        sparse: sparse_factor.SparsePlan,
        plan: plan_type,
        f: tree_kinetic_state.ForceInput,
        flag: int,
    ):
        block, logical = wp.tid()
        lanes = _lanes()
        if lanes[1] == 1 and logical != 0:
            return
        group = block
        if wp.static(pack > 0):
            group = block * wp.static(pack) + lanes[2]
        address = base + wp.uint64(4) * wp.uint64(total) * wp.uint64(group)
        for _ in range(wp.static(repeat)):
            run_pass(address, sparse, plan, f, flag, group)

    run.__name__ = run.__qualname__ = f"tree_scan_{item.name}_{arm.replace('-', '_')}_{layout.bodies}_{layout.lanes}"
    return wp.kernel(module="unique", enable_backward=False)(run)


def block_dim(item: Pass, plan: TreePlan, layout: TreeLayout) -> int:
    """Threads per block: the layout's, or 64 where a chain packing spans two warps."""
    if item.plan_struct == "chain":
        packing = plan.chain_packing(2)
        return max(layout.block, packing.lanes if packing is not None else 32)
    return layout.block


def _with_preludes(layout: TreeLayout, text: str, preludes) -> str:
    from newton._src.solvers.feather_pgs.device_preludes import dual_native, lane_group  # noqa: PLC0415

    return dual_native(layout.render(text), preludes=(lane_group(layout.lanes), *preludes))


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
    sparse: sparse_factor.SparsePlan
    held: object
    scan: ts.ScanPlan
    force: tree_kinetic_state.ForceInput
    twists: wp.array


def inputs(item: Pass, plan: TreePlan, layout: TreeLayout, groups: int, seed: int, device) -> Inputs:
    total = layout.sizes()["TOTAL"]
    scratch = wp.array(random_scratch(layout, groups, seed).reshape(-1), dtype=float, device=device)
    held = chain_plan(plan, device) if item.plan_struct == "chain" else kinetic_plan(plan, device)
    rng = np.random.default_rng(seed + 1)
    sparse = sparse_factor.SparsePlan()
    sparse.joint_child = wp.array(np.arange(groups * plan.count, dtype=np.int32), dtype=int, device=device)
    force = tree_kinetic_state.ForceInput()
    force.publish_twists = 1
    force.S = wp.array(
        rng.standard_normal((groups * plan.count, 6)).astype(np.float32), dtype=wp.spatial_vector, device=device
    )
    force.vhat = wp.array(rng.standard_normal(groups * plan.count).astype(np.float32), dtype=float, device=device)
    twists = wp.zeros(groups * plan.count, dtype=wp.spatial_vector, device=device)
    force.predicted_twist = twists
    return Inputs(scratch, total, sparse, held, ts.device_plan(plan, device), force, twists)


def launch(
    item: Pass, plan: TreePlan, layout: TreeLayout, arm: str, data: Inputs, groups: int, flag: int, device, repeat=1
):
    """Run one arm of a pass over ``groups`` trees in place on ``data``."""
    kernel = pass_kernel(item, layout, arm, plan, repeat)
    pack = 1 if item.plan_struct == "chain" else layout.pack
    structure = data.scan if arm == "generated-scan" else data.held
    wp.launch_tiled(
        kernel,
        dim=[groups // pack],
        inputs=[wp.uint64(data.scratch.ptr), data.total, data.sparse, structure, data.force, flag],
        block_dim=block_dim(item, plan, layout),
        device=device,
    )


def compare(item: Pass, plan: TreePlan, layout: TreeLayout, groups: int, flag: int, seed: int, device):
    """Run the existing text and both generated dialects on the same inputs; return their outputs per arm."""
    results = {}
    for arm in ("existing", "generated-held", "generated-scan"):
        data = inputs(item, plan, layout, groups, seed, device)
        launch(item, plan, layout, arm, data, groups, flag, device)
        results[arm] = (data.scratch.numpy().copy(), data.twists.numpy().copy())
    return results


# ---------------------------------------------------------------------------------------------------------------------
# The general route: whole articulations through the existing serial kernels and their scan configurations
# ---------------------------------------------------------------------------------------------------------------------


def articulated_model(shapes, worlds: int, device, seed: int = 0):
    """A model of the named tree shapes per world: free roots, revolute, prismatic and ball joints, random inertia."""
    import newton  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    scene = newton.ModelBuilder()
    for shape in shapes:
        parents = SHAPES[shape]
        bodies, joints = [], []
        for index, parent in enumerate(parents):
            inertia = np.diag(rng.uniform(0.02, 0.09, 3))
            body = scene.add_link(
                mass=float(rng.uniform(0.3, 2.0)),
                com=wp.vec3(*rng.uniform(-0.05, 0.05, 3)),
                inertia=wp.mat33(inertia),
            )
            bodies.append(body)
            anchor = wp.transform(
                wp.vec3(*rng.uniform(-0.2, 0.2, 3)), wp.quat_rpy(*[float(a) for a in rng.uniform(-0.5, 0.5, 3)])
            )
            child_anchor = wp.transform(wp.vec3(*rng.uniform(-0.05, 0.05, 3)), wp.quat_identity())
            if index == 0:
                joint = scene.add_joint_free(body)
            elif index % 7 == 3:
                joint = scene.add_joint_ball(bodies[parent], body, parent_xform=anchor, child_xform=child_anchor)
            elif index % 5 == 2:
                joint = scene.add_joint_prismatic(
                    bodies[parent], body, axis=newton.Axis.Z, parent_xform=anchor, child_xform=child_anchor
                )
            else:
                axis = rng.standard_normal(3)
                joint = scene.add_joint_revolute(
                    bodies[parent],
                    body,
                    axis=tuple(float(a) for a in axis / np.linalg.norm(axis)),
                    parent_xform=anchor,
                    child_xform=child_anchor,
                )
            joints.append(joint)
        scene.add_articulation(joints)
    builder = newton.ModelBuilder()
    builder.replicate(scene, worlds)
    model = builder.finalize(device=device)
    q = model.joint_q.numpy()
    types, q_start = model.joint_type.numpy(), model.joint_q_start.numpy()
    for j in range(model.joint_count):
        s = q_start[j]
        if types[j] in (int(newton.JointType.REVOLUTE), int(newton.JointType.PRISMATIC)):
            q[s] = rng.uniform(-1.0, 1.0)
        elif types[j] == int(newton.JointType.BALL):
            quaternion = rng.standard_normal(4)
            q[s : s + 4] = quaternion / np.linalg.norm(quaternion)
        elif types[j] == int(newton.JointType.FREE):
            quaternion = rng.standard_normal(4)
            q[s : s + 3] = rng.uniform(-1.0, 1.0, 3)
            q[s + 3 : s + 7] = quaternion / np.linalg.norm(quaternion)
    model.joint_q.assign(q)
    model.joint_qd.assign(rng.uniform(-1.0, 1.0, model.joint_dof_count).astype(np.float32))
    return model


@dataclass
class SerialArrays:
    """Inputs and outputs of the serial route's kernels on one model."""

    model: object
    joint_end: wp.array
    body_X_com: wp.array
    is_free_rigid: wp.array
    gravity: wp.array
    joint_articulation: wp.array
    dof_joint: wp.array
    size_groups: list
    articulations: wp.array
    outputs: dict


def serial_arrays(model) -> SerialArrays:
    from newton._src.solvers.feather_pgs import kernels  # noqa: PLC0415

    device = model.device
    starts = model.articulation_start.numpy()
    body_X_com = wp.zeros(model.body_count, dtype=wp.transform, device=device)
    wp.launch(
        kernels.compute_com_transforms, model.body_count, inputs=[model.body_com], outputs=[body_X_com], device=device
    )
    joint_articulation = np.repeat(np.arange(model.articulation_count, dtype=np.int32), np.diff(starts))
    plan = TreePlan.build(model.joint_ancestor.numpy(), np.diff(model.joint_qd_start.numpy()))
    counts = np.diff(model.joint_qd_start.numpy()[starts])
    rng = np.random.default_rng(3)
    n = model.body_count
    outputs = {
        "body_q": wp.zeros(n, dtype=wp.transform, device=device),
        "body_q_com": wp.zeros(n, dtype=wp.transform, device=device),
        "articulation_origin": wp.zeros(model.articulation_count, dtype=wp.vec3, device=device),
        "joint_S_s": wp.zeros(model.joint_dof_count, dtype=wp.spatial_vector, device=device),
        "body_I_s": wp.zeros(n, dtype=wp.spatial_matrix, device=device),
        "body_inertia_terms": wp.zeros((n, 12), dtype=float, device=device),
        "body_v_s": wp.zeros(n, dtype=wp.spatial_vector, device=device),
        "body_f_s": wp.zeros(n, dtype=wp.spatial_vector, device=device),
        "body_a_s": wp.zeros(n, dtype=wp.spatial_vector, device=device),
        "fk_id_cache_valid": wp.zeros(model.articulation_count, dtype=int, device=device),
        "body_ft_s": wp.zeros(n, dtype=wp.spatial_vector, device=device),
        "tau": wp.zeros(model.joint_dof_count, dtype=float, device=device),
        "body_I_c": wp.zeros(n, dtype=wp.spatial_matrix, device=device),
        "body_f_ext": wp.array(rng.standard_normal((n, 6)).astype(np.float32), dtype=wp.spatial_vector, device=device),
        "joint_f": wp.array(rng.standard_normal(model.joint_dof_count).astype(np.float32), dtype=float, device=device),
        "spring": wp.array(rng.uniform(0, 2, model.joint_dof_count).astype(np.float32), dtype=float, device=device),
        "damping": wp.array(rng.uniform(0, 1, model.joint_dof_count).astype(np.float32), dtype=float, device=device),
        "mask": wp.ones(model.articulation_count, dtype=int, device=device),
        "H": {
            int(size): wp.zeros((int(np.sum(counts == size)), int(size), int(size)), dtype=float, device=device)
            for size in np.unique(counts)
        },
        "drive_row": wp.full(model.joint_dof_count, -1, dtype=int, device=device),
        "row_K": wp.zeros(1, dtype=float, device=device),
    }
    return SerialArrays(
        model,
        wp.array(starts[1:].astype(np.int32), dtype=int, device=device),
        body_X_com,
        wp.zeros(model.articulation_count, dtype=int, device=device),
        wp.array([[0.0, 0.0, -9.81]], dtype=wp.vec3, device=device),
        wp.array(joint_articulation, dtype=int, device=device),
        wp.array(plan.dof_node, dtype=int, device=device),
        [
            (int(size), wp.array(np.flatnonzero(counts == size).astype(np.int32), dtype=int, device=device))
            for size in np.unique(counts)
        ],
        wp.array(np.arange(model.articulation_count, dtype=np.int32), dtype=int, device=device),
        outputs,
    )


SERIAL_PASSES = (
    "fk_kinematics",
    "fk_id",
    "tau",
    "composite_inertia",
    "composite_inertia_warp",
    "composite_inertia_compact",
    "crba",
)
"""The general route's passes; the two warp forms of the composite inertia are CUDA only."""


def serial_launch(name: str, arm: str, data: SerialArrays):
    """Launch one serial pass, ``existing`` (``kernels``) or ``generated`` (``tree_passes``), with one launch shape."""
    from newton._src.solvers.feather_pgs import kernels, tree_passes  # noqa: PLC0415

    m, o, device = data.model, data.outputs, data.model.device
    arts = m.articulation_count
    common = [m.articulation_start, data.joint_end, m.joint_type, m.joint_parent, m.joint_child, m.joint_q_start]
    frames = [m.joint_qd_start, m.joint_q, m.joint_qd, m.joint_X_p, m.joint_X_c, data.body_X_com, m.joint_axis]
    if name == "fk_kinematics":
        kernel = kernels.eval_rigid_fk_kinematics if arm == "existing" else tree_passes.fk_kinematics
        wp.launch(
            kernel,
            arts,
            inputs=[*common, *frames, m.joint_dof_dim, m.body_com],
            outputs=[
                o[k]
                for k in (
                    "body_q",
                    "body_q_com",
                    "articulation_origin",
                    "joint_S_s",
                    "body_v_s",
                    "body_a_s",
                    "fk_id_cache_valid",
                )
            ],
            device=device,
        )
    elif name == "fk_id":
        kernel = kernels.eval_rigid_fk_id if arm == "existing" else tree_passes.fk_id
        wp.launch(
            kernel,
            arts,
            inputs=[
                *common,
                *frames,
                m.joint_dof_dim,
                m.body_com,
                m.body_mass,
                m.body_inertia,
                data.is_free_rigid,
                1,
                1,
                0,
                data.gravity,
            ],
            outputs=[
                o[k]
                for k in (
                    "body_q",
                    "body_q_com",
                    "articulation_origin",
                    "joint_S_s",
                    "body_I_s",
                    "body_inertia_terms",
                    "body_v_s",
                    "body_f_s",
                    "body_a_s",
                    "fk_id_cache_valid",
                )
            ],
            device=device,
        )
    elif name == "tau":
        kernel = kernels.eval_rigid_tau if arm == "existing" else tree_passes.inverse_dynamics_tau
        o["body_ft_s"].zero_()
        inputs = [
            m.articulation_start,
            data.joint_end,
            m.joint_type,
            m.joint_parent,
            m.joint_child,
            data.joint_articulation,
            m.joint_qd_start,
            m.joint_q_start,
            m.joint_dof_dim,
            o["joint_f"],
            m.joint_q,
            m.joint_qd,
            o["spring"],
            m.joint_q,
            o["damping"],
            o["joint_S_s"],
            o["body_f_s"],
            o["body_f_ext"],
            m.body_flags,
            o["body_q"],
            m.body_com,
            o["articulation_origin"],
        ]
        if arm != "existing":
            inputs.append(0)
        wp.launch(kernel, arts, inputs=inputs, outputs=[o["body_ft_s"], o["tau"]], device=device)
    elif name == "composite_inertia":
        if arm == "existing":
            wp.launch(
                kernels.compute_composite_inertia,
                arts,
                inputs=[
                    m.articulation_start,
                    data.joint_end,
                    o["mask"],
                    m.joint_ancestor,
                    m.joint_child,
                    o["body_I_s"],
                ],
                outputs=[o["body_I_c"]],
                device=device,
            )
        else:
            wp.launch(
                tree_passes.composite_inertia,
                arts,
                inputs=[m.articulation_start, data.joint_end, o["mask"], m.joint_parent, m.joint_child, o["body_I_s"]],
                outputs=[o["body_I_c"]],
                device=device,
            )
    elif name in ("composite_inertia_warp", "composite_inertia_compact"):
        # The solver's CUDA form: one warp per articulation, lanes over the 36 channels.
        from newton._src.solvers.feather_pgs.solver_feather_pgs import (  # noqa: PLC0415
            _COMPOSITE_INERTIA_WARPS_PER_BLOCK as warps,
        )
        from newton._src.solvers.feather_pgs.solver_feather_pgs import (  # noqa: PLC0415
            _get_composite_inertia_warp_kernel,
        )

        compact = name == "composite_inertia_compact"
        if arm == "existing":
            kernel = _get_composite_inertia_warp_kernel(str(getattr(device, "arch", "")), warps, compact_terms=compact)
        else:
            kernel = tree_passes.composite_inertia_lanes(warps, compact)
        source = [m.body_mass, o["body_inertia_terms"]] if compact else [o["body_I_s"]]
        wp.launch_tiled(
            kernel,
            dim=[(arts + warps - 1) // warps],
            inputs=[
                arts,
                data.articulations,
                m.articulation_start,
                data.joint_end,
                m.joint_ancestor,
                m.joint_child,
                *source,
            ],
            outputs=[o["body_I_c"]],
            block_dim=32 * warps,
            device=device,
        )
    elif name == "crba":
        # One launch per size group, as the solver groups articulations by coordinate count.
        for dofs, group_to_art in data.size_groups:
            _crba_launch(arm, data, dofs, group_to_art)
    else:
        raise KeyError(name)


def _crba_launch(arm, data, dofs, group_to_art):
    from newton._src.solvers.feather_pgs import kernels, tree_passes  # noqa: PLC0415

    m, o = data.model, data.outputs
    first = _articulation_dof_start(m)
    tail = [o["joint_S_s"], o["body_I_c"], group_to_art, dofs, 0, o["drive_row"], o["row_K"]]
    topology = [o["mask"], m.joint_ancestor, m.joint_child, m.joint_qd_start, m.joint_dof_dim]
    if arm == "existing":
        kernel, inputs = kernels.crba_fill_par_dof, [m.articulation_start, first, *topology, *tail]
    else:
        kernel, inputs = tree_passes.crba_fill, [first, *topology, m.articulation_start, data.dof_joint, *tail]
    wp.launch(kernel, group_to_art.size * dofs, inputs=inputs, outputs=[o["H"][dofs]], device=m.device)


def _numpy(value):
    if isinstance(value, dict):
        return np.concatenate([value[size].numpy().reshape(-1) for size in sorted(value)])
    return value.numpy().copy()


def _articulation_dof_start(model):
    cached = getattr(model, "_tree_scan_dof_start", None)
    if cached is None:
        starts = model.articulation_start.numpy()
        cached = wp.array(model.joint_qd_start.numpy()[starts[:-1]].astype(np.int32), dtype=int, device=model.device)
        model._tree_scan_dof_start = cached
    return cached


SERIAL_OUTPUTS = {
    "fk_kinematics": ("body_q", "body_q_com", "articulation_origin", "joint_S_s", "body_v_s", "body_a_s"),
    "fk_id": (
        "body_q",
        "body_q_com",
        "joint_S_s",
        "body_I_s",
        "body_inertia_terms",
        "body_v_s",
        "body_f_s",
        "body_a_s",
    ),
    "tau": ("tau",),
    "composite_inertia": ("body_I_c",),
    "composite_inertia_warp": ("body_I_c",),
    "composite_inertia_compact": ("body_I_c",),
    "crba": ("H",),
}


def serial_compare(model):
    """Run each serial pass's two arms in turn on identical inputs; return ``{pass: {output: (existing, generated)}}``.

    The arms share the outputs of earlier passes as inputs (the existing arm's), so each comparison is one pass's.
    """
    results = {}
    for name in SERIAL_PASSES:
        if name.startswith("composite_inertia_") and not model.device.is_cuda:
            continue
        arms = {}
        for arm in ("existing", "generated"):
            data = serial_arrays(model)
            # Earlier passes, existing arm, as inputs.
            for previous in ("fk_id", "tau", "composite_inertia"):
                if SERIAL_PASSES.index(previous) < SERIAL_PASSES.index(name):
                    serial_launch(previous, "existing", data)
            serial_launch(name, arm, data)
            arms[arm] = {key: _numpy(data.outputs[key]) for key in SERIAL_OUTPUTS[name]}
        results[name] = {key: (arms["existing"][key], arms["generated"][key]) for key in SERIAL_OUTPUTS[name]}
    return results
