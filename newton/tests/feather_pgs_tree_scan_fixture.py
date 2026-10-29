# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared machinery of the tree-scan and tree-pass tests and ``tools/fpgs_bench/tree_scan_bench.py``.

Tree shapes, every held-tree scan configuration compiled as a stand-alone
native on a held tree's scratch layout (to compare with a float64 reference or
time with one launch), articulated models of the shapes, and the general
route's passes on a solver's own arrays (:class:`GeneralRoute`), with the tape
and central-difference gradients of each. It defines no tests.
"""

import functools
from dataclasses import dataclass
from unittest import mock

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import sparse_factor, tree_kinetic_state
from newton._src.solvers.feather_pgs import tree_scan as ts
from newton._src.solvers.feather_pgs.device_preludes import device_branches
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
EXTERNAL = ts.add(ts.Group("wrench", 6))
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
        Pass("external_level", EXTERNAL, ts.Direction.UP, ts.Schedule.LEVEL),
        Pass("external_chain", EXTERNAL, ts.Direction.UP, ts.Schedule.HEAVY_CHAIN),
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
    if item.combine is EXTERNAL:
        return ts.Storage({"wrench": "s[6*({node})+({k})]"})
    return ts.Storage(
        {
            "wrench": f"s[{layout.wrench_offset}+6*({{node}})+({{k}})]",
            "moment": f"s[{layout.moment_offset}+13*({{node}})+({{k}})]",
        }
    )


# One native per pass with one signature, ``(address, plan, flag)``; ``flag`` is ``refresh`` for the subtree
# reductions, ``nonzero`` for the external ones and the twist publication switch for the twist sum.
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
        return _TWIST_SCOPE + device_branches(
            "    if(f.publish_twists!=0) {\n        SYNC();\n" + scan.cuda + "    }\n", ""
        )
    if reduction:
        if item.combine is EXTERNAL:
            return _REDUCE_SCOPE + "if(nonzero) {\n" + scan.text + "}\n"
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
            p: sparse_factor.SparsePlan,
            plan: ts.ScanPlan,
            f: tree_kinetic_state.ForceInput,
            flag: int,
            group: int,
        ): ...

    else:

        @wp.func_native(snippet)
        def run_pass(
            address: wp.uint64,
            p_unused: sparse_factor.SparsePlan,
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
        sparse: sparse_factor.SparsePlan,
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
    sparse: sparse_factor.SparsePlan
    scan: ts.ScanPlan
    force: tree_kinetic_state.ForceInput
    twists: wp.array


def inputs(plan: TreePlan, layout: TreeLayout, groups: int, seed: int, device) -> Inputs:
    total = layout.sizes()["TOTAL"]
    scratch = wp.array(random_scratch(layout, groups, seed).reshape(-1), dtype=float, device=device)
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


# ---------------------------------------------------------------------------------------------------------------------
# The general route
# ---------------------------------------------------------------------------------------------------------------------


def articulated_model(shapes, worlds: int, device, seed: int = 0, requires_grad: bool = False):
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
    model = builder.finalize(device=device, requires_grad=requires_grad)
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
class GeneralRoute:
    """A model's general-route passes on a solver's own arrays and binding (``_dynamics``), with random
    external wrenches, generalized forces, and passive springs and dampers on the one-coordinate axes."""

    model: object
    solver: object
    state: object
    control: object
    aug: object
    """The augmented state: the solver, or with gradients the output state (``_prepare_augmented_state``)."""
    size_groups: list
    """Per coordinate count: ``(size, group_to_art)``, the articulations of that size."""
    H: dict
    """Per coordinate count: the joint-space inertias of its articulations."""


def general_route(model, seed: int = 3) -> GeneralRoute:
    """The general route of ``model`` (see :class:`GeneralRoute`)."""
    import newton  # noqa: PLC0415
    from newton.solvers import SolverFeatherPGS  # noqa: PLC0415

    device = model.device
    with mock.patch.dict(SolverFeatherPGS._route_pins, {"evaluator": "ordinary"}):
        solver = SolverFeatherPGS(model)
    state, control = model.state(), model.control()
    aug = solver._prepare_augmented_state(state, model.state(), control)
    rng = np.random.default_rng(seed)
    if model.body_count:
        state.body_f.assign(rng.standard_normal((model.body_count, 6)).astype(np.float32))
    dofs = model.joint_dof_count
    control.joint_f.assign(rng.standard_normal(dofs).astype(np.float32))
    dof_joint = np.repeat(np.arange(model.joint_count), np.diff(model.joint_qd_start.numpy()))
    axes = np.isin(
        model.joint_type.numpy()[dof_joint], (int(newton.JointType.REVOLUTE), int(newton.JointType.PRISMATIC))
    )
    for name, low, high in (
        ("_passive_spring_stiffness", 0.5, 2.0),
        ("_passive_spring_ref", -0.5, 0.5),
        ("_passive_joint_damping", 0.1, 0.5),
    ):
        values = np.where(axes, rng.uniform(low, high, dofs), 0.0).astype(np.float32)
        setattr(solver, name, wp.array(values, dtype=float, device=device, requires_grad=model.requires_grad))
    solver.mass_update_mask.fill_(1)
    starts = model.articulation_start.numpy()
    counts = np.diff(model.joint_qd_start.numpy()[starts])
    sizes = [int(size) for size in np.unique(counts)]
    return GeneralRoute(
        model,
        solver,
        state,
        control,
        aug,
        [(size, wp.array(np.flatnonzero(counts == size).astype(np.int32), dtype=int, device=device)) for size in sizes],
        {
            size: wp.zeros(
                (int(np.sum(counts == size)), size, size), dtype=float, device=device, requires_grad=model.requires_grad
            )
            for size in sizes
        },
    )


GENERAL_PASSES = ("fk_id", "tau", "composite_inertia", "crba")
"""The general route's passes, in step order."""


def pass_launch(route: GeneralRoute, name: str, *, block_dim: int | None = None, **values):
    """One general-route pass as a launch callable, bound once to the solver's binding of the route's state
    (``values``: its switches), with the solver's block sizes or ``block_dim``."""
    from newton._src.solvers.feather_pgs import tree_passes  # noqa: PLC0415

    m, solver, device = route.model, route.solver, route.model.device
    dynamics = solver._dynamics(route.state, route.aug, route.control, **values)
    kernels = {
        "fk_id": (tree_passes.fk_id, tree_passes.SERIAL_BLOCK_DIM),
        "tau": (tree_passes.inverse_dynamics_tau, tree_passes.SERIAL_BLOCK_DIM),
        "composite_inertia": (tree_passes.composite_inertia, tree_passes.SERIAL_BLOCK_DIM),
    }
    if name in kernels:
        kernel, solver_block = kernels[name]
        return functools.partial(
            wp.launch,
            kernel,
            m.articulation_count,
            inputs=[dynamics],
            block_dim=block_dim or solver_block,
            device=device,
        )
    if name != "crba":
        raise KeyError(name)
    launches = [
        functools.partial(
            wp.launch,
            tree_passes.crba_fill,
            group_to_art.size * size,
            inputs=[dynamics, solver.body_I_c, group_to_art, size],
            outputs=[route.H[size]],
            block_dim=block_dim or 128,
            device=device,
        )
        for size, group_to_art in route.size_groups
    ]
    return lambda: [launch() for launch in launches]


def run_pass(route: GeneralRoute, name: str, **values):
    """Launch one general-route pass (:func:`pass_launch`)."""
    pass_launch(route, name, **values)()


def run_inputs(route: GeneralRoute, name: str):
    """Run the passes whose outputs ``name`` reads, with every inertia record."""
    for previous in GENERAL_PASSES[: GENERAL_PASSES.index(name)]:
        run_pass(route, previous, write_inertia_records=1)


def joint_space_inertias(route: GeneralRoute) -> np.ndarray:
    """The route's joint-space inertias by articulation, zero-padded to the widest (Newton's mass-matrix layout)."""
    width = max(size for size, _ in route.size_groups)
    H = np.zeros((route.model.articulation_count, width, width), dtype=np.float32)
    for size, group_to_art in route.size_groups:
        H[group_to_art.numpy(), :size, :size] = route.H[size].numpy()
    return H


GRADIENT_INPUTS = {
    "fk_id": ("joint_q", "joint_qd", "body_mass", "body_inertia"),
    "tau": ("joint_q", "joint_qd", "joint_f", "joint_S_s", "body_f_s", "body_f", "body_q"),
    "composite_inertia": ("_body_inertia_terms",),
    "crba": ("joint_S_s", "body_I_c"),
}
"""Each pass's inputs whose gradients the tests check: the state's, the control's, the augmented state's,
the solver's or the model's arrays."""

GRADIENT_OUTPUTS = {
    "fk_id": ("body_q", "body_q_com", "joint_S_s", "_body_inertia_terms", "body_v_s", "body_f_s", "body_a_s"),
    "tau": ("body_ft_s", "joint_tau"),
    "composite_inertia": ("body_I_c",),
    "crba": (),
}
"""Each pass's outputs, a random linear form of which the gradient tests differentiate (CRBA: its matrices)."""


def owner_array(route: GeneralRoute, key: str) -> wp.array:
    """The array ``key`` of the route's state, control, augmented state, solver or model, in that order."""
    for owner in (route.state, route.control, route.aug, route.solver, route.model):
        value = getattr(owner, key, None)
        if isinstance(value, wp.array):
            return value
    raise KeyError(key)


def _outputs(route: GeneralRoute, name: str) -> list:
    if name == "crba":
        return [route.H[size] for size, _ in route.size_groups]
    return [owner_array(route, key) for key in GRADIENT_OUTPUTS[name]]


def _form(model, name: str, seed: int):
    """A route with ``name``'s inputs in place, and random weights of a linear form of its outputs."""
    route = general_route(model)
    run_inputs(route, name)
    rng = np.random.default_rng(seed)
    return route, [(array, rng.standard_normal(array.numpy().shape)) for array in _outputs(route, name)]


def pass_gradients(model, name: str, seed: int = 5) -> dict:
    """The gradients of a random linear form of pass ``name``'s outputs with respect to its inputs
    (:data:`GRADIENT_INPUTS`), by Warp's tape (``model`` requires gradients)."""
    route, weights = _form(model, name, seed)
    inputs = {key: owner_array(route, key) for key in GRADIENT_INPUTS[name]}
    tape = wp.Tape()
    with tape:
        run_pass(route, name, write_inertia_records=1)
    for array in inputs.values():
        array.grad.zero_()
    tape.backward(
        grads={array: wp.array(w.astype(np.float32), dtype=array.dtype, device=model.device) for array, w in weights}
    )
    gradients = {key: array.grad.numpy().astype(np.float64) for key, array in inputs.items()}
    tape.zero()
    return gradients


def pass_difference(model, name: str, key: str, step: float, indices, seed: int = 5) -> np.ndarray:
    """The same gradient with respect to input ``key`` by central differences, at the flat ``indices``
    (``nan`` elsewhere)."""
    route, weights = _form(model, name, seed)
    target = owner_array(route, key)
    base = target.numpy().copy()

    def outputs(values):
        target.assign(values.reshape(base.shape))
        for array, _ in weights:
            array.zero_()
        run_pass(route, name, write_inertia_records=1)
        return [array.numpy().astype(np.float64) for array, _ in weights]

    flat = base.reshape(-1)
    gradient = np.full(flat.size, np.nan)
    for k in indices:
        values = flat.copy()
        values[k] += step
        plus = outputs(values)
        values[k] -= 2.0 * step
        minus = outputs(values)
        gradient[k] = sum(float(np.sum((a - b) * w)) for a, b, (_, w) in zip(plus, minus, weights, strict=True)) / (
            2.0 * step
        )
    target.assign(base)
    return gradient.reshape(base.shape)
