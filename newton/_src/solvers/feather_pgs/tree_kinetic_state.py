# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers

# SPDX-License-Identifier: Apache-2.0

"""Generated current/next tree moments, bias and fused state publication."""

import dataclasses
import functools
import hashlib
from types import SimpleNamespace

import warp as wp

from ...core import memory
from ...core.execution import Device, Use, Work, shape
from ...sim import BodyFlags, JointType
from . import (
    cached_force,
    field_group,
)
from . import tree_scan as ts
from .device_preludes import (
    CUDA_ONLY,
    HELD,
    STATUS_CLOCK,
    device_branches,
    dual_native,
    memory_words,
    substitute,
)
from .factor import kernels
from .factor.kernels import Clock, clock_changed, clock_refresh
from .factor.walk import (
    WALK_COORDINATE,
    WALK_FRAME,
    WALK_LINK,
    WALK_NOW,
    WALK_UNPACKED,
    WalkTrees,
    walk_down_prelude,
    walk_loads_prelude,
)
from .field_group import Field
from .kernel_template import KernelTemplate, items, python_kernel
from .publication_data import PublicationData
from .spatial_laws import (
    GEOMETRY_REQUEST,
    INERTIA_CHANNELS,
    SPATIAL,
    body_wrench,
    inertia_at,
    inertia_matrix,
    inertia_record,
    joint_columns,
    joint_frame_pose,
    joint_motion,
    joint_transform,
    pack_inertia,
    point_velocity,
)
from .tree_kinetic_layout import REFERENCE_LAYOUT, TreeLayout
from .tree_plan import TreePlan

HELD_PLAN = (
    *ts.SCAN_PLAN,
    # Each coordinate's joint coordinate in the state bank (group-major) and each group's free-root slots.
    Field("q_index", wp.int32, ("group_dof",), owner="plan"),
    Field("root_slot", wp.int32, ("root_slot",), owner="plan"),
    # Each local body's first local coordinate and coordinate count (its joint's coordinates).
    Field("body_dof_start", wp.int32, ("node",), owner="plan"),
    Field("body_dof_count", wp.int32, ("node",), owner="plan"),
    # Forest groups: the articulation of each root, each body's root index, each root's local body, each free
    # root's local body and coordinate offset, and whether a coordinate takes passive springs.
    Field("arts", wp.int32, ("group", "root"), owner="plan"),
    Field("root_of", wp.int32, ("node",), owner="plan"),
    Field("root_local", wp.int32, ("root",), owner="plan"),
    Field("free_root_local", wp.int32, ("free_root",), owner="plan"),
    Field("free_root_dof", wp.int32, ("free_root",), owner="plan"),
    Field("dof_passive", wp.int32, ("dof",), owner="plan"),
    # The trees' state blocks where they exceed shared memory (TreeLayout.global_block): a tree's TOTAL words.
    Field("scratch", wp.float32, ("scratch",), owner="plan"),
)
"""A held group's template tree as its kernels read it: the scan plan (:data:`.tree_scan.SCAN_PLAN`) and the
group's coordinate maps."""

HeldPlan = field_group.struct("HeldPlan", HELD_PLAN, __name__)


KINETIC_CACHE = field_group.FieldGroup(
    Field("bias", wp.float32, ("group", "dof"), fill=None, owner="cache"),
    Field("com_offset", wp.vec3, ("joint",), owner="factor", attr="joint_com_offset"),
    Field("diagonal", wp.float32, ("group", "diagonal"), fill=None, owner="cache"),
    Field("current_valid", wp.int32, ("articulation",), owner="factor", attr="fk_id_cache_valid"),
    Field("geometry_valid", wp.int32, ("group",), owner="cache"),
    Field("generation", wp.int64, ("group",), owner="cache"),
    Field("geometry_generation", wp.int64, ("group",), fill=-1, owner="cache"),
    Field("status", wp.int32, ("group",), owner="cache"),
    Field("source_q", wp.uint64, ("group",), owner="cache"),
    Field("source_qd", wp.uint64, ("group",), owner="cache"),
    Field("stage_clocks", wp.int64, ("group", "stage"), owner="step"),
)
"""The held tree's cache. ``diagonal`` is a compact diagonal's K-free mass ``H_ii + R_ii`` (a group without one has
none). Geometry, forces
and source tags follow factor groups (the cache allocates them);
current validity follows articulation IDs, shared with the ordinary FK/ID cache (the solver's); the stage clocks
are the owner's."""

KineticData = field_group.struct("KineticData", KINETIC_CACHE, __name__)


@wp.struct
class StateBank:
    """The generalized and public body state of one state bank, read word for word."""

    joint_q: wp.array[float]
    joint_qd: wp.array[float]
    body_q: wp.array[wp.transform]
    body_qd: wp.array[wp.spatial_vector]


GROUP_TREES = (
    Field("dof_joint", wp.int32, ("dof",), owner="host"),
    Field("group_to_art", wp.int32, ("art",), owner="step"),
    Field("art_to_world", wp.int32, ("articulation",), owner="forest"),
    Field("art_dof_start", wp.int32, ("articulation",), owner="forest", attr="articulation_dof_start"),
    Field("art_joint_start", wp.int32, ("articulation_bound",), owner="model", attr="articulation_start"),
    Field("joint_child", wp.int32, ("joint",), owner="model"),
)
"""A held group's trees: each group's articulation and its world, the articulations' first coordinate and joint, the
joints' bodies and the template tree's joint of each coordinate."""

GroupTrees = field_group.struct("GroupTrees", GROUP_TREES, __name__)


@wp.struct
class HeldResponse:
    """A held group's validity and its status by tree, which the predictor reads."""

    valid: wp.array[int]
    status: wp.array[int]


FORCE_INPUT = (
    Field("joint_q", wp.float32, ("coord",), owner="state_in"),
    Field("joint_qd", wp.float32, ("dof",), owner="state_in"),
    Field("predictor_qd", wp.float32, ("dof",), owner="step"),
    Field("joint_f", wp.float32, ("dof",), owner="control"),
    Field("body_f", wp.spatial_vector, ("body",), owner="state_in"),
    Field("body_flags", wp.int32, ("body",), owner="model"),
    Field("stiffness", wp.float32, ("dof",), owner="factor", attr="passive_spring_stiffness"),
    Field("reference", wp.float32, ("dof",), owner="factor", attr="passive_spring_ref"),
    Field("damping", wp.float32, ("dof",), owner="factor", attr="passive_joint_damping"),
    Field("u0", wp.float32, ("dof",), owner="state_aug", attr="joint_tau"),
    Field("S", wp.spatial_vector, ("dof",), owner="state_aug", attr="joint_S_s"),
    Field("kinematic_joint", wp.int32, ("joint",), owner="factor", attr="kinematic_joint_mask"),
    Field("qdd", wp.float32, ("dof",), owner="state_aug", attr="joint_qdd"),
    Field("vhat", wp.float32, ("dof",), attr="v_hat"),
    Field("dt", wp.float32, owner="step"),
    Field("stage_clocks", wp.int64, ("group", "stage"), owner="step"),
    Field("predicted_twist", wp.spatial_vector, ("body",), owner="step"),
    Field("publish_twists", wp.int32, owner="step"),
    Field("drive", wp.vec4, ("dof",), owner="factor", attr="aug_drive"),
    Field("drive_row", wp.int32, ("dof",), owner="factor", attr="drive_row_by_dof"),
    Field("row_K", wp.float32, ("drive_row",), owner="factor", attr="aug_row_K"),
    Field("row_Kd", wp.float32, ("drive_row",), owner="factor", attr="aug_row_Kd"),
    Field("row_H", wp.float32, ("drive_row",), owner="factor", attr="aug_row_H"),
    Field("row_dv", wp.float32, ("drive_row",), owner="factor", attr="aug_row_dv"),
    Field("frame_status", wp.int32, ("status",), owner="factor", attr="walk_release_status"),
)
"""The held-tree predictor's inputs: the step's state and control, the solver's passive and kinematic tables, the
augmented state and the drive rows it writes, the border's status word (:func:`border_solve`), and the step's
values."""

ForceInput = field_group.struct("ForceInput", FORCE_INPUT, __name__)


@wp.struct
class TreeGroup:
    """One tree group in a finish or repair: its threads, articulation, refresh and state block.

    Admission fills every field but the state block, which a group admitted
    for a rebuild allocates next; every later stage reads the group from here.
    """

    address: wp.uint64
    lane: int
    stride: int
    group: int
    art: int
    ds: int
    js: int
    refresh: int
    # Alternate repair motion: whether the motion velocity differs from the state bank's.
    motion_changed: int
    # Whether this launch rebuilds the group's current state.
    admitted: int


@wp.func_native(dual_native("    return wp::int64(CLOCK());\n", preludes=(STATUS_CLOCK,)))
def _clock() -> wp.int64: ...


_LANES_SOURCE = "    return wp::vec_t<3,int>(LANE,STRIDE,SUB);\n"
_SYNC_SOURCE = "    SYNC();\n"
_SUB_SOURCE = "    return SUB;\n"
_LANE_STRIDE = "    const int lane=LANE,stride=STRIDE;\n"


_DIFFERENT_MOTION_SOURCE = r"""
    if(state_qd.data==motion_qd.data)return 0;
    int different=0;
    for(int i=LANE;i<count;i+=STRIDE)
        different|=state_qd.data[start+i]!=motion_qd.data[start+i];
    return ANY(different);
"""


@wp.func
def body_dynamics(
    body: int,
    articulation: int,
    pose: wp.transform,
    pose_com: wp.transform,
    origin: wp.vec3,
    velocity: wp.spatial_vector,
    acceleration: wp.spatial_vector,
    mass: float,
    inertia: wp.mat33,
    com: wp.vec3,
    gravity: wp.vec3,
    write_matrix: int,
    write_record: int,
    publish_twist: int,
    body_I_s: wp.array[wp.spatial_matrix],
    body_inertia_terms: wp.array[inertia_record],
    body_f_s: wp.array[wp.spatial_vector],
    body_qd: wp.array[wp.spatial_vector],
):
    """Store a body's inverse-dynamics wrench about the origin (:func:`.spatial_laws.body_wrench`); optionally
    its inertia there and its COM twist.

    ``write_matrix`` stores the dense spatial inertia, ``write_record`` the inertia record. A body outside an
    articulation (``articulation < 0``) only publishes its COM twist.
    """
    if articulation < 0:
        if publish_twist != 0:
            linear = point_velocity(velocity, wp.transform_point(pose, com))
            body_qd[body] = wp.spatial_vector(linear, wp.spatial_bottom(velocity))
        return
    offset = wp.transform_get_translation(pose_com) - origin
    rotation = wp.transform_get_rotation(pose_com)
    if write_matrix != 0 or write_record != 0:
        body_mass, moment, rotational = inertia_at(mass, inertia, rotation, offset)
        if write_matrix != 0:
            body_I_s[body] = inertia_matrix(body_mass, moment, rotational)
        if write_record != 0:
            body_inertia_terms[body] = pack_inertia(body_mass, moment, rotational)
    body_f_s[body] = body_wrench(mass, rotation, inertia, offset, velocity, acceleration, gravity)
    if publish_twist != 0:
        linear = point_velocity(velocity, wp.transform_point(pose, com) - origin)
        body_qd[body] = wp.spatial_vector(linear, wp.spatial_bottom(velocity))


@functools.cache
def _thread_helpers(layout):
    """Lane, barrier and sub-tree natives for one tree layout."""
    lanes_source = layout.native(_LANES_SOURCE)
    sync_source = layout.native(_SYNC_SOURCE)
    sub_source = layout.native(_SUB_SOURCE)
    motion_source = layout.native(_DIFFERENT_MOTION_SOURCE)

    @wp.func_native(lanes_source)
    def _lanes() -> wp.vec3i: ...

    @wp.func_native(sync_source)
    def _sync(): ...

    @wp.func_native(sub_source)
    def _sub() -> int: ...

    @wp.func_native(motion_source)
    def _different_motion(state_qd: wp.array[float], motion_qd: wp.array[float], start: int, count: int) -> int: ...

    return _lanes, _sync, _sub, _different_motion


# Unpacked (one tree per block) helpers under their module-level names, for the
# tools tests that drive the natives directly.
_lanes, _sync, _sub, _different_motion = _thread_helpers(REFERENCE_LAYOUT)


@wp.func_native(
    "return cache.source_q.data[group]==reinterpret_cast<uint64_t>(q.data)&&cache.source_qd.data[group]==reinterpret_cast<uint64_t>(qd.data);"
)
def _same_source(cache: KineticData, group: int, q: wp.array[float], qd: wp.array[float]) -> int: ...


@wp.func_native(
    "cache.source_q.data[group]=reinterpret_cast<uint64_t>(q.data);cache.source_qd.data[group]=reinterpret_cast<uint64_t>(qd.data);"
)
def _stamp_source(cache: KineticData, group: int, q: wp.array[float], qd: wp.array[float]): ...


_BANK_MATCHES = r"""
    bool equal=true;
    const unsigned* a=reinterpret_cast<const unsigned*>(q.data+first_q);
    const unsigned* b=reinterpret_cast<const unsigned*>(bank.joint_q.data+first_q);
    for(int k=lane;k<$COORDINATES;k+=stride)equal&=a[k]==b[k];
    a=reinterpret_cast<const unsigned*>(qd.data+first_dof);
    b=reinterpret_cast<const unsigned*>(bank.joint_qd.data+first_dof);
    for(int k=lane;k<$DOFS;k+=stride)equal&=a[k]==b[k];
    for(int k=lane;k<7*$BODIES;k+=stride) {
        const int body=children.data[first_joint+k/7];
        equal&=reinterpret_cast<const unsigned*>(body_q.data+body)[k%7]==
            reinterpret_cast<const unsigned*>(bank.body_q.data+body)[k%7];
    }
    for(int k=lane;k<6*$BODIES;k+=stride) {
        const int body=children.data[first_joint+k/6];
        equal&=reinterpret_cast<const unsigned*>(body_qd.data+body)[k%6]==
            reinterpret_cast<const unsigned*>(bank.body_qd.data+body)[k%6];
    }
    return ALL(equal);
"""


@functools.cache
def _bank_binding(coordinates, layout):
    """Compare a group's input bank with the bank its cache was published into, word for word.

    A group's coordinates (free-root positions and quaternions included), its
    velocities and its bodies' public poses and twists are compared as 32-bit
    words, and the group's lanes stride the words and vote. Equal banks are
    identical inputs, so a group that matches keeps the cache the last finish
    published from those words, as a group stepped from the bank that finish
    wrote does. A repair would compute it from the same words with the same
    stage source in another kernel, whose floating-point contraction may
    differ: keeping it is exact up to that reordering (float-reorder), not
    bitwise.
    """

    @wp.func_native(layout.native(_LANE_STRIDE + _BANK_MATCHES, COORDINATES=str(coordinates)))
    def matches(
        bank: StateBank,
        first_q: int,
        first_dof: int,
        first_joint: int,
        q: wp.array[float],
        qd: wp.array[float],
        children: wp.array[int],
        body_q: wp.array[wp.transform],
        body_qd: wp.array[wp.spatial_vector],
    ) -> int: ...

    return matches


# A tree's block of words (TreeLayout): one store and one load of a vector's components as consecutive floats from
# float ``word`` of the block at ``address``, for each vector type a block holds (a ``vec3``, a spatial vector, a
# transform, an inertia record). Each caller names its word from the layout's offsets.
_STORE_WORDS = "float* s=reinterpret_cast<float*>(address)+word;for(unsigned k=0;k<sizeof(value)/4;++k)s[k]=value[k];"
_LOAD_WORDS = (
    "auto value=like;const float* s=reinterpret_cast<const float*>(address)+word;"
    "for(unsigned k=0;k<sizeof(value)/4;++k)value[k]=s[k];return value;"
)


@wp.func_native(_STORE_WORDS)
def store_words(address: wp.uint64, word: int, value: wp.vec3): ...


@wp.func_native(_STORE_WORDS)
def store_words(address: wp.uint64, word: int, value: wp.spatial_vector): ...


@wp.func_native(_STORE_WORDS)
def store_words(address: wp.uint64, word: int, value: wp.transform): ...


@wp.func_native(_STORE_WORDS)
def store_words(address: wp.uint64, word: int, value: inertia_record): ...


@wp.func_native(_LOAD_WORDS)
def load_words(address: wp.uint64, word: int, like: wp.vec3) -> wp.vec3: ...


@wp.func_native(_LOAD_WORDS)
def load_words(address: wp.uint64, word: int, like: wp.spatial_vector) -> wp.spatial_vector: ...


@wp.func_native(_LOAD_WORDS)
def load_words(address: wp.uint64, word: int, like: wp.transform) -> wp.transform: ...


# ---------------------------------------------------------------------------------------------------------------------
# The tree's scans: tree_scan configurations on the state block, sized by the layout's names
# ---------------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class HeldSchedules:
    """The schedules of a held group's scans (:class:`.tree_scan.Schedule`), chosen from its plan's data."""

    pose: ts.Schedule
    motion: ts.Schedule
    reduce: ts.Schedule
    """The bias wrench and inertia moment sums of the collection."""

    @property
    def chains(self) -> bool:
        """Whether a scan runs on the heavy chains (whose texts address the block's threads)."""
        return ts.Schedule.HEAVY_CHAIN in (self.pose, self.motion, self.reduce)

    @property
    def tag(self) -> str:
        """The kernel-name suffix of the schedules."""
        return "_" + "_".join(s.value for s in (self.pose, self.motion, self.reduce))


def held_lanes(tree: TreePlan, groups: int, device: Device) -> int:
    """Threads per tree of a held group of ``groups`` trees shaped like ``tree`` on ``device``, from the tree's data.

    The group's scans shorten with its lanes (a scan's work: as many lanes as bodies, from 8, two warps past 32
    bodies), widened until the group fills whole 32-thread blocks. Measured on the RTX PRO 6000 (finish, predict and
    repair of 4096 trees at each lane width, and the construction-time timings the replaced lane choice recorded in
    the gate's tasks): this width ran within 3% of the fastest width of every tree of up to 32 bodies. The
    coordinates do not set it: the 13-body, 21-coordinate Franka forest runs 27% slower at 32 lanes than at 16, a
    one-body tree 8% slower at 4 lanes than at 8. A one-body rest tree (no coordinates: its finish publishes a pose)
    takes one lane, bitwise with eight (`79e9e7b9`). On the CPU each tree takes a block of its own (32 lanes, two warps
    past 32 bodies): one thread runs it serially.
    """
    if not device.cuda:
        return 64 if tree.count > 32 else 32
    least = 1 if tree.count == 1 and int(tree.dof_start[-1]) == 0 else 8
    lanes = shape(Work(groups, Use.SCAN, width=tree.count, least=least, most=64), device).lanes
    while lanes < 32 and groups % (32 // lanes):
        lanes *= 2
    return lanes


def held_schedules(tree: TreePlan, lanes: int, *, chains: bool) -> HeldSchedules:
    """The schedules of a held group's scans at ``lanes`` lanes per tree, from its tree's data
    (:func:`.tree_scan.choose_schedule`).

    ``chains``: the launch may give each tree its own block, which the heavy-chain scans address.
    """
    own_block = chains and lanes >= 32

    def choose(combine, direction):
        return ts.choose_schedule(tree, direction, combine, cuda=True, lanes=lanes, chains=own_block)

    return HeldSchedules(
        choose(ts.TRANSFORM, ts.Direction.DOWN),
        choose(_MOTION, ts.Direction.DOWN),
        choose(_REDUCE, ts.Direction.UP),
    )


def publishes_twists(layout: TreeLayout, schedules: HeldSchedules, *, cuda: bool) -> bool:
    """Whether the predictor can publish the predicted body twists: on CUDA, for a warp-sized tree whose scans keep
    to their lane groups (the twist sums are a register scan of :attr:`TreeLayout.lanes` lanes, which has no CPU
    form: :func:`_twist_block`)."""
    return cuda and layout.lanes <= 32 and not schedules.chains


_SIZES = {"nodes": "$BODIES", "rounds": "$ROUNDS", "max_depth": "($LAST_PARENT+1)", "windows": "$WINDOWS"}
"""The layout's size names in place of a plan's literals: one text serves every tree of a layout."""
_ANY_TREE = TreePlan.build([-1, 0])
"""Any parent-first plan stands for the layout's tree once the sizes are names."""
_POSE_SLOTS = ts.Storage({"pose": "s[7*({node})+({k})]"})
_MOTION_SLOTS = ts.Storage(
    {"velocity": "s[$MOTION+6*({node})+({k})]", "acceleration": "s[$MOTION+$SIX_BODIES+6*({node})+({k})]"}
)
_REDUCE_SLOTS = ts.Storage({"wrench": "s[$WRENCH+6*({node})+({k})]", "moment": "s[$MOMENT+13*({node})+({k})]"})
_MOTION = ts.TWIST.with_zero_acceleration()
"""The motion scan's combine where no joint has an own bias acceleration (:attr:`.TreeLayout.own_bias`)."""
_REDUCE = ts.add(ts.Group("wrench", 6), ts.Group("moment", INERTIA_CHANNELS, "refresh"))
_TWIST_SUM = ts.add(ts.Group("twist", 6))
# Predicted body twists for the resolved-world classifier: each lane sums its body's own axes against the final
# predictor velocity, then pointer-jumps up the tree through shuffles (velocities at a common origin add).
_TWIST_SOURCE = (
    "{{const int first=ds+plan.body_dof_start.data[{node}],count=plan.body_dof_count.data[{node}];"
    "for(int d=0;d<count;++d){{const wp::spatial_vector axis=f.S.data[first+d];const float speed=f.vhat.data[first+d];"
    "for(int k=0;k<6;++k){value}[k]+=axis[k]*speed;}}}}"
)
_TWIST_SINK = (
    "{{wp::spatial_vector twist;for(int k=0;k<6;++k)twist[k]={value}[k];"
    "f.predicted_twist.data[p.joint_child.data[js+{node}]]=twist;}}"
)


def _held_scan(combine, direction, schedule, storage, access, **options) -> ts.ScanSource:
    return ts.tree_scan(
        _ANY_TREE, combine, direction, schedule, storage=storage, access=access, sizes=_SIZES, lanes="$LANES", **options
    )


def _down(layout: TreeLayout, combine, schedule: ts.Schedule, storage: ts.Storage) -> str:
    """A pose or motion scan's native body over the state block ``address``, the plan struct named ``p``."""
    scan = _held_scan(combine, ts.Direction.DOWN, schedule, storage, ts.PlanAccess().renamed("p"), level_csr=False)
    return layout.native("    float* s=reinterpret_cast<float*>(address);\n" + scan.text, preludes=scan.preludes)


def _reduction(combine, schedule: ts.Schedule, storage: ts.Storage) -> str:
    """A subtree-sum fragment: ``s``, ``lane``, ``stride`` and the plan struct ``plan`` in scope."""
    return _held_scan(combine, ts.Direction.UP, schedule, storage, ts.PlanAccess(), level_csr=False).text


def _twist_block() -> str:
    """The predictor's twist publication, a register scan of warp-sized trees: CUDA only, so a CPU build of a
    predictor that holds it fails (:func:`publishes_twists` leaves it out of a CPU owner's predictor)."""
    scan = _held_scan(
        _TWIST_SUM,
        ts.Direction.DOWN,
        ts.Schedule.JUMP_SHFL,
        None,
        ts.PlanAccess(),
        source=_TWIST_SOURCE,
        sink=_TWIST_SINK,
        cpu=False,
    )
    return device_branches("    if(f.publish_twists!=0) {\n        SYNC();\n" + scan.cuda + "    }\n", CUDA_ONLY)


_COLLECT = r"""    float* s=reinterpret_cast<float*>(address);
    const int lane=LANE,stride=STRIDE;
$CLOCK0$REDUCE
$CLOCK1    for(int dof=lane;dof<$DOFS;dof+=stride) {
        const int body=p.dof_joint.data[dof];
        float bias=0.0f;
        for(int k=0;k<6;++k)bias+=s[$AXIS+6*dof+k]*s[$WRENCH+6*body+k];
        cache.bias.data[group*$DOFS+dof]=bias;
        if(!wp::isfinite(bias))RAISE(cache.status.data[group],1);
        if(refresh) {
            // The coordinate's diagonal entry of H, S^T (I S) with I its body's subtree moment, is all a group reads of
            // H (the walk refreshes from the step's columns): the drives' K-free mass H_ii + R_ii on their rows (the
            // saturation decision's inertia), and a compact diagonal's K-free mass (_DIAGONAL).
            const float* t=s+$MOMENT+13*body,*axis=s+$AXIS+6*dof;
            const wp::vec3 l(axis[0],axis[1],axis[2]),w(axis[3],axis[4],axis[5]),h(t[1],t[2],t[3]);
            const wp::vec3 linear=t[0]*l+wp::cross(w,h);
            wp::vec3 angular=wp::cross(h,l);
            for(int i=0;i<3;++i)for(int j=0;j<3;++j)angular[i]+=t[4+3*i+j]*w[j];
            float mass=0.0f;
            for(int k=0;k<3;++k)mass+=axis[k]*linear[k];
            for(int k=0;k<3;++k)mass+=axis[3+k]*angular[k];
            const int drive=data.drive_row.data[ds+dof];
            if(drive>=0)data.row_H.data[drive]=mass+data.armature.data[ds+dof];
$DIAGONAL            if(!wp::isfinite(mass))RAISE(cache.status.data[group],2);
        }
    }
    SYNC();
$CLOCK2    if(lane==0) {
        cache.generation.data[group]+=1;
        $CURRENT_VALID
        cache.geometry_valid.data[group]=refresh&&cache.status.data[group]==0;
        if(refresh)cache.geometry_generation.data[group]=cache.generation.data[group];
    }
"""

_DIAGONAL = "            cache.diagonal.data[group*$DOFS+dof]=mass+data.armature.data[ds+dof];\n"
"""A compact diagonal's K-free mass ``H_ii + R_ii``, for a group with a diagonal root (:func:`get_helpers`'s
``diagonal``): its predictor divides by it with the drives' ``K`` (:func:`_diagonal_mass`), its refresh adds ``K`` and
inverts (the rows' inverse mass)."""

_FORCE = cached_force.ForceOperands(
    q="f.joint_q.data",
    state_qd="f.joint_qd.data",
    predictor_qd="f.predictor_qd.data",
    stiffness="f.stiffness.data",
    reference="f.reference.data",
    damping="f.damping.data",
    dt="f.dt",
)

_HALF = r"""    $PHYSICAL_TABLE

    const int art=p.group_to_art.data[group];
    const int ds=p.art_dof_start.data[art],js=p.art_joint_start.data[art];
    const int lane=LANE,stride=STRIDE;
#if defined(__CUDA_ARCH__)
    $SCRATCH
    __shared__ int ${FLAG}_all[$PACK];int& $FLAG=${FLAG}_all[SUB];
#else
    float s[$PRED_SCRATCH];int $FLAG=0;
#endif
"""
"""The head of the predictor's two halves around the walk's solve (Z4.2, :func:`_half_kernel`): the tree's
articulation, its lanes, its block of words ``s`` and the half's one flag a tree, ``$FLAG``."""

_FORCE_HALF = r"""    if(lane==0){nonzero=0;$CLEAR_SLIPS}
    SYNC();
    for(int local=lane;local<$BODIES;local+=stride) {
        const int body=p.joint_child.data[js+local];
        // A kinematic body takes no external wrench, as in ordinary dynamics
        // its wrench must not reach its ancestors.
        auto input=f.body_f.data[body];
        if((f.body_flags.data[body]&$KINEMATIC_FLAG)!=0){for(int k=0;k<6;++k)input[k]=0.0f;}
        bool active=false;for(int k=0;k<6;++k)active|=input[k]!=0.0f;
        if(active){RAISE(nonzero,1);$FRAME}
        for(int k=0;k<6;++k)s[6*local+k]=input[k];
    }
    SYNC();
$CARRY
    for(int i=lane;i<$DOFS;i+=stride) {
        const int dof=ds+i,body=p.dof_joint.data[i];
        float value=-cache.bias.data[group*$DOFS+i]+f.joint_f.data[dof]+f.u0.data[dof];
        if(nonzero){$PROJECT}
        if($SPRING_GUARD)value+=$PASSIVE;
        trial.force.data[dof]=value;
    }
"""
"""The predictor's force half (Z4.2): ``J^T W`` of the external wrenches in the walk's local frames (a tree whose bodies
take none, ``nonzero`` clear, skips it), then the force on each coordinate: bias, joint and control forces, the body
wrenches projected (``s[6 body + k]``, dead once projected) and the passive springs, stored where it is computed, to
``trial.force`` at its global coordinate, for the walk's solve (:func:`get_walk_solve_kernel`), which answers in
``trial.response``.

The wrenches never meet at the articulation's origin. There a coordinate's float32 force ``S . (f, n_O)`` rounds terms
of the lever about the origin times the force, whatever its own size: the coordinates near a loaded tip, whose forces
are a link's lever, err as much as the root's, a covector no wrench makes, which ``H^-1`` amplifies along a chain's soft
modes (``long_chain(106)`` without armature: the step 1.1e-4 off float64; 2.9e-5 here). Each COM wrench moves to its
body's frame (:data:`_EXTERNAL_FRAME`), the walk's carry takes each to its parent's (:data:`_EXTERNAL_CARRY`) and each
coordinate reads its column there (:data:`_EXTERNAL_PROJECT`): every lever is one link long, so a coordinate's rounding
is the size of its own force. The frames, levers and columns are the rows' seeds' (``WALK_UP``)."""

_EXTERNAL_FRAME = r"""float P[4];WALK_READ_POINT(WALK_LDG,P,walk.now_link.data+$WALK_LINK*(js+local)+$WALK_FRAME);
            const wp::vec3 c=cache.com_offset.data[js+local];
            input=Spatial::wrench_shift(input,wp::vec3(c[0]-P[0],c[1]-P[1],c[2]-P[2]));"""
"""A body's external wrench at its frame: the force keeps, the moment gains the centre of mass less the frame point
(:func:`.spatial_laws.wrench_shift`), both the step's, the articulation's own (relative to its origin), as the walk's
refresh forms its inertia's lever (with ``factor.walk.WalkTrees.now_link``)."""

_EXTERNAL_CARRY = r"""    if(nonzero) {
        if(lane==0) {
            for(int local=$BODIES-1;local>0;--local) {
                const int up=plan.parent.data[local];
                if(up<0)continue;
                float l[6];
                WALK_READ_LINK(WALK_LDG,l,walk.now_link.data,js+local,js+up);
                const float* q=s+6*local;
                float* w=s+6*up;
                $EDGE
            }
        }
        SYNC();
    }
"""
"""The walk's carry of the subtree wrenches, leaf first, one lane a tree (the walk's own order, one text for both
devices): each body's wrench passes its joint into its parent's at the parent's frame (:data:`.tree_scan.WHITEN`'s joint, ``(f, t
- d x f)``), ``d`` the parent's frame point less its own (``WALK_READ_LINK``, the step's links)."""

_DIAGONAL_PROJECT = r"""const float* c=reinterpret_cast<const float*>(walk.now.data)+$WALK_NOW*dof;
            const bool turn=c[3]!=0.0f||c[4]!=0.0f||c[5]!=0.0f;
            float S[6];for(int k=0;k<6;++k)S[k]=(k<3)!=turn?c[k]:0.0f;
            const float* w=s+6*body;
            value+=$DOT;"""
"""A diagonal coordinate's share of the subtree wrench at its body's frame: its one joint's column there, the half of
its step column ``WALK_NOW`` its joint moves along (a turn's angular half, else a slide's linear half: the walk's
record names the same half, :func:`.factor.walk.pack_walk_records`), with no record to read."""

_EXTERNAL_PROJECT = r"""float S[6];
            WALK_READ_COLUMN(WALK_LDG,S,reinterpret_cast<const float*>(walk.now.data)+$WALK_NOW*dof,walk.data.data+$WALK_COORDINATE*dof,$HALF);
            const float* w=s+6*body;
            value+=$DOT;"""
"""A coordinate's share of the subtree wrench at its body's frame: its step column there (``WALK_READ_COLUMN``, the
rows' read) dotted as every walk dots (:func:`.tree_scan.dot`): the rows' ``J`` entry. A prescribed root's coordinates,
which no walk holds, respond with zero (:func:`get_walk_solve_kernel`): their force is not read."""

_RESPONSE_STAGE = r"""$START    for(int col=lane;col<$DOFS;col+=stride) {
        $TRANSPOSE_ACTION
        const int dof=ds+col;
        const bool valid=cache.current_valid.data[art]&&d.valid.data[group]&&!(d.status.data[group])&&!cache.status.data[group]&&wp::isfinite(value);
        f.qdd.data[dof]=valid?value:NAN;
        f.vhat.data[dof]=$VELOCITY;
        if(!valid)RAISE(cache.status.data[group],4);
        $DECIDE
    }
    SYNC();
"""
"""A response's readout: each coordinate's ``qdd`` (its response word, ``NAN`` where the tree is not valid) and
``v_hat``; the trial's also takes the drives' decision (``$DECIDE``)."""

_RESPONSE_HALF = r"""    const auto MOVE=[&](int bits){RAISE(moved,bits);};
    if(lane==0)moved=0;
$CLOCK0    for(int i=lane;i<$DOFS;i+=stride) {
        s[$SIX_BODIES+i]=trial.force.data[ds+i];
        s[$WHITE+i]=$RESPONSE;
    }
    SYNC();
$CLOCK1    $TRIAL
$CLOCK4    if(moved!=0) {
    $SATURATE
    $BORDER
    $SECOND_RESPONSE
    }
    // A prescribed (kinematic) free root keeps its given velocity: no transport.
    $TRANSPORT_BLOCK
$TWIST_BLOCK$CLOCK5"""
"""The predictor's response half (Z4.2): the force and the walk's response read back into the block (a diagonal root's
its one division, :data:`_DIAGONAL_TRIAL`), then the trial's response (``qdd``, ``v_hat``), the drives' decision and
the border's decided step, the root transport and the twists."""


@wp.struct
class TrialWords:
    """The trial's words between the predictor's halves and the walk's solve (Z4.2): each coordinate's force and its
    response ``P^T L^-T L^-1 P f``, at its global coordinate."""

    force: wp.array[float]
    response: wp.array[float]


QUIET_DRIVE = r"""const wp::vec4 drive=f.drive.data[dof];
        if(drive[0]>0.0f&&drive[2]>0.0f) {
            const float u0=drive[3],limited=Spatial::limited_drive_force(u0,drive[2]);
            if(u0!=limited||wp::abs(Spatial::drive_step_force(u0,limited,drive[0],value,0.0f))>drive[2])MOVE(1);
        }
"""
"""The test of a drive with nothing to do, where ``qdd`` is written: no excess, and the trial leaves it within its limit.
A world every drive of which passes reads nothing else of the decision and takes no second stage."""

SATURATE_DRIVES = r"""    for(int i=lane;i<$DOFS;i+=stride) {
        const int dof=ds+i,row=f.drive_row.data[dof];
        float released=0.0f,kept=0.0f;
        if(row>=0) {
            const int root=$ROOT;
            const wp::vec4 drive=f.drive.data[dof];
            const float K=drive[0],passive=drive[1],limit=drive[2],u0=drive[3];
            const float all=f.row_K.data[row];
            float decided=all;
            if(K>0.0f&&limit>0.0f) {
                const float qdd=f.qdd.data[dof];
                const float limited=Spatial::limited_drive_force(u0,limit);
                const float inertia=u0!=limited?f.row_H.data[row]+passive:0.0f;
                const float force=Spatial::drive_step_force(u0,limited,K,qdd,inertia);
                float change=0.0f;
                if(wp::abs(force)>limit) {
                    decided=passive;
                    change=wp::sign(force)*limit-limited;
                } else if(u0!=limited) {
                    const float rows=f.row_dv.data[row]/f.dt;
                    change=Spatial::free_drive_force(u0,limited,K,qdd,rows,inertia,limit)-limited;
                }
                if(change!=0.0f||decided!=all) {
                    f.u0.data[dof]+=change;
                    s[$SIX_BODIES+i]+=change;
                    MOVE(2<<root);
                }
            }
            f.row_Kd.data[row]=decided;
            released=all-decided;
            kept=decided;
        }
        s[$WHITE+i]=released;
        s[i]=kept;
    }
    SYNC();
"""
"""The drives' saturation decision (:func:`.spatial_laws.drive_step_force`), read from the trial solve just made, the
step's: a drive past its limit becomes the constant ``+-limit`` with no ``K``, and a drive within it takes back what its
limit held off (:func:`.spatial_laws.free_drive_force`). Each changes the force by the same amount the trial solved
with (``s[$SIX_BODIES + i]``); each drive's decided ``K`` goes to ``row_Kd``. For the border (:func:`border_solve`) each
coordinate leaves in the dead response words its released stiffness (``K - Kd``, nonzero on a slip) and in the dead body
wrench words its decided ``K``. ``moved``: bit 0, the trial found something to do (:data:`QUIET_DRIVE`); bit ``1 + r``,
root ``r``'s decided step differs from its trial. A world with nothing to do pays its :data:`QUIET_DRIVE` tests and no
launch, and a captured graph has nothing to skip."""


def _root(layout: TreeLayout) -> str:
    """Native expression of the root of coordinate ``i``."""
    return "plan.root_of.data[p.dof_joint.data[i]]" if layout.forest else "0"


def _root_articulation(layout: TreeLayout, r: int) -> str:
    """Native expression of root ``r``'s articulation."""
    return f"plan.arts.data[group*{len(layout.roots)}+{r}]" if layout.forest else "art"


def _placed_block(layout: TreeLayout, name: str, words: str) -> str:
    """Native statements binding ``float* name`` to this tree's block of ``words`` floats, the one placement rule: shared
    memory, packed trees in consecutive blocks of one array, unless the device has no room for the group's state
    (:attr:`.TreeLayout.global_block`), then the tree's words of the plan's global scratch. Every barrier the stages
    take fences global memory too, so no stage reads where its block lies; both hold the same values. ``plan`` and
    ``group`` are in scope."""
    if layout.global_block:
        return f"float* {name}=plan.scratch.data+group*{words};"
    return f"__shared__ float {name}_all[{words}*$PACK];float* {name}={name}_all+SUB*{words};"


# A CPU launch runs its blocks one after another on the launching thread, each done with the block before the next, so
# its block is static storage. Concurrent CPU launches of one kernel from several threads would share it (the JIT has no
# thread-local storage); only direct-construction tests build CPU owners.
_STORAGE_SOURCE = r"""
#if defined(__CUDA_ARCH__)
    $BLOCK
    return reinterpret_cast<uint64_t>(values);
#else
    static float values[$TOTAL];return reinterpret_cast<uint64_t>(values);
#endif
"""


def _native_clocks(clocked, owner, *, stages, column):
    """Return a native's cycle-counter fragments ``CLOCK0`` to ``CLOCK<stages>``, empty unless ``clocked``.

    Each fragment reads the cycle counter at a stage boundary; the last also
    writes lane 0's ``stages`` stage durations to ``<owner>.stage_clocks[group,
    column:column + stages]``.
    """
    if not clocked:
        return {f"CLOCK{k}": "" for k in range(stages + 1)}
    copies = ",".join(f"c{k}=c0" for k in range(1, stages + 1))
    fragments = {"CLOCK0": f"    long long c0=CLOCK(),{copies};\n"}
    for k in range(1, stages + 1):
        fragments[f"CLOCK{k}"] = f"    c{k}=CLOCK();\n"
    durations = "".join(f"o[{k}]=c{k + 1}-c{k};" for k in range(stages))
    fragments[f"CLOCK{stages}"] += (
        f"    if(lane==0&&{owner}.stage_clocks.shape[0]>group){{long long* o={owner}.stage_clocks.data"
        f"+group*{owner}.stage_clocks.shape[1]+{column};{durations}}}\n"
    )
    return fragments


def _current_valid(layout):
    """Native statement marking the group's articulations current when no stage reported a failure."""
    if not layout.forest:
        return "cache.current_valid.data[p.group_to_art.data[group]]=cache.status.data[group]==0;"
    roots = len(layout.roots)
    return (
        f"for(int r=0;r<{roots};++r)"
        f"cache.current_valid.data[plan.arts.data[group*{roots}+r]]=cache.status.data[group]==0;"
    )


@functools.cache
def get_helpers(
    layout: TreeLayout,
    schedules: HeldSchedules,
    body_velocity: bool = False,
    *,
    clocked: bool = False,
    diagonal: bool = False,
):
    """Bind one physical tree evaluation to its storage and its scans' schedules.

    ``clocked`` adds the collection (``stage_clocks`` columns 13-14) and body
    term (16-21) counters; ``diagonal`` has the collection keep a compact
    diagonal's K-free mass (:data:`_DIAGONAL`). They change the emitted
    code, so they are arguments (cache-key members), never environment reads.
    """
    wrench_word, moment_word = layout.wrench_offset, layout.moment_offset
    pose_source = _down(layout, ts.TRANSFORM, schedules.pose, _POSE_SLOTS)
    motion_source = _down(layout, ts.TWIST if layout.own_bias else _MOTION, schedules.motion, _MOTION_SLOTS)
    reduction = _reduction(_REDUCE, schedules.reduce, _REDUCE_SLOTS)
    collect_source = layout.native(
        _COLLECT,
        preludes=(STATUS_CLOCK,),
        # A rest tree (no coordinates) collects nothing: no bias, no H_ii.
        REDUCE=reduction if layout.dofs else "",
        CURRENT_VALID=_current_valid(layout),
        DIAGONAL=_DIAGONAL if diagonal else "",
        **_native_clocks(clocked, "cache", stages=2, column=13),
    )

    @wp.func_native(layout.native(_STORAGE_SOURCE, BLOCK=_placed_block(layout, "values", "$TOTAL")))
    def _storage(plan: HeldPlan, group: int) -> wp.uint64: ...

    @wp.func_native(pose_source)
    def _scan_poses(address: wp.uint64, p: HeldPlan): ...

    @wp.func_native(motion_source)
    def _scan_motion(address: wp.uint64, p: HeldPlan): ...

    # A root-only free body is a free rigid body (in a forest, a free root may
    # be one): the matrix-free contact path reads its spatial inertia and bias
    # wrench every substep, so publish them through the ordinary free-rigid
    # finalization; the runtime free-rigid test skips tree bodies. A single
    # free-root tree of several bodies never takes this branch (it cost Cassie
    # flat and Humanoid 3-4% in the finish kernel).
    free_rigid = (layout.bodies == 1 and layout.free_root) or (layout.forest and layout.free_root_count > 0)

    @wp.func
    def _body_terms(
        g: TreeGroup,
        body: int,
        local: int,
        pose: wp.transform,
        pose_com: wp.transform,
        origin: wp.vec3,
        velocity: wp.spatial_vector,
        acceleration: wp.spatial_vector,
        data: PublicationData,
        cache: KineticData,
    ):
        """Publish one body's twist and store its bias wrench, moments and COM offset.

        ``clocked`` records the cycles of the terms, the publication and the
        stores in ``stage_clocks`` columns 16-18 (local body 0) or 19-21 (a
        free rigid body).
        """
        if wp.static(clocked):
            clock_b0 = _clock()
        radius = wp.transform_get_translation(pose_com) - origin
        rotation = wp.transform_get_rotation(pose_com)
        mass = data.body_mass[body]
        inertia_local = data.body_inertia[body]
        record = inertia_record()
        if g.refresh != 0:
            total, moment, rotational = inertia_at(mass, inertia_local, rotation, radius)
            record = pack_inertia(total, moment, rotational)
        wrench = body_wrench(mass, rotation, inertia_local, radius, velocity, acceleration, data.gravity[0])
        if wp.static(clocked):
            clock_b1 = _clock()
        public_radius = wp.transform_point(pose, data.body_com[body]) - origin
        data.body_qd[body] = wp.spatial_vector(point_velocity(velocity, public_radius), wp.spatial_bottom(velocity))
        if wp.static(body_velocity):
            data.body_v_s[body] = velocity
        # The body's centre of mass relative to its articulation's origin, by its joint: the predictor's external
        # wrenches shift there, and the walk's refresh forms its inertia's lever from it (walk_step_kernel's now_com).
        cache.com_offset[g.js + local] = radius
        if wp.static(clocked):
            clock_b2 = _clock()
        store_words(g.address, wp.static(wrench_word) + 6 * local, wrench)
        if g.refresh != 0:
            store_words(g.address, wp.static(moment_word) + 13 * local, record)
        if wp.static(free_rigid):
            if data.is_free_rigid[data.body_to_articulation[body]] != 0:
                data.body_q_com[body] = pose_com
                data.body_a_s[body] = acceleration
                body_dynamics(
                    body,
                    data.body_to_articulation[body],
                    pose,
                    pose_com,
                    origin,
                    velocity,
                    acceleration,
                    data.body_mass[body],
                    data.body_inertia[body],
                    data.body_com[body],
                    data.gravity[0],
                    1,
                    0,
                    1,
                    data.body_I_s,
                    data.body_inertia_terms,
                    data.body_f_s,
                    data.body_qd,
                )
        if wp.static(clocked):
            clock_b3 = _clock()
            if data.stage_clocks.shape[0] > g.group:
                column = int(-1)
                if local == 0:
                    column = 16
                elif data.is_free_rigid[data.body_to_articulation[body]] != 0:
                    column = 19
                if column >= 0:
                    data.stage_clocks[g.group, column] = clock_b1 - clock_b0
                    data.stage_clocks[g.group, column + 1] = clock_b2 - clock_b1
                    data.stage_clocks[g.group, column + 2] = clock_b3 - clock_b2

    @wp.func_native(collect_source)
    def _collect(
        address: wp.uint64,
        group: int,
        ds: int,
        p: GroupTrees,
        plan: HeldPlan,
        data: PublicationData,
        cache: KineticData,
        refresh: int,
    ): ...

    return SimpleNamespace(
        _body_terms=_body_terms,
        _collect=_collect,
        _scan_motion=_scan_motion,
        _scan_poses=_scan_poses,
        _storage=_storage,
    )


@functools.cache
def get_state_kernel(
    finish: bool,
    schedules: HeldSchedules,
    *,
    layout: TreeLayout = REFERENCE_LAYOUT,
    body_velocity: bool = False,
    motion_velocity: bool = False,
    bank_coordinates: int = 0,
    clocked: bool = False,
    diagonal: bool = False,
    two_half: bool = False,
):
    """Build a complete repair, or with ``finish`` a complete finish, from its stage functions.

    Each stage reads its group from the :class:`TreeGroup` admission returns.
    Admission binds the group's threads and, in a repair, skips a group whose
    current cache already holds its state bank. A finish then integrates with
    the original generalized laws: accelerations, free-root transport and
    joints. Both store relative joint poses; scan poses and publish
    articulation origins; store joint axes and relative velocities and scan
    motion; publish bodies with their bias wrenches and moments; and collect
    bias, geometry and validity. A rest tree (no coordinates) publishes its
    poses, a zero twist, its origin and its validity alone.

    Every finish and repair takes one launch ABI, which keeps each stage one
    call below the kernel: after the six common arguments, the alternate
    motion velocity ``motion_qd`` and the ``bank`` a repair compares its input
    bank with. A kernel leaves the inputs of the other forms unread.

    ``bank_coordinates`` (a group's coordinate count) builds the repair that
    compares input banks inside its blocks: a valid group stamped with the
    ``bank`` argument compares the two banks (see :func:`_bank_binding`) and,
    when they are equal, is restamped and takes the early exit. ``schedules``
    are the scans' schedules;
    ``clocked`` selects the stage counters (``stage_clocks`` columns 0-7 for
    the stages, 22-24 for motion; see :func:`get_helpers`); ``diagonal`` keeps
    a compact diagonal's K-free mass (:data:`_DIAGONAL`); ``two_half`` (the group
    holds a joint whose step column has two halves at its frame,
    :data:`.tree_scan.WALK_BOTH`) forms the walk's step columns a second time
    at the anchor's rotation, where every other group stores S's halves.
    """
    if finish and motion_velocity:
        raise ValueError("Alternate motion velocity is a repair-only contract")
    if bank_coordinates and (finish or motion_velocity):
        raise ValueError("A compared input bank is a contract of the repair of ordinary motion")
    h = get_helpers(layout, schedules, body_velocity, clocked=clocked, diagonal=diagonal)
    _lanes, _sync, _sub, _different_motion = _thread_helpers(layout)
    banked = bank_coordinates != 0
    _bank_matches = _bank_binding(bank_coordinates, layout) if banked else None
    scan_poses, scan_motion, collect, body_terms = h._scan_poses, h._scan_motion, h._collect, h._body_terms
    _storage = h._storage
    # The words of a tree's block (TreeLayout): poses (7 a body) from 0, then each body's twist and bias acceleration
    # (6 each), the axes (6 a coordinate) and each root's origin and shift (3 each).
    motion_word, bias_word = 7 * layout.bodies, 13 * layout.bodies
    axis_word, origin_word = layout.axis_offset, layout.origin_offset
    bodies, dofs, pack, free_root = layout.bodies, layout.dofs, layout.pack, layout.free_root
    roots, free_roots, forest, own_bias = len(layout.roots), layout.free_root_count, layout.forest, layout.own_bias
    # A rest tree (no coordinates: a body or a weld chain on the world) publishes its poses, a zero twist and its
    # validity; it integrates nothing, has no motion and collects nothing (no bias, no H_ii, no walk geometry).
    rest = dofs == 0
    from .spatial_laws import store_columns  # noqa: PLC0415

    @wp.func
    def admit(
        block: int,
        logical_lane: int,
        p: GroupTrees,
        plan: HeldPlan,
        data: PublicationData,
        cache: KineticData,
        requests: wp.array[int],
        clock: wp.array[Clock],
        motion_qd: wp.array[float],
        bank: StateBank,
    ) -> TreeGroup:
        """Bind the group's threads; admit it unless a repair finds its current cache valid.

        A repair that compares banks keeps the cache of a group stamped with
        ``bank`` whose input bank holds the same words, and stamps it with the
        input bank.
        """
        g = TreeGroup()
        lanes = _lanes()
        g.lane = lanes[0]
        g.stride = lanes[1]
        # Where a tree takes one lane, its slot in the block is its logical lane: every thread on CUDA (32 one-lane
        # trees a block), the first alone on the CPU (one thread is the block's one group, its other lanes exit).
        if g.stride == 1 and logical_lane != lanes[2]:
            return g
        g.group = block * wp.static(pack) + lanes[2]
        g.art = p.group_to_art[g.group]
        g.ds = p.art_dof_start[g.art]
        g.js = p.art_joint_start[g.art]
        if wp.static(motion_velocity):
            g.motion_changed = _different_motion(data.joint_qd, motion_qd, g.ds, dofs)
        # A repair (before the walk's refresh advances the clock) refreshes this step's geometry where the step
        # refreshes; the finish (after it) computes the next step's on the next step's cadence.
        refresh = clock_refresh(clock[0], data.dt)
        if wp.static(finish):
            refresh = clock[0].cadence
        valid = cache.current_valid[g.art]
        if wp.static(forest):
            for root in range(1, wp.static(roots)):
                valid = wp.min(valid, cache.current_valid[plan.arts[g.group, root]])
        if not wp.static(finish):
            if clock_changed(clock[0], data.dt) != 0:
                valid = 0
            refresh = int(refresh != 0 or (requests[g.art] & GEOMETRY_REQUEST) != 0)
            if wp.static(forest):
                for root in range(1, wp.static(roots)):
                    refresh = int(refresh != 0 or (requests[plan.arts[g.group, root]] & GEOMETRY_REQUEST) != 0)
            source_matches = _same_source(cache, g.group, data.joint_q, data.joint_qd)
            if wp.static(banked):
                rebind = int(0)
                if valid != 0 and source_matches == 0:
                    if _same_source(cache, g.group, bank.joint_q, bank.joint_qd) != 0:
                        source_matches = _bank_matches(
                            bank,
                            data.joint_q_start[g.js],
                            g.ds,
                            g.js,
                            data.joint_q,
                            data.joint_qd,
                            data.joint_child,
                            data.body_q,
                            data.body_qd,
                        )
                        rebind = source_matches
            if (
                valid != 0
                and source_matches != 0
                and g.motion_changed == 0
                and (
                    refresh == 0
                    or (
                        cache.geometry_valid[g.group] != 0
                        and cache.geometry_generation[g.group] == cache.generation[g.group]
                    )
                )
            ):
                if wp.static(banked):
                    if rebind != 0 and g.lane == 0:
                        _stamp_source(cache, g.group, data.joint_q, data.joint_qd)
                return g
        g.refresh = refresh
        if g.lane == 0:
            cache.current_valid[g.art] = 0
            if wp.static(forest):
                for root in range(1, wp.static(roots)):
                    cache.current_valid[plan.arts[g.group, root]] = 0
        g.admitted = 1
        return g

    @wp.func
    def accelerations(g: TreeGroup, data: PublicationData):
        """Recover each coordinate's acceleration from its solved velocity."""
        for local in range(g.lane, dofs, g.stride):
            dof = g.ds + local
            kernels._light_update_qdd_from_velocity(
                dof,
                data.joint_qd,
                data.kinematic_dof_mask,
                1.0 / data.dt,
                data.v_out,
                data.joint_qdd,
            )
        _sync()

    @wp.func
    def transport(g: TreeGroup, plan: HeldPlan, data: PublicationData):
        """Remove the free roots' velocity transport from their accelerations."""
        if wp.static(not forest):
            if wp.static(free_root) and g.lane == 0:
                kernels._light_remove_free_root_transport_from_qdd(
                    plan.root_slot[g.group],
                    data.free_root_joint_indices,
                    data.joint_qd_start,
                    data.kinematic_joint_mask,
                    data.joint_qd,
                    data.joint_qdd,
                )
        else:
            if g.lane < wp.static(free_roots):
                kernels._light_remove_free_root_transport_from_qdd(
                    plan.root_slot[g.group * wp.static(free_roots) + g.lane],
                    data.free_root_joint_indices,
                    data.joint_qd_start,
                    data.kinematic_joint_mask,
                    data.joint_qd,
                    data.joint_qdd,
                )
        _sync()

    @wp.func
    def integrate(g: TreeGroup, data: PublicationData):
        """Integrate every joint's coordinates into the next state bank."""
        for local in range(g.lane, bodies, g.stride):
            joint = g.js + local
            kernels._light_integrate_generalized_joints(
                joint,
                data.joint_type,
                data.joint_parent,
                data.joint_child,
                data.joint_q_start,
                data.joint_qd_start,
                data.kinematic_joint_mask,
                data.joint_dof_dim,
                data.body_com,
                data.joint_X_c,
                data.joint_q,
                data.joint_qd,
                data.joint_qdd,
                data.dt,
                data.joint_q_new,
                data.joint_qd_new,
            )
        _sync()

    @wp.func
    def relative_poses(g: TreeGroup, plan: HeldPlan, data: PublicationData, q: wp.array[float]):
        """Store each joint's pose in its parent; a root's translation goes to its origin slot."""
        for local in range(g.lane, bodies, g.stride):
            joint = g.js + local
            transform = joint_transform(
                data.joint_type[joint],
                data.joint_axis,
                data.joint_qd_start[joint],
                data.joint_dof_dim[joint, 0],
                data.joint_dof_dim[joint, 1],
                q,
                data.joint_q_start[joint],
            )
            relative = joint_frame_pose(data.joint_X_p[joint], transform, data.joint_X_c[joint])
            if wp.static(not forest):
                if local == 0:
                    store_words(g.address, wp.static(origin_word + 3), wp.transform_get_translation(relative))
                    relative = wp.transform(wp.vec3(), wp.transform_get_rotation(relative))
            else:
                if plan.parent[local] < 0:
                    store_words(
                        g.address,
                        wp.static(origin_word) + 3 * (2 * plan.root_of[local] + 1),
                        wp.transform_get_translation(relative),
                    )
                    relative = wp.transform(wp.vec3(), wp.transform_get_rotation(relative))
            store_words(g.address, 7 * local, relative)
        _sync()

    @wp.func
    def poses(g: TreeGroup, plan: HeldPlan, data: PublicationData):
        """Complete the pose scan and publish each articulation's origin."""
        scan_poses(g.address, plan)
        if wp.static(not forest):
            if g.lane == 0:
                root_body = data.joint_child[g.js]
                origin = wp.transform_point(load_words(g.address, 0, wp.transform()), data.body_com[root_body])
                store_words(g.address, wp.static(origin_word), origin)
                data.articulation_origin[g.art] = origin + load_words(g.address, wp.static(origin_word + 3), wp.vec3())
        else:
            for root in range(g.lane, wp.static(roots), g.stride):
                root_local = plan.root_local[root]
                root_body = data.joint_child[g.js + root_local]
                origin = wp.transform_point(
                    load_words(g.address, 7 * root_local, wp.transform()), data.body_com[root_body]
                )
                store_words(g.address, wp.static(origin_word) + 3 * (2 * root), origin)
                data.articulation_origin[plan.arts[g.group, root]] = origin + load_words(
                    g.address, wp.static(origin_word) + 3 * (2 * root + 1), wp.vec3()
                )
        _sync()

    @wp.func
    def motion(
        g: TreeGroup,
        origin: wp.vec3,
        shift: wp.vec3,
        plan: HeldPlan,
        data: PublicationData,
        q: wp.array[float],
        qd: wp.array[float],
    ):
        """Store each joint's axes, relative velocity and (:attr:`.TreeLayout.own_bias`) own bias acceleration at the
        origin of the coordinates ``q`` and velocities ``qd``, then complete the motion scan.

        ``clocked`` records local body 0's anchor, joint-motion and store
        cycles in ``stage_clocks`` columns 22-24.
        """
        for local in range(g.lane, bodies, g.stride):
            if wp.static(clocked):
                clock_m0 = _clock()
            joint = g.js + local
            parent = plan.parent[local]
            if wp.static(forest):
                origin = load_words(g.address, wp.static(origin_word) + 3 * (2 * plan.root_of[local]), wp.vec3())
                shift = load_words(g.address, wp.static(origin_word) + 3 * (2 * plan.root_of[local] + 1), wp.vec3())
            anchor = data.joint_X_p[joint]
            if parent >= 0:
                anchor = load_words(g.address, 7 * parent, wp.transform()) * anchor
            else:
                anchor = wp.transform(wp.transform_get_translation(anchor) - shift, wp.transform_get_rotation(anchor))
            # The joint's frame point (shifted with the root): its anchor, a free joint's child's centre of mass.
            point = wp.transform_get_translation(anchor)
            anchor = wp.transform(point - origin, wp.transform_get_rotation(anchor))
            if wp.static(clocked):
                clock_m1 = _clock()
            velocity, bias = joint_motion(
                data.joint_type[joint],
                data.joint_axis,
                data.joint_dof_dim[joint, 0],
                data.joint_dof_dim[joint, 1],
                anchor,
                q,
                data.joint_q_start[joint],
                qd,
                data.joint_qd_start[joint],
                data.joint_S_s,
            )
            if wp.static(clocked):
                clock_m2 = _clock()
            start = data.joint_qd_start[joint]
            lin, ang = data.joint_dof_dim[joint, 0], data.joint_dof_dim[joint, 1]
            # The walk's step geometry (tree_factor.WalkTrees.now): each coordinate's column at its joint's frame, so
            # a walk takes it there without S's lever about the origin, and the frame's world point (a free joint's
            # child's centre of mass). A column of one half is that half of S at the frame, the same words (S carries
            # the anchor's point only into a turn's linear half, which the frame drops): the walk reads S itself, its
            # half the record's, and only a model with a column of two halves keeps the frame's columns (walk_now).
            columns = data.walk_now.shape[0] > 0
            for k in range(lin + ang):
                column = data.joint_S_s[start + k]
                store_words(g.address, wp.static(axis_word) + 6 * (start - g.ds + k), column)
                if wp.static(not two_half):
                    if columns:
                        if k < lin:
                            data.walk_now[start + k] = wp.spatial_vector(wp.spatial_top(column), wp.vec3())
                        else:
                            data.walk_now[start + k] = wp.spatial_vector(wp.vec3(), wp.spatial_bottom(column))
            if data.walk_now_link.shape[0] > 0:
                kind = data.joint_type[joint]
                free = kind == JointType.FREE or kind == JointType.DISTANCE
                if free:
                    point = wp.transform_point(
                        load_words(g.address, 7 * local, wp.transform()), data.body_com[data.joint_child[joint]]
                    )
                if wp.static(two_half):
                    # A group with a joint whose column has two halves at its frame (a D6 joint's turns after its
                    # slides, a free joint's turns below a root): the columns again, at the anchor's rotation alone.
                    l0, l1, l2, a0, a1, a2 = joint_columns(
                        kind,
                        data.joint_axis,
                        lin,
                        ang,
                        wp.transform(wp.vec3(), wp.transform_get_rotation(anchor)),
                        q,
                        data.joint_q_start[joint],
                        start,
                    )
                    if free:
                        # A free joint turns about the articulation's origin: the same twist at its frame.
                        a0 = wp.spatial_vector(wp.cross(wp.spatial_bottom(a0), point - origin), wp.spatial_bottom(a0))
                        a1 = wp.spatial_vector(wp.cross(wp.spatial_bottom(a1), point - origin), wp.spatial_bottom(a1))
                        a2 = wp.spatial_vector(wp.cross(wp.spatial_bottom(a2), point - origin), wp.spatial_bottom(a2))
                    if columns:
                        store_columns(data.walk_now, start, lin, ang, l0, l1, l2, a0, a1, a2)
                # The frame point is the articulation's, relative to its origin (WalkTrees.now_origin, the
                # articulation_origin poses() published), as the body's centre of mass the publication stores: their
                # differences keep their digits wherever the articulation stands.
                frame = point - origin
                for k in range(3):
                    data.walk_now_link[joint, wp.static(WALK_FRAME) + k] = frame[k]
            store_words(g.address, wp.static(motion_word) + 6 * local, velocity)
            if wp.static(own_bias):
                store_words(g.address, wp.static(bias_word) + 6 * local, bias)
            if wp.static(clocked):
                clock_m3 = _clock()
                if local == 0 and data.stage_clocks.shape[0] > g.group:
                    data.stage_clocks[g.group, 22] = clock_m1 - clock_m0
                    data.stage_clocks[g.group, 23] = clock_m2 - clock_m1
                    data.stage_clocks[g.group, 24] = clock_m3 - clock_m2
        _sync()
        scan_motion(g.address, plan)

    @wp.func
    def publish(
        g: TreeGroup, origin: wp.vec3, shift: wp.vec3, plan: HeldPlan, data: PublicationData, cache: KineticData
    ):
        """Publish each body's pose and twist with its bias wrench and moments."""
        for local in range(g.lane, bodies, g.stride):
            body = data.joint_child[g.js + local]
            if wp.static(forest):
                origin = load_words(g.address, wp.static(origin_word) + 3 * (2 * plan.root_of[local]), wp.vec3())
                shift = load_words(g.address, wp.static(origin_word) + 3 * (2 * plan.root_of[local] + 1), wp.vec3())
            pose = load_words(g.address, 7 * local, wp.transform())
            data.body_q[body] = wp.transform(
                wp.transform_get_translation(pose) + shift, wp.transform_get_rotation(pose)
            )
            if wp.static(rest):
                data.body_qd[body] = wp.spatial_vector()
                if wp.static(body_velocity):
                    data.body_v_s[body] = wp.spatial_vector()
            else:
                body_terms(
                    g,
                    body,
                    local,
                    pose,
                    pose * data.body_X_com[body],
                    origin,
                    load_words(g.address, wp.static(motion_word) + 6 * local, wp.spatial_vector()),
                    load_words(g.address, wp.static(bias_word) + 6 * local, wp.spatial_vector()),
                    data,
                    cache,
                )
        _sync()

    @wp.func
    def close(
        g: TreeGroup,
        p: GroupTrees,
        plan: HeldPlan,
        data: PublicationData,
        cache: KineticData,
        q: wp.array[float],
        qd: wp.array[float],
    ):
        """Collect bias, geometry and validity, and record the state bank the cache now holds."""
        collect(g.address, g.group, g.ds, p, plan, data, cache, g.refresh)
        if g.lane == 0:
            if wp.static(motion_velocity):
                # An override source must not be reused when a later begin returns to
                # ordinary motion, even if no finish occurred between the two calls.
                if g.motion_changed == 0:
                    _stamp_source(cache, g.group, q, data.joint_qd)
                else:
                    _stamp_source(cache, g.group, q, qd)
            else:
                _stamp_source(cache, g.group, q, qd)

    # One launch ABI for every finish and repair: motion_qd is the velocity an
    # alternate-motion repair reads, bank the bank a comparing repair reads.
    # Every other kernel leaves them unread.
    def kinetic_state(
        p: GroupTrees,
        plan: HeldPlan,
        data: PublicationData,
        cache: KineticData,
        requests: wp.array[int],
        clock: wp.array[Clock],
        motion_qd: wp.array[float],
        bank: StateBank,
    ):
        block, logical_lane = wp.tid()
        g = admit(
            block,
            logical_lane,
            p,
            plan,
            data,
            cache,
            requests,
            clock,
            motion_qd,
            bank,
        )
        if g.admitted == 0:
            return
        # Stage cycles of lane 0, stage_clocks columns 0-7: accelerations,
        # root transport, joint integration, relative poses, poses, motion,
        # publication, collection.
        if wp.static(clocked):
            clock_0 = _clock()
        g.address = _storage(plan, g.group)
        q = data.joint_q
        qd = data.joint_qd
        if wp.static(finish):
            if wp.static(not rest):
                accelerations(g, data)
            if wp.static(clocked):
                clock_i1 = _clock()
            if wp.static(not rest):
                transport(g, plan, data)
            if wp.static(clocked):
                clock_i2 = _clock()
            if wp.static(not rest):
                integrate(g, data)
            q = data.joint_q_new
            qd = data.joint_qd_new
        if wp.static(clocked):
            clock_1 = _clock()
            if not wp.static(finish):
                clock_i1 = clock_0
                clock_i2 = clock_0
        relative_poses(g, plan, data, q)
        if wp.static(clocked):
            clock_2 = _clock()
        poses(g, plan, data)
        origin = load_words(g.address, wp.static(origin_word), wp.vec3())
        shift = load_words(g.address, wp.static(origin_word + 3), wp.vec3())
        if wp.static(motion_velocity):
            qd = motion_qd
        if wp.static(clocked):
            clock_3 = _clock()
        if wp.static(not rest):
            motion(g, origin, shift, plan, data, q, qd)
        if wp.static(clocked):
            clock_4 = _clock()
        publish(g, origin, shift, plan, data, cache)
        if wp.static(clocked):
            clock_5 = _clock()
        close(g, p, plan, data, cache, q, qd)
        if wp.static(clocked):
            clock_6 = _clock()
            if g.lane == 0 and data.stage_clocks.shape[0] > g.group:
                data.stage_clocks[g.group, 0] = clock_i1 - clock_0
                data.stage_clocks[g.group, 1] = clock_i2 - clock_i1
                data.stage_clocks[g.group, 2] = clock_1 - clock_i2
                data.stage_clocks[g.group, 3] = clock_2 - clock_1
                data.stage_clocks[g.group, 4] = clock_3 - clock_2
                data.stage_clocks[g.group, 5] = clock_4 - clock_3
                data.stage_clocks[g.group, 6] = clock_5 - clock_4
                data.stage_clocks[g.group, 7] = clock_6 - clock_5

    name = f"tree_kinetic_finish{bodies}" if finish else f"tree_kinetic_repair{bodies}"
    if layout.forest:
        name += f"_forest{roots}"
    if layout.pack > 1:
        name += f"_l{layout.lanes}"
    name += schedules.tag
    if motion_velocity:
        name += "_motion"
    if banked:
        name += "_banked"
    if diagonal:
        name += "_diagonal"
    if two_half:
        name += "_halves"
    return python_kernel(kinetic_state, name)


_EDGE_UP = ts.WHITEN.joint(ts.LOADED.lever, "p")
"""A joint of the border's column walk up, on a loaded link ``l``."""
_HELD_COLUMN = [
    f"float x = {ts.dot(ts.LOADED.s, ts.channels('p'))};",
    "if (c == i) x += 1.0f;",
    *ts.WHITEN.step(ts.LOADED, "p", value="x", out="z", index="q"),
]
"""A slip's column step, ``L^-1 P e_i`` walked up its root path: the unit covector a seed at the slip's coordinate,
added there only (no ``0 +`` after it: the sign of a zero footprint is the walk's own)."""


_HELD = """    // The border (SAT_PHYS section 1.2, W9): a tree's decided step on its all-K walk, with no factor array. The release
    // R = L_F^-1 L_K, built from the all-K walk's columns and the K-free pivot, is the decided factor's whitening; the
    // predictor's step takes R then R^T between its two walks, the rows R, the leave R^T, the audit R^-T (WalkRelease).
    // The tree's segment (one warp, or a packed tree's lanes; on the CPU the one thread, a segment of one lane) holds
    // the tree's walk: each coordinate's record and body and each body's parent, first coordinate and link, loaded in
    // one parallel trip, lane l holding items l, l + WIDTH, ... in HOLD_WORDS words. The slips sit the same way: slip j
    // on lane j % WIDTH, in its slot j / WIDTH of HOLD_SLOTS (on CUDA one slot at the class's slip capacity; the CPU's
    // lane holds every slip, the lanes as a loop). Then the release is built coordinate-major (HELD_BUILD), and the
    // decided step's two tree walks (WalkClass.held: the class's straight-line text) run on every lane of the segment
    // in lockstep, reading each step's values by shuffle from their fixed lane, with no global load on a chain; between
    // them HELD_APPLY applies R^T R.
    constexpr int HOLD_WORDS = ($items + WIDTH - 1) / WIDTH, HOLD_SLIPS = $slips;
    constexpr int HOLD_SLOTS = (HOLD_SLIPS + WIDTH - 1) / WIDTH;
    // FRAME_STATUS(bit): raise a bit of the border's status word, which every tree's lane 0 may write at once.
    const auto FRAME_STATUS = [&](const int bit) { RAISE(f.frame_status.data[0], bit); };
    struct WalkHold {
        float r[HOLD_WORDS][13];
        float l[HOLD_WORDS][3];
    };
    // A held array (a word a lane) or a slotted one (a slip a lane) is the lane group's HELD: item x sits in lane
    // x % WIDTH's word or slot x / WIDTH, read by HELD_PICK(v, x) and, a row of a record or a link, HELD_ROW(v, x, out).
    // The tree's path is its class's, each body's parent and first coordinate in constant memory (WalkClass's): no
    // segment holds it.
    const auto HOLD_LOAD = [&](WalkHold& h, const float* data, const float* link, const int* parent, const int n,
                               const int nb) {
        #pragma unroll
        for (int k = 0; k < HOLD_WORDS; ++k) {
            const int x = lane + k * WIDTH;
            if (x < n) {
                float d[$unpacked];
                WALK_READ(WALK_LDG, d, data + $coordinate * x, $half);
                #pragma unroll
                for (int m = 0; m < 13; ++m) h.r[k][m] = d[m];
            } else {
                #pragma unroll
                for (int m = 0; m < 13; ++m) h.r[k][m] = 0.0f;
            }
            if (x < nb) {
                float l[6];
                WALK_READ_LINK(WALK_LDG, l, link, x, parent[x]);
                h.l[k][0] = l[0];
                h.l[k][1] = l[1];
                h.l[k][2] = l[2];
            } else {
                h.l[k][0] = h.l[k][1] = h.l[k][2] = 0.0f;
            }
        }
    };
    // HELD_APPLY(a, u, slips): u <- R^T R u, every slip's release on its own lane and slot at once (at most HOLD_SLIPS,
    // as HELD_BUILD keeps). R_j is applied leaf first and R_j^T root first along slip j's root path (the release's slip
    // steps, :func:`.tree_scan.slip`). Slips that share a coordinate share their path from it to their root
    // (release.root, their group), so a coordinate's depth is the same on every path through it: slip j takes the
    // coordinate at depth d (0 at the root) on turn (deepest - d) + rank for R, d + (size - 1 - rank) for R^T, its rank
    // among its group's slips in order. Every coordinate then takes its slips' steps in the group's order (R: the
    // group's slips in order, R^T: last first), each slip's chain in its own order, with no two slips on one coordinate
    // in a turn, over deepest + the largest group turns a direction. Groups touch disjoint coordinates, so their R_j
    // commute: (H + R + K_F)^-1 x = P^T L^-T R^T R L^-1 P x, since R^T R = L_K^T A_F^-1 L_K.
    const auto HELD_APPLY = [&](const int a, float* u, const int slips) {
        auto& release = walk.release;
        const int depth = release.step.shape[2], cap = release.dof.shape[1];
        bool mine[HOLD_SLOTS];
        const wp::vec4* steps[HOLD_SLOTS];
        int len[HOLD_SLOTS], group[HOLD_SLOTS], rank[HOLD_SLOTS], size[HOLD_SLOTS], lead[HOLD_SLOTS];
        float rs[HOLD_SLOTS];
        #pragma unroll
        for (int m = 0; m < HOLD_SLOTS; ++m) {
            const int j = lane + m * WIDTH;
            mine[m] = j < slips;
            const size_t at = ((size_t)a * cap + (mine[m] ? j : 0)) * depth;
            steps[m] = release.step.data + at;
            len[m] = 0;
            group[m] = -1;
            if (mine[m]) {
                group[m] = release.root.data[a * cap + j];
                while (len[m] < depth && WALK_BITS(steps[m][len[m]][0]) >= 0) ++len[m];
            }
            rank[m] = 0;
            size[m] = 0;
        }
        int deepest = 0, widest = 0;
        #pragma unroll 1
        for (int k = 0; k < slips; ++k) {
            const int gk = HELD_PICK(group, k), lk = HELD_PICK(len, k);
            #pragma unroll
            for (int m = 0; m < HOLD_SLOTS; ++m)
                if (mine[m] && gk == group[m]) {
                    rank[m] += k < lane + m * WIDTH ? 1 : 0;
                    ++size[m];
                }
            deepest = lk > deepest ? lk : deepest;
        }
        #pragma unroll 1
        for (int k = 0; k < slips; ++k) {
            const int sk = HELD_PICK(size, k);
            widest = sk > widest ? sk : widest;
        }
        const int turns = deepest + widest - 1;
        #pragma unroll
        for (int m = 0; m < HOLD_SLOTS; ++m) {
            rs[m] = 0.0f;
            lead[m] = deepest - len[m] + rank[m];
        }
        #pragma unroll 1
        for (int T = 0; T < turns; ++T) {
            #pragma unroll
            for (int m = 0; m < HOLD_SLOTS; ++m) {
                const int t = T - lead[m];
                if (mine[m] && t >= 0 && t < len[m]) {
                    int c;
                    float b, ar, g;
                    WALK_STEP(steps[m] + t, c, b, ar, g);
                    float& r = rs[m];
                    $apply_r
                }
            }
            WARP_SYNC();
        }
        #pragma unroll
        for (int m = 0; m < HOLD_SLOTS; ++m) {
            rs[m] = 0.0f;
            lead[m] = size[m] - 1 - rank[m];
        }
        #pragma unroll 1
        for (int T = 0; T < turns; ++T) {
            #pragma unroll
            for (int m = 0; m < HOLD_SLOTS; ++m) {
                const int t = len[m] - 1 - (T - lead[m]);
                if (mine[m] && T >= lead[m] && t >= 0) {
                    int c;
                    float b, ar, g;
                    WALK_STEP(steps[m] + t, c, b, ar, g);
                    float& r = rs[m];
                    $apply_rt
                }
            }
            WARP_SYNC();
        }
    };
    // HELD_BUILD(h, parent, first, a, n, nb, data, released, kept): the release of the tree's slips, coordinate-major. The slips are
    // its coordinates whose released stiffness is positive, root-most first (ascending), slip j on lane j % WIDTH in
    // slot j / WIDTH (at most HOLD_SLIPS, the class's slip capacity, and the release's; a slip past them keeps its K
    // and raises status bit 0). A pivot that is not positive raises bit 1 (SAT_PHYS falsifier 1: the split
    // capacitance not positive definite). One leaf-first sweep over the coordinates the slips' paths hold: at each,
    // every slip on it takes its column step (b = L^-1 P e_i) from the held record; then in slip order slip k takes
    // the rank-1 downdate's recurrence step (Gill, Golub, Murray and Saunders, with the split pivot P0 + Kd at the
    // slip: the pivot without K, so no cancellation; b, a = beta d^1/2, g = (d / dbar)^1/2 into walk.release) and
    // hands its b, a, g and the coordinate's new pivot to the later slips, and each later slip on the coordinate
    // applies R_k's step to its column with its pair state (R_k meets a column only where both paths hold the
    // coordinate: from their meeting point; b_j' = R_<j b_j, W9 P4). The recurrence takes one rsqrt of each new pivot
    // (the CPU: one over a square root) and multiplies. Every value is a register: the columns, the pivots (one a
    // coordinate), the pair states; nothing of the release is read back. Returns the slips.
    const auto HELD_BUILD = [&](const WalkHold& h, const int* parent, const int* first, const int a, const int n,
                                const int nb, const float* data, const float* released, const float* kept) -> int {
        auto& release = walk.release;
        const int depth = release.step.shape[2];
        const int cap = a < release.dof.shape[0] ? release.dof.shape[1] : 0;
        // Each slot's slip: the (lane + m WIDTH)-th coordinate, ascending, whose released stiffness is positive.
        int is[HOLD_SLOTS], total = 0;
        #pragma unroll
        for (int m = 0; m < HOLD_SLOTS; ++m) is[m] = -1;
        #pragma unroll
        for (int k = 0; k < HOLD_WORDS; ++k) {
            const int x = lane + k * WIDTH;
            const unsigned candidates = BALLOT(x < n && released[x] > 0.0f);
            const int count = COUNT(candidates);
            #pragma unroll
            for (int m = 0; m < HOLD_SLOTS; ++m) {
                const int want = lane + m * WIDTH - total;
                if (want >= 0 && want < count) {
                    unsigned rest = candidates;
                    for (int q = 0; q < want; ++q) rest &= rest - 1u;
                    is[m] = k * WIDTH + FIRST(rest);
                }
            }
            total += count;
        }
        const int most = cap < HOLD_SLIPS ? cap : HOLD_SLIPS, slips = total < most ? total : most;
        if (total > slips && lane == 0) FRAME_STATUS(1);
        // Each slip's own values, loaded once: its released and kept stiffness and its pivot without K; its column's
        // state, value and pair states; its next coordinate, leaf first along its path (-1 once past its root), its
        // steps so far and its root.
        float alphas[HOLD_SLOTS], keeps[HOLD_SLOTS], pivots[HOLD_SLOTS], ps[HOLD_SLOTS][6], zs[HOLD_SLOTS];
        float pairs[HOLD_SLOTS][HOLD_SLIPS];
        int next[HOLD_SLOTS], ts[HOLD_SLOTS], roots[HOLD_SLOTS];
        size_t ats[HOLD_SLOTS];
        #pragma unroll
        for (int m = 0; m < HOLD_SLOTS; ++m) {
            if (lane + m * WIDTH >= slips) is[m] = -1;
            const int i = is[m];
            alphas[m] = keeps[m] = pivots[m] = 0.0f;
            if (i >= 0) {
                alphas[m] = -released[i];
                keeps[m] = kept[i];
                pivots[m] = WALK_LOAD_PIVOT(data + $coordinate * i);
            }
            #pragma unroll
            for (int k = 0; k < 6; ++k) ps[m][k] = 0.0f;
            zs[m] = 0.0f;
            #pragma unroll
            for (int k = 0; k < HOLD_SLIPS; ++k) pairs[m][k] = 0.0f;
            next[m] = roots[m] = i;
            ts[m] = 0;
            ats[m] = ((size_t)a * cap + (i >= 0 ? lane + m * WIDTH : 0)) * depth;
        }
        const int deepest = slips > 0 ? HELD_PICK(is, slips - 1) : -1;
        #pragma unroll 1
        for (int b = nb - 1; b >= 0 && slips > 0; --b) {
            const int f0 = first[b], f1 = first[b + 1];
            if (f0 > deepest) continue;
            bool in[HOLD_SLOTS], any = false;
            int start[HOLD_SLOTS];
            #pragma unroll
            for (int m = 0; m < HOLD_SLOTS; ++m) {
                in[m] = next[m] >= f0 && next[m] < f1;
                if (BALLOT(in[m]) != 0u) any = true;
                start[m] = in[m] ? next[m] : -1;
            }
            if (!any) continue;
            #pragma unroll 1
            for (int c = f1 - 1; c >= f0; --c) {
                bool on[HOLD_SLOTS];
                unsigned slipping = 0u;
                #pragma unroll
                for (int m = 0; m < HOLD_SLOTS; ++m) {
                    on[m] = c <= start[m];
                    slipping |= BALLOT(on[m]) << (m * WIDTH);
                }
                if (slipping == 0u) continue;
                float d[13];
                HELD_ROW(h.r, c, d);
                #pragma unroll
                for (int m = 0; m < HOLD_SLOTS; ++m)
                    if (on[m]) {
                        const int i = is[m];
                        float* const p = ps[m];
                        float& z = zs[m];
                        int& root = roots[m];
                        $held_step
                    }
                // The coordinate's pivot and its square root (1/ell: the record's d[12] is 1/sqrt(P)), each slip's
                // step taking one rsqrt of its new pivot for the rest.
                float pivot = 1.0f / (d[12] * d[12]), root = 1.0f / d[12];
                #pragma unroll
                for (int k = 0; k < HOLD_SLIPS; ++k) {
                    if (k >= slips) break;
                    if (!((slipping >> k) & 1u)) continue;
                    float bk = 0.0f, ak = 0.0f, gk = 0.0f, dk = 0.0f, rk = 0.0f;
                    if (lane == k % WIDTH) {
                        const int i = is[k / WIDTH];
                        float& alpha = alphas[k / WIDTH];
                        const float z = zs[k / WIDTH];
                        int& t = ts[k / WIDTH];
                        const size_t at = ats[k / WIDTH];
                        const float dd = pivot, scale = root;
                        float dbar = dd + alpha * (z * scale) * (z * scale);
                        if (c == i) dbar = pivots[k / WIDTH] + keeps[k / WIDTH];
                        if (!(dbar > 0.0f) || !wp::isfinite(dbar)) FRAME_STATUS(2);
#if defined(__CUDA_ARCH__)
                        const float r = rsqrtf(dbar), inverse = r * r;
#else
                        const float r = 1.0f / sqrtf(dbar), inverse = r * r;
#endif
                        const float beta = alpha * z * scale * inverse;
                        alpha = alpha * dd * inverse;
                        bk = z;
                        ak = beta * scale;
                        gk = scale * r;
                        dk = dbar;
                        rk = dbar * r;
                        if (t < depth) release.step.data[at + t] = wp::vec4(WALK_FLOAT(c), bk, ak, gk);
                        ++t;
                    }
                    bk = READ(bk, k);
                    ak = READ(ak, k);
                    gk = READ(gk, k);
                    pivot = READ(dk, k);
                    root = READ(rk, k);
                    #pragma unroll
                    for (int m = 0; m < HOLD_SLOTS; ++m)
                        if (on[m] && lane + m * WIDTH > k) {
                            float& z = zs[m];
                            float* const pair = pairs[m];
                            $build_r
                        }
                }
            }
            const int up = parent[b];
            const int last = first[up + 1] - 1;
            float l[3];
            HELD_ROW(h.l, b, l);
            #pragma unroll
            for (int m = 0; m < HOLD_SLOTS; ++m)
                if (in[m]) {
                    float* const p = ps[m];
                    if (up >= 0) {
                        $edge_p
                    }
                    next[m] = up >= 0 ? last : -1;
                }
        }
        #pragma unroll
        for (int m = 0; m < HOLD_SLOTS; ++m)
            if (is[m] >= 0) {
                if (ts[m] > depth) FRAME_STATUS(1);
                release.dof.data[a * cap + lane + m * WIDTH] = is[m];
                release.root.data[a * cap + lane + m * WIDTH] = roots[m];
                for (int t = ts[m]; t < depth; ++t) release.step.data[ats[m] + t] = wp::vec4(WALK_FLOAT(-1), 0.0f, 0.0f, 0.0f);
            }
        if (lane == 0) release.count.data[a] = slips;
        WARP_SYNC();
        return slips;
    };
"""


def _walk_roots(layout: TreeLayout) -> list[tuple[int, int, int]]:
    """``(root, coordinate offset, family)`` of each root the predictor solves: one tree, or each root of a forest. A
    prescribed root (family 0) responds with zero; a diagonal root (a negative family, :func:`_diagonal_roots`) by one
    division."""
    return list(zip(range(len(layout.roots)), layout.root_offsets, layout.root_families, strict=True))


def _walked(layout: TreeLayout, classes: tuple) -> list[tuple[int, int, int]]:
    """``(root, coordinate offset, class index)`` of each root the walk solves, its index into the walked classes
    (:func:`.factor.walk.walk_down_prelude`'s ``cls``)."""
    walked, k = [], 0
    for (r, offset, _), cls in zip(_walk_roots(layout), classes, strict=True):
        if cls is not None:
            walked.append((r, offset, k))
            k += 1
    return walked


def _diagonal_roots(layout: TreeLayout) -> tuple[int, ...]:
    """The roots whose walk is the compact inverse: a compact diagonal family's (a negative family), one-coordinate
    trees on a fixed base whose mass matrix is its diagonal. No walk holds them: the predictor answers a force on one by
    one division by its mass (:func:`_diagonal_mass`), the rows step it by the compact inverse
    (:func:`.walk_rows.key_step`) and the sweep leaves it by the same."""
    return tuple(r for r, _, family in _walk_roots(layout) if family < 0)


def _on_diagonal(layout: TreeLayout, index: str) -> str:
    """Native test of whether coordinate ``index`` of the group lies on a diagonal root (:func:`_diagonal_roots`):
    compiled for one tree, the forest's root of the coordinate against the diagonal roots' mask otherwise."""
    roots = _diagonal_roots(layout)
    if not layout.forest:
        return "true" if roots else "false"
    return f"((({sum(1 << r for r in roots)})>>{_root(layout).replace('[i]', f'[{index}]')})&1)"


def _diagonal_mass(index: str, stiffness: str) -> str:
    """Native expression of a diagonal coordinate's mass: its K-free mass, which the group's collection keeps
    (``cache.diagonal``, :data:`_DIAGONAL`), plus the drive stiffness ``stiffness`` (zero without a drive)."""
    return f"(cache.diagonal.data[group*$DOFS+{index}]+({stiffness}))"


_DIAGONAL_TRIAL = (
    "$ON?trial.force.data[ds+i]/"
    + _diagonal_mass("i", "f.drive_row.data[ds+i]>=0?f.row_K.data[f.drive_row.data[ds+i]]:0.0f")
    + ":trial.response.data[ds+i]"
)
"""A diagonal coordinate's trial response, its force over its all-``K`` mass (the trial's walk on the compact inverse);
another reads the walk's solve."""

_DIAGONAL_DECIDED = r"""    for(int i=lane;i<$DOFS;i+=stride)
        if($ON&&(moved&(2<<$ROOT)))s[$WHITE+i]=s[$SIX_BODIES+i]/$MASS;
    SYNC();
"""
"""A diagonal coordinate's decided step on a root the decision moved: its decided force over its mass with its decided
``K`` (the drive's kept stiffness, ``s[i]``, zero without a drive), as the border's walk takes a walked root's."""


def border_solve(layout: TreeLayout, classes: tuple) -> str:
    """Native statements of the decided step (FL3.9, W9), one text for both devices (:data:`_HELD`). Each changed
    root's segment holds its walk and builds its release (into :class:`.factor.walk.WalkRelease`, from the marks
    :data:`SATURATE_DRIVES` left) coordinate-major, a slip a lane and slot, every value a register (``HELD_BUILD``);
    then it takes its decided step from the force words into the response words: the class's walk up, ``R^T R`` with
    every slip on its lane (``HELD_APPLY``), the walk down. A root whose step its trial already gives (bit ``1 + r`` of
    ``moved`` clear) keeps its trial's response. The releases are built for every root first: the marks of one root
    lie in the states of another's walk. Nothing is added to the block (H8: g1's predictor sits at the RTX PRO 6000's
    64 KB carveout line): the walks' join states and twists lie in the dead body words. A diagonal root's decided step
    is one division a coordinate (:data:`_DIAGONAL_DECIDED`), ahead of the walked roots'."""

    decided = ""
    if _diagonal_roots(layout):
        decided = substitute(
            _DIAGONAL_DECIDED,
            layout.sizes(),
            ON=_on_diagonal(layout, "i"),
            ROOT=_root(layout),
            MASS=substitute(_diagonal_mass("i", "s[i]"), layout.sizes()),
        )
    roots = [(r, offset) for r, offset, _ in _walked(layout, classes)]
    if not roots:
        return decided
    tree = (
        "const int a={art};const int j0=walk.joint_start.data[a],d0=walk.dof_start.data[a],n=walk.dofs.data[a];"
        "const float* const data=walk.data.data+(size_t)d0*$WALK_COORDINATE;"
        "const float* const link=walk.link.data+(size_t)j0*$WALK_LINK;"
    )
    hold = "const int nb=walk.joint_end.data[a]-j0;WalkHold h;HOLD_LOAD(h,data,link,HOLD_PARENT_{r},n,nb);"
    # Each walked root's path, its class's table in constant memory.
    tables = "".join(
        f"        {ts.constant_table(f'HOLD_PARENT_{r}', classes[r].parent)}\n"
        f"        {ts.constant_table(f'HOLD_FIRST_{r}', classes[r].first)}\n"
        for r, _ in roots
    )
    # A tree the walk does not hold keeps its trial and flags bit 2.
    frame = (
        "        if(moved&{bit}) {{{tree}\n"
        "        if(d0+n>walk.data.shape[0]) {{if(lane==0) {{FRAME_STATUS(4);moved&=~{bit};}}}}\n"
        "        else if(lane<WIDTH) {{{hold}\n"
        "            slips{r}=HELD_BUILD(h,HOLD_PARENT_{r},HOLD_FIRST_{r},a,n,nb,data,s+$WHITE+{offset},s+{offset});\n"
        "        }}}}\n"
    )

    def step(r, offset):
        # The class's walk up of the force words into the response words, R^T R between the walks, every slip on its
        # lane at once (HELD_APPLY), the walk down in place; the join states and the twists in the dead body words.
        walk_up, walk_down = classes[r].held(f"(s+$SIX_BODIES+{offset})", f"(s+$WHITE+{offset})", "s")
        return (
            f"        if((moved&{2 << r})&&lane<WIDTH) {{{tree.format(art=_root_articulation(layout, r))}{hold.format(r=r)}\n"
            f"{walk_up}\n            HELD_APPLY(a,s+$WHITE+{offset},slips{r});\n{walk_down}\n        }}\n"
        )

    frames = "".join(
        frame.format(
            bit=2 << r, r=r, offset=offset, tree=tree.format(art=_root_articulation(layout, r)), hold=hold.format(r=r)
        )
        for r, offset in roots
    )
    text = (
        decided
        + "    {\n"
        + tables
        + "        "
        + "".join(f"int slips{r}=0;" for r, _ in roots)
        + "\n"
        + frames
        + "        SYNC();\n"
        + "".join(step(r, offset) for r, offset in roots)
        + "        SYNC();\n    }\n"
    )
    return substitute(text, {**layout.sizes(), "WALK_COORDINATE": WALK_COORDINATE, "WALK_LINK": WALK_LINK})


def held_lambdas(layout: TreeLayout, slips: int, half: int = ts.WALK_BOTH) -> str:
    """The border's native lambdas (:data:`_HELD`), both devices, declared once ahead of the trial: the segment's words
    a lane hold the layout's coordinates and bodies (and one past the last body). ``slips`` is the class's slip capacity
    (:meth:`.tree_factor.TreeFactor.slip_capacity`): the build's slips, a lane and slot each. ``half``: the records'
    half as the border reads any tree's (``WALK_READ``, :func:`.tree_scan.record_half`)."""

    return substitute(
        _HELD,
        items=max(layout.dofs, layout.bodies + 1),
        slips=max(1, slips),
        coordinate=WALK_COORDINATE,
        unpacked=WALK_UNPACKED,
        half=half,
        held_step=ts.statements([*_HELD_COLUMN, "root = c;"], 24),
        apply_r=ts.statements(ts.slip("R", "u[c]", "r", "b", "ar", "g", "u[c] = {};"), 20),
        apply_rt=ts.statements(ts.slip("RT", "u[c]", "r", "b", "ar", "g", "u[c] = {};"), 20),
        build_r=ts.statements(ts.slip("R", "z", "pair[k]", "bk", "ak", "gk", "z = {};"), 28),
        edge_p=ts.statements(_EDGE_UP, 24),
    )


def _transport_block(layout: TreeLayout) -> str:
    """Native statements of the free roots' transport of the predicted velocity."""
    if not layout.forest:
        return (
            "if($FREE_ROOT&&lane==0&&f.kinematic_joint.data[js]==0) {\n        "
            + _FORCE.transport("ds", "f.vhat.data[ds+k]")
            + "\n    }"
        )
    return (
        "if(lane<$FREE_ROOTS&&f.kinematic_joint.data[js+plan.free_root_local.data[lane]]==0) {\n"
        "        const int rd=ds+plan.free_root_dof.data[lane];\n        "
        + _FORCE.transport("rd", "f.vhat.data[rd+k]")
        + "\n    }"
    )


PREDICTOR_REGISTERS = 56
"""The register line of a predictor's half: main's g1 predictor before the saturation law, 18 blocks of 64 at 56
registers (H8). The trial and the decision fit it (g1's 52); the border, which about one tree in eighteen runs, keeps
its working values in its stack rather than taking a resident block from every world (W7 Q5.5: the rare branch stays
out of every block)."""


def predictor_work(layout: TreeLayout) -> Work:
    """A predictor half's work: a packed group of trees a block of :attr:`.TreeLayout.block` threads at
    :data:`PREDICTOR_REGISTERS`. Its launch bound is sized for :data:`~.memory.REFERENCE_ARCH` on every device (24
    blocks of 32 threads, 18 of 64 on sm_120), as the kernel's builder takes no device."""
    return Work(0, block=layout.block, registers=PREDICTOR_REGISTERS)


def _predictor_name(prefix: str, layout: TreeLayout, schedules: HeldSchedules | None, response: str, classes) -> str:
    """A predictor launch's name: its stage, the layout's coordinates, forest and packing, the scans' schedules, the
    group's kind (``response``: the sparse class's tree, a dense size group, a compact diagonal) and its walked classes."""
    if response not in ("sparse", "cholesky", "diagonal"):
        raise ValueError("Unknown held tree response representation")
    name = f"{prefix}{layout.dofs}"
    if layout.forest:
        name += f"_forest{len(layout.roots)}"
    if layout.pack > 1:
        name += f"_l{layout.lanes}"
    if schedules is not None:
        name += schedules.tag
    if response != "sparse":
        name += "_" + response
    walked = tuple(cls for cls in classes if cls is not None)
    return name + "_w" + hashlib.sha256(repr(walked).encode()).hexdigest()[:8]


def _half_kernel(prefix, schedules, layout, response, classes, flag, body, preludes=(), **fragments):
    """A half of the predictor (:data:`_HALF` and its ``body``, a packed group of trees a block, its launch bound
    :func:`predictor_work`'s): the force half or the response half."""

    source = layout.native(
        _HALF + body,
        preludes=(*SPATIAL.preludes(), STATUS_CLOCK, *preludes),
        PHYSICAL_TABLE=layout.physical_table(),
        SCRATCH=_placed_block(layout, "s", "$PRED_SCRATCH"),
        FLAG=flag,
        **fragments,
    )
    params = (
        ("p", GroupTrees),
        ("d", HeldResponse),
        ("plan", HeldPlan),
        ("cache", KineticData),
        ("f", ForceInput),
        ("walk", WalkTrees),
        ("trial", TrialWords),
    )
    return KernelTemplate(
        _predictor_name(prefix, layout, schedules, response, classes),
        params,
        source,
        # A packed group of trees per block: the block's group and this thread's tree in it.
        items("group", groups=layout.pack, lanes=layout.lanes),
        composed=True,
        launch_bounds=shape(predictor_work(layout), memory.built_for(None)).bounds,
    ).kernel()


@functools.cache
def get_force_kernel(schedules: HeldSchedules, *, layout: TreeLayout = REFERENCE_LAYOUT, response="sparse", classes):
    """The predictor's force half (Z4.2, :data:`_FORCE_HALF`): the force on every coordinate into ``trial.force``; it
    clears each root's slip count for the response half's border. The columns' half is the walked classes'
    (:func:`.tree_scan.classes_half`), a diagonal root's its joint's (:data:`_DIAGONAL_PROJECT`). A group none of whose
    roots responds (prescribed roots, rest trees) takes no external wrench, which no coordinate of it answers."""

    walked = tuple(cls for cls in classes if cls is not None)
    diagonal = _diagonal_roots(layout)
    external = dict.fromkeys(("FRAME", "CARRY", "PROJECT"), "")
    if walked or diagonal:
        names = {
            **layout.sizes(),
            "WALK_COORDINATE": WALK_COORDINATE,
            "WALK_FRAME": WALK_FRAME,
            "WALK_LINK": WALK_LINK,
            "WALK_NOW": WALK_NOW,
        }
        edge = ts.WHITEN.joint(ts.LOADED.lever, "q", into="w")
        dot = ts.dot(ts.channels("S"), ts.channels("w"))
        project = [substitute(_DIAGONAL_PROJECT, names, DOT=dot)] if diagonal else []
        if walked:
            project.append(substitute(_EXTERNAL_PROJECT, names, HALF=ts.classes_half(walked), DOT=dot))
        external = {
            "FRAME": substitute(_EXTERNAL_FRAME, names),
            "CARRY": substitute(_EXTERNAL_CARRY, names, EDGE=ts.statements(edge, 16)),
            "PROJECT": (
                project[0]
                if len(project) == 1
                else f"if({_on_diagonal(layout, 'i')}){{{project[0]}}}else{{{project[1]}}}"
            ),
        }
    return _half_kernel(
        "tree_kinetic_force",
        schedules,
        layout,
        response,
        classes,
        "nonzero",
        _FORCE_HALF,
        (walk_loads_prelude(),),
        CLEAR_SLIPS="".join(
            f"walk.release.count.data[{_root_articulation(layout, r)}]=0;" for r, _, _ in _walk_roots(layout)
        ),
        KINEMATIC_FLAG=str(int(BodyFlags.KINEMATIC)),
        **external,
        # Passive springs act on no root coordinate.
        SPRING_GUARD=f"i>={layout.root_dofs if layout.free_root else 0}"
        if not layout.forest
        else "plan.dof_passive.data[i]!=0",
        PASSIVE=_FORCE.passive("plan.q_index.data[group*$DOFS+i]", "dof"),
    )


_WALK_SOLVE = r"""    const int group = tid / $ROOT_COUNT, r = tid - group * $ROOT_COUNT;
    if (group >= groups) return;
    const int art = p.group_to_art.data[group];
    switch (r) {
$ROOTS
    }
"""


def walk_solves(layout: TreeLayout) -> bool:
    """Whether a group's predictor launches the walk's solve (:func:`get_walk_solve_kernel`): a root it walks or answers
    with zero, not only diagonal roots."""
    return any(family >= 0 for _, _, family in _walk_roots(layout))


@functools.cache
def get_walk_solve_kernel(*, layout: TreeLayout = REFERENCE_LAYOUT, classes):
    """The walk's solve of the trial (Z4.2, WALK_MAPPING E5): a thread a tree root, the class's walk compiled into
    straight-line code (:meth:`.tree_scan.WalkClass.walk` up to :data:`.tree_scan.WALK_UNROLL` coordinates), the
    walk up of ``trial.force`` into registers and the walk down into ``trial.response``, ``P^T L^-T L^-1 P f`` on the
    all-``K`` walk (:func:`.tree_scan.walk_texts`). A prescribed root responds with zero; a diagonal root holds no walk
    (the response half's division, :data:`_DIAGONAL_TRIAL`), so a group of diagonal roots alone launches none
    (:func:`walk_solves`)."""
    from .tree_scan import WALK_UNROLL  # noqa: PLC0415

    sizes = {"coordinate": WALK_COORDINATE, "link": WALK_LINK}
    # The walk up of the force x into u, then the walk down of u into the trial's response (no prior).
    up = ts.walk_texts(ts.WHITEN, source="$X.u", sink="u[$c] = z;", value="x[$c]", prior="0.0f", **sizes)
    down = ts.walk_texts(
        ts.LEAVE,
        source="$X.u",
        sink="qd($c, q, $X.p);",
        value="u[$c]",
        prior="0.0f",
        end="twist($b, $V);",
        **sizes,
    )
    hand = "WALK_HAND($X, u[$c], 0.0f)"
    walked = {r: k for r, _, k in _walked(layout, classes)}
    cases = []
    for r, offset, family in _walk_roots(layout):
        if family < 0:
            # A diagonal root holds no walk: the response half divides its force by its mass.
            body = ""
        elif r not in walked:
            body = (
                f"const int ds = p.art_dof_start.data[art];"
                f"for (int k = 0; k < {layout.root_coordinates[r]}; ++k) trial.response.data[ds + {offset} + k] = 0.0f;"
            )
        else:
            cls = classes[r]
            body = (
                f"const int a = {_root_articulation(layout, r)};\n"
                "const int j0 = walk.joint_start.data[a], d0 = walk.dof_start.data[a];\n"
                f"const float* __restrict__ data = walk.data.data + (size_t)d0 * {WALK_COORDINATE};\n"
                f"const float* __restrict__ link = walk.link.data + (size_t)j0 * {WALK_LINK};\n"
                "const float* __restrict__ x = trial.force.data + d0;\n"
                "float* __restrict__ out = trial.response.data + d0;\n"
                f"float u[{max(cls.coordinates, 1)}];\n"
                "const auto qd = [&](const int c, const float q, const float) { out[c] = q; };\n"
                "const auto twist = [](const int, const float*) {};\n"
                + cls.walk_twice(up, down, hand=hand, registers=memory.REGISTERS_PER_THREAD)
                + "\n"
            )
        cases.append(f"    case {r}: {{\n{body}\n        break;\n    }}")
    walked_classes = tuple(cls for cls in classes if cls is not None)
    return KernelTemplate(
        _predictor_name("tree_kinetic_walk", layout, None, "sparse", classes),
        (("groups", int), ("p", GroupTrees), ("plan", HeldPlan), ("walk", WalkTrees), ("trial", TrialWords)),
        substitute(_WALK_SOLVE, ROOT_COUNT=len(_walk_roots(layout)), ROOTS="\n".join(cases)),
        items("tid", flat=True),
        preludes=(walk_down_prelude(walked_classes, WALK_UNROLL),) if walked_classes else (),
        dual=True,
        native_name="tree_kinetic_walk_native",
    ).kernel()


@functools.cache
def get_predictor_kernel(
    schedules: HeldSchedules,
    *,
    layout: TreeLayout = REFERENCE_LAYOUT,
    response="sparse",
    clocked: bool = False,
    twists: bool,
    classes: tuple,
    slips: int,
    side: bool = True,
):
    """The predictor's response half (Z4.2, :data:`_RESPONSE_HALF`), after the force half (:func:`get_force_kernel`) and
    the walk's solve (:func:`get_walk_solve_kernel`): the trial's response read back into the block, ``qdd`` and
    ``v_hat``, the drives' saturation decision, and a world whose drives moved takes its decided step by the border on
    the walk (:func:`border_solve`), from the trial's force plus the decision's change of it.

    ``schedules`` are the scans' schedules; ``clocked`` adds the predictor stage counters (see :func:`get_helpers`);
    ``twists`` holds the twist publication (:func:`publishes_twists`), which only CUDA builds; ``classes`` holds each
    root's walking class (:class:`.tree_scan.WalkClass`, ``None`` for a prescribed root); ``slips`` the classes'
    slip capacity, the border's build lanes (:func:`held_lambdas`); ``side`` the factor's
    :attr:`.tree_factor.TreeFactor.walk_side`. ``response`` names the group's kind in the kernel's name only.
    """

    if len(classes) != len(_walk_roots(layout)):
        raise ValueError("The predictor needs a walking class (or None) for each root")
    clocks = _native_clocks(clocked, "f", stages=5, column=8)
    velocity = _FORCE.velocity("dof", "f.qdd.data[dof]")

    def stage(action, start="", decide=""):
        return substitute(
            _RESPONSE_STAGE, layout.sizes(), START=start, TRANSPOSE_ACTION=action, VELOCITY=velocity, DECIDE=decide
        )

    # The decided step's response: the border's words, on the roots it changed.
    changed = f"if(!(moved&(2<<{_root(layout).replace('[i]', '[col]')})))continue;"
    return _half_kernel(
        "tree_kinetic_predict",
        schedules,
        layout,
        response,
        classes,
        "moved",
        _RESPONSE_HALF,
        (walk_loads_prelude(), HELD, memory_words(memory.Place.READ_ONLY)),
        TRIAL=held_lambdas(layout, slips, ts.record_half(side))
        + stage("float value=s[$WHITE+col];", clocks["CLOCK2"] + clocks["CLOCK3"], QUIET_DRIVE),
        RESPONSE=(
            substitute(_DIAGONAL_TRIAL, layout.sizes(), ON=_on_diagonal(layout, "i"))
            if _diagonal_roots(layout)
            else "trial.response.data[ds+i]"
        ),
        SATURATE=substitute(SATURATE_DRIVES, layout.sizes(), ROOT=_root(layout)),
        BORDER=border_solve(layout, classes),
        SECOND_RESPONSE=stage(changed + "float value=s[$WHITE+col];"),
        TRANSPORT_BLOCK=_transport_block(layout),
        TWIST_BLOCK=_twist_block() if twists else "",
        **{name: text for name, text in clocks.items() if name not in ("CLOCK2", "CLOCK3")},
    )
