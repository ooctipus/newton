# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers

# SPDX-License-Identifier: Apache-2.0

"""Generated current/next tree moments, bias and fused state publication."""

import dataclasses
import functools
import hashlib
from types import SimpleNamespace

import warp as wp

from ...sim import BodyFlags, JointType
from . import (
    cached_force,
    field_group,
    kernels,
    sparse_factor,
)
from . import tree_scan as ts
from .device_preludes import CUDA_ONLY, DevicePrelude, device_branches, substitute
from .field_group import Field
from .kernel_template import Dispatch, KernelTemplate, python_kernel
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
    Field("com_offset", wp.vec3, ("group", "body"), fill=None, owner="cache"),
    Field("geometric", wp.float32, ("group", "entry"), fill=None, owner="cache"),
    Field("current_valid", wp.int32, ("articulation",), owner="factor", attr="fk_id_cache_valid"),
    Field("geometry_valid", wp.int32, ("group",), owner="cache"),
    Field("generation", wp.int64, ("group",), owner="cache"),
    Field("geometry_generation", wp.int64, ("group",), fill=-1, owner="cache"),
    Field("status", wp.int32, ("group",), owner="cache"),
    Field("source_q", wp.uint64, ("group",), owner="cache"),
    Field("source_qd", wp.uint64, ("group",), owner="cache"),
    Field("stage_clocks", wp.int64, ("group", "stage"), owner="step"),
)
"""The held tree's cache. Geometry, forces and source tags follow factor groups (the cache allocates them);
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


@wp.struct
class HeldResponse:
    lower: wp.array3d[float]
    inverse: wp.array[float]
    valid: wp.array[int]
    status: wp.array[int]
    # Forest groups: one held factor family per root and each root's index in it.
    lower1: wp.array3d[float]
    lower2: wp.array3d[float]
    lower3: wp.array3d[float]
    factor_index: wp.array2d[int]


def root_field(name: str, root: int) -> str:
    """The :class:`HeldResponse` field of a forest's root ``root``."""
    return name if root == 0 else f"{name}{root}"


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


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    return wp::int64(clock64());
#else
    return wp::int64(0);
#endif
""")
def _clock() -> wp.int64: ...


_LANES_SOURCE = r"""
#if defined(__CUDA_ARCH__)
    return wp::vec_t<3,int>(LANE,STRIDE,SUB);
#else
    return wp::vec_t<3,int>(0,1,0);
#endif
"""

_SYNC_SOURCE = r"""
#if defined(__CUDA_ARCH__)
    SYNC();
#endif
"""

_SUB_SOURCE = r"""
#if defined(__CUDA_ARCH__)
    return SUB;
#else
    return 0;
#endif
"""

_LANE_STRIDE = r"""
#if defined(__CUDA_ARCH__)
    const int lane=LANE,stride=STRIDE;
#else
    const int lane=0,stride=1;
#endif
"""


_DIFFERENT_MOTION_SOURCE = r"""
    if(state_qd.data==motion_qd.data)return 0;
    int different=0;
#if defined(__CUDA_ARCH__)
    for(int i=LANE;i<count;i+=STRIDE)
        different|=state_qd.data[start+i]!=motion_qd.data[start+i];
    return ANY(different);
#else
    for(int i=0;i<count;++i)different|=state_qd.data[start+i]!=motion_qd.data[start+i];
    return different;
#endif
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
#if defined(__CUDA_ARCH__)
    return ALL(equal);
#else
    return equal?1:0;
#endif
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


# A launch block's vote, not a tree's (the prelude's ANY): a packed layout's
# block is one warp of several trees, an unpacked block one tree.
_BLOCK_VOTE = r"""
#if defined(__CUDA_ARCH__)
    if constexpr(LANES<32)return __any_sync(0xffffffffu,value);
    else return __syncthreads_or(value);
#else
    return value;
#endif
"""


@functools.cache
def _block_vote(layout):
    """Whether any thread of a launch block passes a nonzero value; every thread of the block must call it.

    A packed layout's block is one warp of several trees, so the vote spans the
    warp; an unpacked block holds one tree and votes over the block.
    """

    @wp.func_native(layout.native(_BLOCK_VOTE))
    def block_any(value: int) -> int: ...

    return block_any


@wp.func_native("float* s=reinterpret_cast<float*>(address);for(int k=0;k<7;++k)s[7*body+k]=pose[k];")
def _store_pose(address: wp.uint64, body: int, pose: wp.transform): ...


@wp.func_native(
    "float* s=reinterpret_cast<float*>(address);wp::transform p;for(int k=0;k<7;++k)p[k]=s[7*body+k];return p;"
)
def _load_pose(address: wp.uint64, body: int) -> wp.transform: ...


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
    external: ts.Schedule
    """The predictor's external wrench sums."""

    @property
    def chains(self) -> bool:
        """Whether a scan runs on the heavy chains (whose texts address the block's threads)."""
        return ts.Schedule.HEAVY_CHAIN in (self.pose, self.motion, self.reduce, self.external)

    @property
    def tag(self) -> str:
        """The kernel-name suffix of the schedules."""
        return "_" + "_".join(s.value for s in (self.pose, self.motion, self.reduce, self.external))


def held_lanes(tree: TreePlan, groups: int, *, cuda: bool = True) -> int:
    """Threads per tree of a held group of ``groups`` trees shaped like ``tree``, from the tree's data.

    A tree of more than 32 bodies takes two warps. Any other takes the narrowest power of two of at least 8 lanes
    covering its bodies, widened until the group fills whole 32-thread blocks. Measured on the RTX PRO 6000 (finish,
    predict and repair of 4096 trees at each lane width, and the construction-time timings the replaced lane choice
    recorded in the gate's tasks): this width ran within 3% of the fastest width of every tree of up to 32 bodies.
    The coordinates do not set it: the 13-body, 21-coordinate Franka forest runs 27% slower at 32 lanes than at 16,
    a one-body tree 8% slower at 4 lanes than at 8. On the CPU (``cuda`` false) each tree takes a block of its own:
    one thread runs it serially.
    """
    bodies = tree.count
    if bodies > 32:
        return 64
    if not cuda:
        return 32
    lanes = max(8, 1 << (bodies - 1).bit_length())
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
        choose(_EXTERNAL, ts.Direction.UP),
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
_EXTERNAL_SLOTS = ts.Storage({"wrench": "s[6*({node})+({k})]"})
_MOTION = ts.TWIST.with_zero_acceleration()
"""The motion scan's combine where no joint has an own bias acceleration (:attr:`.TreeLayout.own_bias`)."""
_REDUCE = ts.add(ts.Group("wrench", 6), ts.Group("moment", INERTIA_CHANNELS, "refresh"))
_EXTERNAL = ts.add(ts.Group("wrench", 6))
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


_COLLECT = r"""    $PHYSICAL_TABLE

    float* s=reinterpret_cast<float*>(address);
#if defined(__CUDA_ARCH__)
    const int lane=LANE,stride=STRIDE;
#else
    const int lane=0,stride=1;
#endif
$CLOCK0$REDUCE
$CLOCK1    for(int dof=lane;dof<$DOFS;dof+=stride) {
        const int body=p.dof_joint.data[dof];
        float bias=0.0f;
        for(int k=0;k<6;++k)bias+=s[$AXIS+6*dof+k]*s[$WRENCH+6*body+k];
        cache.bias.data[group*$DOFS+dof]=bias;
        if(!wp::isfinite(bias)) {
#if defined(__CUDA_ARCH__)
            atomicOr(&cache.status.data[group],1);
#else
            cache.status.data[group]|=1;
#endif
        }
        if(refresh) {
            const float* t=s+$MOMENT+13*body,*axis=s+$AXIS+6*dof;
            const wp::vec3 l(axis[0],axis[1],axis[2]),w(axis[3],axis[4],axis[5]),h(t[1],t[2],t[3]);
            const wp::vec3 linear=t[0]*l+wp::cross(w,h);
            wp::vec3 angular=wp::cross(h,l);
            for(int i=0;i<3;++i)for(int j=0;j<3;++j)angular[i]+=t[4+3*i+j]*w[j];
            for(int k=0;k<3;++k){s[6*dof+k]=linear[k];s[6*dof+3+k]=angular[k];}
        }
    }
#if defined(__CUDA_ARCH__)
    SYNC();
#endif
$CLOCK2    if(refresh)for(int e=lane;e<$NNZ;e+=stride) {
        const int row=$PHYSICAL_ROW,col=$PHYSICAL_COL,src=p.source.data[e];
        const int projection=src==col?row:col;
        float value=0.0f;
        for(int k=0;k<6;++k)value+=s[$AXIS+6*projection+k]*s[6*src+k];
        cache.geometric.data[group*$NNZ+e]=value;
        if(!wp::isfinite(value)) {
#if defined(__CUDA_ARCH__)
            atomicOr(&cache.status.data[group],2);
#else
            cache.status.data[group]|=2;
#endif
        }
    }
#if defined(__CUDA_ARCH__)
    SYNC();
#endif
$CLOCK3    if(lane==0) {
        cache.generation.data[group]+=1;
        $CURRENT_VALID
        cache.geometry_valid.data[group]=refresh&&cache.status.data[group]==0;
        if(refresh)cache.geometry_generation.data[group]=cache.generation.data[group];
    }
"""

_FORCE = cached_force.ForceOperands(
    q="f.joint_q.data",
    state_qd="f.joint_qd.data",
    predictor_qd="f.predictor_qd.data",
    stiffness="f.stiffness.data",
    reference="f.reference.data",
    damping="f.damping.data",
    dt="f.dt",
)

_FORCE_STAGE = r"""    for(int i=lane;i<$DOFS;i+=stride) {
        const int dof=ds+i,body=p.dof_joint.data[i];
        float value=-cache.bias.data[group*$DOFS+i]+f.joint_f.data[dof]+f.u0.data[dof];
        if(nonzero){$PROJECT}
        if($SPRING_GUARD)value+=$PASSIVE;
        s[$SIX_BODIES+i]=value;
    }
#if defined(__CUDA_ARCH__)
    SYNC();
#endif
"""
"""The predictor's force on each coordinate, ``s[$SIX_BODIES + i]``: bias, joint and control forces, the body wrenches
projected (``s[6 body + k]``, dead once projected) and the passive springs."""

_RESPONSE_STAGE = r"""$START_FORWARD    $FORWARD_STAGE
$START_RESPONSE    for(int col=lane;col<$DOFS;col+=stride) {
        $TRANSPOSE_ACTION
        const int dof=ds+$RESPONSE_PHYSICAL;
        const bool valid=cache.current_valid.data[art]&&d.valid.data[group]&&!($STATUS)&&!cache.status.data[group]&&wp::isfinite(value);
        f.qdd.data[dof]=valid?value:NAN;
        f.vhat.data[dof]=$VELOCITY;
        if(!valid) {
#if defined(__CUDA_ARCH__)
            atomicOr(&cache.status.data[group],4);
#else
            cache.status.data[group]|=4;
#endif
        }
        $DECIDE
    }
#if defined(__CUDA_ARCH__)
    SYNC();
#endif
"""
_FORCE_HALF = r"""    $PHYSICAL_TABLE

    const int art=p.group_to_art.data[group];
    const int ds=p.art_dof_start.data[art],js=p.art_joint_start.data[art];
#if defined(__CUDA_ARCH__)
    const int lane=LANE,stride=STRIDE;
    $SCRATCH
    __shared__ int nonzero_all[$PACK];int& nonzero=nonzero_all[SUB];
    __shared__ int moved_all[$PACK];int& moved=moved_all[SUB];
    const auto MOVE=[&](int bits){atomicOr(&moved,bits);};
#else
    const int lane=0,stride=1;
    float s[$PRED_SCRATCH];int nonzero=0;int moved=0;
    const auto MOVE=[&](int bits){moved|=bits;};
#endif
    if(lane==0){nonzero=0;moved=0;$CLEAR_SLIPS}
$CLOCK0#if defined(__CUDA_ARCH__)
    SYNC();
#endif
    for(int local=lane;local<$BODIES;local+=stride) {
        const int body=p.joint_child.data[js+local];
        // A kinematic body takes no external wrench, as in ordinary dynamics
        // its wrench must not reach its ancestors.
        auto input=f.body_f.data[body];
        if((f.body_flags.data[body]&$KINEMATIC_FLAG)!=0){for(int k=0;k<6;++k)input[k]=0.0f;}
        $SHIFT
        bool active=false;for(int k=0;k<6;++k)active|=input[k]!=0.0f;
        if(active) {
#if defined(__CUDA_ARCH__)
            atomicOr(&nonzero,1);
#else
            nonzero=1;
#endif
        }
    }
#if defined(__CUDA_ARCH__)
    SYNC();
#endif
$CLOCK1$REDUCE
$FORCE_STAGE    for(int i=lane;i<$DOFS;i+=stride)trial.force.data[ds+i]=s[$SIX_BODIES+i];
"""
"""The predictor's force half (Z4.2): the external wrenches' reduction and the force on each coordinate
(:data:`_FORCE_STAGE`), stored to ``trial.force`` at its global coordinate for the walk's solve (:func:`get_walk_solve_kernel`),
which answers in ``trial.response``."""

_RESPONSE_HALF = r"""    $PHYSICAL_TABLE

    const int art=p.group_to_art.data[group];
    const int ds=p.art_dof_start.data[art],js=p.art_joint_start.data[art];
#if defined(__CUDA_ARCH__)
    const int lane=LANE,stride=STRIDE;
    $SCRATCH
    __shared__ int moved_all[$PACK];int& moved=moved_all[SUB];
    const auto MOVE=[&](int bits){atomicOr(&moved,bits);};
#else
    const int lane=0,stride=1;
    float s[$PRED_SCRATCH];int moved=0;
    const auto MOVE=[&](int bits){moved|=bits;};
#endif
    if(lane==0)moved=0;
$CLOCK0    for(int i=lane;i<$DOFS;i+=stride) {
        s[$SIX_BODIES+i]=trial.force.data[ds+i];
        s[$WHITE+i]=trial.response.data[ds+i];
    }
#if defined(__CUDA_ARCH__)
    SYNC();
#endif
$CLOCK1    $TRIAL
$CLOCK4    if(moved!=0) {
    $SATURATE
    $BORDER
    $SECOND_RESPONSE
    }
    // A prescribed (kinematic) free root keeps its given velocity: no transport.
    $TRANSPORT_BLOCK
$TWIST_BLOCK$CLOCK5"""
"""The predictor's response half (Z4.2): the force and the walk's response read back into the block, then the trial's
response (``qdd``, ``v_hat``), the drives' decision and the border's decided step, the root transport and the twists."""


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


_STORE_TERMS_SOURCE = r"""
    float* s=reinterpret_cast<float*>(address);
    for(int k=0;k<6;++k)s[$WRENCH+6*body+k]=wrench[k];
    if(refresh)for(int k=0;k<13;++k)s[$MOMENT+13*body+k]=record[k];
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
    fragments = {"CLOCK0": f"#if defined(__CUDA_ARCH__)\n    long long c0=clock64(),{copies};\n#endif\n"}
    for k in range(1, stages + 1):
        fragments[f"CLOCK{k}"] = f"#if defined(__CUDA_ARCH__)\n    c{k}=clock64();\n#endif\n"
    durations = "".join(f"o[{k}]=c{k + 1}-c{k};" for k in range(stages))
    fragments[f"CLOCK{stages}"] += (
        f"#if defined(__CUDA_ARCH__)\n    if(lane==0&&{owner}.stage_clocks.shape[0]>group){{long long* o={owner}.stage_clocks.data"
        f"+group*{owner}.stage_clocks.shape[1]+{column};{durations}}}\n#endif\n"
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
):
    """Bind one physical tree evaluation to its storage and its scans' schedules.

    ``clocked`` adds the collection (``stage_clocks`` columns 13-15) and body
    term (16-21) counters. Both arguments change the emitted code, so they are
    arguments (cache-key members), never environment reads.
    """
    pose_source = _down(layout, ts.TRANSFORM, schedules.pose, _POSE_SLOTS)
    motion_source = _down(layout, ts.TWIST if layout.own_bias else _MOTION, schedules.motion, _MOTION_SLOTS)
    reduction = _reduction(_REDUCE, schedules.reduce, _REDUCE_SLOTS)
    collect_source = layout.native(
        _COLLECT,
        PHYSICAL_TABLE=layout.physical_table(),
        PHYSICAL_ROW=layout.physical("p.row.data[e]"),
        PHYSICAL_COL=layout.physical("p.col.data[e]"),
        REDUCE=reduction,
        CURRENT_VALID=_current_valid(layout),
        **_native_clocks(clocked, "cache", stages=3, column=13),
    )

    @wp.func_native(layout.native(_STORAGE_SOURCE, BLOCK=_placed_block(layout, "values", "$TOTAL")))
    def _storage(plan: HeldPlan, group: int) -> wp.uint64: ...

    @wp.func_native(
        layout.render("float* s=reinterpret_cast<float*>(address)+$MOTION;for(int k=0;k<6;++k)s[6*body+k]=v[k];")
    )
    def _store_motion(address: wp.uint64, body: int, v: wp.spatial_vector): ...

    @wp.func_native(
        layout.render(
            "float* s=reinterpret_cast<float*>(address)+$MOTION+$SIX_BODIES*acceleration;wp::spatial_vector v;for(int k=0;k<6;++k)v[k]=s[6*body+k];return v;"
        )
    )
    def _load_motion(address: wp.uint64, body: int, acceleration: int) -> wp.spatial_vector: ...

    @wp.func_native(
        layout.render(
            "float* s=reinterpret_cast<float*>(address)+$MOTION+$SIX_BODIES;for(int k=0;k<6;++k)s[6*body+k]=a[k];"
        )
    )
    def _store_bias(address: wp.uint64, body: int, a: wp.spatial_vector): ...

    @wp.func_native(
        layout.render("float* s=reinterpret_cast<float*>(address)+$ORIGIN+3*shift;for(int k=0;k<3;++k)s[k]=v[k];")
    )
    def _store_origin(address: wp.uint64, shift: int, v: wp.vec3): ...

    @wp.func_native(
        layout.render("float* s=reinterpret_cast<float*>(address)+$ORIGIN+3*shift;return wp::vec3(s[0],s[1],s[2]);")
    )
    def _load_origin(address: wp.uint64, shift: int) -> wp.vec3: ...

    @wp.func_native(
        layout.render("float* s=reinterpret_cast<float*>(address)+$AXIS;for(int k=0;k<6;++k)s[6*dof+k]=v[k];")
    )
    def _store_axis(address: wp.uint64, dof: int, v: wp.spatial_vector): ...

    @wp.func_native(pose_source)
    def _scan_poses(address: wp.uint64, p: HeldPlan): ...

    @wp.func_native(motion_source)
    def _scan_motion(address: wp.uint64, p: HeldPlan): ...

    @wp.func_native(layout.render(_STORE_TERMS_SOURCE))
    def _store_terms(
        address: wp.uint64, body: int, wrench: wp.spatial_vector, record: inertia_record, refresh: int
    ): ...

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
        cache.com_offset[g.group, local] = radius
        if wp.static(clocked):
            clock_b2 = _clock()
        _store_terms(g.address, local, wrench, record, g.refresh)
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
        address: wp.uint64, group: int, p: sparse_factor.SparsePlan, plan: HeldPlan, cache: KineticData, refresh: int
    ): ...

    return SimpleNamespace(
        _body_terms=_body_terms,
        _collect=_collect,
        _load_motion=_load_motion,
        _load_origin=_load_origin,
        _scan_motion=_scan_motion,
        _scan_poses=_scan_poses,
        _storage=_storage,
        _store_axis=_store_axis,
        _store_bias=_store_bias,
        _store_motion=_store_motion,
        _store_origin=_store_origin,
        _store_terms=_store_terms,
    )


@functools.cache
def get_state_kernel(
    finish: bool,
    schedules: HeldSchedules,
    *,
    layout: TreeLayout = REFERENCE_LAYOUT,
    body_velocity: bool = False,
    motion_velocity: bool = False,
    listed: bool = False,
    bank_coordinates: int = 0,
    clocked: bool = False,
):
    """Build a complete repair, or with ``finish`` a complete finish, from its stage functions.

    Each stage reads its group from the :class:`TreeGroup` admission returns.
    Admission binds the group's threads and, in a repair, skips a group whose
    current cache already holds its state bank. A finish then integrates with
    the original generalized laws: accelerations, free-root transport and
    joints. Both store relative joint poses; scan poses and publish
    articulation origins; store joint axes and relative velocities and scan
    motion; publish bodies with their bias wrenches and moments; and collect
    bias, geometry and validity.

    Every finish and repair takes one launch ABI, which keeps each stage one
    call below the kernel: after the six common arguments, the alternate
    motion velocity ``motion_qd``, a listed repair's block list
    ``repair_blocks`` and its count ``repair_count``, and the ``bank`` a
    repair compares its input bank with. A kernel leaves the inputs of the
    other forms unread.

    ``listed`` builds the repair over the block list of
    :func:`get_classify_kernel` and :func:`get_bank_kernel`: logical block
    ``item`` repairs listed block ``repair_blocks[item]`` and items past
    ``repair_count[0]`` return at once, so a launch over every block on a
    capped grid (Warp's grid-stride loop) costs little more than the listed
    blocks. Each listed group runs the repair unchanged, early exit included.
    Like the classification, it is a contract of ordinary motion.

    ``bank_coordinates`` (a group's coordinate count) builds the one-pass
    repair that compares input banks inside its blocks: a valid group stamped
    with the ``bank`` argument compares the two banks (see
    :func:`_bank_binding`) and, when they are equal, is restamped and takes the
    early exit. It serves launches too small to classify, where a comparison
    pass of its own would cost a launch. Classified launches compare in
    :func:`get_bank_kernel` instead. ``schedules`` are the scans' schedules;
    ``clocked`` selects the stage counters (``stage_clocks`` columns 0-7 for
    the stages, 22-24 for motion; see :func:`get_helpers`).
    """
    if finish and motion_velocity:
        raise ValueError("Alternate motion velocity is a repair-only contract")
    if finish and listed:
        raise ValueError("A listed launch is a repair-only contract")
    if listed and motion_velocity:
        raise ValueError("A listed launch is a contract of the repair of ordinary motion")
    if bank_coordinates and (finish or motion_velocity or listed):
        raise ValueError("A compared input bank is a contract of the one-pass repair of ordinary motion")
    h = get_helpers(layout, schedules, body_velocity, clocked=clocked)
    _lanes, _sync, _sub, _different_motion = _thread_helpers(layout)
    banked = bank_coordinates != 0
    _bank_matches = _bank_binding(bank_coordinates, layout) if banked else None
    scan_poses, scan_motion, collect, body_terms = h._scan_poses, h._scan_motion, h._collect, h._body_terms
    _storage, _store_motion, _load_motion, _store_bias = h._storage, h._store_motion, h._load_motion, h._store_bias
    _store_origin, _load_origin, _store_axis = h._store_origin, h._load_origin, h._store_axis
    bodies, dofs, pack, free_root = layout.bodies, layout.dofs, layout.pack, layout.free_root
    roots, free_roots, forest, own_bias = len(layout.roots), layout.free_root_count, layout.forest, layout.own_bias
    from .tree_factor import WALK_FRAME, store_walk_columns  # noqa: PLC0415 (tree_factor imports this module)

    @wp.func
    def admit(
        block: int,
        logical_lane: int,
        p: sparse_factor.SparsePlan,
        plan: HeldPlan,
        data: PublicationData,
        cache: KineticData,
        requests: wp.array[int],
        global_refresh: int,
        motion_qd: wp.array[float],
        repair_blocks: wp.array[int],
        repair_count: wp.array[int],
        bank: StateBank,
    ) -> TreeGroup:
        """Bind the group's threads; admit it unless a repair finds its current cache valid.

        A listed repair takes its block from the list. A repair that compares
        banks keeps the cache of a group stamped with ``bank`` whose input bank
        holds the same words, and stamps it with the input bank.
        """
        g = TreeGroup()
        lanes = _lanes()
        g.lane = lanes[0]
        g.stride = lanes[1]
        if g.stride == 1 and logical_lane != 0:
            return g
        if wp.static(listed):
            if block >= repair_count[0]:
                return g
            block = repair_blocks[block]
        g.group = block * wp.static(pack) + lanes[2]
        g.art = p.group_to_art[g.group]
        g.ds = p.art_dof_start[g.art]
        g.js = p.art_joint_start[g.art]
        if wp.static(motion_velocity):
            g.motion_changed = _different_motion(data.joint_qd, motion_qd, g.ds, dofs)
        refresh = int(global_refresh != 0)
        valid = cache.current_valid[g.art]
        if wp.static(forest):
            for root in range(1, wp.static(roots)):
                valid = wp.min(valid, cache.current_valid[plan.arts[g.group, root]])
        if not wp.static(finish):
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
                    _store_origin(g.address, 1, wp.transform_get_translation(relative))
                    relative = wp.transform(wp.vec3(), wp.transform_get_rotation(relative))
            else:
                if plan.parent[local] < 0:
                    _store_origin(g.address, 2 * plan.root_of[local] + 1, wp.transform_get_translation(relative))
                    relative = wp.transform(wp.vec3(), wp.transform_get_rotation(relative))
            _store_pose(g.address, local, relative)
        _sync()

    @wp.func
    def poses(g: TreeGroup, plan: HeldPlan, data: PublicationData):
        """Complete the pose scan and publish each articulation's origin."""
        scan_poses(g.address, plan)
        if wp.static(not forest):
            if g.lane == 0:
                root_body = data.joint_child[g.js]
                origin = wp.transform_point(_load_pose(g.address, 0), data.body_com[root_body])
                _store_origin(g.address, 0, origin)
                data.articulation_origin[g.art] = origin + _load_origin(g.address, 1)
        else:
            for root in range(g.lane, wp.static(roots), g.stride):
                root_local = plan.root_local[root]
                root_body = data.joint_child[g.js + root_local]
                origin = wp.transform_point(_load_pose(g.address, root_local), data.body_com[root_body])
                _store_origin(g.address, 2 * root, origin)
                data.articulation_origin[plan.arts[g.group, root]] = origin + _load_origin(g.address, 2 * root + 1)
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
                origin = _load_origin(g.address, 2 * plan.root_of[local])
                shift = _load_origin(g.address, 2 * plan.root_of[local] + 1)
            anchor = data.joint_X_p[joint]
            if parent >= 0:
                anchor = _load_pose(g.address, parent) * anchor
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
            count = data.joint_dof_dim[joint, 0] + data.joint_dof_dim[joint, 1]
            for k in range(count):
                _store_axis(g.address, start - g.ds + k, data.joint_S_s[start + k])
            if data.walk_now.shape[0] > 0:
                # The walk's step geometry (tree_factor.WalkTrees.now): the same columns at the joint's frame, so a
                # walk takes them there without S's lever about the origin, and the frame's world point.
                kind = data.joint_type[joint]
                free = kind == JointType.FREE or kind == JointType.DISTANCE
                if free:
                    point = wp.transform_point(_load_pose(g.address, local), data.body_com[data.joint_child[joint]])
                lin, ang = data.joint_dof_dim[joint, 0], data.joint_dof_dim[joint, 1]
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
                store_walk_columns(data.walk_now, data.joint_qd.shape[0], start, lin, ang, l0, l1, l2, a0, a1, a2)
                frame = point + shift
                for k in range(3):
                    data.walk_now_link[joint, wp.static(WALK_FRAME) + k] = frame[k]
            _store_motion(g.address, local, velocity)
            if wp.static(own_bias):
                _store_bias(g.address, local, bias)
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
                origin = _load_origin(g.address, 2 * plan.root_of[local])
                shift = _load_origin(g.address, 2 * plan.root_of[local] + 1)
            pose = _load_pose(g.address, local)
            data.body_q[body] = wp.transform(
                wp.transform_get_translation(pose) + shift, wp.transform_get_rotation(pose)
            )
            body_terms(
                g,
                body,
                local,
                pose,
                pose * data.body_X_com[body],
                origin,
                _load_motion(g.address, local, 0),
                _load_motion(g.address, local, 1),
                data,
                cache,
            )
        _sync()

    @wp.func
    def close(
        g: TreeGroup,
        p: sparse_factor.SparsePlan,
        plan: HeldPlan,
        data: PublicationData,
        cache: KineticData,
        q: wp.array[float],
        qd: wp.array[float],
    ):
        """Collect bias, geometry and validity, and record the state bank the cache now holds."""
        collect(g.address, g.group, p, plan, cache, g.refresh)
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
    # alternate-motion repair reads, repair_blocks and repair_count a listed
    # repair's block list, bank the bank a comparing repair reads. Every other
    # kernel leaves them unread.
    def kinetic_state(
        p: sparse_factor.SparsePlan,
        plan: HeldPlan,
        data: PublicationData,
        cache: KineticData,
        requests: wp.array[int],
        global_refresh: int,
        motion_qd: wp.array[float],
        repair_blocks: wp.array[int],
        repair_count: wp.array[int],
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
            global_refresh,
            motion_qd,
            repair_blocks,
            repair_count,
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
            accelerations(g, data)
            if wp.static(clocked):
                clock_i1 = _clock()
            transport(g, plan, data)
            if wp.static(clocked):
                clock_i2 = _clock()
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
        origin = _load_origin(g.address, 0)
        shift = _load_origin(g.address, 1)
        if wp.static(motion_velocity):
            qd = motion_qd
        if wp.static(clocked):
            clock_3 = _clock()
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
    if listed:
        name += "_listed"
    if banked:
        name += "_banked"
    return python_kernel(kinetic_state, name)


@functools.cache
def get_bank_kernel(*, layout: TreeLayout = REFERENCE_LAYOUT, coordinates: int = 1):
    """Compare the input banks of the blocks :func:`get_classify_kernel` listed for comparison.

    Logical block ``item`` takes listed block ``compare_blocks[item]`` (items
    past ``counts[1]`` return at once, so a capped grid strides the list). A
    group of that block whose repair would exit early but for its stamp, which
    names the ``bank`` argument (the step's output bank, into which the last
    finish published) instead of the input bank, compares the two banks (see
    :func:`_bank_binding`); when they hold the same words its stamp moves to
    the input bank. The block is then appended to the repair list
    (``repair_blocks``, counted in ``counts[0]``) unless every group in it takes
    its early exit, so the classification never lists it there itself and no
    block is listed twice. A pass of its own, rather than the comparison
    inside the listed repair, keeps that repair's registers: inside, the
    comparison raised a chain-scan repair's from 80 to 151 on a 13-body chain
    (sm_120).
    """
    _lanes, _, _, _ = _thread_helpers(layout)
    matches = _bank_binding(coordinates, layout)
    block_any = _block_vote(layout)
    pack, roots, forest = layout.pack, len(layout.roots), layout.forest

    def bank_pass(
        p: sparse_factor.SparsePlan,
        plan: HeldPlan,
        data: PublicationData,
        cache: KineticData,
        requests: wp.array[int],
        global_refresh: int,
        bank: StateBank,
        compare_blocks: wp.array[int],
        repair_blocks: wp.array[int],
        counts: wp.array[int],
    ):
        item, logical_lane = wp.tid()
        lanes = _lanes()
        lane, stride = lanes[0], lanes[1]
        if stride == 1 and logical_lane != 0:
            return
        if item >= counts[1]:
            return
        block = compare_blocks[item]
        group = block * wp.static(pack) + lanes[2]
        art = p.group_to_art[group]
        valid = cache.current_valid[art]
        refresh = int(global_refresh != 0 or (requests[art] & GEOMETRY_REQUEST) != 0)
        if wp.static(forest):
            for root in range(1, wp.static(roots)):
                valid = wp.min(valid, cache.current_valid[plan.arts[group, root]])
                refresh = int(refresh != 0 or (requests[plan.arts[group, root]] & GEOMETRY_REQUEST) != 0)
        current = int(valid != 0)
        if current != 0 and refresh != 0:
            current = int(
                cache.geometry_valid[group] != 0 and cache.geometry_generation[group] == cache.generation[group]
            )
        exits = int(0)
        if current != 0:
            exits = _same_source(cache, group, data.joint_q, data.joint_qd)
            if exits == 0 and _same_source(cache, group, bank.joint_q, bank.joint_qd) != 0:
                js = p.art_joint_start[art]
                exits = matches(
                    bank,
                    data.joint_q_start[js],
                    p.art_dof_start[art],
                    js,
                    data.joint_q,
                    data.joint_qd,
                    data.joint_child,
                    data.body_q,
                    data.body_qd,
                )
                if exits != 0 and lane == 0:
                    _stamp_source(cache, group, data.joint_q, data.joint_qd)
        # Every thread of the block votes (no thread returned since the block-uniform item test).
        if block_any(int(exits == 0)) != 0 and logical_lane == 0:
            repair_blocks[wp.atomic_add(counts, 0, 1)] = block

    name = f"tree_kinetic_bank{layout.bodies}"
    if forest:
        name += f"_forest{roots}"
    if pack > 1:
        name += f"_l{layout.lanes}"
    return python_kernel(bank_pass, name)


@functools.cache
def get_classify_kernel(*, layout: TreeLayout = REFERENCE_LAYOUT):
    """List the repair blocks holding a group whose repair would not take its early exit; one thread per block.

    A group takes its early exit when its cache and its forest roots' caches
    are valid, it is stamped with the input bank, and the step does not
    refresh it or its geometry is current. A group that meets every condition
    but the stamp, which names the ``bank`` argument instead, is left to the
    comparison: its block goes to ``compare_blocks`` (counted in
    ``counts[1]``), and :func:`get_bank_kernel` appends the block to the
    repair list unless every group in it then exits. Any other block with a
    group that does not exit goes to ``repair_blocks`` (counted in
    ``counts[0]``). The lists have no set order. Passing the input bank as
    ``bank`` lists nothing for comparison. The listed repair
    (``get_state_kernel(listed=True)``) runs each listed group's repair
    unchanged; blocks share no cache, state or publication entries, so every
    group computes the same values as in the one-pass launch.

    The classification is a contract of ordinary motion. A motion-velocity
    layout's repair runs on every step, and its groups also exit only where
    every prescaled velocity compares equal to the input velocity; the owner
    keeps its one-pass launch (see
    :class:`~.tree_kinetic_owner.TreeKineticState`).
    """
    pack, roots, forest = layout.pack, len(layout.roots), layout.forest

    def classify(
        p: sparse_factor.SparsePlan,
        plan: HeldPlan,
        data: PublicationData,
        cache: KineticData,
        requests: wp.array[int],
        global_refresh: int,
        bank: StateBank,
        repair_blocks: wp.array[int],
        compare_blocks: wp.array[int],
        counts: wp.array[int],
    ):
        block = wp.tid()
        needed = int(0)
        compared = int(0)
        for sub in range(wp.static(pack)):
            group = block * wp.static(pack) + sub
            art = p.group_to_art[group]
            valid = cache.current_valid[art]
            refresh = int(global_refresh != 0 or (requests[art] & GEOMETRY_REQUEST) != 0)
            if wp.static(forest):
                for root in range(1, wp.static(roots)):
                    valid = wp.min(valid, cache.current_valid[plan.arts[group, root]])
                    refresh = int(refresh != 0 or (requests[plan.arts[group, root]] & GEOMETRY_REQUEST) != 0)
            current = int(valid != 0)
            if current != 0 and refresh != 0:
                current = int(
                    cache.geometry_valid[group] != 0 and cache.geometry_generation[group] == cache.generation[group]
                )
            own = int(0)
            if current != 0:
                own = _same_source(cache, group, data.joint_q, data.joint_qd)
            exits = own
            if exits == 0:
                candidate = int(0)
                if current != 0 and own == 0:
                    candidate = _same_source(cache, group, bank.joint_q, bank.joint_qd)
                if candidate != 0:
                    compared = 1
                else:
                    needed = 1
        if compared != 0:
            compare_blocks[wp.atomic_add(counts, 1, 1)] = block
        elif needed != 0:
            repair_blocks[wp.atomic_add(counts, 0, 1)] = block

    name = f"tree_kinetic_classify{layout.bodies}"
    if forest:
        name += f"_forest{roots}"
    if pack > 1:
        name += f"_l{layout.lanes}"
    return python_kernel(classify, name)


def _solve_operations(layout: TreeLayout, classes: tuple) -> dict[str, str]:
    """The operands of the predictor's force (:data:`_FORCE_STAGE`) and of its response's status and velocity; the
    response itself is the walk's solve (:func:`get_walk_solve_kernel`)."""
    solve = {
        "PROJECT": cached_force.project_wrench("f.S.data[dof]", "s[6*body+k]", "value"),
        # Passive springs act on no root coordinate.
        "SPRING_GUARD": f"i>={layout.root_dofs if layout.free_root else 0}"
        if not layout.forest
        else "plan.dof_passive.data[i]!=0",
        "PASSIVE": _FORCE.passive("plan.q_index.data[group*$DOFS+i]", "dof"),
        "VELOCITY": _FORCE.velocity("dof", "f.qdd.data[dof]"),
        "STATUS": "d.status.data[group]",
    }
    return {name: substitute(text, layout.sizes()) for name, text in solve.items()}


_BORDER = """    // The border (SAT_PHYS section 1.2, W9): a tree's decided step on its all-K walk, with no factor array. The release
    // R = L_F^-1 L_K, built from the all-K walk's columns and the K-free pivot, is the decided factor's whitening; the
    // predictor's step takes R then R^T between its two walks, the rows R, the leave R^T, the audit R^-T (WalkRelease).
    // BORDER_FETCH(address, bytes, lane, stride): the lanes prefetch the 128-byte lines of [address, address + bytes)
    // into L1 (CUDA; nothing on the CPU).
    const auto BORDER_FETCH = [&](const void* address, const int bytes, const int lane, const int stride) {
#if defined(__CUDA_ARCH__)
        const char* const first = reinterpret_cast<const char*>(reinterpret_cast<size_t>(address) & ~size_t(127));
        const int lines = (int)((reinterpret_cast<const char*>(address) + bytes - first + 127) / 128);
        for (int k = lane; k < lines; k += stride)
            asm volatile("prefetch.global.L1 [%0];" ::"l"(first + 128 * k));
#endif
    };
    // FRAME_STATUS(bit): raise a bit of the border's status word, which every tree's lane 0 may write at once.
    const auto FRAME_STATUS = [&](const int bit) {
#if defined(__CUDA_ARCH__)
        atomicOr(&f.frame_status.data[0], bit);
#else
        f.frame_status.data[0] |= bit;
#endif
    };
    // The release of articulation a's slips, its coordinates whose released stiffness released[c] = K - Kd is positive
    // (kept[c] = Kd), root-most first (ascending coordinate), built on the block's lanes in three rounds. A slip's group
    // is its root (walk.release.root, its path's root-most coordinate): two slips' paths meet if and only if their roots
    // are equal, so groups touch disjoint coordinates and their lanes share one dense column and one pivot array (2n
    // floats).
    // BORDER_SLIPS(a, d0, n, released, fits), lane 0: the slips into walk.release.dof, .root and .count. A slip past the
    // release's capacity or depth, or every slip of a tree whose work does not fit (fits false), keeps its K and flags
    // bit 0 of f.frame_status.
    const auto BORDER_SLIPS = [&](const int a, const int d0, const int n, const float* released, const bool fits) {
        auto& release = walk.release;
        const int depth = release.b.shape[2];
        const int cap = a < release.dof.shape[0] ? release.dof.shape[1] : 0;
        int s = 0;
        #pragma unroll 1
        for (int i = 0; i < n; ++i) {
            if (!(released[i] > 0.0f))
                continue;
            int length = 0, root = i;
            #pragma unroll 1
            for (int body = walk.dof_body.data[d0 + i], c = i; body >= 0;) {
                #pragma unroll 1
                for (; c >= WALK_FIRST_OF(body); --c, ++length)
                    root = c;
                body = WALK_PARENT_OF(body);
                if (body >= 0)
                    c = WALK_FIRST_OF(body + 1) - 1;
            }
            if (!fits || s >= cap || length > depth) {
                FRAME_STATUS(1);
                continue;
            }
            release.dof.data[a * release.dof.shape[1] + s] = i;
            release.root.data[a * release.dof.shape[1] + s] = root;
            ++s;
        }
        release.count.data[a] = s;
    };
    // BORDER_COLUMN(a, d0, data, link, k), a lane a slip: slip k's column b = L^-1 P e_i walked up its root path
    // (WALK_SEED's step), one state of six, into walk.release.b leaf first. The lanes load every slip's column of a
    // world at once.
    const auto BORDER_COLUMN = [&](const int a, const int d0, const float* data, const float* link, const int k) {
        auto& release = walk.release;
        const int depth = release.b.shape[2], cap = release.dof.shape[1];
        float* const column = release.b.data + ((size_t)a * cap + k) * depth;
        const int i = release.dof.data[a * cap + k];
        float p[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float x = 1.0f;
        int t = 0;
        #pragma unroll 1
        for (int body = walk.dof_body.data[d0 + i], c = i; body >= 0;) {
            #pragma unroll 1
            for (; c >= WALK_FIRST_OF(body); --c, ++t) {
                float d[$unpacked];
                WALK_LOAD_RECORD(d, data + $coordinate * c);
                const float z = (x + ((d[0] * p[0] + d[1] * p[1] + d[2] * p[2]) + (d[3] * p[3] + d[4] * p[4] + d[5] * p[5])))
                                * d[12];
                x = 0.0f;
                for (int q = 0; q < 6; ++q)
                    p[q] -= d[6 + q] * z;
                column[t] = z;
            }
            const int up = WALK_PARENT_OF(body);
            if (up >= 0) {
                float l[6];
                WALK_LOAD_LINK(l, link, body, up);
                const float fx = p[0], fy = p[1], fz = p[2];
                p[3] -= l[1] * fz - l[2] * fy;
                p[4] -= l[2] * fx - l[0] * fz;
                p[5] -= l[0] * fy - l[1] * fx;
                c = WALK_FIRST_OF(up + 1) - 1;
            }
            body = up;
        }
    };
    // BORDER_GROUP(a, d0, data, released, kept, pivot, column, l, s), a lane a group: if slip l is its group's first,
    // the group's slips in order, each column into the dense column, the group's releases before it applied (b_j' =
    // R_<j b_j, still inside its path, W9 P4), then the rank-1 downdate's recurrence along its path, leaf first, with the
    // split pivot P0 + Kd at the slip (Gill, Golub, Murray and Saunders; the pivot without K, so no cancellation): b,
    // a = beta d^1/2, g = (d / dbar)^1/2 into walk.release, each coordinate's pivot updated and the column cleared behind
    // it. A pivot that is not positive flags bit 1 (SAT_PHYS falsifier 1: the split capacitance not positive definite).
    const auto BORDER_GROUP = [&](const int a, const int d0, const float* data, const float* released,
                                  const float* kept, float* pivot, float* column, const int l, const int s) {
        auto& release = walk.release;
        const int depth = release.b.shape[2], cap = release.dof.shape[1];
        const int* const group = release.root.data + a * cap;
        #pragma unroll 1
        for (int k = 0; k < l; ++k)
            if (group[k] == group[l])
                return;
        #pragma unroll 1
        for (int j = l; j < s; ++j) {
            if (group[j] != group[l])
                continue;
            const size_t at = ((size_t)a * cap + j) * depth;
            const int i = release.dof.data[a * cap + j], body0 = walk.dof_body.data[d0 + i];
            int t = 0;
            #pragma unroll 1
            for (int body = body0, c = i; body >= 0;) {
                #pragma unroll 1
                for (; c >= WALK_FIRST_OF(body); --c, ++t)
                    column[c] = release.b.data[at + t];
                body = WALK_PARENT_OF(body);
                if (body >= 0)
                    c = WALK_FIRST_OF(body + 1) - 1;
            }
            #pragma unroll 1
            for (int k = l; k < j; ++k) {
                if (group[k] != group[l])
                    continue;
                WALK_SLIP(release, ((size_t)a * cap + k) * depth, column, 1);
            }
            float alpha = -released[i];
            t = 0;
            #pragma unroll 1
            for (int body = body0, c = i; body >= 0;) {
                #pragma unroll 1
                for (; c >= WALK_FIRST_OF(body); --c, ++t) {
                    const float d = pivot[c], scale = sqrtf(d), b = column[c];
                    float dbar = d + alpha * (b * scale) * (b * scale);
                    if (c == i)
                        dbar = WALK_LOAD_PIVOT(data + $coordinate * i) + kept[i];
                    if (!(dbar > 0.0f) || !wp::isfinite(dbar))
                        FRAME_STATUS(2);
                    const float beta = alpha * b * scale / dbar;
                    alpha = alpha * d / dbar;
                    release.coordinate.data[at + t] = c;
                    release.b.data[at + t] = b;
                    release.a.data[at + t] = beta * scale;
                    release.g.data[at + t] = scale / sqrtf(dbar);
                    pivot[c] = dbar;
                    column[c] = 0.0f;
                }
                body = WALK_PARENT_OF(body);
                if (body >= 0)
                    c = WALK_FIRST_OF(body + 1) - 1;
            }
            #pragma unroll 1
            for (; t < depth; ++t)
                release.coordinate.data[at + t] = -1;
        }
    };
    // BORDER_APPLY(a, u, l, s), a lane a group, between the decided step's two walks: if slip l is its group's first,
    // u <- R^T R u on the group's coordinates, R_j for its slips in order (WALK_SLIP), then R_j^T the last first
    // (WALK_SLIP_T), each along its path. Groups touch disjoint coordinates, so their R_j commute:
    // (H + R + K_F)^-1 x = P^T L^-T R^T R L^-1 P x, since R^T R = L_K^T A_F^-1 L_K.
    const auto BORDER_APPLY = [&](const int a, float* u, const int l, const int s) {
        auto& release = walk.release;
        const int depth = release.b.shape[2], cap = release.dof.shape[1];
        const int* const group = release.root.data + a * cap;
        #pragma unroll 1
        for (int k = 0; k < l; ++k)
            if (group[k] == group[l])
                return;
        #pragma unroll 1
        for (int j = l; j < s; ++j)
            if (group[j] == group[l])
                WALK_SLIP(release, ((size_t)a * cap + j) * depth, u, 1);
        #pragma unroll 1
        for (int j = s - 1; j >= l; --j)
            if (group[j] == group[l])
                WALK_SLIP_T(release, ((size_t)a * cap + j) * depth, u);
    };"""


@functools.cache
def walk_tree_prelude() -> DevicePrelude:
    """The predictor's border, for both devices: :func:`.tree_factor.walk_paths_prelude`'s path table and walks (the
    slips' paths read it) and the border's build (:data:`_BORDER`). The whole-tree walks are the class emitter's
    (:func:`walk_tree`)."""
    from .tree_factor import WALK_COORDINATE, WALK_UNPACKED, walk_paths_prelude  # noqa: PLC0415

    walk = walk_paths_prelude(1)
    tree = substitute(_BORDER, coordinate=WALK_COORDINATE, unpacked=WALK_UNPACKED)
    return DevicePrelude("walk_tree", walk.source + "\n" + tree, cpu=walk.cpu + "\n" + tree)


_HELD = """    // The border held by the tree's segment (CUDA, every tree whose coordinates and bodies fit $HOLD words a lane): the
    // segment's lanes (one warp, or a packed tree's lanes) load the tree's walk, each coordinate's record and body and
    // each body's parent, first coordinate and link, into registers in one parallel trip, lane l holding coordinates and
    // bodies l, l + WIDTH, ...; then the release is built coordinate-major, a lane a slip (HELD_BUILD), and the
    // decided step's two tree walks run on every lane of the segment in lockstep, reading each step's values by
    // shuffle, with no global load on a chain; between them the release's primitive applies R^T R (BORDER_APPLY).
    struct WalkHold {
        float r[$HOLD][13];
        int parent[$HOLD], first[$HOLD];
        float l[$HOLD][3];
    };
    // HOLD_PICK(v, x): word x of a held array (x the same on every lane): its slot's register, from lane x % WIDTH.
    const auto HOLD_PICK = [&](const auto& v, const int x) {
        auto t = v[0];
        #pragma unroll
        for (int k = 1; k < $HOLD; ++k)
            if (x >= k * WIDTH) t = v[k];
        return READ(t, x & (WIDTH - 1));
    };
    const auto HOLD_LOAD = [&](WalkHold& h, const float* data, const float* link, const int* path, const int n,
                               const int nb) {
        #pragma unroll
        for (int k = 0; k < $HOLD; ++k) {
            const int x = lane + k * WIDTH;
            if (x < n) {
                float d[$unpacked];
                WALK_LOAD_RECORD(d, data + $coordinate * x);
                #pragma unroll
                for (int m = 0; m < 13; ++m) h.r[k][m] = d[m];
            } else {
                #pragma unroll
                for (int m = 0; m < 13; ++m) h.r[k][m] = 0.0f;
            }
            if (x < nb) {
                h.parent[k] = __ldg(path + 2 * x);
                h.first[k] = __ldg(path + 2 * x + 1);
                float l[6];
                WALK_LOAD_LINK(l, link, x, h.parent[k]);
                h.l[k][0] = l[0];
                h.l[k][1] = l[1];
                h.l[k][2] = l[2];
            } else {
                h.parent[k] = -1;
                h.first[k] = n;
                h.l[k][0] = h.l[k][1] = h.l[k][2] = 0.0f;
            }
        }
    };
    // HOLD_RECORD(h, c, d): coordinate c's s, ut and 1 / ell into d[0..12].
    const auto HOLD_RECORD = [&](const WalkHold& h, const int c, float* d) {
        #pragma unroll
        for (int m = 0; m < 13; ++m) {
            float t = h.r[0][m];
            #pragma unroll
            for (int k = 1; k < $HOLD; ++k)
                if (c >= k * WIDTH) t = h.r[k][m];
            d[m] = READ(t, c & (WIDTH - 1));
        }
    };
    const auto HOLD_LINK = [&](const WalkHold& h, const int b, float* l) {
        #pragma unroll
        for (int m = 0; m < 3; ++m) {
            float t = h.l[0][m];
            #pragma unroll
            for (int k = 1; k < $HOLD; ++k)
                if (b >= k * WIDTH) t = h.l[k][m];
            l[m] = READ(t, b & (WIDTH - 1));
        }
    };
    // HELD_UP(h, n, nb, x, u, state): WALK_UP_TREE on the held walk; lane 0 writes u and the join states.
    const auto HELD_UP = [&](const WalkHold& h, const int n, const int nb, const float* x, float* u, float* state) {
        for (int k = lane; k < 6 * nb; k += WIDTH) state[k] = 0.0f;
        WARP_SYNC();
        float p[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll 1
        for (int b = nb - 1; b >= 0; --b) {
            for (int k = 0; k < 6; ++k) p[k] += state[6 * b + k];
            const int f0 = HOLD_PICK(h.first, b), f1 = HOLD_PICK(h.first, b + 1);
            #pragma unroll 1
            for (int c = f1 - 1; c >= f0; --c) {
                float d[13];
                HOLD_RECORD(h, c, d);
                const float z = (x[c] + ((d[0] * p[0] + d[1] * p[1] + d[2] * p[2]) + (d[3] * p[3] + d[4] * p[4] + d[5] * p[5])))
                                * d[12];
                for (int k = 0; k < 6; ++k) p[k] -= d[6 + k] * z;
                if (lane == 0) u[c] = z;
            }
            const int up = HOLD_PICK(h.parent, b);
            float l[3];
            HOLD_LINK(h, b, l);
            if (up >= 0) {
                const float fx = p[0], fy = p[1], fz = p[2];
                p[3] -= l[1] * fz - l[2] * fy;
                p[4] -= l[2] * fx - l[0] * fz;
                p[5] -= l[0] * fy - l[1] * fx;
            }
            if (up != b - 1) {
                if (up >= 0) {
                    if (lane == 0)
                        for (int k = 0; k < 6; ++k) state[6 * up + k] += p[k];
                    WARP_SYNC();
                }
                for (int k = 0; k < 6; ++k) p[k] = 0.0f;
            }
        }
        WARP_SYNC();
    };
    // HELD_DOWN(h, n, nb, u, qd, twist): WALK_DOWN_TREE on the held walk, u read into the lanes first (u == qd allowed);
    // lane 0 writes qd, every lane the twists (one value).
    const auto HELD_DOWN = [&](const WalkHold& h, const int n, const int nb, const float* u, float* qd, float* twist) {
        float uh[$HOLD];
        #pragma unroll
        for (int k = 0; k < $HOLD; ++k) uh[k] = lane + k * WIDTH < n ? u[lane + k * WIDTH] : 0.0f;
        WARP_SYNC();
        float V[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll 1
        for (int b = 0; b < nb; ++b) {
            const int up = HOLD_PICK(h.parent, b);
            float l[3];
            HOLD_LINK(h, b, l);
            if (up < 0) {
                for (int k = 0; k < 6; ++k) V[k] = 0.0f;
            } else {
                if (up != b - 1)
                    for (int k = 0; k < 6; ++k) V[k] = twist[6 * up + k];
                V[0] += l[1] * V[5] - l[2] * V[4];
                V[1] += l[2] * V[3] - l[0] * V[5];
                V[2] += l[0] * V[4] - l[1] * V[3];
            }
            const int f0 = HOLD_PICK(h.first, b), f1 = HOLD_PICK(h.first, b + 1);
            #pragma unroll 1
            for (int c = f0; c < f1; ++c) {
                float d[13];
                HOLD_RECORD(h, c, d);
                const float uc = HOLD_PICK(uh, c);
                const float q = (uc - ((d[6] * V[0] + d[7] * V[1] + d[8] * V[2]) + (d[9] * V[3] + d[10] * V[4] + d[11] * V[5])))
                                * d[12];
                if (lane == 0) qd[c] = q;
                for (int k = 0; k < 6; ++k) V[k] += d[k] * q;
            }
            for (int k = 0; k < 6; ++k) twist[6 * b + k] = V[k];
        }
        WARP_SYNC();
    };
    // HELD_BUILD(h, a, n, nb, data, released, kept): the release of the tree's slips, coordinate-major. The slips are
    // its coordinates whose released stiffness is positive, root-most first (ascending), one a lane (lane j holds slip
    // j: at most $SLIPS, the class's slip capacity, and the release's; a slip past them keeps its K and raises status
    // bit 0). One leaf-first sweep over the coordinates the slips' paths hold: at each, every slip on it takes its
    // column step (b = L^-1 P e_i) from the held record; then in slip order slip k takes its recurrence step on its
    // lane and hands its b, a, g and the coordinate's new pivot to the later lanes, and each later slip on the
    // coordinate applies R_k's step to its column with its pair state (R_k meets a column only where both paths hold
    // the coordinate: from their meeting point). Every value is a register: the columns, the pivots (one a coordinate),
    // the pair states; nothing of the release is read back. The column and R_k's steps are the serial text's
    // (BORDER_COLUMN, BORDER_GROUP); the recurrence takes one rsqrt of each new pivot and multiplies where the serial
    // text takes two square roots and two divides (rounding). Returns the slips.
    const auto HELD_BUILD = [&](const WalkHold& h, const int a, const int n, const int nb, const float* data,
                                const float* released, const float* kept) -> int {
        auto& release = walk.release;
        const int depth = release.b.shape[2];
        const int cap = a < release.dof.shape[0] ? release.dof.shape[1] : 0;
        unsigned long long candidates = 0ull;
        #pragma unroll
        for (int k = 0; k < $HOLD; ++k) {
            const int x = lane + k * WIDTH;
            candidates |= (unsigned long long)BALLOT(x < n && released[x] > 0.0f) << (k * WIDTH);
        }
        const int total = COUNT(candidates);
        const int slips = min(total, min(cap, $SLIPS));
        if (total > slips && lane == 0) FRAME_STATUS(1);
        int i = -1;
        if (lane < slips) {
            unsigned long long rest = candidates;
            for (int q = 0; q < lane; ++q) rest &= rest - 1ull;
            i = FIRST(rest);
        }
        const bool mine = i >= 0;
        // The slip's own values, loaded once: its released and kept stiffness and its pivot without K.
        float alpha = 0.0f, keep = 0.0f, pivot0 = 0.0f;
        if (mine) {
            alpha = -released[i];
            keep = kept[i];
            pivot0 = WALK_LOAD_PIVOT(data + $coordinate * i);
        }
        float p[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, z = 0.0f, pair[$SLIPS];
        #pragma unroll
        for (int k = 0; k < $SLIPS; ++k) pair[k] = 0.0f;
        // next: the slip's next coordinate, leaf first along its path (-1 once past its root); t: its steps so far.
        int next = i, t = 0, root = i;
        const size_t at = ((size_t)a * cap + (mine ? lane : 0)) * depth;
        const int deepest = slips > 0 ? READ(i, slips - 1) : -1;
        #pragma unroll 1
        for (int b = nb - 1; b >= 0 && slips > 0; --b) {
            const int f0 = HOLD_PICK(h.first, b), f1 = HOLD_PICK(h.first, b + 1);
            if (f0 > deepest) continue;
            const bool in = next >= f0 && next < f1;
            if (BALLOT(in) == 0u) continue;
            const int start = in ? next : -1;
            #pragma unroll 1
            for (int c = f1 - 1; c >= f0; --c) {
                const bool on = c <= start;
                const unsigned slipping = BALLOT(on);
                if (slipping == 0u) continue;
                float d[13];
                HOLD_RECORD(h, c, d);
                if (on) {
                    z = ((c == i ? 1.0f : 0.0f) + ((d[0] * p[0] + d[1] * p[1] + d[2] * p[2]) + (d[3] * p[3] + d[4] * p[4] + d[5] * p[5])))
                        * d[12];
                    for (int q = 0; q < 6; ++q) p[q] -= d[6 + q] * z;
                    root = c;
                }
                // The coordinate's pivot and its square root (1/ell: the record's d[12] is 1/sqrt(P)), each slip's
                // step taking one rsqrt of its new pivot for the rest.
                float pivot = 1.0f / (d[12] * d[12]), root = 1.0f / d[12];
                #pragma unroll
                for (int k = 0; k < $SLIPS; ++k) {
                    if (k >= slips) break;
                    if (!((slipping >> k) & 1u)) continue;
                    float bk = 0.0f, ak = 0.0f, gk = 0.0f, dk = 0.0f, rk = 0.0f;
                    if (lane == k) {
                        const float dd = pivot, scale = root;
                        float dbar = dd + alpha * (z * scale) * (z * scale);
                        if (c == i) dbar = pivot0 + keep;
                        if (!(dbar > 0.0f) || !wp::isfinite(dbar)) FRAME_STATUS(2);
                        const float r = rsqrtf(dbar), inverse = r * r;
                        const float beta = alpha * z * scale * inverse;
                        alpha = alpha * dd * inverse;
                        bk = z;
                        ak = beta * scale;
                        gk = scale * r;
                        dk = dbar;
                        rk = dbar * r;
                        if (t < depth) {
                            release.coordinate.data[at + t] = c;
                            release.b.data[at + t] = bk;
                            release.a.data[at + t] = ak;
                            release.g.data[at + t] = gk;
                        }
                        ++t;
                    }
                    bk = READ(bk, k);
                    ak = READ(ak, k);
                    gk = READ(gk, k);
                    pivot = READ(dk, k);
                    root = READ(rk, k);
                    if (on && lane > k) {
                        const float x = z - bk * pair[k];
                        pair[k] += ak * x;
                        z = gk * x;
                    }
                }
            }
            const int up = HOLD_PICK(h.parent, b);
            const int last = HOLD_PICK(h.first, up + 1) - 1;
            float l[3];
            HOLD_LINK(h, b, l);
            if (in) {
                if (up >= 0) {
                    const float fx = p[0], fy = p[1], fz = p[2];
                    p[3] -= l[1] * fz - l[2] * fy;
                    p[4] -= l[2] * fx - l[0] * fz;
                    p[5] -= l[0] * fy - l[1] * fx;
                }
                next = up >= 0 ? last : -1;
            }
        }
        if (mine) {
            if (t > depth) FRAME_STATUS(1);
            release.dof.data[a * cap + lane] = i;
            release.root.data[a * cap + lane] = root;
            for (int tt = t; tt < depth; ++tt) release.coordinate.data[at + tt] = -1;
        }
        if (lane == 0) release.count.data[a] = slips;
        WARP_SYNC();
        return slips;
    };
"""


def _held_words(layout: TreeLayout) -> int:
    """Words a lane holds of a tree's walk in the border's held form (:data:`_HELD`): its coordinates and bodies (and
    one past the last body) over the segment's lanes, at most two; 0 for a tree past that (the serial text)."""
    width = min(layout.lanes, 32)
    words = -(-max(layout.dofs, layout.bodies + 1) // width)
    return words if words <= 2 else 0


def _walk_roots(layout: TreeLayout) -> list[tuple[int, int, int]]:
    """``(root, coordinate offset, family)`` of each root the predictor solves: one tree, or each root of a forest (a
    prescribed root, family 0, responds with zero)."""
    if layout.forest:
        return list(zip(range(len(layout.roots)), layout.root_offsets, layout.root_families, strict=True))
    return [(0, 0, layout.dofs)]


PREDICTOR_AHEAD = 0
"""Items whose inputs the predictor's walks hold in registers ahead of the chain (:data:`.tree_factor.WALK_AHEAD`):
none, as for :data:`PREDICTOR_UNROLL`: the predictor's launch bound holds 56 registers on g1's block, and the walk's
touches (prefetch hints, no register) still ask L1 for its lines ahead."""

PREDICTOR_UNROLL = 0
"""Coordinates up to which the predictor's walks are straight-line code (:data:`.tree_factor.WALK_UNROLL`): none. The
predictor's launch bound holds 56 registers on g1's block (:func:`_predictor_bounds`), and a straight walk's hoisted
loads spill there (g1 3,808 B of stack, offline sm_120); the loop form keeps a chain's state in registers, its loads
on the path the group's lanes fetched into L1 (WALK_MAPPING §4: B3 against B4)."""


_UP_STEP = """{
    const float z = WALK_MUL(
        $X.u + (($X.s.x * $V[0] + $X.s.y * $V[1] + $X.s.z * $V[2]) + ($X.s.w * $V[3] + $X.t.x * $V[4] + $X.t.y * $V[5])),
        $X.e);
    u[$c] = z;
    $V[0] -= $X.t.z * z;
    $V[1] -= $X.t.w * z;
    $V[2] -= $X.w.x * z;
    $V[3] -= $X.w.y * z;
    $V[4] -= $X.w.z * z;
    $V[5] -= $X.w.w * z;
}"""
"""A coordinate of the walk up (leaf first, ``u = L^-1 P x``): ``z_c = (x_c + s_c . p) / ell_c``, ``p -= ut_c z_c``, ``p``
its body's state, ``z_c`` rounded on its own (``WALK_MUL``); its record and ``x_c`` fetched ahead of the chain
(``$X``, :meth:`.tree_factor.WalkClass.walk`)."""

_UP_JOINT = """{
    $P[0] += $V[0];
    $P[1] += $V[1];
    $P[2] += $V[2];
    $P[3] += $V[3] - ($L.y * $V[2] - $L.z * $V[1]);
    $P[4] += $V[4] - ($L.z * $V[0] - $L.x * $V[2]);
    $P[5] += $V[5] - ($L.x * $V[1] - $L.y * $V[0]);
}"""
"""A joint of the walk up: the body's state ``(f, t)`` passes to its parent's frame, ``t - d x f`` (``$L`` the body's
link, fetched ahead), and adds into it."""


def _walked(layout: TreeLayout, classes: tuple) -> list[tuple[int, int, int]]:
    """``(root, coordinate offset, class index)`` of each root the walk solves, its index into the walked classes
    (:func:`.tree_factor.walk_down_prelude`'s ``cls``)."""
    walked, k = [], 0
    for (r, offset, _), cls in zip(_walk_roots(layout), classes, strict=True):
        if cls is not None:
            walked.append((r, offset, k))
            k += 1
    return walked


def _walk_head(layout: TreeLayout, r: int) -> str:
    """Native declarations of root ``r``'s tree in the walk: its articulation, first joint and coordinate, records and
    links (read-only)."""
    from .tree_factor import WALK_COORDINATE, WALK_LINK  # noqa: PLC0415

    return (
        f"const int a={_root_articulation(layout, r)};"
        "const int j0=walk.joint_start.data[a],d0=walk.dof_start.data[a];"
        f"const float* __restrict__ data=walk.data.data+(size_t)d0*{WALK_COORDINATE};"
        f"const float* __restrict__ link=walk.link.data+(size_t)j0*{WALK_LINK};"
    )


def walk_fetch(layout: TreeLayout, classes: tuple, guard: str = "") -> str:
    """Native statements of every lane of the tree's group asking L1 for the lines of each walked root's records and
    links, one parallel trip, before lane 0 walks (WALK_MAPPING §4, B5); a hint, so no barrier. ``guard`` formats with
    a root's ``bit`` (the decided step's changed roots)."""
    from .tree_factor import WALK_COORDINATE, WALK_LINK  # noqa: PLC0415

    text = "".join(
        f"    {guard.format(bit=2 << r)}{{{_walk_head(layout, r)}"
        f"WALK_FETCH(data,walk.dofs.data[a]*{4 * WALK_COORDINATE},lane,stride);"
        f"WALK_FETCH(link,(walk.joint_end.data[a]-j0)*{4 * WALK_LINK},lane,stride);}}\n"
        for r, _, _ in _walked(layout, classes)
    )
    return f"#if defined(__CUDA_ARCH__)\n{text}#endif\n"


def walk_up(cls, unroll: int = PREDICTOR_UNROLL, ahead: int | None = PREDICTOR_AHEAD) -> str:
    """Native statements of the walk up of class ``cls`` (the emitter's leaf-first walk, straight-line up to
    ``unroll`` coordinates, its records, links and ``x`` loaded ``ahead`` items ahead of the chain and asked of L1
    before that, as the walk down's): ``u = L^-1 P x``, with ``x``, ``u``, ``data`` and ``link`` in scope."""
    from string import Template  # noqa: PLC0415

    from .tree_factor import WALK_COORDINATE, WALK_LINK  # noqa: PLC0415

    step = Template(_UP_STEP).safe_substitute(coordinate=WALK_COORDINATE)
    joint = Template(_UP_JOINT).safe_substitute(link=WALK_LINK)
    inputs = {
        "fetch_joint": "WALK_LINK_IN(link, $b, $up)",
        "fetch_step": f"WALK_IN(data + {WALK_COORDINATE} * ($c), x[$c], 0.0f, $half)",
        "touch_joint": f"WALK_TOUCH(link + {WALK_LINK} * ($b));",
        "touch_step": f"WALK_TOUCH(data + {WALK_COORDINATE} * ($c));",
    }
    return (
        "{\n#if defined(__CUDA_ARCH__)\n"
        + cls.walk(joint, step, root_first=False, cuda=True, unroll=unroll, ahead=ahead, **inputs)
        + "\n#else\n"
        + cls.walk(joint, step, root_first=False, cuda=False, unroll=unroll, ahead=ahead, **inputs)
        + "\n#endif\n}"
    )


def walk_tree(layout: TreeLayout, classes: tuple, *, up: bool, down: bool, guard: str = "") -> str:
    """Native statements of lane 0 walking each walked root's tree (Z4.2): the walk up of the force words
    ``s[$SIX_BODIES + i]`` into the response words ``s[$WHITE + i]`` (:func:`walk_up`), then the walk down in place
    (``WALK_DOWN``, the one walk down of :func:`.tree_factor.walk_down_prelude`): ``P^T L^-T L^-1 P f`` on the all-``K``
    walk, its states in registers, no word of the block read but the force and response words. A prescribed root
    responds with zero. ``guard`` formats with a root's ``bit``."""
    blocks = []
    walked = {r: k for r, _, k in _walked(layout, classes)}
    for r, offset, _family in _walk_roots(layout):
        if r not in walked:
            if up and down:
                blocks.append(f"for(int k=0;k<{layout.root_coordinates[r]};++k)s[$WHITE+{offset}+k]=0.0f;")
            continue
        body = [_walk_head(layout, r), f"const float* x=s+$SIX_BODIES+{offset};", f"float* u=s+$WHITE+{offset};"]
        if up:
            body.append(walk_up(classes[r]))
        if down:
            body.append(
                f"WALK_DOWN({walked[r]},data,link,[&](const int c){{return u[c];}},[](const int){{return 0.0f;}},"
                "[&](const int c,const float q,const float){u[c]=q;},[](const int,const float*){});"
            )
        blocks.append(guard.format(bit=2 << r) + "{\n" + "\n".join(body) + "\n}")
    text = "if(lane==0) {\n" + "\n".join(blocks) + "\n}\n#if defined(__CUDA_ARCH__)\nSYNC();\n#endif\n"
    return substitute(text, layout.sizes())


PREDICTOR_THREADS = 1152
"""Threads an SM keeps of a predictor: main's g1 predictor before the saturation law, 18 blocks of 64 at 56 registers
(H8). The predictor's launch bound (:func:`_predictor_bounds`) holds every layout there or at the block limit."""

BLOCK_LIMIT = 24
"""Resident blocks an sm_120 SM keeps at most; ptxas drops a launch bound that asks for more."""


def _predictor_bounds(layout: TreeLayout) -> tuple[int, int]:
    """The predictor's launch bound: its block and the blocks :data:`PREDICTOR_THREADS` holds, at most
    :data:`BLOCK_LIMIT`. The trial and the decision fit those registers (g1's 52); the border, which about one tree in
    eighteen runs, keeps its working values in its stack rather than taking a resident block from every world (W7
    Q5.5: the rare branch stays out of every block)."""
    return layout.block, min(BLOCK_LIMIT, PREDICTOR_THREADS // layout.block)


def border_solve(layout: TreeLayout, classes: tuple) -> str:
    """Native statements of the decided step (FL3.9, W9). The block's lanes build each changed root's release (into
    :class:`.tree_factor.WalkRelease`, from the marks :data:`SATURATE_DRIVES` left): on CUDA, for a tree whose walk a
    lane holds in two words (:data:`_HELD`), the tree's segment holds the walk in registers and builds the release
    coordinate-major, a lane a slip, every value a register (``HELD_BUILD``); otherwise lane 0 lists the slips and their
    groups (``BORDER_SLIPS``), a lane a slip walks the columns (``BORDER_COLUMN``) and a lane a group runs its slips'
    recurrences (``BORDER_GROUP``). Then each changed root takes its decided step from the force words into the response
    words: the walk up (held: the segment in lockstep; else lane 0's, :func:`walk_tree`), ``R^T R`` by the release's
    one primitive, a lane a group (``BORDER_APPLY``: ``WALK_SLIP``, ``WALK_SLIP_T``), the walk down. A root whose step
    its trial already gives (bit ``1 + r`` of ``moved`` clear) keeps its trial's response. The releases are built for
    every root first: the marks of one root lie in the states of another's walk. Nothing is added to the block (H8: g1's
    predictor sits at the RTX PRO 6000's 64 KB carveout line): the serial build's work, the pivots and a dense column,
    lies in the dead body-wrench words, else in the walk's scratch; the held build's is registers."""
    from .tree_factor import WALK_COORDINATE, WALK_LINK  # noqa: PLC0415

    roots = [(r, offset) for r, offset, _ in _walked(layout, classes)]
    if not roots:
        return ""
    head = (
        "const int a={art};const int j0=walk.joint_start.data[a],d0=walk.dof_start.data[a],n=walk.dofs.data[a];"
        "WALK_TREE(reinterpret_cast<const int*>(walk.path.data+j0),walk.joint_end.data[a]-j0,n);"
        "const float* const data=walk.data.data+(size_t)d0*$WALK_COORDINATE;"
        "const float* const link=walk.link.data+(size_t)j0*$WALK_LINK;"
    )
    sync = "\n#if defined(__CUDA_ARCH__)\n            SYNC();\n#endif\n"

    def each(form):
        return "".join(
            form.format(bit=2 << r, r=r, offset=offset, head=head.format(art=_root_articulation(layout, r)))
            for r, offset in roots
        )

    # A tree the walk does not hold keeps its trial and flags bit 2. Its work needs 2n floats; a tree that has no room
    # for them keeps its K and flags bit 0.
    frame = (
        "        if(moved&{bit}) {{{head}\n"
        "        if(d0+n>walk.data.shape[0]) {{if(lane==0) {{FRAME_STATUS(4);moved&=~{bit};}}}}\n"
        "        else {{\n"
        "            const int room=$SIX_BODIES-$DOFS;\n"
        "            float* const work=2*n<=room?s+$DOFS:reinterpret_cast<float*>(walk.scratch.data+j0);\n"
        "            if(lane==0)BORDER_SLIPS(a,d0,n,s+$WHITE+{offset},2*n<=room||2*n<=6*(walk.joint_end.data[a]-j0));"
        + sync
        + "            slips{r}=walk.release.count.data[a];\n"
        "            if(slips{r}>0) {{\n"
        "            for(int c=lane;c<n;c+=stride) {{"
        "const float inverse=WALK_LOAD_INVERSE(data+$WALK_COORDINATE*c);work[c]=1.0f/(inverse*inverse);work[n+c]=0.0f;}}\n"
        "            for(int k=lane;k<slips{r};k+=stride)BORDER_COLUMN(a,d0,data,link,k);"
        + sync
        + "            for(int k=lane;k<slips{r};k+=stride)"
        "BORDER_GROUP(a,d0,data,s+$WHITE+{offset},s+{offset},work,work+n,k,slips{r});" + sync + "        }}}}}}\n"
    )
    apply = (
        "        if(moved&{bit}) {{const int a={art};"
        "for(int k=lane;k<slips{r};k+=stride)BORDER_APPLY(a,s+$WHITE+{offset},k,slips{r});}}\n"
    )
    serial_apply = "".join(
        apply.format(bit=2 << r, r=r, offset=offset, art=_root_articulation(layout, r)) for r, offset in roots
    )
    declarations = "".join(f"int slips{r}=0;" for r, _ in roots)
    guard = "if(moved&{bit})"
    up = walk_fetch(layout, classes, guard) + walk_tree(layout, classes, up=True, down=False, guard=guard)
    down = walk_tree(layout, classes, up=False, down=True, guard=guard)
    sync_all = "#if defined(__CUDA_ARCH__)\n        SYNC();\n#endif\n"
    serial = f"        {declarations}\n{each(frame)}{sync_all}{up}{serial_apply}{sync_all}{down}"
    words = _held_words(layout)
    if not words:
        text = "    {\n" + serial + "    }\n"
        return substitute(text, {**layout.sizes(), "WALK_COORDINATE": WALK_COORDINATE, "WALK_LINK": WALK_LINK})
    nb = "const int nb=walk.joint_end.data[a]-j0;"
    hold = "WalkHold h;HOLD_LOAD(h,data,link,reinterpret_cast<const int*>(walk.path.data+j0),n,nb);"
    held_frame = (
        "        if(moved&{bit}) {{{head}\n"
        "        if(d0+n>walk.data.shape[0]) {{if(lane==0) {{FRAME_STATUS(4);moved&=~{bit};}}}}\n"
        "        else if(lane<WIDTH) {{" + nb + hold + "\n"
        "            slips{r}=HELD_BUILD(h,a,n,nb,data,s+$WHITE+{offset},s+{offset});\n"
        "        }}}}\n"
    )
    held_step = (
        "        if((moved&{bit})&&lane<WIDTH) {{{head}" + nb + hold + "\n"
        "            HELD_UP(h,n,nb,s+$SIX_BODIES+{offset},s+$WHITE+{offset},s);\n"
        # R^T R between the walks by the release's one primitive (WALK_SLIP, WALK_SLIP_T), a lane a group.
        "            for(int k=lane;k<slips{r};k+=WIDTH)BORDER_APPLY(a,s+$WHITE+{offset},k,slips{r});\n"
        "            WARP_SYNC();\n"
        "            HELD_DOWN(h,n,nb,s+$WHITE+{offset},s+$WHITE+{offset},s);\n"
        "        }}\n"
    )
    held = f"        {declarations}\n{each(held_frame)}        SYNC();\n{each(held_step)}        SYNC();\n"
    text = f"    {{\n#if defined(__CUDA_ARCH__)\n{held}#else\n{serial}#endif\n    }}\n"
    return substitute(text, {**layout.sizes(), "WALK_COORDINATE": WALK_COORDINATE, "WALK_LINK": WALK_LINK})


def held_lambdas(layout: TreeLayout, slips: int) -> str:
    """The held walk's native lambdas (:data:`_HELD`) for the layout's words a lane, CUDA only, declared once ahead of the
    trial; nothing for a tree past two words a lane (:func:`_held_words`). ``slips`` is the class's slip capacity
    (:meth:`.tree_factor.TreeFactor.slip_capacity`): the build's lanes, one a slip, at most the segment's width."""
    from .tree_factor import WALK_COORDINATE, WALK_UNPACKED  # noqa: PLC0415

    words = _held_words(layout)
    if not words:
        return ""
    bound = max(1, min(slips, layout.lanes, 32))
    lambdas = substitute(_HELD, HOLD=words, SLIPS=bound, coordinate=WALK_COORDINATE, unpacked=WALK_UNPACKED)
    return f"#if defined(__CUDA_ARCH__)\n{lambdas}\n#endif\n"


def clear_slips(layout: TreeLayout) -> str:
    """Native statements of lane 0 clearing each root's slip count (``walk.release.count``): only a changed root's border
    sets it."""
    return "".join(f"walk.release.count.data[{_root_articulation(layout, r)}]=0;" for r, _, _ in _walk_roots(layout))


def _predictor_operations(layout: TreeLayout, schedules: HeldSchedules) -> dict[str, str]:
    """The fragments of the predictor's force half that the external wrenches and the root transport take."""
    reduction = "    if(nonzero) {\n" + _reduction(_EXTERNAL, schedules.external, _EXTERNAL_SLOTS) + "    }\n"
    return {
        "PHYSICAL_TABLE": layout.physical_table(),
        "KINEMATIC_FLAG": str(int(BodyFlags.KINEMATIC)),
        "SHIFT": cached_force.shift_wrench(
            "input", "cache.com_offset.data[group*$BODIES+local]", "s[6*local+k]", "s[6*local+3+k]"
        ),
        "REDUCE": reduction,
    }


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


def _predictor_name(prefix: str, layout: TreeLayout, schedules: HeldSchedules | None, response: str, classes) -> str:
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


def _predictor_params(response: str):
    from .tree_factor import WalkTrees  # noqa: PLC0415

    data_type = sparse_factor.SparseData if response == "sparse" else HeldResponse
    return (
        ("p", sparse_factor.SparsePlan),
        ("d", data_type),
        ("plan", HeldPlan),
        ("cache", KineticData),
        ("f", ForceInput),
        ("walk", WalkTrees),
        ("trial", TrialWords),
    )


def _tiled(layout: TreeLayout, name: str, params, source: str):
    _, _, _sub, _ = _thread_helpers(layout)
    return KernelTemplate(
        name,
        params,
        source,
        # A packed group of trees per block: the block's group and this thread's tree in it.
        Dispatch(
            "block, _ = wp.tid()\ngroup = block * wp.static(pack) + _sub()\n$CALL",
            ("group",),
            (("group", int),),
            constants=(("pack", layout.pack), ("_sub", _sub)),
        ),
        composed=True,
        launch_bounds=_predictor_bounds(layout),
    ).kernel()


@functools.cache
def get_force_kernel(schedules: HeldSchedules, *, layout: TreeLayout = REFERENCE_LAYOUT, response="sparse", classes):
    """The predictor's force half (Z4.2, :data:`_FORCE_HALF`): the force on every coordinate into ``trial.force``; it
    clears each root's slip count for the response half's border."""
    if response not in ("sparse", "cholesky", "diagonal"):
        raise ValueError("Unknown held tree response representation")
    force = {
        name: text
        for name, text in _solve_operations(layout, classes).items()
        if name in ("PROJECT", "SPRING_GUARD", "PASSIVE")
    }
    source = layout.native(
        _FORCE_HALF,
        preludes=(*SPATIAL.preludes(),),
        SCRATCH=_placed_block(layout, "s", "$PRED_SCRATCH"),
        CLEAR_SLIPS=clear_slips(layout),
        FORCE_STAGE=substitute(_FORCE_STAGE, layout.sizes(), **force),
        CLOCK0="",
        CLOCK1="",
        **_predictor_operations(layout, schedules),
    )
    name = _predictor_name("tree_kinetic_force", layout, schedules, response, classes)
    return _tiled(layout, name, _predictor_params(response), source)


_WALK_SOLVE = r"""    const int group = tid / $ROOT_COUNT, r = tid - group * $ROOT_COUNT;
    if (group >= groups) return;
    const int art = p.group_to_art.data[group];
    switch (r) {
$ROOTS
    }
"""


@functools.cache
def get_walk_solve_kernel(*, layout: TreeLayout = REFERENCE_LAYOUT, classes):
    """The walk's solve of the trial (Z4.2, WALK_MAPPING E5): a thread a tree root, the class's walk compiled into
    straight-line code (:meth:`.tree_factor.WalkClass.walk` up to :data:`.tree_factor.WALK_UNROLL` coordinates), the
    walk up of ``trial.force`` into registers and the walk down into ``trial.response``, ``P^T L^-T L^-1 P f`` on the
    all-``K`` walk. A prescribed root responds with zero."""
    from .kernel_template import THREADS  # noqa: PLC0415
    from .tree_factor import WALK_COORDINATE, WALK_LINK, WALK_UNROLL, WalkTrees, walk_down_prelude  # noqa: PLC0415

    walked = {r: k for r, _, k in _walked(layout, classes)}
    cases = []
    for r, offset, _family in _walk_roots(layout):
        if r not in walked:
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
                + walk_up(cls, WALK_UNROLL, None)
                + f"\nWALK_DOWN({walked[r]}, data, link, [&](const int c) {{ return u[c]; }},"
                " [](const int) -> float { return 0.0f; }, [&](const int c, const float q, const float) { out[c] = q; },"
                " [](const int, const float*) {});"
            )
        cases.append(f"    case {r}: {{\n{body}\n        break;\n    }}")
    walked_classes = tuple(cls for cls in classes if cls is not None)
    source = substitute(_WALK_SOLVE, ROOT_COUNT=len(_walk_roots(layout)), ROOTS="\n".join(cases))
    name = _predictor_name("tree_kinetic_walk", layout, None, "sparse", classes)
    return KernelTemplate(
        name,
        (
            ("groups", int),
            ("p", sparse_factor.SparsePlan),
            ("plan", HeldPlan),
            ("walk", WalkTrees),
            ("trial", TrialWords),
        ),
        source,
        THREADS,
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
):
    """The predictor's response half (Z4.2, :data:`_RESPONSE_HALF`), after the force half (:func:`get_force_kernel`) and
    the walk's solve (:func:`get_walk_solve_kernel`): the trial's response read back into the block, ``qdd`` and
    ``v_hat``, the drives' saturation decision, and a world whose drives moved takes its decided step by the border on
    the walk (:func:`border_solve`), from the trial's force plus the decision's change of it.

    ``schedules`` are the scans' schedules; ``clocked`` adds the predictor stage counters (see :func:`get_helpers`);
    ``twists`` holds the twist publication (:func:`publishes_twists`), which only CUDA builds; ``classes`` holds each
    root's walking class (:class:`.tree_factor.WalkClass`, ``None`` for a prescribed root); ``slips`` the classes'
    slip capacity, the border's build lanes (:func:`held_lambdas`). ``response`` names only the factor data still bound
    as ``d``.
    """
    if response not in ("sparse", "cholesky", "diagonal"):
        raise ValueError("Unknown held tree response representation")
    if len(classes) != len(_walk_roots(layout)):
        raise ValueError("The predictor needs a walking class (or None) for each root")
    clocks = _native_clocks(clocked, "f", stages=5, column=8)
    trial = _solve_operations(layout, classes)
    response_ops = {
        "FORWARD_STAGE": "",
        "TRANSPOSE_ACTION": "float value=s[$WHITE+col];",
        "RESPONSE_PHYSICAL": "col",
        "STATUS": trial["STATUS"],
        "VELOCITY": trial["VELOCITY"],
    }
    # The decided step's response: the border's words, on the roots it changed.
    changed = f"if(!(moved&(2<<{_root(layout).replace('[i]', '[col]')})))continue;"
    second = {**response_ops, "TRANSPOSE_ACTION": changed + "float value=s[$WHITE+col];"}
    from .tree_factor import walk_down_prelude  # noqa: PLC0415

    walked = tuple(cls for cls in classes if cls is not None)
    source = layout.native(
        _RESPONSE_HALF,
        preludes=(
            *SPATIAL.preludes(),
            walk_tree_prelude(),
            *((walk_down_prelude(walked, PREDICTOR_UNROLL, PREDICTOR_AHEAD),) if walked else ()),
        ),
        PHYSICAL_TABLE=layout.physical_table(),
        SCRATCH=_placed_block(layout, "s", "$PRED_SCRATCH"),
        TRIAL=held_lambdas(layout, slips)
        + substitute(
            _RESPONSE_STAGE,
            layout.sizes(),
            **response_ops,
            START_FORWARD=clocks["CLOCK2"],
            START_RESPONSE=clocks["CLOCK3"],
            DECIDE=QUIET_DRIVE,
        ),
        SATURATE=substitute(SATURATE_DRIVES, layout.sizes(), ROOT=_root(layout)),
        BORDER=border_solve(layout, classes),
        SECOND_RESPONSE=substitute(
            _RESPONSE_STAGE, layout.sizes(), **second, START_FORWARD="", START_RESPONSE="", DECIDE=""
        ),
        TRANSPORT_BLOCK=_transport_block(layout),
        TWIST_BLOCK=_twist_block() if twists else "",
        **{name: text for name, text in clocks.items() if name not in ("CLOCK2", "CLOCK3")},
    )
    name = _predictor_name("tree_kinetic_predict", layout, schedules, response, classes)
    return _tiled(layout, name, _predictor_params(response), source)
