# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers

# SPDX-License-Identifier: Apache-2.0

"""Generated current/next tree moments, bias and fused state publication."""

import dataclasses
import functools
from types import SimpleNamespace

import warp as wp

from ...sim import BodyFlags
from . import cached_force, field_group, kernels, response_operator, sparse_factor
from . import tree_scan as ts
from .device_preludes import CUDA_ONLY, device_branches
from .field_group import Field
from .kernel_template import Dispatch, KernelTemplate, python_kernel
from .publication_data import PublicationData
from .spatial_laws import (
    INERTIA_CHANNELS,
    SPATIAL,
    body_wrench,
    inertia_at,
    inertia_record,
    joint_frame_pose,
    joint_motion,
    joint_transform,
    pack_inertia,
    point_velocity,
)
from .tree_kinetic_layout import REFERENCE_LAYOUT, TreeLayout
from .tree_passes import body_dynamics
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
)
"""A held group's template tree as its kernels read it: the scan plan (:data:`.tree_scan.SCAN_PLAN`) and the
group's coordinate maps."""

HeldPlan = field_group.struct("HeldPlan", HELD_PLAN, __name__)


KINETIC_CACHE = field_group.FieldGroup(
    Field("bias", wp.float32, ("group", "dof"), fill=None, grad=False, owner="cache"),
    Field("com_offset", wp.vec3, ("group", "body"), fill=None, grad=False, owner="cache"),
    Field("geometric", wp.float32, ("group", "entry"), fill=None, grad=False, owner="cache"),
    Field("current_valid", wp.int32, ("articulation",), owner="solver", attr="_fk_id_cache_valid"),
    Field("geometry_valid", wp.int32, ("group",), grad=False, owner="cache"),
    Field("generation", wp.int64, ("group",), grad=False, owner="cache"),
    Field("geometry_generation", wp.int64, ("group",), fill=-1, grad=False, owner="cache"),
    Field("status", wp.int32, ("group",), grad=False, owner="cache"),
    Field("source_q", wp.uint64, ("group",), grad=False, owner="cache"),
    Field("source_qd", wp.uint64, ("group",), grad=False, owner="cache"),
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


FORCE_INPUT = (
    Field("joint_q", wp.float32, ("coord",), owner="state_in"),
    Field("joint_qd", wp.float32, ("dof",), owner="state_in"),
    Field("predictor_qd", wp.float32, ("dof",), owner="step"),
    Field("joint_f", wp.float32, ("dof",), owner="control"),
    Field("body_f", wp.spatial_vector, ("body",), owner="state_in"),
    Field("body_flags", wp.int32, ("body",), owner="model"),
    Field("stiffness", wp.float32, ("dof",), attr="_passive_spring_stiffness"),
    Field("reference", wp.float32, ("dof",), attr="_passive_spring_ref"),
    Field("damping", wp.float32, ("dof",), attr="_passive_joint_damping"),
    Field("u0", wp.float32, ("dof",), owner="state_aug", attr="joint_tau"),
    Field("S", wp.spatial_vector, ("dof",), owner="state_aug", attr="joint_S_s"),
    Field("kinematic_joint", wp.int32, ("joint",), attr="_kinematic_joint_mask"),
    Field("qdd", wp.float32, ("dof",), owner="state_aug", attr="joint_qdd"),
    Field("vhat", wp.float32, ("dof",), attr="v_hat"),
    Field("dt", wp.float32, owner="step"),
    Field("stage_clocks", wp.int64, ("group", "stage"), owner="step"),
    Field("predicted_twist", wp.spatial_vector, ("body",), owner="step"),
    Field("publish_twists", wp.int32, owner="step"),
)
"""The held-tree predictor's inputs: the step's state and control, the solver's passive and kinematic tables, the
augmented state it writes, and the step's values."""

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


# The default form's release (see _STORAGE_SOURCE): its free never links on
# the CPU and is kept verbatim for the CUDA tree kernels' content hashes.
@wp.func_native(r"""
#if !defined(__CUDA_ARCH__)
    free(reinterpret_cast<void*>(address));
#endif
""")
def _release(address: wp.uint64): ...


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


def held_schedules(tree: TreePlan, lanes: int, *, chains: bool, reference: bool = False) -> HeldSchedules:
    """The schedules of a held group's scans at ``lanes`` lanes per tree, from its tree's data
    (:func:`.tree_scan.choose_schedule`).

    ``chains``: the launch may give each tree its own block, which the heavy-chain scans address. ``reference``
    selects the shared-memory jumps and depth levels the A/B tests compare against.
    """
    if reference:
        return HeldSchedules(ts.Schedule.JUMP_SMEM, ts.Schedule.JUMP_SMEM, ts.Schedule.LEVEL, ts.Schedule.LEVEL)
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

_PREDICT = r"""    $PHYSICAL_TABLE

    const int art=p.group_to_art.data[group];
    const int ds=p.art_dof_start.data[art],js=p.art_joint_start.data[art];
#if defined(__CUDA_ARCH__)
    const int lane=LANE,stride=STRIDE;
    __shared__ float s_all[$PRED_SCRATCH*$PACK];__shared__ int nonzero_all[$PACK];
    float* s=s_all+SUB*$PRED_SCRATCH;int& nonzero=nonzero_all[SUB];
#else
    const int lane=0,stride=1;
    float s[$PRED_SCRATCH];int nonzero=0;
#endif
    if(lane==0)nonzero=0;
$CLOCK0#if defined(__CUDA_ARCH__)
    SYNC();
#endif
    for(int local=lane;local<$BODIES;local+=stride) {
        const int body=p.joint_child.data[js+local];
        // A kinematic body takes no external wrench, as in ordinary dynamics
        // (tree_passes.net_wrench): its wrench must not reach its ancestors.
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
    for(int i=lane;i<$DOFS;i+=stride) {
        const int dof=ds+i,body=p.dof_joint.data[i];
        float value=-cache.bias.data[group*$DOFS+i]+f.joint_f.data[dof]+f.u0.data[dof];
        if(nonzero){$PROJECT}
        if($SPRING_GUARD)value+=$PASSIVE;
        s[$SIX_BODIES+i]=value;
    }
#if defined(__CUDA_ARCH__)
    SYNC();
#endif
$CLOCK2    $FORWARD_STAGE
$CLOCK3    for(int col=lane;col<$DOFS;col+=stride) {
        $TRANSPOSE_ACTION
        const int dof=ds+$RESPONSE_PHYSICAL;
        const bool valid=cache.current_valid.data[art]&&d.valid.data[group]&&!d.status.data[group]&&!cache.status.data[group]&&wp::isfinite(value);
        f.qdd.data[dof]=valid?value:NAN;
        f.vhat.data[dof]=$VELOCITY;
        if(!valid) {
#if defined(__CUDA_ARCH__)
            atomicOr(&cache.status.data[group],4);
#else
            cache.status.data[group]|=4;
#endif
        }
    }
#if defined(__CUDA_ARCH__)
    SYNC();
#endif
$CLOCK4    // A prescribed (kinematic) free root keeps its given velocity: no transport.
    $TRANSPORT_BLOCK
$TWIST_BLOCK$CLOCK5"""

# The default form, which CUDA owners launch, keeps one tree's state block in
# shared memory; packed trees take consecutive blocks of one array. Its CPU
# branch never links: Warp's CPU JIT links only a curated C runtime subset,
# without malloc and free. That branch and the free in _release stay verbatim
# because this text is part of every CUDA tree kernel's content hash (its
# kernel-cache key). An owner on the CPU builds the host=True form below.
_STORAGE_SOURCE = r"""
#if defined(__CUDA_ARCH__)
    __shared__ float values[$TOTAL*$PACK];
    return reinterpret_cast<uint64_t>(values+SUB*$TOTAL);
#else
    return reinterpret_cast<uint64_t>(malloc($TOTAL*sizeof(float)));
#endif
"""

# The host=True form keeps the CPU scratch in static storage: a CPU launch runs
# its blocks one after another on the launching thread, and each block is done
# with the scratch before the next. Concurrent CPU launches of one kernel from
# several threads would share it (the JIT has no thread-local storage); only
# direct-construction tests build CPU owners.
_HOST_STORAGE_SOURCE = r"""
#if defined(__CUDA_ARCH__)
    __shared__ float values[$TOTAL*$PACK];
    return reinterpret_cast<uint64_t>(values+SUB*$TOTAL);
#else
    static float values[$TOTAL];return reinterpret_cast<uint64_t>(values);
#endif
"""


def _host_storage(layout):
    """Return the scratch of an owner on the CPU, which the kernels release with ``_keep_static``."""

    @wp.func_native(layout.native(_HOST_STORAGE_SOURCE))
    def _storage() -> wp.uint64: ...

    return _storage


@wp.func_native("")
def _keep_static(address: wp.uint64): ...


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
    host: bool = False,
):
    """Bind one physical tree evaluation to its storage and its scans' schedules.

    ``clocked`` adds the collection (``stage_clocks`` columns 13-15) and body
    term (16-21) counters. Both arguments change the emitted code, so they are
    arguments (cache-key members), never environment reads. ``host`` keeps
    the CPU scratch in static storage (see ``_HOST_STORAGE_SOURCE``); the
    default is the form CUDA owners launch.
    """
    pose_source = _down(layout, ts.TRANSFORM, schedules.pose, _POSE_SLOTS)
    motion_source = _down(layout, _MOTION, schedules.motion, _MOTION_SLOTS)
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

    @wp.func_native(layout.native(_STORAGE_SOURCE))
    def _storage() -> wp.uint64: ...

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
    free_rigid = (layout.bodies == 1 and layout.root_dofs == 6) or (layout.forest and layout.free_roots > 0)

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
        _storage=_host_storage(layout) if host else _storage,
        _store_axis=_store_axis,
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
    host: bool = False,
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
    the stages, 22-24 for motion; see :func:`get_helpers`); ``host`` the
    static CPU scratch of an owner on the CPU.
    """
    if finish and motion_velocity:
        raise ValueError("Alternate motion velocity is a repair-only contract")
    if finish and listed:
        raise ValueError("A listed launch is a repair-only contract")
    if listed and motion_velocity:
        raise ValueError("A listed launch is a contract of the repair of ordinary motion")
    if bank_coordinates and (finish or motion_velocity or listed):
        raise ValueError("A compared input bank is a contract of the one-pass repair of ordinary motion")
    h = get_helpers(layout, schedules, body_velocity, clocked=clocked, host=host)
    release = _keep_static if host else _release
    _lanes, _sync, _sub, _different_motion = _thread_helpers(layout)
    banked = bank_coordinates != 0
    _bank_matches = _bank_binding(bank_coordinates, layout) if banked else None
    scan_poses, scan_motion, collect, body_terms = h._scan_poses, h._scan_motion, h._collect, h._body_terms
    _storage, _store_motion, _load_motion = h._storage, h._store_motion, h._load_motion
    _store_origin, _load_origin, _store_axis = h._store_origin, h._load_origin, h._store_axis
    bodies, dofs, pack, free_root = layout.bodies, layout.dofs, layout.pack, layout.root_dofs == 6
    roots, free_roots, forest = len(layout.roots), layout.free_roots, layout.forest

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
            refresh = int(refresh != 0 or requests[g.art] != 0)
            if wp.static(forest):
                for root in range(1, wp.static(roots)):
                    refresh = int(refresh != 0 or requests[plan.arts[g.group, root]] != 0)
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
                data.angular_damping,
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
        g: TreeGroup, origin: wp.vec3, shift: wp.vec3, plan: HeldPlan, data: PublicationData, qd: wp.array[float]
    ):
        """Store each joint's axes and relative velocity at the origin, then complete the motion scan.

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
            anchor = wp.transform(wp.transform_get_translation(anchor) - origin, wp.transform_get_rotation(anchor))
            if wp.static(clocked):
                clock_m1 = _clock()
            velocity = joint_motion(
                data.joint_type[joint],
                data.joint_axis,
                data.joint_dof_dim[joint, 0],
                data.joint_dof_dim[joint, 1],
                anchor,
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
            _store_motion(g.address, local, velocity)
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
        g.address = _storage()
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
        motion(g, origin, shift, plan, data, qd)
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
        release(g.address)

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
        refresh = int(global_refresh != 0 or requests[art] != 0)
        if wp.static(forest):
            for root in range(1, wp.static(roots)):
                valid = wp.min(valid, cache.current_valid[plan.arts[group, root]])
                refresh = int(refresh != 0 or requests[plan.arts[group, root]] != 0)
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
            refresh = int(global_refresh != 0 or requests[art] != 0)
            if wp.static(forest):
                for root in range(1, wp.static(roots)):
                    valid = wp.min(valid, cache.current_valid[plan.arts[group, root]])
                    refresh = int(refresh != 0 or requests[plan.arts[group, root]] != 0)
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


@functools.cache
def get_predictor_kernel(
    schedules: HeldSchedules,
    *,
    layout: TreeLayout = REFERENCE_LAYOUT,
    response="sparse",
    reference: bool = False,
    clocked: bool = False,
    twists: bool,
):
    """Apply current external/control force and both unchanged W actions.

    ``schedules`` are the scans' schedules (the external wrench sums take ``external``); ``reference`` keeps the
    serial one-lane held solve on CUDA; ``clocked`` adds the predictor stage counters (see :func:`get_helpers`);
    ``twists`` holds the twist publication (:func:`publishes_twists`), which only CUDA builds.
    """
    reduction = "    if(nonzero) {\n" + _reduction(_EXTERNAL, schedules.external, _EXTERNAL_SLOTS) + "    }\n"
    inverse = response_operator.SparseInverseFactor(
        "d.W.data+group*$NNZ",
        response_operator.IndexedSupport(
            "$DOFS", "p.index.data", "p.inverse_count.data", "p.inverse_nodes.data", "$INVERSE_WIDTH"
        ),
    )
    operations = {
        "PHYSICAL_TABLE": layout.physical_table(),
        "KINEMATIC_FLAG": str(int(BodyFlags.KINEMATIC)),
        "SHIFT": cached_force.shift_wrench(
            "input", "cache.com_offset.data[group*$BODIES+local]", "s[6*local+k]", "s[6*local+3+k]"
        ),
        "PROJECT": cached_force.project_wrench("f.S.data[dof]", "s[6*body+k]", "value"),
        # Passive springs act on no root coordinate.
        "SPRING_GUARD": f"i>={layout.root_dofs}" if not layout.forest else "plan.dof_passive.data[i]!=0",
        "PASSIVE": _FORCE.passive("plan.q_index.data[group*$DOFS+i]", "dof"),
        "VELOCITY": _FORCE.velocity("dof", "f.qdd.data[dof]"),
        "TRANSPORT_BLOCK": (
            "if($FREE_ROOT&&lane==0&&f.kinematic_joint.data[js]==0) {\n        "
            + _FORCE.transport("ds", "f.vhat.data[ds+k]")
            + "\n    }"
            if not layout.forest
            else "if(lane<$FREE_ROOTS&&f.kinematic_joint.data[js+plan.free_root_local.data[lane]]==0) {\n"
            "        const int rd=ds+plan.free_root_dof.data[lane];\n        "
            + _FORCE.transport("rd", "f.vhat.data[rd+k]")
            + "\n    }"
        ),
        "REDUCE": reduction,
        "TRANSPOSE_ACTION": inverse.transpose("col", lambda index: f"s[$WHITE+{index}]"),
        "RESPONSE_PHYSICAL": layout.physical("col"),
    }
    forward = inverse.forward("row", lambda index: f"s[$SIX_BODIES+{layout.physical(index)}]")
    operations["FORWARD_STAGE"] = (
        """for(int row=lane;row<$DOFS;row+=stride) {
        """
        + forward
        + """
        s[$WHITE+row]=value;
    }
#if defined(__CUDA_ARCH__)
    SYNC();
#endif
"""
    )
    data_type = sparse_factor.SparseData
    if response == "cholesky":
        # Held factor families applied in physical coordinates. On CUDA every
        # family of at most one warp of coordinates is solved lane-parallel
        # (one lane per row, one shuffle per step); larger families and the
        # CPU twin keep the serial one-lane law. A prescribed root (family 0)
        # responds with zero and a diagonal-mass root (negative family) with
        # its stored inverse. ``reference`` keeps the serial form on CUDA for
        # comparison (the owner reads FEATHER_PGS_TREE_REFERENCE at construction).
        roots = len(layout.roots)
        if not layout.forest:
            families = [(0, layout.dofs, f"d.lower.data+group*{layout.dofs**2}", layout.dofs)]
        else:
            families = []
            for r, (offset, family) in enumerate(zip(layout.root_offsets, layout.root_families, strict=True)):
                lower = "d.lower" if r == 0 else f"d.lower{r}"
                pointer = f"{lower}.data+d.factor_index.data[group*{roots}+{r}]*{family * family}"
                families.append((offset, family, pointer, layout.root_coordinates[r] if family == 0 else abs(family)))
        serial, parallel = [], []
        for offset, family, pointer, width in families:
            if family == 0:
                serial.append(f"if(lane==0){{for(int k=0;k<{width};++k)s[$WHITE+{offset}+k]=0.0f;}}")
                parallel.append(f"if(lane<{width})s[$WHITE+{offset}+lane]=0.0f;")
                continue
            if family < 0:
                serial.append(
                    f"if(lane==0){{for(int k=0;k<{width};++k)"
                    f"s[$WHITE+{offset}+k]=d.inverse.data[ds+{offset}+k]*s[$SIX_BODIES+{offset}+k];}}"
                )
                parallel.append(
                    f"if(lane<{width})s[$WHITE+{offset}+lane]=d.inverse.data[ds+{offset}+lane]*s[$SIX_BODIES+{offset}+lane];"
                )
                continue
            held = response_operator.CholeskySolve(family, pointer)

            def white(index, o=offset):
                return f"s[$WHITE+{o}+{index}]"

            serial.append(
                "if(lane==0){\n"
                + held.whiten(lambda index, o=offset: f"s[$SIX_BODIES+{o}+{index}]", white)
                + held.leave(white, white)
                + "\n}"
            )
            if family <= min(32, layout.lanes) and not reference:
                parallel.append(
                    response_operator.LaneCholeskySolve(family, pointer).solve(
                        f"$SIX_BODIES+{offset}", f"$WHITE+{offset}"
                    )
                )
            else:
                parallel.append(serial[-1])
        operations["FORWARD_STAGE"] = (
            "#if defined(__CUDA_ARCH__)\n"
            + "\n".join(parallel)
            + "\n#else\n"
            + "\n".join(serial)
            + "\n#endif\n#if defined(__CUDA_ARCH__)\nSYNC();\n#endif\n"
        )
        operations["TRANSPOSE_ACTION"] = "float value=s[$WHITE+col];"
        operations["RESPONSE_PHYSICAL"] = "col"
        data_type = HeldResponse
    elif response == "diagonal":
        operations["FORWARD_STAGE"] = ""
        operations["TRANSPOSE_ACTION"] = "float value=d.inverse.data[ds+col]*s[$SIX_BODIES+col];"
        operations["RESPONSE_PHYSICAL"] = "col"
        data_type = HeldResponse
    elif response != "sparse":
        raise ValueError("Unknown held tree response representation")
    operations["TWIST_BLOCK"] = _twist_block() if twists else ""
    source = layout.native(
        _PREDICT, preludes=SPATIAL.preludes(), **operations, **_native_clocks(clocked, "f", stages=5, column=8)
    )

    _, _, _sub, _ = _thread_helpers(layout)
    name = f"tree_kinetic_predict{layout.dofs}"
    if layout.forest:
        name += f"_forest{len(layout.roots)}"
    if layout.pack > 1:
        name += f"_l{layout.lanes}"
    name += schedules.tag
    if response != "sparse":
        name += "_" + response
    return KernelTemplate(
        name,
        (
            ("p", sparse_factor.SparsePlan),
            ("d", data_type),
            ("plan", HeldPlan),
            ("cache", KineticData),
            ("f", ForceInput),
        ),
        source,
        # A packed group of trees per block: the block's group and this thread's tree in it.
        Dispatch(
            "block, _ = wp.tid()\ngroup = block * wp.static(pack) + _sub()\n$CALL",
            ("group",),
            (("group", int),),
            constants=(("pack", layout.pack), ("_sub", _sub)),
        ),
        composed=True,
    ).kernel()
