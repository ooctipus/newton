# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers

# SPDX-License-Identifier: Apache-2.0

"""Generated current/next tree moments, bias and fused state publication."""

import functools
from textwrap import indent
from types import SimpleNamespace

import warp as wp

from . import cached_force, response_operator, sparse_factor, tree_chain_scan
from .cached_state import StateOperands, StateStages, make_body_terms, make_state_kernel
from .tree_kinetic_layout import REFERENCE_LAYOUT, TreeLayout


@wp.func
def _inertia_action(rotation: wp.quat, inertia: wp.mat33, value: wp.vec3):
    """Apply a body-frame inertia to a world-frame vector."""
    return wp.quat_rotate(rotation, inertia * wp.quat_rotate_inv(rotation, value))


@wp.struct
class KineticPlan:
    parent: wp.array[int]
    depth: wp.array[int]
    jump: wp.array2d[int]
    children_offsets: wp.array[int]
    children: wp.array[int]
    q_index: wp.array[int]
    root_slot: wp.array[int]
    # Depth-first order of the local bodies and, per lane of that order, the
    # lane past its subtree: every subtree is the lane interval [lane, dfs_end).
    dfs_order: wp.array[int]
    dfs_end: wp.array[int]
    # Each local body's first local DOF and DOF count (its joint's coordinates).
    body_dof_start: wp.array[int]
    body_dof_count: wp.array[int]
    # Forest groups: the articulation of each root, each body's root index,
    # each root's local body, each free root's local body and coordinate
    # offset, and whether a coordinate takes passive springs.
    arts: wp.array2d[int]
    root_of: wp.array[int]
    root_local: wp.array[int]
    free_root_local: wp.array[int]
    free_root_dof: wp.array[int]
    dof_passive: wp.array[int]


@wp.struct
class KineticData:
    # Geometry, forces and source tags follow factor groups. Current validity
    # follows articulation IDs shared with the ordinary FK/ID cache.
    bias: wp.array2d[float]
    com_offset: wp.array2d[wp.vec3]
    geometric: wp.array2d[float]
    current_valid: wp.array[int]
    geometry_valid: wp.array[int]
    generation: wp.array[wp.int64]
    geometry_generation: wp.array[wp.int64]
    status: wp.array[int]
    source_q: wp.array[wp.uint64]
    source_qd: wp.array[wp.uint64]
    snapshot_q: wp.array2d[float]
    snapshot_qd: wp.array2d[float]
    snapshot_body_q: wp.array2d[wp.transform]
    snapshot_body_qd: wp.array2d[wp.spatial_vector]
    stage_clocks: wp.array2d[wp.int64]


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


@wp.struct
class ForceInput:
    joint_q: wp.array[float]
    joint_qd: wp.array[float]
    predictor_qd: wp.array[float]
    joint_f: wp.array[float]
    body_f: wp.array[wp.spatial_vector]
    body_flags: wp.array[int]
    stiffness: wp.array[float]
    reference: wp.array[float]
    damping: wp.array[float]
    u0: wp.array[float]
    S: wp.array[wp.spatial_vector]
    kinematic_joint: wp.array[int]
    qdd: wp.array[float]
    vhat: wp.array[float]
    dt: float
    stage_clocks: wp.array2d[wp.int64]
    predicted_twist: wp.array[wp.spatial_vector]
    publish_twists: int


_LANES_SOURCE = r"""
#if defined(__CUDA_ARCH__)
    return wp::vec_t<3,int>($LANE,$STRIDE,$SUB);
#else
    return wp::vec_t<3,int>(0,1,0);
#endif
"""

_SYNC_SOURCE = r"""
#if defined(__CUDA_ARCH__)
    $SYNC;
#endif
"""

_SUB_SOURCE = r"""
#if defined(__CUDA_ARCH__)
    return $SUB;
#else
    return 0;
#endif
"""

_THREADS = r"""
#if defined(__CUDA_ARCH__)
    const int lane=$LANE,stride=$STRIDE;
#else
    const int lane=0,stride=1;
#endif
"""


_DIFFERENT_MOTION_SOURCE = r"""
    if(state_qd.data==motion_qd.data)return 0;
    int different=0;
#if defined(__CUDA_ARCH__)
    for(int i=$LANE;i<count;i+=$STRIDE)
        different|=state_qd.data[start+i]!=motion_qd.data[start+i];
    return $ANY different);
#else
    for(int i=0;i<count;++i)different|=state_qd.data[start+i]!=motion_qd.data[start+i];
    return different;
#endif
"""


@functools.cache
def _thread_helpers(layout):
    """Lane, barrier and sub-tree natives for one tree layout."""
    lanes_source = layout.render(_LANES_SOURCE)
    sync_source = layout.render(_SYNC_SOURCE)
    sub_source = layout.render(_SUB_SOURCE)
    motion_source = layout.render(_DIFFERENT_MOTION_SOURCE)

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


@functools.cache
def _copy_binding(dofs, bodies, plan_type, layout):
    """Compare fixed scalar coordinates on a changed bank inside its repair block."""
    lanes = layout.render(_THREADS)

    match_source = (
        lanes
        + r"""
    bool equal=true;
    for(int local=lane;local<$DOFS;local+=stride) {
        const int slot=group*$DOFS+local;
        equal&=q.data[plan.q_index.data[slot]]==cache.snapshot_q.data[slot];
        equal&=qd.data[first+local]==cache.snapshot_qd.data[slot];
    }
    for(int local=lane;local<$BODIES;local+=stride) {
        const int slot=group*$BODIES+local,body=children.data[first_joint+local];
        for(int k=0;k<7;++k)equal&=body_q.data[body][k]==cache.snapshot_body_q.data[slot][k];
        for(int k=0;k<6;++k)equal&=body_qd.data[body][k]==cache.snapshot_body_qd.data[slot][k];
    }
#if defined(__CUDA_ARCH__)
    return $ALL equal);
#else
    return equal?1:0;
#endif
""".replace("$DOFS", str(dofs)).replace("$BODIES", str(bodies))
    )
    match_source = layout.render(match_source)

    @wp.func_native(match_source)
    def matches(
        plan: plan_type,
        cache: KineticData,
        group: int,
        first: int,
        q: wp.array[float],
        qd: wp.array[float],
        children: wp.array[int],
        first_joint: int,
        body_q: wp.array[wp.transform],
        body_qd: wp.array[wp.spatial_vector],
    ) -> int: ...

    store_source = (
        lanes
        + r"""
    for(int local=lane;local<$DOFS;local+=stride) {
        const int slot=group*$DOFS+local;
        cache.snapshot_q.data[slot]=q.data[plan.q_index.data[slot]];
        cache.snapshot_qd.data[slot]=qd.data[first+local];
    }
    for(int local=lane;local<$BODIES;local+=stride) {
        const int slot=group*$BODIES+local,body=children.data[first_joint+local];
        cache.snapshot_body_q.data[slot]=body_q.data[body];
        cache.snapshot_body_qd.data[slot]=body_qd.data[body];
    }
""".replace("$DOFS", str(dofs)).replace("$BODIES", str(bodies))
    )

    @wp.func_native(store_source)
    def store(
        plan: plan_type,
        cache: KineticData,
        group: int,
        first: int,
        q: wp.array[float],
        qd: wp.array[float],
        children: wp.array[int],
        first_joint: int,
        body_q: wp.array[wp.transform],
        body_qd: wp.array[wp.spatial_vector],
    ): ...

    return matches, store


@wp.func_native("float* s=reinterpret_cast<float*>(address);for(int k=0;k<7;++k)s[7*body+k]=pose[k];")
def _store_pose(address: wp.uint64, body: int, pose: wp.transform): ...


@wp.func_native(
    "float* s=reinterpret_cast<float*>(address);wp::transform p;for(int k=0;k<7;++k)p[k]=s[7*body+k];return p;"
)
def _load_pose(address: wp.uint64, body: int) -> wp.transform: ...


_POSE = r"""
    float* s=reinterpret_cast<float*>(address);
#if defined(__CUDA_ARCH__)
    const int body=$LANE;
    wp::transform value;
    if(body<$BODIES)for(int k=0;k<7;++k)value[k]=s[7*body+k];
    for(int round=0;round<$ROUNDS;++round) {
        const int parent=body<$BODIES?p.jump.data[round*$BODIES+body]:-1;
        wp::transform ancestor;
        if(parent>=0)for(int k=0;k<7;++k)ancestor[k]=s[7*parent+k];
        $SYNC;
        if(parent>=0)value=wp::transform_multiply(ancestor,value);
        if(body<$BODIES)for(int k=0;k<7;++k)s[7*body+k]=value[k];
        $SYNC;
    }
#else
    wp::transform old[$BODIES];
    for(int round=0;round<$ROUNDS;++round) {
        for(int body=0;body<$BODIES;++body)for(int k=0;k<7;++k)old[body][k]=s[7*body+k];
        for(int body=0;body<$BODIES;++body) {
            const int parent=p.jump.data[round*$BODIES+body];
            const wp::transform v=parent>=0?wp::transform_multiply(old[parent],old[body]):old[body];
            for(int k=0;k<7;++k)s[7*body+k]=v[k];
        }
    }
#endif
"""

_MOTION = r"""
    float* s=reinterpret_cast<float*>(address)+$MOTION;
#if defined(__CUDA_ARCH__)
    const int body=$LANE;
    float v[6]={},a[6]={};
    if(body<$BODIES)for(int k=0;k<6;++k){v[k]=s[6*body+k];s[$SIX_BODIES+6*body+k]=0.0f;}
    $SYNC;
    for(int round=0;round<$ROUNDS;++round) {
        const int parent=body<$BODIES?p.jump.data[round*$BODIES+body]:-1;
        float pv[6],pa[6];
        if(parent>=0)for(int k=0;k<6;++k){pv[k]=s[6*parent+k];pa[k]=s[$SIX_BODIES+6*parent+k];}
        $SYNC;
        if(parent>=0) {
            const wp::vec3 pl(pv[0],pv[1],pv[2]),pw(pv[3],pv[4],pv[5]);
            const wp::vec3 vl(v[0],v[1],v[2]),vw(v[3],v[4],v[5]);
            const wp::vec3 c=wp::cross(pw,vl)+wp::cross(pl,vw),d=wp::cross(pw,vw);
            for(int k=0;k<3;++k){a[k]=pa[k]+a[k]+c[k];a[k+3]=pa[k+3]+a[k+3]+d[k];}
            for(int k=0;k<6;++k)v[k]=pv[k]+v[k];
        }
        if(body<$BODIES)for(int k=0;k<6;++k){s[6*body+k]=v[k];s[$SIX_BODIES+6*body+k]=a[k];}
        $SYNC;
    }
#else
    float ov[$BODIES][6],oa[$BODIES][6];
    for(int body=0;body<$BODIES;++body)for(int k=0;k<6;++k)s[$SIX_BODIES+6*body+k]=0.0f;
    for(int round=0;round<$ROUNDS;++round) {
        for(int body=0;body<$BODIES;++body)for(int k=0;k<6;++k){ov[body][k]=s[6*body+k];oa[body][k]=s[$SIX_BODIES+6*body+k];}
        for(int body=0;body<$BODIES;++body) {
            const int parent=p.jump.data[round*$BODIES+body];
            if(parent<0)continue;
            const float* pv=ov[parent],*pa=oa[parent],*v=ov[body];
            const wp::vec3 pl(pv[0],pv[1],pv[2]),pw(pv[3],pv[4],pv[5]);
            const wp::vec3 vl(v[0],v[1],v[2]),vw(v[3],v[4],v[5]);
            const wp::vec3 c=wp::cross(pw,vl)+wp::cross(pl,vw),d=wp::cross(pw,vw);
            for(int k=0;k<3;++k){s[$SIX_BODIES+6*body+k]=pa[k]+oa[body][k]+c[k];s[$SIX_BODIES_PLUS3+6*body+k]=pa[k+3]+oa[body][k+3]+d[k];}
            for(int k=0;k<6;++k)s[6*body+k]=pv[k]+v[k];
        }
    }
#endif
"""

# Register scans for warp-sized layouts: each lane carries its body's value and
# pointer-jumps to the 2^round-th ancestor through shuffles, with no shared-memory
# round trip and no barrier per round. Lanes past the last body idle in place.
_POSE_SHFL = r"""
    float* s=reinterpret_cast<float*>(address);
#if defined(__CUDA_ARCH__)
    const int lane=$LANE,body=lane;
    wp::transform value=wp::transform_identity<float>();
    if(body<$BODIES)for(int k=0;k<7;++k)value[k]=s[7*body+k];
    int parent=body<$BODIES?p.parent.data[body]:-1;
    for(int round=0;round<$ROUNDS;++round) {
        const int source=parent>=0?parent:lane;
        wp::transform ancestor;
        for(int k=0;k<7;++k)ancestor[k]=__shfl_sync($MASK,value[k],source$SHFL);
        const int next=__shfl_sync($MASK,parent,source$SHFL);
        if(parent>=0)value=wp::transform_multiply(ancestor,value);
        parent=parent>=0?next:-1;
    }
    if(body<$BODIES)for(int k=0;k<7;++k)s[7*body+k]=value[k];
    $SYNC;
#else
    wp::transform old[$BODIES];
    for(int round=0;round<$ROUNDS;++round) {
        for(int body=0;body<$BODIES;++body)for(int k=0;k<7;++k)old[body][k]=s[7*body+k];
        for(int body=0;body<$BODIES;++body) {
            const int parent=p.jump.data[round*$BODIES+body];
            const wp::transform v=parent>=0?wp::transform_multiply(old[parent],old[body]):old[body];
            for(int k=0;k<7;++k)s[7*body+k]=v[k];
        }
    }
#endif
"""

_MOTION_SHFL = r"""
    float* s=reinterpret_cast<float*>(address)+$MOTION;
#if defined(__CUDA_ARCH__)
    const int lane=$LANE,body=lane;
    float v[6]={},a[6]={};
    if(body<$BODIES)for(int k=0;k<6;++k)v[k]=s[6*body+k];
    int parent=body<$BODIES?p.parent.data[body]:-1;
    for(int round=0;round<$ROUNDS;++round) {
        const int source=parent>=0?parent:lane;
        float pv[6],pa[6];
        for(int k=0;k<6;++k){pv[k]=__shfl_sync($MASK,v[k],source$SHFL);pa[k]=__shfl_sync($MASK,a[k],source$SHFL);}
        const int next=__shfl_sync($MASK,parent,source$SHFL);
        if(parent>=0) {
            const wp::vec3 pl(pv[0],pv[1],pv[2]),pw(pv[3],pv[4],pv[5]);
            const wp::vec3 vl(v[0],v[1],v[2]),vw(v[3],v[4],v[5]);
            const wp::vec3 c=wp::cross(pw,vl)+wp::cross(pl,vw),d=wp::cross(pw,vw);
            for(int k=0;k<3;++k){a[k]=pa[k]+a[k]+c[k];a[k+3]=pa[k+3]+a[k+3]+d[k];}
            for(int k=0;k<6;++k)v[k]=pv[k]+v[k];
        }
        parent=parent>=0?next:-1;
    }
    if(body<$BODIES)for(int k=0;k<6;++k){s[6*body+k]=v[k];s[$SIX_BODIES+6*body+k]=a[k];}
    $SYNC;
#else
    float ov[$BODIES][6],oa[$BODIES][6];
    for(int body=0;body<$BODIES;++body)for(int k=0;k<6;++k)s[$SIX_BODIES+6*body+k]=0.0f;
    for(int round=0;round<$ROUNDS;++round) {
        for(int body=0;body<$BODIES;++body)for(int k=0;k<6;++k){ov[body][k]=s[6*body+k];oa[body][k]=s[$SIX_BODIES+6*body+k];}
        for(int body=0;body<$BODIES;++body) {
            const int parent=p.jump.data[round*$BODIES+body];
            if(parent<0)continue;
            const float* pv=ov[parent],*pa=oa[parent],*v=ov[body];
            const wp::vec3 pl(pv[0],pv[1],pv[2]),pw(pv[3],pv[4],pv[5]);
            const wp::vec3 vl(v[0],v[1],v[2]),vw(v[3],v[4],v[5]);
            const wp::vec3 c=wp::cross(pw,vl)+wp::cross(pl,vw),d=wp::cross(pw,vw);
            for(int k=0;k<3;++k){s[$SIX_BODIES+6*body+k]=pa[k]+oa[body][k]+c[k];s[$SIX_BODIES_PLUS3+6*body+k]=pa[k+3]+oa[body][k+3]+d[k];}
            for(int k=0;k<6;++k)s[6*body+k]=pv[k]+v[k];
        }
    }
#endif
"""


_COLLECT = r"""    $PHYSICAL_DECL

    float* s=reinterpret_cast<float*>(address);
#if defined(__CUDA_ARCH__)
    const int lane=$LANE,stride=$STRIDE;
#else
    const int lane=0,stride=1;
#endif
$CLOCKC0$REDUCE
$CLOCKC1    for(int dof=lane;dof<$DOFS;dof+=stride) {
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
    $SYNC;
#endif
$CLOCKC2    if(refresh)for(int e=lane;e<$NNZ;e+=stride) {
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
    $SYNC;
#endif
$CLOCKC3    if(lane==0) {
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

_PREDICT = r"""    $PHYSICAL_DECL

    const int art=p.group_to_art.data[group];
    const int ds=p.art_dof_start.data[art],js=p.art_joint_start.data[art];
#if defined(__CUDA_ARCH__)
    const int lane=$LANE,stride=$STRIDE;
    $PRED_SCRATCH_DECL
#else
    const int lane=0,stride=1;
    float s[$PRED_SCRATCH];int nonzero=0;
#endif
    if(lane==0)nonzero=0;
$CLOCK0#if defined(__CUDA_ARCH__)
    $SYNC;
#endif
    for(int local=lane;local<$BODIES;local+=stride) {
        const int body=p.joint_child.data[js+local];
        const auto input=f.body_f.data[body];
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
    $SYNC;
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
    $SYNC;
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
    $SYNC;
#endif
$CLOCK4    // A prescribed (kinematic) free root keeps its given velocity: no transport.
    $TRANSPORT_BLOCK
$TWIST_BLOCK$CLOCK5"""

_LEVEL_REDUCE = r"""    for(int level=$LAST_PARENT;level>=0;--level) {
        for(int body=lane;body<$BODIES;body+=stride)if(plan.depth.data[body]==level) {
            for(int pos=plan.children_offsets.data[body];pos<plan.children_offsets.data[body+1];++pos) {
                const int child=plan.children.data[pos];
                for(int k=0;k<6;++k)s[$WRENCH+6*body+k]+=s[$WRENCH+6*child+k];
                if(refresh)for(int k=0;k<13;++k)s[$MOMENT+13*body+k]+=s[$MOMENT+13*child+k];
            }
        }
#if defined(__CUDA_ARCH__)
        $SYNC;
#endif
    }
"""

# Subtree sums for warp-sized layouts: lanes walk the bodies in depth-first
# order, so a subtree is a lane interval. Each lane builds the sums of the
# 1, 2, 4, ... lanes starting at itself (shuffle-down, zero past the group),
# then assembles its interval from the binary digits of its length: one
# shuffle per digit, additions only, no barrier and no cancellation. The
# CPU twin keeps the depth-level loop.
_INTERVAL_REDUCE = (
    r"""
#if defined(__CUDA_ARCH__)
    {
    const int dfs_body=lane<$BODIES?plan.dfs_order.data[lane]:-1;
    const int dfs_len=lane<$BODIES?plan.dfs_end.data[lane]-lane:0;
    const int dfs_count=refresh?19:6;
    for(int c=0;c<19;++c) {
        if(c>=dfs_count)break;
        const int slot=c<6?$WRENCH+6*dfs_body+c:$MOMENT+13*dfs_body+(c-6);
        float w[$WINDOWS];
        w[0]=dfs_body>=0?s[slot]:0.0f;
        #pragma unroll
        for(int b=1;b<$WINDOWS;++b){
            const float t=__shfl_down_sync($MASK,w[b-1],1<<(b-1)$SHFL);
            w[b]=w[b-1]+(lane+(1<<(b-1))<$LANES?t:0.0f);
        }
        float sum=0.0f;
        int offset=lane;
        #pragma unroll
        for(int b=$WINDOWS-1;b>=0;--b){
            const float t=__shfl_sync($MASK,w[b],offset$SHFL);
            if((dfs_len>>b)&1){sum+=t;offset+=1<<b;}
        }
        if(dfs_body>=0)s[slot]=sum;
    }
    $SYNC;
    }
#else
"""
    + _LEVEL_REDUCE
    + "#endif\n"
)


# Predicted body twists for the resolved-world classifier, published by the
# predictor from the tree it already holds: each lane sums its body's own
# axes against the final predictor velocity, then pointer-jumps up the tree
# through shuffles (velocities at a common origin add). Warp-sized plans only.
_TWIST_BLOCK = r"""
#if defined(__CUDA_ARCH__)
    if(f.publish_twists!=0) {
        $SYNC;
        float tw[6]={};
        if(lane<$BODIES) {
            const int first=ds+plan.body_dof_start.data[lane],count=plan.body_dof_count.data[lane];
            for(int d=0;d<count;++d) {
                const wp::spatial_vector axis=f.S.data[first+d];
                const float speed=f.vhat.data[first+d];
                for(int k=0;k<6;++k)tw[k]+=axis[k]*speed;
            }
        }
        int twist_parent=lane<$BODIES?plan.parent.data[lane]:-1;
        for(int round=0;round<$ROUNDS;++round) {
            const int source=twist_parent>=0?twist_parent:lane;
            float ptw[6];
            for(int k=0;k<6;++k)ptw[k]=__shfl_sync($MASK,tw[k],source$SHFL);
            const int next=__shfl_sync($MASK,twist_parent,source$SHFL);
            if(twist_parent>=0)for(int k=0;k<6;++k)tw[k]+=ptw[k];
            twist_parent=twist_parent>=0?next:-1;
        }
        if(lane<$BODIES) {
            wp::spatial_vector twist;
            for(int k=0;k<6;++k)twist[k]=tw[k];
            f.predicted_twist.data[p.joint_child.data[js+lane]]=twist;
        }
    }
#endif
"""

_EXTERNAL_REDUCE = r"""    if(nonzero)for(int level=$LAST_PARENT;level>=0;--level) {
        for(int local=lane;local<$BODIES;local+=stride)if(plan.depth.data[local]==level) {
            for(int pos=plan.children_offsets.data[local];pos<plan.children_offsets.data[local+1];++pos) {
                const int child=plan.children.data[pos];
                for(int k=0;k<6;++k)s[6*local+k]+=s[6*child+k];
            }
        }
#if defined(__CUDA_ARCH__)
        $SYNC;
#endif
    }
"""


_STORAGE_SOURCE = r"""
#if defined(__CUDA_ARCH__)
    $STATE_SCRATCH_DECL
#else
    return reinterpret_cast<uint64_t>(malloc($TOTAL*sizeof(float)));
#endif
"""

_STORE_TERMS_SOURCE = r"""
    float* s=reinterpret_cast<float*>(address);
    for(int k=0;k<6;++k)s[$WRENCH+6*body+k]=wrench[k];
    if(refresh) {
        float* t=s+$MOMENT+13*body;
        t[0]=mass;
        for(int k=0;k<3;++k)t[1+k]=mass*radius[k];
        for(int i=0;i<3;++i)for(int j=0;j<3;++j)
            t[4+3*i+j]=inertia.data[i][j]+mass*((i==j?wp::dot(radius,radius):0.0f)-radius[i]*radius[j]);
    }
"""


@functools.cache
def get_helpers(
    layout: TreeLayout = REFERENCE_LAYOUT,
    chain_scan: bool = False,
    body_velocity: bool = False,
    *,
    reference: bool = False,
    clocked: bool = False,
):
    """Bind one physical tree evaluation to its storage and traversal schedule.

    ``reference`` keeps the shared-memory scans and the depth-level subtree
    reduction on warp-sized layouts (the forms the A/B tests compare against);
    ``clocked`` adds the collection stage counters. Both change the emitted
    code, so they are arguments (cache-key members), never environment reads.
    """
    plan_type = tree_chain_scan.ChainPlan if chain_scan else KineticPlan
    # A warp-sized layout scans in registers through shuffles; two-warp layouts
    # keep the shared-memory jump table.
    register_scan = not chain_scan and layout.lanes <= 32 and not reference
    pose_source = layout.render(tree_chain_scan._POSE if chain_scan else _POSE_SHFL if register_scan else _POSE)
    motion_source = layout.render(tree_chain_scan._MOTION if chain_scan else _MOTION_SHFL if register_scan else _MOTION)
    interval_reduce = not chain_scan and layout.lanes <= 32 and not reference
    reduction = (
        tree_chain_scan.reduction_source(moments=True, layout=layout)
        if chain_scan
        else layout.render(_INTERVAL_REDUCE if interval_reduce else _LEVEL_REDUCE)
    )
    collect_text = _COLLECT.replace("$REDUCE", reduction)
    for k in range(4):
        statement = ""
        if clocked and k == 0:
            statement = "#if defined(__CUDA_ARCH__)\n    long long c0=clock64(),c1=c0,c2=c0,c3=c0;\n#endif\n"
        elif clocked:
            statement = f"#if defined(__CUDA_ARCH__)\n    c{k}=clock64();\n#endif\n"
        if clocked and k == 3:
            statement += (
                "#if defined(__CUDA_ARCH__)\n    if(lane==0&&cache.stage_clocks.shape[0]>group){long long* o=cache.stage_clocks.data"
                "+group*cache.stage_clocks.shape[1]+13;o[0]=c1-c0;o[1]=c2-c1;o[2]=c3-c2;}\n#endif\n"
            )
        collect_text = collect_text.replace(f"$CLOCKC{k}", statement)
    collect_source = layout.render(collect_text)

    @wp.func_native(layout.render(_STORAGE_SOURCE))
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
    def _scan_poses(address: wp.uint64, p: plan_type): ...

    @wp.func_native(motion_source)
    def _scan_motion(address: wp.uint64, p: plan_type): ...

    @wp.func_native(layout.render(_STORE_TERMS_SOURCE))
    def _store_terms(
        address: wp.uint64,
        body: int,
        mass: float,
        radius: wp.vec3,
        inertia: wp.mat33,
        wrench: wp.spatial_vector,
        refresh: int,
    ): ...

    from . import kernels  # noqa: PLC0415

    store = "_store_terms(address, local, mass, radius, inertia_com, wrench, refresh)"
    if (layout.bodies == 1 and layout.root_dofs == 6) or (layout.forest and layout.free_roots > 0):
        # A root-only free body is a free rigid body (in a forest, a free root
        # may be one): the matrix-free contact path reads its spatial inertia
        # and bias wrench every substep, so publish them through the ordinary
        # free-rigid finalization; the runtime free-rigid test skips tree
        # bodies. A single free-root tree of several bodies never takes this
        # branch (it cost Cassie flat and Humanoid 3-4% in the finish kernel).
        store += """
if data.is_free_rigid[data.body_to_articulation[body]] != 0:
    data.body_q_com[body] = pose_com
    data.body_a_s[body] = acceleration
    kernels.finalize_body_dynamics_values(
        body,
        data.body_to_articulation[body],
        pose,
        pose_com,
        data.body_com,
        data.body_mass,
        data.body_inertia,
        data.is_free_rigid,
        origin,
        0,
        0,
        data.gravity,
        velocity,
        acceleration,
        data.body_I_s,
        data.body_inertia_terms,
        data.body_f_s,
        data.body_qd,
    )"""
    _body_terms = make_body_terms(
        "address: wp.uint64, body: int, local: int, world: int, pose: wp.transform, pose_com: wp.transform, origin: wp.vec3, velocity: wp.spatial_vector, acceleration: wp.spatial_vector, data: PublicationData, cache: KineticData, refresh: int",
        dict(globals(), _store_terms=_store_terms, _inertia_action=_inertia_action, kernels=kernels),
        refresh="refresh",
        cache="cache",
        com_offset="radius",
        body_velocity=body_velocity,
        store=store,
        clocked=clocked,
    )

    @wp.func_native(collect_source)
    def _collect(
        address: wp.uint64, group: int, p: sparse_factor.SparsePlan, plan: plan_type, cache: KineticData, refresh: int
    ): ...

    return SimpleNamespace(
        plan_type=plan_type,
        _body_terms=_body_terms,
        _collect=_collect,
        _load_motion=_load_motion,
        _load_origin=_load_origin,
        _scan_motion=_scan_motion,
        _scan_poses=_scan_poses,
        _storage=_storage,
        _store_axis=_store_axis,
        _store_motion=_store_motion,
        _store_origin=_store_origin,
        _store_terms=_store_terms,
    )


@functools.cache
def get_state_kernel(
    finish: bool,
    chain_scan: bool = False,
    *,
    layout: TreeLayout = REFERENCE_LAYOUT,
    body_velocity: bool = False,
    motion_velocity: bool = False,
    copy_reuse: bool = False,
    reference: bool = False,
    clocked: bool = False,
):
    """Build complete repair/finish using original generalized integration laws.

    ``reference`` and ``clocked`` select the comparison forms and the stage
    counters (see :func:`get_helpers`).
    """
    if finish and motion_velocity:
        raise ValueError("Alternate motion velocity is a repair-only contract")
    if copy_reuse and (layout.root_dofs or motion_velocity):
        raise ValueError("Copied-state reuse requires fixed scalar coordinates and ordinary motion")
    h = get_helpers(layout, chain_scan, body_velocity, reference=reference, clocked=clocked)
    plan_type = h.plan_type
    _lanes, _sync, _sub, _different_motion = _thread_helpers(layout)
    pack = layout.pack
    scan_poses, scan_motion, collect = h._scan_poses, h._scan_motion, h._collect
    _storage, _store_motion, _load_motion = h._storage, h._store_motion, h._load_motion
    _store_origin, _load_origin, _store_axis = h._store_origin, h._load_origin, h._store_axis
    _body_terms = h._body_terms
    bodies, dofs, free_root = layout.bodies, layout.dofs, layout.root_dofs == 6
    operands = StateOperands(
        bodies,
        dofs,
        "js + local",
        "ds + local",
        "if wp.static(free_root) and lane == 0" if not layout.forest else "if lane < wp.static(free_roots)",
        "plan.root_slot[group]" if not layout.forest else "plan.root_slot[group * wp.static(free_roots) + lane]",
    )
    roots, free_roots, forest = len(layout.roots), layout.free_roots, layout.forest
    motion_setup = "motion_changed = _different_motion(data.joint_qd, motion_qd, ds, dofs)\n" if motion_velocity else ""
    motion_reuse = "        and motion_changed == 0\n" if motion_velocity else ""
    motion_bind = "qd = motion_qd" if motion_velocity else ""
    copy_setup, copy_store, copy_rebind = "", "", ""
    if copy_reuse:
        _copy_matches, _copy_store = _copy_binding(dofs, bodies, plan_type, layout)
        copy_setup = """    if cache.current_valid[art] != 0 and source_matches == 0:
        source_matches = _copy_matches(plan, cache, group, ds, data.joint_q, data.joint_qd, data.joint_child, js, data.body_q, data.body_qd)
"""
        copy_rebind = """        if lane == 0:
            _stamp_source(cache, group, data.joint_q, data.joint_qd)
"""
        copy_store = "_copy_store(plan, cache, group, ds, q, qd, data.joint_child, js, data.body_q, data.body_qd)\n"
    motion_clock = ["pass"] * 4
    if clocked:
        motion_clock = [f"clock_m{k} = _clock()" for k in range(4)]
        motion_clock[3] += (
            "\n    if local == 0 and data.stage_clocks.shape[0] > group:\n"
            "        data.stage_clocks[group, 22] = clock_m1 - clock_m0\n"
            "        data.stage_clocks[group, 23] = clock_m2 - clock_m1\n"
            "        data.stage_clocks[group, 24] = clock_m3 - clock_m2"
        )
    stamp = "_stamp_source(cache, group, q, qd)"
    if motion_velocity:
        # An override source must not be reused when a later begin returns to
        # ordinary motion, even if no finish occurred between the two calls.
        stamp = """if motion_changed == 0:
    _stamp_source(cache, group, q, data.joint_qd)
else:
    _stamp_source(cache, group, q, qd)"""
    stages = StateStages(
        setup=f"""
block, logical_lane = wp.tid()
lanes = _lanes()
lane, stride = lanes[0], lanes[1]
if stride == 1 and logical_lane != 0:
    return
group = block * wp.static(pack) + lanes[2]
art = p.group_to_art[group]
ds, js = p.art_dof_start[art], p.art_joint_start[art]
{motion_setup}\
refresh = int(global_refresh != 0)
valid = cache.current_valid[art]
if wp.static(forest):
    for root in range(1, wp.static(roots)):
        valid = wp.min(valid, cache.current_valid[plan.arts[group, root]])
if not wp.static(finish):
    refresh = int(refresh != 0 or requests[art] != 0)
    if wp.static(forest):
        for root in range(1, wp.static(roots)):
            refresh = int(refresh != 0 or requests[plan.arts[group, root]] != 0)
    source_matches = _same_source(cache, group, data.joint_q, data.joint_qd)
{copy_setup}\
    if (
        valid != 0
        and source_matches != 0
{motion_reuse}\
        and (
            refresh == 0
            or (
                cache.geometry_valid[group] != 0 and cache.geometry_generation[group] == cache.generation[group]
            )
        )
    ):
{copy_rebind}\
        return
if lane == 0:
    cache.current_valid[art] = 0
    if wp.static(forest):
        for root in range(1, wp.static(roots)):
            cache.current_valid[plan.arts[group, root]] = 0
""",
        relative="""
if wp.static(not forest):
    if local == 0:
        _store_origin(address, 1, wp.transform_get_translation(relative))
        relative = wp.transform(wp.vec3(), wp.transform_get_rotation(relative))
else:
    if plan.parent[local] < 0:
        _store_origin(address, 2 * plan.root_of[local] + 1, wp.transform_get_translation(relative))
        relative = wp.transform(wp.vec3(), wp.transform_get_rotation(relative))
""",
        poses=f"""
scan_poses(address, plan)
if wp.static(not forest):
    if lane == 0:
        root_body = data.joint_child[js]
        origin = wp.transform_point(_load_pose(address, 0), data.body_com[root_body])
        _store_origin(address, 0, origin)
        data.articulation_origin[art] = origin + _load_origin(address, 1)
else:
    for root in range(lane, wp.static(roots), stride):
        root_local = plan.root_local[root]
        root_body = data.joint_child[js + root_local]
        origin = wp.transform_point(_load_pose(address, root_local), data.body_com[root_body])
        _store_origin(address, 2 * root, origin)
        data.articulation_origin[plan.arts[group, root]] = origin + _load_origin(address, 2 * root + 1)
_sync()
origin = _load_origin(address, 0)
shift = _load_origin(address, 1)
{motion_bind}
""",
        motion=f"""
for local in range(lane, bodies, stride):
    {motion_clock[0]}
    joint = js + local
    parent = plan.parent[local]
    if wp.static(forest):
        origin = _load_origin(address, 2 * plan.root_of[local])
        shift = _load_origin(address, 2 * plan.root_of[local] + 1)
    anchor = data.joint_X_p[joint]
    if parent >= 0:
        anchor = _load_pose(address, parent) * anchor
    else:
        anchor = wp.transform(wp.transform_get_translation(anchor) - shift, wp.transform_get_rotation(anchor))
    anchor = wp.transform(wp.transform_get_translation(anchor) - origin, wp.transform_get_rotation(anchor))
    {motion_clock[1]}
    velocity = kernels.jcalc_motion(
        data.joint_type[joint],
        data.joint_axis,
        data.joint_dof_dim[joint, 0],
        data.joint_dof_dim[joint, 1],
        anchor,
        qd,
        data.joint_qd_start[joint],
        data.joint_S_s,
    )
    {motion_clock[2]}
    start = data.joint_qd_start[joint]
    count = data.joint_dof_dim[joint, 0] + data.joint_dof_dim[joint, 1]
    for k in range(count):
        _store_axis(address, start - ds + k, data.joint_S_s[start + k])
    _store_motion(address, local, velocity)
    {motion_clock[3]}
_sync()
scan_motion(address, plan)
""",
        bodies="""
for local in range(lane, bodies, stride):
    body = data.joint_child[js + local]
    if wp.static(forest):
        origin = _load_origin(address, 2 * plan.root_of[local])
        shift = _load_origin(address, 2 * plan.root_of[local] + 1)
    pose = _load_pose(address, local)
    data.body_q[body] = wp.transform(
        wp.transform_get_translation(pose) + shift, wp.transform_get_rotation(pose)
    )
    _body_terms(
        address,
        body,
        local,
        group,
        pose,
        pose * data.body_X_com[body],
        origin,
        _load_motion(address, local, 0),
        _load_motion(address, local, 1),
        data,
        cache,
        refresh,
    )
_sync()
""",
        collect="""
collect(address, group, p, plan, cache, refresh)
"""
        + copy_store
        + """
if lane == 0:
"""
        + indent(stamp, "    "),
    )
    name = f"tree_kinetic_finish{bodies}" if finish else f"tree_kinetic_repair{bodies}"
    if layout.forest:
        name += f"_forest{roots}"
    if layout.pack > 1:
        name += f"_l{layout.lanes}"
    if chain_scan:
        name += "_chain"
    if copy_reuse:
        name += "_copies"
    signature = "p: sparse_factor.SparsePlan, plan: plan_type, data: PublicationData, cache: KineticData, requests: wp.array[int], global_refresh: int"
    if motion_velocity:
        name += "_motion"
        signature += ", motion_qd: wp.array[float]"
    return make_state_kernel(
        name,
        signature,
        operands,
        stages,
        dict(globals(), **locals()),
        finish=finish,
        clocked=clocked,
    )


@functools.cache
def get_predictor_kernel(
    chain_scan: bool = False,
    *,
    layout: TreeLayout = REFERENCE_LAYOUT,
    response="sparse",
    reference: bool = False,
    clocked: bool = False,
):
    """Apply current external/control force and both unchanged W actions.

    ``reference`` keeps the serial one-lane held solve on CUDA; ``clocked``
    adds the predictor stage counters (see :func:`get_helpers`).
    """
    plan_type = tree_chain_scan.ChainPlan if chain_scan else KineticPlan
    reduction = (
        tree_chain_scan.reduction_source(moments=False, layout=layout)
        if chain_scan
        else layout.render(_EXTERNAL_REDUCE)
    )
    if chain_scan:
        reduction = "if(nonzero) {\n" + reduction + "}\n"
    inverse = response_operator.SparseInverseFactor(
        "d.W.data+group*$NNZ",
        response_operator.IndexedSupport(
            "$DOFS", "p.index.data", "p.inverse_count.data", "p.inverse_nodes.data", "$INVERSE_WIDTH"
        ),
    )
    physical = (
        f"{layout.dofs - 1}-$index"
        if layout.permutation == tuple(range(layout.dofs - 1, -1, -1))
        else "physical[$index]"
    )
    operations = {
        "SHIFT": cached_force.shift_wrench(
            "input", "cache.com_offset.data[group*$BODIES+local]", "s[6*local+k]", "s[6*local+3+k]"
        ),
        "PROJECT": cached_force.project_wrench("f.S.data[dof]", "s[6*body+k]", "value"),
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
        "TRANSPOSE_ACTION": inverse.transpose("col", "s[$WHITE+$index]"),
        "RESPONSE_PHYSICAL": "$PHYSICAL_INDEX",
    }
    forward = inverse.forward("row", f"s[$SIX_BODIES+{physical}]")
    operations["FORWARD_STAGE"] = (
        """for(int row=lane;row<$DOFS;row+=stride) {
        """
        + forward
        + """
        s[$WHITE+row]=value;
    }
#if defined(__CUDA_ARCH__)
    $SYNC;
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
            serial.append(
                "if(lane==0){\n"
                + held.forward(f"s[$SIX_BODIES+{offset}+$index]", f"s[$WHITE+{offset}+$index]")
                + held.transpose(f"s[$WHITE+{offset}+$index]", f"s[$WHITE+{offset}+$index]")
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
            + "\n#endif\n#if defined(__CUDA_ARCH__)\n$SYNC;\n#endif\n"
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
    operations["TWIST_BLOCK"] = _TWIST_BLOCK if (layout.lanes <= 32 and not chain_scan) else ""
    source = _PREDICT
    for k in range(6):
        statement = ""
        if clocked and k == 0:
            statement = (
                "#if defined(__CUDA_ARCH__)\n    long long c0=clock64(),c1=c0,c2=c0,c3=c0,c4=c0,c5=c0;\n#endif\n"
            )
        elif clocked:
            statement = f"#if defined(__CUDA_ARCH__)\n    c{k}=clock64();\n#endif\n"
        if clocked and k == 5:
            statement += (
                "#if defined(__CUDA_ARCH__)\n    if(lane==0&&f.stage_clocks.shape[0]>group){long long* o=f.stage_clocks.data"
                "+group*f.stage_clocks.shape[1]+8;o[0]=c1-c0;o[1]=c2-c1;o[2]=c3-c2;o[3]=c4-c3;o[4]=c5-c4;}\n#endif\n"
            )
        source = source.replace(f"$CLOCK{k}", statement)
    for name, operation in operations.items():
        source = source.replace("$" + name, operation)
    source = layout.render(source)

    @wp.func_native(source)
    def native(
        group: int,
        p: sparse_factor.SparsePlan,
        d: data_type,
        plan: plan_type,
        cache: KineticData,
        f: ForceInput,
    ): ...

    _, _, _sub, _ = _thread_helpers(layout)
    pack = layout.pack

    def predict(p: sparse_factor.SparsePlan, d: data_type, plan: plan_type, cache: KineticData, f: ForceInput):
        block, _ = wp.tid()
        group = block * wp.static(pack) + _sub()
        native(group, p, d, plan, cache, f)

    predict.__name__ = predict.__qualname__ = f"tree_kinetic_predict{layout.dofs}"
    if layout.forest:
        predict.__name__ = predict.__qualname__ = predict.__name__ + f"_forest{len(layout.roots)}"
    if layout.pack > 1:
        predict.__name__ = predict.__qualname__ = predict.__name__ + f"_l{layout.lanes}"
    if chain_scan:
        predict.__name__ = predict.__qualname__ = predict.__name__ + "_chain"
    if response != "sparse":
        predict.__name__ = predict.__qualname__ = predict.__name__ + "_" + response
    return wp.kernel(module="unique", enable_backward=False)(predict)
