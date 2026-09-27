# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Packed heavy-chain schedules for the existing complete kinetic owners.

Only tree traversal changes. All poses, common-origin motion, bias, moments,
held-factor actions and publication retain their original physical meanings.
"""

import functools

import numpy as np
import warp as wp


@wp.struct
class ChainPlan:
    parent: wp.array[int]
    depth: wp.array[int]
    jump: wp.array2d[int]
    children_offsets: wp.array[int]
    children: wp.array[int]
    q_index: wp.array[int]
    root_slot: wp.array[int]
    chain_meta: wp.array[int]
    light_offsets: wp.array[int]
    light_children: wp.array[int]
    stage_width: wp.array[int]
    chain_levels: int
    chain_width: int


def make_plan(parent):
    """Pack topology-derived heavy chains into two warps, or reject the plan.

    Each metadata row contains body, segment start, chain length, light depth,
    and the actual body anchoring the chain head. There are no task-name or
    joint-count rules in the decomposition; the existing owner admits G1.
    """
    parent = np.asarray(parent, dtype=np.int32)
    count = parent.size
    if parent.ndim != 1 or not count or count > 64:
        raise ValueError("Packed chains require one nonempty tree of at most64 bodies")
    if parent[0] != -1 or np.any(parent[1:] < 0) or np.any(parent[1:] >= np.arange(1, count)):
        raise ValueError("Packed chains require one parent-first tree")
    children = [[] for _ in range(count)]
    for body in range(1, count):
        children[int(parent[body])].append(body)
    size = np.ones(count, np.int32)
    for body in range(count - 1, 0, -1):
        size[parent[body]] += size[body]
    heavy = np.full(count, -1, np.int32)
    for body, edges in enumerate(children):
        if edges:
            heavy[body] = max(edges, key=lambda child: int(size[child]))
    chains = []
    pending = [(0, 0)]
    while pending:
        head, level = pending.pop()
        chain, body = [], head
        while body >= 0:
            chain.append(body)
            pending.extend((child, level + 1) for child in children[body] if child != heavy[body])
            body = int(heavy[body])
        width = 1 << (len(chain) - 1).bit_length()
        if width > 32:
            raise ValueError("A heavy chain cannot cross a hardware warp")
        chains.append((head, level, chain, width))
    chains.sort(key=lambda chain: (-chain[3], chain[0]))
    meta = np.full((64, 5), -1, np.int32)
    meta[:, 1] = np.arange(64)
    meta[:, 2] = 0
    used = [0, 0]
    for head, level, chain, width in chains:
        candidates = [warp for warp in range(2) if used[warp] + width <= 32]
        if not candidates:
            raise ValueError("Padded heavy chains do not fit two warps")
        warp = candidates[0]
        start = 32 * warp + used[warp]
        used[warp] += width
        for offset, body in enumerate(chain):
            meta[start + offset] = (body, start, len(chain), level, parent[head])
    light, offsets = [], [0]
    for body, edges in enumerate(children):
        # Keep the original descending child order for the branch attachments.
        light.extend(child for child in reversed(edges) if child != heavy[body])
        offsets.append(len(light))
    levels = 1 + max(chain[1] for chain in chains)
    return {
        "meta": meta,
        "light_offsets": np.asarray(offsets, np.int32),
        "light_children": np.asarray(light, np.int32),
        "levels": levels,
        "stage_width": np.asarray(
            [max(chain[3] for chain in chains if chain[1] == level) for level in range(levels)], np.int32
        ),
        "widths": np.asarray([chain[3] for chain in chains], np.int32),
    }


_POSE = r"""
    float* s=reinterpret_cast<float*>(address);
#if defined(__CUDA_ARCH__)
    const int lane=threadIdx.x,body=p.chain_meta.data[5*lane];
    const int start=p.chain_meta.data[5*lane+1],position=lane-start;
    const int level=p.chain_meta.data[5*lane+3],anchor=p.chain_meta.data[5*lane+4];
    wp::transform value=wp::transform_identity<float>();
    if(body>=0)for(int k=0;k<7;++k)value[k]=s[7*body+k];
    for(int offset=1;offset<p.chain_width;offset*=2) {
        const bool combine=body>=0&&position>=offset;
        const int source=combine?((lane-offset)&31):(lane&31);
        wp::transform ancestor;
        for(int k=0;k<7;++k)ancestor[k]=__shfl_sync(0xffffffffu,value[k],source,32);
        if(combine)value=wp::transform_multiply(ancestor,value);
    }
    for(int stage=0;stage<p.chain_levels;++stage) {
        if(body>=0&&level==stage) {
            if(anchor>=0) {
                wp::transform ancestor;
                for(int k=0;k<7;++k)ancestor[k]=s[7*anchor+k];
                value=wp::transform_multiply(ancestor,value);
            }
            for(int k=0;k<7;++k)s[7*body+k]=value[k];
        }
        __syncthreads();
    }
#else
    wp::transform values[64],old[64];
    for(int lane=0;lane<64;++lane) {
        values[lane]=wp::transform_identity<float>();
        const int body=p.chain_meta.data[5*lane];
        if(body>=0)for(int k=0;k<7;++k)values[lane][k]=s[7*body+k];
    }
    for(int offset=1;offset<p.chain_width;offset*=2) {
        for(int lane=0;lane<64;++lane)old[lane]=values[lane];
        for(int lane=0;lane<64;++lane)if(p.chain_meta.data[5*lane]>=0&&lane-p.chain_meta.data[5*lane+1]>=offset)
            values[lane]=wp::transform_multiply(old[lane-offset],old[lane]);
    }
    for(int stage=0;stage<p.chain_levels;++stage)for(int lane=0;lane<64;++lane) {
        const int body=p.chain_meta.data[5*lane],anchor=p.chain_meta.data[5*lane+4];
        if(body>=0&&p.chain_meta.data[5*lane+3]==stage) {
            if(anchor>=0) {
                wp::transform ancestor;
                for(int k=0;k<7;++k)ancestor[k]=s[7*anchor+k];
                values[lane]=wp::transform_multiply(ancestor,values[lane]);
            }
            for(int k=0;k<7;++k)s[7*body+k]=values[lane][k];
        }
    }
#endif
"""


@wp.func_native(_POSE)
def scan_poses(address: wp.uint64, p: ChainPlan): ...


_COMPOSE_MOTION = r"""
    const wp::vec3 pl(pv[0],pv[1],pv[2]),pw(pv[3],pv[4],pv[5]);
    const wp::vec3 vl(v[0],v[1],v[2]),vw(v[3],v[4],v[5]);
    const wp::vec3 c=wp::cross(pw,vl)+wp::cross(pl,vw),d=wp::cross(pw,vw);
    for(int k=0;k<3;++k){a[k]=pa[k]+a[k]+c[k];a[k+3]=pa[k+3]+a[k+3]+d[k];}
    for(int k=0;k<6;++k)v[k]=pv[k]+v[k];
"""

_MOTION = r"""
    float* s=reinterpret_cast<float*>(address)+308;
    auto compose=[](float* v,float* a,const float* pv,const float* pa) {
COMPOSE_MOTION
    };
#if defined(__CUDA_ARCH__)
    const int lane=threadIdx.x,body=p.chain_meta.data[5*lane];
    const int position=lane-p.chain_meta.data[5*lane+1];
    const int level=p.chain_meta.data[5*lane+3],anchor=p.chain_meta.data[5*lane+4];
    float v[6]={},a[6]={};
    if(body>=0)for(int k=0;k<6;++k)v[k]=s[6*body+k];
    for(int offset=1;offset<p.chain_width;offset*=2) {
        const bool combine=body>=0&&position>=offset;
        const int source=combine?((lane-offset)&31):(lane&31);
        float pv[6],pa[6];
        for(int k=0;k<6;++k) {
            pv[k]=__shfl_sync(0xffffffffu,v[k],source,32);
            pa[k]=__shfl_sync(0xffffffffu,a[k],source,32);
        }
        if(combine)compose(v,a,pv,pa);
    }
    for(int stage=0;stage<p.chain_levels;++stage) {
        if(body>=0&&level==stage) {
            if(anchor>=0) {
                float pv[6],pa[6];
                for(int k=0;k<6;++k){pv[k]=s[6*anchor+k];pa[k]=s[264+6*anchor+k];}
                compose(v,a,pv,pa);
            }
            for(int k=0;k<6;++k){s[6*body+k]=v[k];s[264+6*body+k]=a[k];}
        }
        __syncthreads();
    }
#else
    float v[64][6]={},a[64][6]={},ov[64][6],oa[64][6];
    for(int lane=0;lane<64;++lane) {
        const int body=p.chain_meta.data[5*lane];
        if(body>=0)for(int k=0;k<6;++k)v[lane][k]=s[6*body+k];
    }
    for(int offset=1;offset<p.chain_width;offset*=2) {
        for(int lane=0;lane<64;++lane)for(int k=0;k<6;++k){ov[lane][k]=v[lane][k];oa[lane][k]=a[lane][k];}
        for(int lane=0;lane<64;++lane)if(p.chain_meta.data[5*lane]>=0&&lane-p.chain_meta.data[5*lane+1]>=offset)
            compose(v[lane],a[lane],ov[lane-offset],oa[lane-offset]);
    }
    for(int stage=0;stage<p.chain_levels;++stage)for(int lane=0;lane<64;++lane) {
        const int body=p.chain_meta.data[5*lane],anchor=p.chain_meta.data[5*lane+4];
        if(body>=0&&p.chain_meta.data[5*lane+3]==stage) {
            if(anchor>=0) {
                float pv[6],pa[6];
                for(int k=0;k<6;++k){pv[k]=s[6*anchor+k];pa[k]=s[264+6*anchor+k];}
                compose(v[lane],a[lane],pv,pa);
            }
            for(int k=0;k<6;++k){s[6*body+k]=v[lane][k];s[264+6*body+k]=a[lane][k];}
        }
    }
#endif
""".replace("COMPOSE_MOTION", _COMPOSE_MOTION)


@wp.func_native(_MOTION)
def scan_motion(address: wp.uint64, p: ChainPlan): ...


def reduction_source(*, moments):
    """Add light subtrees before an inclusive heavy-chain suffix; never subtract."""
    address = "(k<6?1094+6*BODY+k:1358+13*BODY+k-6)" if moments else "(6*BODY+k)"
    count = "(refresh?19:6)" if moments else "6"
    vector = r"""
    {
    float values[COMPONENTS];
    for(int stage=plan.chain_levels-1;stage>=0;--stage) {
        const bool active=body>=0&&level==stage;
        const int width=plan.stage_width.data[stage];
        #pragma unroll
        for(int k=0;k<COMPONENTS;++k)values[k]=active?s[slot(body,k)]:0.0f;
        if(active)for(int edge=first;edge<last;++edge) {
            const int child=plan.light_children.data[edge];
            #pragma unroll
            for(int k=0;k<COMPONENTS;++k)values[k]+=s[slot(child,k)];
        }
        for(int offset=1;offset<width;offset*=2) {
            const bool combine=active&&position+offset<length;
            const int source=combine?((thread+offset)&31):(thread&31);
            #pragma unroll
            for(int k=0;k<COMPONENTS;++k) {
                const float next=__shfl_sync(0xffffffffu,values[k],source,32);
                if(combine)values[k]+=next;
            }
        }
        if(active) {
            #pragma unroll
            for(int k=0;k<COMPONENTS;++k)s[slot(body,k)]=values[k];
        }
        __syncthreads();
    }
    }
"""
    # Fixed component counts expose independent channels to the compiler. A
    # runtime component-major loop serialized every shuffle chain and repeated
    # the light-child list/address walk for all 6 or 19 channels. The new loop
    # order preserves each channel's exact light-child and suffix-add order.
    cuda = vector.replace("COMPONENTS", "6")
    if moments:
        cuda = "if(refresh) " + vector.replace("COMPONENTS", "19") + " else " + cuda
    source = r"""
    {
    auto slot=[](int body,int k){return ADDRESS;};
#if defined(__CUDA_ARCH__)
    const int thread=threadIdx.x,body=plan.chain_meta.data[5*thread];
    const int position=thread-plan.chain_meta.data[5*thread+1];
    const int length=plan.chain_meta.data[5*thread+2],level=plan.chain_meta.data[5*thread+3];
    const int first=body>=0?plan.light_offsets.data[body]:0;
    const int last=body>=0?plan.light_offsets.data[body+1]:0;
VECTOR_REDUCTION
#else
    for(int stage=plan.chain_levels-1;stage>=0;--stage) {
        for(int k=0;k<COMPONENTS;++k) {
            float value[64]={},old[64];
            for(int thread=0;thread<64;++thread) {
                const int body=plan.chain_meta.data[5*thread];
                if(body>=0&&plan.chain_meta.data[5*thread+3]==stage) {
                    value[thread]=s[slot(body,k)];
                    for(int edge=plan.light_offsets.data[body];edge<plan.light_offsets.data[body+1];++edge)
                        value[thread]+=s[slot(plan.light_children.data[edge],k)];
                }
            }
            for(int offset=1;offset<plan.stage_width.data[stage];offset*=2) {
                for(int thread=0;thread<64;++thread)old[thread]=value[thread];
                for(int thread=0;thread<64;++thread) {
                    const int body=plan.chain_meta.data[5*thread],position=thread-plan.chain_meta.data[5*thread+1];
                    if(body>=0&&plan.chain_meta.data[5*thread+3]==stage&&position+offset<plan.chain_meta.data[5*thread+2])
                        value[thread]+=old[thread+offset];
                }
            }
            for(int thread=0;thread<64;++thread) {
                const int body=plan.chain_meta.data[5*thread];
                if(body>=0&&plan.chain_meta.data[5*thread+3]==stage)s[slot(body,k)]=value[thread];
            }
        }
    }
#endif
    }
"""
    return (
        source.replace("VECTOR_REDUCTION", cuda)
        .replace("ADDRESS", address.replace("BODY", "body"))
        .replace("COMPONENTS", count)
    )


@functools.cache
def get_collect(source):
    """Replace only the original collection traversal, preserving every consumer."""
    start = source.index("    for(int level=9;level>=0;--level) {")
    end = source.index("    for(int dof=lane;dof<43;dof+=stride) {", start)
    code = source[:start] + reduction_source(moments=True) + source[end:]
    from .g1_kinetic_state import KineticData  # noqa: PLC0415
    from .sparse_factor import SparsePlan  # noqa: PLC0415

    @wp.func_native(code)
    def collect(address: wp.uint64, group: int, p: SparsePlan, plan: ChainPlan, cache: KineticData, refresh: int): ...

    return collect


def predictor_source(source):
    """Preserve the zero-external-force shortcut and both original W actions."""
    start = source.index("    if(nonzero)for(int level=9;level>=0;--level) {")
    end = source.index("    for(int i=lane;i<43;i+=stride) {", start)
    return source[:start] + "if(nonzero) {\n" + reduction_source(moments=False) + "}\n" + source[end:]
