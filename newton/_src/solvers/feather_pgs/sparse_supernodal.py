# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Opt-in multifrontal refresh with the original packed inverse-whitener ABI.

Panels stop at branch junctions. Only descendant updates enter separator
storage; original mass entries are assembled exactly once by their column
owner. The reverse pass composes complete ancestor inverses, not responses.
"""

from functools import cache

import numpy as np
import warp as wp

from .sparse_factor import SparseData, SparsePlan

# Reverse-coordinate chains of the already admitted 434-entry SparsePlan.
_BLOCKS = tuple(
    tuple(range(a, b))
    for a, b in (
        (0, 3),
        (3, 5),
        (5, 7),
        (7, 12),
        (12, 15),
        (15, 17),
        (17, 19),
        (19, 24),
        (24, 25),
        (25, 31),
        (31, 37),
        (37, 43),
    )
)
_ROOT = tuple(range(37, 43))
_SEPARATORS = (
    *(((*range(7, 12), 24, *_ROOT),) * 3),
    (24, *_ROOT),
    *(((*range(19, 24), 24, *_ROOT),) * 3),
    (24, *_ROOT),
    _ROOT,
    _ROOT,
    _ROOT,
    (),
)
_PARENTS = (3, 3, 3, 8, 7, 7, 7, 8, 11, 11, 11, -1)
# Four fixed workers. The first two process one finger fan each; the last
# two process the independent legs. No warp waits on another within a stage.
_STAGES = (((0, 1, 2), (4, 5, 6), (9,), (10,)), ((3,), (7,), (), ()), ((8,), (), (), ()), ((11,), (), (), ()))


@cache
def schedule():
    """Return the immutable exact panel schedule and its packed output map."""
    pattern = np.zeros((43, 43), dtype=bool)
    offsets = []
    offset = 0
    for block, separator in zip(_BLOCKS, _SEPARATORS, strict=True):
        for j, col in enumerate(block):
            pattern[list(block[j:] + separator), col] = True
        offsets.append(offset)
        offset += len(separator) * (len(separator) + 1) // 2
    rows, cols = np.nonzero(pattern)
    index = np.full((43, 43), -1, dtype=np.int32)
    index[rows, cols] = np.arange(len(rows))
    index.setflags(write=False)
    if len(rows) != 434 or offset != 587:
        raise ValueError("Unexpected sparse panel storage")
    return {
        "blocks": _BLOCKS,
        "separators": _SEPARATORS,
        "parents": _PARENTS,
        "stages": _STAGES,
        "offsets": tuple(offsets),
        "index": index,
    }


def validate_plan(index):
    """Reject any pattern not covered completely by the emitted panel owner."""
    if not np.array_equal(index, schedule()["index"]):
        raise ValueError("Supernodal refresh requires the admitted 434-entry pattern")


def _cuda_helpers():
    """Keep one bounded element-parallel routine rather than twelve expansions."""
    return r"""
    struct Panels {
        static __device__ __forceinline__ int tri(int r,int c){return r*(r+1)/2+c;}
        static __device__ __forceinline__ int node(int r,int b,int start,int s,int first){
            if(r<b)return start+r;
            r-=b;
            if(s==12){if(r<5)return first+r;if(r==5)return 24;return 31+r;}
            if(s==7)return r==0?24:36+r;
            return 37+r;
        }
        static __device__ __noinline__ void factor(
            float* a,float* updates,float* f,float* C,float* W,int* bad,
            const int* index,const int* rc,int lane,int start,int b,int s,int first,
            int offset,int child0,int child1,int child2){
            const int n=b+s,entries=n*(n+1)/2;
            #pragma unroll 1
            for(int e=lane;e<entries;e+=32){
                const int pair=rc[e],r=pair>>4,c=pair&15;
                float value=c<b?a[index[node(r,b,start,s,first)*43+start+c]]:0.0f;
                if(child0>=0)value+=updates[child0+e];
                if(child1>=0)value+=updates[child1+e];
                if(child2>=0)value+=updates[child2+e];
                f[e]=value;
            }
            __syncwarp(0xffffffffu);
            #pragma unroll 1
            for(int k=0;k<b;++k){
                if(lane==0){
                    const float value=f[tri(k,k)];
                    if(!(value>0.0f)||!wp::isfinite(value))atomicExch(bad,1);
                    f[tri(k,k)]=wp::sqrt(value);
                }
                __syncwarp(0xffffffffu);
                const int row=k+1+lane;
                if(row<n)f[tri(row,k)]/=f[tri(k,k)];
                __syncwarp(0xffffffffu);
                #pragma unroll 1
                for(int e=lane;e<entries;e+=32){
                    const int pair=rc[e],r=pair>>4,c=pair&15;
                    if(c>k)f[e]-=f[tri(r,k)]*f[tri(c,k)];
                }
                __syncwarp(0xffffffffu);
            }
            #pragma unroll 1
            for(int e=lane;e<s*(s+1)/2;e+=32){
                const int pair=rc[e],r=pair>>4,c=pair&15;
                updates[offset+e]=f[tri(r+b,c+b)];
            }
            if(lane<b*(b+1)/2)C[lane]=f[lane];
            __syncwarp(0xffffffffu);
            // Preserve C while the same front becomes C^-1 and E=V*C^-1.
            #pragma unroll 1
            for(int c=b-1;c>=0;--c){
                if(lane<n&&lane>=c){
                    float value=lane<b?(lane==c?1.0f:0.0f):f[tri(lane,c)];
                    #pragma unroll 1
                    for(int k=c+1;k<b&&k<=lane;++k)value-=f[tri(lane,k)]*C[tri(k,c)];
                    f[tri(lane,c)]=value/C[tri(c,c)];
                }
                __syncwarp(0xffffffffu);
            }
            #pragma unroll 1
            for(int e=lane;e<entries;e+=32){
                const int pair=rc[e],r=pair>>4,c=pair&15;
                if(c<b){
                    const int entry=index[node(r,b,start,s,first)*43+start+c];
                    const float value=f[e];a[entry]=value;
                    if(r<b){W[entry]=value;if(!wp::isfinite(value))atomicExch(bad,1);}
                }
            }
        }
        static __device__ __noinline__ void inverse(
            float* a,float* f,float* W,int* bad,const int* index,const int* rc,
            int lane,int start,int b,int s,int first){
            const int coupling=b*s;
            #pragma unroll 1
            for(int e=lane;e<coupling;e+=32){
                const int r=e/b,c=e-r*b;
                f[e]=a[index[node(r+b,b,start,s,first)*43+start+c]];
            }
            #pragma unroll 1
            for(int e=lane;e<s*(s+1)/2;e+=32){
                const int pair=rc[e],r=pair>>4,c=pair&15;
                f[coupling+e]=a[index[node(r+b,b,start,s,first)*43+node(c+b,b,start,s,first)]];
            }
            __syncwarp(0xffffffffu);
            #pragma unroll 1
            for(int e=lane;e<coupling;e+=32){
                const int r=e/b,c=e-r*b;
                float value=0.0f;
                #pragma unroll 1
                for(int k=0;k<=r;++k)value-=f[coupling+tri(r,k)]*f[k*b+c];
                const int entry=index[node(r+b,b,start,s,first)*43+start+c];
                a[entry]=value;W[entry]=value;
                if(!wp::isfinite(value))atomicExch(bad,1);
            }
        }
    };
"""


def _factor_cuda(panel):
    block, separator = _BLOCKS[panel], _SEPARATORS[panel]
    children = [child for child, parent in enumerate(_PARENTS) if parent == panel]
    for child in children:
        if _SEPARATORS[child] != block + separator:
            raise ValueError("Child separator must cover its parent's complete front")
    offsets = [schedule()["offsets"][child] for child in children]
    offsets.extend([-1] * (3 - len(offsets)))
    args = (
        block[0],
        len(block),
        len(separator),
        separator[0] if separator else 0,
        schedule()["offsets"][panel],
        *offsets,
    )
    return (
        "Panels::factor(a,updates,front+warp*120,diagonal+warp*21,d.W.data+group*434,&bad,p.index.data,tri_rc,lane,"
        + ",".join(map(str, args))
        + ");"
    )


def _inverse_cuda(panel):
    block, separator = _BLOCKS[panel], _SEPARATORS[panel]
    args = (block[0], len(block), len(separator), separator[0])
    return (
        "Panels::inverse(a,front+warp*120,d.W.data+group*434,&bad,p.index.data,tri_rc,lane,"
        + ",".join(map(str, args))
        + ");"
    )


def _cpu_panels():
    """Execute the same assembled fronts and ancestor composition on CPU."""
    out = []
    order = [p for stage in _STAGES for worker in stage for p in worker]
    for panel in order:
        block, separator = _BLOCKS[panel], _SEPARATORS[panel]
        b, s = len(block), len(separator)
        nodes = block + separator
        n = len(nodes)
        out.extend(["{", f"const int nodes[{n}]={{{','.join(map(str, nodes))}}};", f"float v[{n * n}]={{}};"])
        out.append(
            f"for(int r=0;r<{n};++r)for(int c=0;c<{b}&&c<=r;++c)v[r*{n}+c]=a[p.index.data[nodes[r]*43+nodes[c]]];"
        )
        for child, parent in enumerate(_PARENTS):
            if parent != panel:
                continue
            child_s = _SEPARATORS[child]
            start = nodes.index(child_s[0])
            off = schedule()["offsets"][child]
            out.append(
                f"for(int r={start};r<{n};++r)for(int c={start};c<=r;++c)v[r*{n}+c]+=updates[{off}+(r-{start})*(r-{start}+1)/2+c-{start}];"
            )
        out.append(
            f"for(int k=0;k<{b};++k){{float &pivot=v[k*{n}+k];if(!(pivot>0.0f)||!wp::isfinite(pivot))bad=1;pivot=wp::sqrt(pivot);for(int r=k+1;r<{n};++r)v[r*{n}+k]/=pivot;for(int r=k+1;r<{n};++r)for(int c=k+1;c<=r;++c)v[r*{n}+c]-=v[r*{n}+k]*v[c*{n}+k];}}"
        )
        off = schedule()["offsets"][panel]
        out.append(f"for(int r=0;r<{s};++r)for(int c=0;c<=r;++c)updates[{off}+r*(r+1)/2+c]=v[(r+{b})*{n}+c+{b}];")
        out.append(
            f"float e[{n * b}]={{}};for(int r=0;r<{n};++r)for(int c={b - 1};c>=0;--c){{if(r<c)continue;float value=r<{b}?(r==c?1.0f:0.0f):v[r*{n}+c];for(int k=c+1;k<{b}&&k<=r;++k)value-=e[r*{b}+k]*v[k*{n}+c];e[r*{b}+c]=value/v[c*{n}+c];const int entry=p.index.data[nodes[r]*43+nodes[c]];a[entry]=e[r*{b}+c];if(r<{b}){{d.W.data[group*434+entry]=e[r*{b}+c];if(!wp::isfinite(e[r*{b}+c]))bad=1;}}}}"
        )
        out.append("}")
    for stage in reversed(_STAGES[:-1]):
        for worker in stage:
            for panel in worker:
                block, separator = _BLOCKS[panel], _SEPARATORS[panel]
                b, s = len(block), len(separator)
                out.extend(
                    [
                        "{",
                        f"const int bs[{b}]={{{','.join(map(str, block))}}};",
                        f"const int ss[{s}]={{{','.join(map(str, separator))}}};",
                        f"float e[{s * b}];for(int r=0;r<{s};++r)for(int c=0;c<{b};++c)e[r*{b}+c]=a[p.index.data[ss[r]*43+bs[c]]];",
                    ]
                )
                out.append(
                    f"for(int r=0;r<{s};++r)for(int c=0;c<{b};++c){{float value=0.0f;for(int k=0;k<=r;++k)value-=a[p.index.data[ss[r]*43+ss[k]]]*e[k*{b}+c];const int entry=p.index.data[ss[r]*43+bs[c]];a[entry]=value;d.W.data[group*434+entry]=value;if(!wp::isfinite(value))bad=1;}}"
                )
                out.append("}")
    return "\n".join(out)


def native_source(*, geometric=False):
    """Emit one complete refresh with no factor-sized global intermediate."""
    source = r"""
    const int art=p.group_to_art.data[group];
    if(mask.data[art]==0)return;
    const int world=p.art_to_world.data[art];
    const int ds=p.art_dof_start.data[art],js=p.art_joint_start.data[art];
#if defined(__CUDA_ARCH__)
    const int tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    __shared__ float a[434],updates[587];
    __shared__ float front[480], diagonal[84];
    __shared__ int tri_rc[120];
    __shared__ int bad;
    if(tid==0){bad=0;d.valid.data[world]=0;}
    if(tid<120){
        int r=tid>=36?8:0;
        if(tid>=(r+4)*(r+5)/2)r+=4;
        if(tid>=(r+2)*(r+3)/2)r+=2;
        if(tid>=(r+1)*(r+2)/2)r+=1;
        tri_rc[tid]=(r<<4)|(tid-r*(r+1)/2);
    }
#else
    const int tid=0;
    float a[434],updates[587];
    int bad=0;d.valid.data[world]=0;
#endif
"""
    if not geometric:
        source += r"""
#if defined(__CUDA_ARCH__)
    __shared__ float force[258];
    for(int col=tid;col<43;col+=128){
#else
    float force[258];
    for(int col=0;col<43;++col){
#endif
        const int body=p.joint_child.data[js+p.dof_joint.data[col]];
        const float* inertia=reinterpret_cast<const float*>(&I.data[body]);
        const float* screw=reinterpret_cast<const float*>(&S.data[ds+col]);
        for(int r=0;r<6;++r){float value=0.0f;for(int c=0;c<6;++c)value+=inertia[6*r+c]*screw[c];force[6*col+r]=value;}
    }
#if defined(__CUDA_ARCH__)
    __syncthreads();
#endif
"""
    source += r"""
#if defined(__CUDA_ARCH__)
    for(int e=tid;e<434;e+=128){
#else
    for(int e=0;e<434;++e){
#endif
        const int row=p.row.data[e],col=p.col.data[e];
        const int original_row=42-row,original_col=42-col;
"""
    if geometric:
        source += "float value=I.data[group*434+e];\n"
    else:
        source += r"""
        const int src=p.source.data[e];
        const int projection=src==original_col?original_row:original_col;
        const float* screw=reinterpret_cast<const float*>(&S.data[ds+projection]);
        float value=0.0f;for(int k=0;k<6;++k)value+=screw[k]*force[6*src+k];
"""
    source += r"""
        if(row==col){
            value+=R.data[group*43+original_row];
            int drive=drive_row.data[ds+original_row];
            if(drive<0&&!parallel_drives){
                for(int r=0;r<drive_counts.data[art];++r){const int entry=art*drive_stride+r;if(drive_dofs.data[entry]==ds+original_row){drive=entry;break;}}
            }
            if(drive>=0&&K.data[drive]>0.0f)value+=K.data[drive];
        }
        a[e]=value;
    }
#if defined(__CUDA_ARCH__)
    __syncthreads();
"""
    for stage in _STAGES:
        for worker, panels in enumerate(stage):
            if panels:
                source += (
                    f"if(warp=={worker}){{\n"
                    + "\n__syncwarp(0xffffffffu);\n".join(_factor_cuda(p) for p in panels)
                    + "\n}\n"
                )
        source += "__syncthreads();\n"
    for stage in reversed(_STAGES[:-1]):
        for worker, panels in enumerate(stage):
            if panels:
                source += (
                    f"if(warp=={worker}){{\n"
                    + "\n__syncwarp(0xffffffffu);\n".join(_inverse_cuda(p) for p in panels)
                    + "\n}\n"
                )
        source += "__syncthreads();\n"
    source += "#else\n" + _cpu_panels() + "\n#endif\n"
    source += "if(tid==0){if(bad)atomicOr(&d.status.data[world],1);d.valid.data[world]=bad==0;}\n"
    # CPU compilation has no CUDA atomic primitive; one thread owns the world.
    source = source.replace("if(tid==0){if(bad)atomicOr", "if(tid==0){if(bad)wp::atomic_or")
    return "#if defined(__CUDA_ARCH__)\n" + _cuda_helpers() + "\n#endif\n" + source


@cache
def get_refresh_kernel(geometric=False):
    """Bind the original refresh ABI to complete panels."""
    inertia_type = wp.array2d[float] if geometric else wp.array[wp.spatial_matrix]

    @wp.func_native(native_source(geometric=geometric))
    def native(
        group: int,
        p: SparsePlan,
        d: SparseData,
        mask: wp.array[int],
        S: wp.array[wp.spatial_vector],
        I: inertia_type,
        R: wp.array2d[float],
        drive_row: wp.array[int],
        K: wp.array[float],
        drive_counts: wp.array[int],
        drive_dofs: wp.array[int],
        drive_stride: int,
        parallel_drives: int,
    ): ...

    def refresh(
        p: SparsePlan,
        d: SparseData,
        mask: wp.array[int],
        S: wp.array[wp.spatial_vector],
        I: inertia_type,
        R: wp.array2d[float],
        drive_row: wp.array[int],
        K: wp.array[float],
        drive_counts: wp.array[int],
        drive_dofs: wp.array[int],
        drive_stride: int,
        parallel_drives: int,
    ):
        group, _ = wp.tid()
        native(group, p, d, mask, S, I, R, drive_row, K, drive_counts, drive_dofs, drive_stride, parallel_drives)

    refresh.__name__ = refresh.__qualname__ = ("g1_kinetic_" if geometric else "") + "sparse_supernodal43_434"
    return wp.kernel(enable_backward=False, module="unique")(refresh)
