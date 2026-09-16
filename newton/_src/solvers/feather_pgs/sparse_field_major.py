# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Research-only world-lane metric GS plus charged cooperative publication.

No production dispatch is installed. Packing is an explicit, untimed component
input conversion, not a claimed retired producer. The canonical held W remains
unchanged and the separate decode must be included in every component timing.
"""

from functools import cache

import numpy as np
import warp as wp

from .sparse_factor import SparseData, SparsePlan
from .sparse_metric_tangents import get_fragments


@wp.struct
class FieldMajorRows:
    Z: wp.array3d[float]
    support: wp.array2d[int]
    incident: wp.array2d[float]
    rhs: wp.array2d[float]
    diagonal: wp.array2d[float]
    row_type: wp.array2d[int]
    parent: wp.array2d[int]
    mu: wp.array2d[float]
    impulses: wp.array2d[float]
    cross: wp.array2d[float]
    du: wp.array2d[float]


def pack_rows(data, rhs, diagonal, row_type, parent, mu, *, device=None):
    """Create explicit component inputs from original arrays; never time-exclude decode."""
    device = data.Z.device if device is None else device
    worlds = int(data.Z.shape[0])
    if tuple(data.Z.shape) != (worlds, 100, 18):
        raise ValueError("Field-major component requires original capacity100/support18 rows")
    packed = FieldMajorRows()
    packed.Z = wp.array(np.ascontiguousarray(data.Z.numpy().transpose(2, 1, 0)), dtype=float, device=device)
    for name, values, dtype in (
        ("support", data.support, int),
        ("incident", data.incident, float),
        ("rhs", rhs, float),
        ("diagonal", diagonal, float),
        ("row_type", row_type, int),
        ("parent", parent, int),
        ("mu", mu, float),
    ):
        if tuple(values.shape) != (worlds, 100):
            raise ValueError(f"Field-major component requires {name}[worlds,100]")
        setattr(packed, name, wp.array(np.ascontiguousarray(values.numpy().T), dtype=dtype, device=device))
    packed.impulses = wp.empty((100, worlds), dtype=float, device=device)
    packed.cross = wp.empty((100, worlds), dtype=float, device=device)
    packed.du = wp.empty((43, worlds), dtype=float, device=device)
    return packed


def metric_root_source():
    """Reuse the frozen native16 root, guards and repair without changing its law."""
    _, update = get_fragments(100)
    begin = "                                r1+=contact_cross[row]*change0;"
    end = "\n                            }\n                        }\n                        accepted=__shfl_sync"
    if update.count(begin) != 1 or update.count(end) != 1:
        raise RuntimeError("Original metric-root source seam changed")
    root = update[update.index(begin) : update.index(end)]
    return (
        root.replace("contact_cross[row+2]", "F(cross,row+2)")
        .replace("contact_cross[row+1]", "F(cross,row+1)")
        .replace("contact_cross[row]", "F(cross,row)")
        .replace("diagonal.data[base+row+1]", "F(diagonal,row+1)")
        .replace("diagonal.data[base+row+2]", "F(diagonal,row+2)")
    )


def native_source():
    """Return one fixed block32 lane/world translation of the original transactions."""
    source = r"""
    if(group>=p.group_to_art.shape[0])return;
    const int art=p.group_to_art.data[group],world=p.art_to_world.data[art];
    const int count=counts.data[world],stride=f.rhs.shape[1],base=world*100;
    if(!d.valid.data[world] || d.status.data[world] || count>100)return;
    #define F(field,row) f.field.data[(row)*stride+world]
    #define Z(row,k) f.Z.data[((k)*100+(row))*stride+world]
    #if defined(__CUDA_ARCH__)
    const int lane=threadIdx.x&31;
    __shared__ float du[43][32];
    #define DU(k) du[k][lane]
    #else
    float du[43];
    #define DU(k) du[k]
    // Warp's CPU headers omit these CUDA/libm names. Keep the native CUDA
    // expressions untouched; provide finite-equivalent CPU test helpers.
    const auto fmaxf=[](float x,float y){return x!=x?y:y!=y?x:(x>y?x:y);};
    const auto fminf=[](float x,float y){return x!=x?y:y!=y?x:(x<y?x:y);};
    const auto hypotf=[](float x,float y){
        const float a=wp::abs(x),b=wp::abs(y),hi=wp::max(a,b),lo=wp::min(a,b);
        if(hi==0.0f)return 0.0f;
        const float ratio=lo/hi;return hi*wp::sqrt(1.0f+ratio*ratio);
    };
    #endif
    int bad=0;
    for(int row=0;row<count;++row) {
        const int type=F(row_type,row);
        if(type!=0 && type!=2 && type!=3)bad=1;
        if(type==2) {
            const int par=F(parent,row);
            if(par<0 || par+2>=count || (row!=par+1 && row!=par+2))bad=1;
            else if(F(row_type,par)!=0 || F(row_type,par+1)!=2 || F(row_type,par+2)!=2 ||
                    F(parent,par+1)!=par || F(parent,par+2)!=par)bad=1;
        }
    }
    if(bad) {
        #if defined(__CUDA_ARCH__)
        atomicOr(&d.status.data[world],4);
        #else
        d.status.data[world]|=4;
        #endif
        return;
    }
    for(int k=0;k<43;++k)DU(k)=0.0f;
    for(int row=0;row<count;++row)F(impulses,row)=impulses.data[base+row];
    unsigned int ready0=0u,ready1=0u,ready2=0u,ready3=0u;
    for(int iteration=0;iteration<iterations;++iteration) {
        int changed=0,skip_until=0;
        for(int row=0;row<count;++row) {
            // Keep the row-slot induction variable common across active world
            // lanes while retaining each world's complete triplet transaction.
            if(row<skip_until)continue;
            const int type=F(row_type,row);
            if(type==0 && row+2<count && iteration>=friction_start && omega==1.0f &&
               F(row_type,row+1)==2 && F(row_type,row+2)==2 &&
               F(parent,row+1)==row && F(parent,row+2)==row &&
               F(support,row)==F(support,row+1) && F(support,row)==F(support,row+2)) {
                const float friction=F(mu,row+1);
                if(isfinite(friction) && friction>=0.0f && friction==F(mu,row+2)) {
                    const int tpl=F(support,row),length=p.support_count.data[tpl];
                    float r0=0.0f;
                    #pragma unroll 1
                    for(int k=0;k<length;++k)r0+=Z(row,k)*DU(p.support_nodes.data[tpl*18+k]);
                    r0=r0+F(incident,row)+F(rhs,row);
                    const float old0=F(impulses,row),old1=F(impulses,row+1),old2=F(impulses,row+2);
                    const float d0=F(diagonal,row);
                    const float next0=fmaxf(old0-r0/d0,0.0f),change0=next0-old0;
                    const float radius=friction*next0;
                    const int normal_ok=isfinite(r0) && isfinite(d0) && d0>0.0f &&
                        isfinite(old0) && isfinite(old1) && isfinite(old2) &&
                        isfinite(next0) && isfinite(change0) && isfinite(radius);
                    if(normal_ok) {
                        int accepted=radius==0.0f,bad_coeff=0;
                        float next1=0.0f,next2=0.0f;
                        const int word=row/32;
                        const unsigned int bit=1u<<(row%32);
                        const unsigned int ready=word==0?ready0:word==1?ready1:word==2?ready2:ready3;
                        const int needs_block=radius>0.0f && (ready&bit)==0u;
                        float r1=0.0f,r2=0.0f,a01=0.0f,a02=0.0f,a12=0.0f;
                        #pragma unroll 1
                        for(int k=0;k<length;++k) {
                            const float z0=Z(row,k);
                            const float z1=radius>0.0f || old1!=0.0f?Z(row+1,k):0.0f;
                            const float z2=radius>0.0f || old2!=0.0f?Z(row+2,k):0.0f;
                            bad_coeff|=!isfinite(z0) || !isfinite(z1) || !isfinite(z2);
                            if(radius>0.0f) {
                                const float value=DU(p.support_nodes.data[tpl*18+k]);
                                r1+=z1*value;r2+=z2*value;
                                if(needs_block){a01+=z0*z1;a02+=z0*z2;a12+=z1*z2;}
                            }
                        }
                        if(radius>0.0f) {
                            if(needs_block) {
                                F(cross,row)=a01;F(cross,row+1)=a02;F(cross,row+2)=a12;
                                if(word==0)ready0|=bit;else if(word==1)ready1|=bit;
                                else if(word==2)ready2|=bit;else ready3|=bit;
                            }
                            r1=r1+F(incident,row+1)+F(rhs,row+1);
                            r2=r2+F(incident,row+2)+F(rhs,row+2);
                            METRIC_ROOT
                        }
                        const float change1=next1-old1,change2=next2-old2;
                        if(accepted && !bad_coeff && isfinite(change1) && isfinite(change2)) {
                            F(impulses,row)=next0;F(impulses,row+1)=next1;F(impulses,row+2)=next2;
                            if(change0!=0.0f || change1!=0.0f || change2!=0.0f) {
                                #pragma unroll 1
                                for(int k=0;k<length;++k) {
                                    const float z0=Z(row,k);
                                    const float z1=radius>0.0f || old1!=0.0f?Z(row+1,k):0.0f;
                                    const float z2=radius>0.0f || old2!=0.0f?Z(row+2,k):0.0f;
                                    DU(p.support_nodes.data[tpl*18+k])+=z0*change0+z1*change1+z2*change2;
                                }
                                changed=1;
                            }
                            skip_until=row+3;
                            continue;
                        }
                    }
                }
            }
            if(type==2 && iteration<friction_start){F(impulses,row)=0.0f;continue;}
            const float denom=F(diagonal,row);
            if(!(denom>0.0f))continue;
            const int tpl=F(support,row),length=p.support_count.data[tpl];
            float dot=0.0f;
            #pragma unroll 1
            for(int k=0;k<length;++k)dot+=Z(row,k)*DU(p.support_nodes.data[tpl*18+k]);
            const float old=F(impulses,row),residual=dot+F(incident,row)+F(rhs,row);
            float next=old+omega*(-residual/denom),sibling_delta=0.0f;
            int sibling=-1;
            if(type==0 || type==3)next=fmaxf(next,0.0f);
            else if(type==2) {
                const int par=F(parent,row);
                const float radius=fmaxf(F(mu,row)*F(impulses,par),0.0f);
                if(radius<=0.0f)next=0.0f;
                else {
                    sibling=row==par+1?par+2:par+1;
                    const float other=F(impulses,sibling),mag=sqrtf(next*next+other*other);
                    if(mag>radius) {
                        const float scale=radius/mag;next*=scale;
                        const float adjusted=other*scale;sibling_delta=adjusted-other;F(impulses,sibling)=adjusted;
                    }
                }
            }
            const float delta=next-old;F(impulses,row)=next;
            if(sibling_delta!=0.0f) {
                const int st=F(support,sibling),sn=p.support_count.data[st];
                #pragma unroll 1
                for(int k=0;k<sn;++k)DU(p.support_nodes.data[st*18+k])+=Z(sibling,k)*sibling_delta;
                changed=1;
            }
            if(delta!=0.0f) {
                #pragma unroll 1
                for(int k=0;k<length;++k)DU(p.support_nodes.data[tpl*18+k])+=Z(row,k)*delta;
                changed=1;
            }
        }
        if(iteration>=friction_start && !changed)break;
    }
    for(int k=0;k<43;++k)f.du.data[k*stride+world]=DU(k);
    #undef DU
    #undef Z
    #undef F
"""
    return source.replace("METRIC_ROOT", metric_root_source())


@cache
def get_solve_kernel():
    """One scalar world per lane; launch ordinary dim=groups, block_dim=32 only."""

    @wp.func_native(native_source())
    def native(
        group: int,
        p: SparsePlan,
        d: SparseData,
        f: FieldMajorRows,
        counts: wp.array[int],
        impulses: wp.array2d[float],
        iterations: int,
        omega: float,
        friction_start: int,
    ): ...

    def solve(
        p: SparsePlan,
        d: SparseData,
        f: FieldMajorRows,
        counts: wp.array[int],
        impulses: wp.array2d[float],
        iterations: int,
        omega: float,
        friction_start: int,
    ):
        group = wp.tid()
        native(group, p, d, f, counts, impulses, iterations, omega, friction_start)

    solve.__name__ = solve.__qualname__ = "sparse_field_major_metric43_s18_c100"
    return wp.kernel(enable_backward=False, module="unique")(solve)


@cache
def get_decode_kernel():
    """Original cooperative W decode and canonical impulse export, always charged."""
    source = r"""
    if(group>=p.group_to_art.shape[0])return;
    const int art=p.group_to_art.data[group],world=p.art_to_world.data[art];
    const int count=counts.data[world],start=p.art_dof_start.data[art],stride=f.rhs.shape[1];
    if(!d.valid.data[world] || d.status.data[world] || count>100)return;
    #if defined(__CUDA_ARCH__)
    const int lane=threadIdx.x&31,step=32;
    #else
    const int lane=0,step=1;
    #endif
    for(int col=lane;col<43;col+=step) {
        float value=0.0f;
        for(int k=0;k<p.inverse_count.data[col];++k) {
            const int row=p.inverse_nodes.data[col*18+k];
            value+=d.W.data[group*434+p.index.data[row*43+col]]*f.du.data[row*stride+world];
        }
        vout.data[start+42-col]=vhat.data[start+42-col]+value;
    }
    for(int row=lane;row<count;row+=step)impulses.data[world*100+row]=f.impulses.data[row*stride+world];
"""

    @wp.func_native(source)
    def native(
        group: int,
        p: SparsePlan,
        d: SparseData,
        f: FieldMajorRows,
        counts: wp.array[int],
        impulses: wp.array2d[float],
        vhat: wp.array[float],
        vout: wp.array[float],
    ): ...

    def decode(
        p: SparsePlan,
        d: SparseData,
        f: FieldMajorRows,
        counts: wp.array[int],
        impulses: wp.array2d[float],
        vhat: wp.array[float],
        vout: wp.array[float],
    ):
        group, _ = wp.tid()
        native(group, p, d, f, counts, impulses, vhat, vout)

    decode.__name__ = decode.__qualname__ = "sparse_field_major_decode43_s18_c100"
    return wp.kernel(enable_backward=False, module="unique")(decode)
