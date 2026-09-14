# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental current packets with CTA-local sparse rows and original PGS."""

from functools import cache

import warp as wp

from .kernels import _FPGS_CONTACT_END_GAP_SLOP
from .sparse_factor import SparseData, SparsePlan


@wp.struct
class PacketInput:
    shape0: wp.array[int]
    shape1: wp.array[int]
    point0: wp.array[wp.vec3]
    point1: wp.array[wp.vec3]
    normal: wp.array[wp.vec3]
    thickness0: wp.array[float]
    thickness1: wp.array[float]
    shape_body: wp.array[int]
    body_q: wp.array[wp.transform]
    screw: wp.array[wp.spatial_vector]
    origin: wp.array[wp.vec3]
    art_a: wp.array[int]
    art_b: wp.array[int]
    cfm: wp.array2d[float]
    phi: wp.array2d[float]
    target: wp.array2d[float]
    restitution: wp.array2d[float]
    dt: float
    threshold: float
    shared_anchor: int
    friction_anchor: int


@wp.kernel(enable_backward=False)
def packet_contacts(
    count: wp.array[int],
    workers: int,
    path: wp.array[int],
    slot: wp.array[int],
    world: wp.array[int],
    needed: wp.array[int],
    packets: wp.array2d[int],
):
    """Store only each original reservation's current raw-contact identity."""
    worker = wp.tid()
    for c in range(worker, wp.min(count[0], path.shape[0]), workers):
        row = slot[c]
        if path[c] == 0 and row >= 0:
            for direction in range(needed[c]):
                if row + direction < packets.shape[1]:
                    packets[world[c], row + direction] = 3 * c + direction


@cache
def get_prefix_kernel():
    """Keep the original lower/upper order in three stable candidate ballots."""
    source = r"""
#if defined(__CUDA_ARCH__)
    const int lane=threadIdx.x&31, art=p.group_to_art.data[group];
    const int world=p.art_to_world.data[art], start=p.art_dof_start.data[art];
    const int capacity=row_type.shape[1], base=world*capacity;
    int count=0;
    for(int batch=0;batch<3;++batch) {
        const int candidate=batch*32+lane, local=candidate/2, side=candidate&1;
        bool active=false; float value=0.0f;
        if(enabled && candidate<86) {
            const int dof=start+local, qi=limit_q.data[dof];
            if(qi>=0) {
                const float bound=side?upper.data[dof]:lower.data[dof], position=q.data[qi];
                value=(side?-1.0f:1.0f)*(position-bound);
                active=isfinite(bound) && (side?position>=bound-gap:position<=bound+gap);
            }
        }
        const unsigned mask=__ballot_sync(0xffffffff,active);
        const int row=count+__popc(mask&((1u<<lane)-1u)); count+=__popc(mask);
        if(active && row<capacity) {
            const int at=base+row;
            d.support.data[at]=-candidate-1;
            row_type.data[at]=3; parent.data[at]=-1; mu.data[at]=0.0f;
            row_beta.data[at]=beta; row_cfm.data[at]=cfm;
            phi.data[at]=value; target.data[at]=0.0f;
        }
    }
    if(lane==0) {
        counter.data[world]=count;
        phase.data[world*phase.shape[1]]=count; phase.data[world*phase.shape[1]+1]=count;
    }
#else
    const int art=p.group_to_art.data[group], world=p.art_to_world.data[art];
    const int start=p.art_dof_start.data[art], capacity=row_type.shape[1], base=world*capacity;
    int count=0;
    if(enabled)for(int candidate=0;candidate<86;++candidate) {
        const int local=candidate/2, side=candidate&1, dof=start+local, qi=limit_q.data[dof];
        if(qi<0)continue;
        const float bound=side?upper.data[dof]:lower.data[dof], position=q.data[qi];
        if(!isfinite(bound) || !(side?position>=bound-gap:position<=bound+gap))continue;
        const int row=count++;
        if(row>=capacity)continue;
        const int at=base+row;
        d.support.data[at]=-candidate-1; row_type.data[at]=3;parent.data[at]=-1;
        mu.data[at]=0.0f;row_beta.data[at]=beta;row_cfm.data[at]=cfm;
        phi.data[at]=(side?-1.0f:1.0f)*(position-bound);target.data[at]=0.0f;
    }
    counter.data[world]=count;
    phase.data[world*phase.shape[1]]=count;phase.data[world*phase.shape[1]+1]=count;
#endif
"""

    @wp.func_native(source)
    def native(
        group: int,
        p: SparsePlan,
        d: SparseData,
        limit_q: wp.array[int],
        lower: wp.array[float],
        upper: wp.array[float],
        q: wp.array[float],
        enabled: int,
        gap: float,
        beta: float,
        cfm: float,
        counter: wp.array[int],
        row_type: wp.array2d[int],
        parent: wp.array2d[int],
        mu: wp.array2d[float],
        row_beta: wp.array2d[float],
        row_cfm: wp.array2d[float],
        phi: wp.array2d[float],
        target: wp.array2d[float],
        phase: wp.array2d[int],
    ): ...

    def prefix(
        p: SparsePlan,
        d: SparseData,
        limit_q: wp.array[int],
        lower: wp.array[float],
        upper: wp.array[float],
        q: wp.array[float],
        vhat: wp.array[float],
        enabled: int,
        gap: float,
        beta: float,
        cfm: float,
        counter: wp.array[int],
        row_type: wp.array2d[int],
        parent: wp.array2d[int],
        mu: wp.array2d[float],
        row_beta: wp.array2d[float],
        row_cfm: wp.array2d[float],
        phi: wp.array2d[float],
        target: wp.array2d[float],
        diagonal: wp.array2d[float],
        phase: wp.array2d[int],
    ):
        group, _ = wp.tid()
        native(
            group,
            p,
            d,
            limit_q,
            lower,
            upper,
            q,
            enabled,
            gap,
            beta,
            cfm,
            counter,
            row_type,
            parent,
            mu,
            row_beta,
            row_cfm,
            phi,
            target,
            phase,
        )

    prefix.__name__ = prefix.__qualname__ = "sparse_packet_prefix43"
    return wp.kernel(enable_backward=False, module="unique")(prefix)


def cuda_source(capacity):
    """Stage current sparse rows once, preserving the original ordered recurrence."""
    return f"""
#if defined(__CUDA_ARCH__)
    const int lane=threadIdx.x&31, art=p.group_to_art.data[group], world=p.art_to_world.data[art];
    const int start=p.art_dof_start.data[art], count=counts.data[world], base=world*{capacity};
    if(!d.valid.data[world] || d.status.data[world] || count>{capacity})return;
    int bad=0;
    for(int r=lane;r<count;r+=32) {{
        const int type=row_type.data[base+r];
        if(type!=0 && type!=2 && type!=3)bad=1;
        if(type==2) {{
            const int par=parent.data[base+r];
            if(par<0 || par+2>=count || (r!=par+1 && r!=par+2))bad=1;
            else if(row_type.data[base+par]!=0 || row_type.data[base+par+1]!=2 || row_type.data[base+par+2]!=2 || parent.data[base+par+1]!=par || parent.data[base+par+2]!=par)bad=1;
        }}
    }}
    if(__ballot_sync(0xffffffff,bad)) {{if(lane==0)atomicOr(&d.status.data[world],4);return;}}
    __shared__ float zrows[{capacity}*18], du[43], lam[{capacity}], norm[{capacity}], bias[{capacity}];
    __shared__ int templates[{capacity}];
    for(int r=0;r<count;) {{
        const int key=d.support.data[base+r];
        if(key<0) {{
            const int candidate=-key-1, local=candidate/2;
            if(candidate>=86 || row_type.data[base+r]!=3) {{if(lane==0)atomicOr(&d.status.data[world],4);return;}}
            const float sign=(candidate&1)?-1.0f:1.0f;
            const int tpl=p.limit_support.data[local], length=p.support_count.data[tpl];
            float z=0.0f;
            if(lane<length) {{const int node=p.support_nodes.data[tpl*18+lane], e=p.index.data[node*43+42-local];if(e>=0)z=sign*d.W.data[group*434+e];}}
            if(lane<18)zrows[r*18+lane]=z;
            float square=z*z;
            for(int shift=16;shift>0;shift>>=1)square+=__shfl_down_sync(0xffffffff,square,shift);
            if(lane==0) {{templates[r]=tpl;norm[r]=square+x.cfm.data[base+r];bias[r]=sign*vhat.data[start+local]+rhs.data[base+r];}}
            ++r;continue;
        }}
        const int c=key/3, direction=key%3;
        if(direction!=0 || row_type.data[base+r]!=0) {{if(lane==0)atomicOr(&d.status.data[world],4);return;}}
        const int aa=x.art_a.data[c], ab=x.art_b.data[c];
        if((aa>=0 && ab>=0 && aa!=ab) || (aa!=art && ab!=art)) {{if(lane==0)atomicOr(&d.status.data[world],2);return;}}
        const int sa=x.shape0.data[c], sb=x.shape1.data[c];
        const int ba=sa>=0?x.shape_body.data[sa]:-1, bb=sb>=0?x.shape_body.data[sb]:-1;
        const int ta=ba>=0?p.body_tag.data[ba]:0, tb=bb>=0?p.body_tag.data[bb]:0;
        if(ta<0 || tb<0) {{if(lane==0)atomicOr(&d.status.data[world],3);return;}}
        const int tpl=p.pair_support.data[ta*4+tb], length=p.support_count.data[tpl];
        const wp::vec3 normal=-x.normal.data[c];
        const wp::vec3 pa=(ba>=0?wp::transform_point(x.body_q.data[ba],x.point0.data[c]):x.point0.data[c])-x.thickness0.data[c]*normal;
        const wp::vec3 pb=(bb>=0?wp::transform_point(x.body_q.data[bb],x.point1.data[c]):x.point1.data[c])+x.thickness1.data[c]*normal;
        const wp::vec3 anchor=0.5f*(pa+pb), pan=x.shared_anchor?anchor:pa, pbn=x.shared_anchor?anchor:pb;
        const wp::vec3 pat=(x.shared_anchor||x.friction_anchor)?anchor:pa, pbt=(x.shared_anchor||x.friction_anchor)?anchor:pb;
        wp::vec3 t0=wp::cross(normal,wp::vec3(1.0f,0.0f,0.0f));
        if(wp::length_sq(t0)<1.0e-12f)t0=wp::cross(normal,wp::vec3(0.0f,1.0f,0.0f));
        t0=wp::normalize(t0);const wp::vec3 t1=wp::normalize(wp::cross(normal,t0));
        float jn=0.0f,jt0=0.0f,jt1=0.0f;
        const int node=lane<length?p.support_nodes.data[tpl*18+lane]:-1, dof=42-node;
        if(lane<length) {{
            const wp::spatial_vector screw=x.screw.data[start+dof];
            const wp::vec3 lin(screw[0],screw[1],screw[2]),ang(screw[3],screw[4],screw[5]);
            const unsigned long long bit=1ull<<dof;
            if(ba>=0 && aa==art && (p.body_mask.data[ba]&bit)) {{
                const wp::vec3 vn=lin+wp::cross(ang,pan-x.origin.data[art]),vt=lin+wp::cross(ang,pat-x.origin.data[art]);
                jn+=wp::dot(normal,vn);jt0+=wp::dot(t0,vt);jt1+=wp::dot(t1,vt);
            }}
            if(bb>=0 && ab==art && (p.body_mask.data[bb]&bit)) {{
                const wp::vec3 vn=lin+wp::cross(ang,pbn-x.origin.data[art]),vt=lin+wp::cross(ang,pbt-x.origin.data[art]);
                jn-=wp::dot(normal,vn);jt0-=wp::dot(t0,vt);jt1-=wp::dot(t1,vt);
            }}
        }}
        float zn=0.0f,zt0=0.0f,zt1=0.0f;
        for(int k=0;k<length;++k) {{
            const float j0=__shfl_sync(0xffffffff,jn,k),j1=__shfl_sync(0xffffffff,jt0,k),j2=__shfl_sync(0xffffffff,jt1,k);
            if(lane<length) {{const int col=p.support_nodes.data[tpl*18+k],e=p.index.data[node*43+col];if(e>=0) {{const float a=d.W.data[group*434+e];zn+=a*j0;zt0+=a*j1;zt1+=a*j2;}}}}
        }}
        const int directions=(r+1<count && row_type.data[base+r+1]==2 && parent.data[base+r+1]==r)?3:1;
        for(int a=0;a<directions;++a) {{
            const int row=r+a;
            if(d.support.data[base+row]!=3*c+a) {{if(lane==0)atomicOr(&d.status.data[world],4);return;}}
            const float z=a==0?zn:(a==1?zt0:zt1),j=a==0?jn:(a==1?jt0:jt1);
            if(lane<18)zrows[row*18+lane]=z;
            float square=z*z,incident=lane<length?j*vhat.data[start+dof]:0.0f;
            for(int shift=16;shift>0;shift>>=1) {{square+=__shfl_down_sync(0xffffffff,square,shift);incident+=__shfl_down_sync(0xffffffff,incident,shift);}}
            if(lane==0) {{
                templates[row]=tpl;norm[row]=square+x.cfm.data[base+row];
                float b=rhs.data[base+row];
                if(a==0) {{
                    const float velocity=incident-x.target.data[base+row],e=x.restitution.data[base+row],phi=x.phi.data[base+row];
                    if(e>0.0f && velocity<-x.threshold && (phi<={float(_FPGS_CONTACT_END_GAP_SLOP):.9g}f || phi+x.dt*velocity<={float(_FPGS_CONTACT_END_GAP_SLOP):.9g}f)) b=-x.target.data[base+row]+e*velocity;
                }}
                bias[row]=incident+b;
            }}
        }}
        r+=directions;
    }}
    for(int k=lane;k<43;k+=32)du[k]=0.0f;
    for(int r=lane;r<count;r+=32)lam[r]=impulses.data[base+r];
    __syncwarp();
    for(int iteration=0;iteration<iterations;++iteration) {{
        int changed=0;
        for(int row=0;row<count;++row) {{
            const int type=row_type.data[base+row];
            if(type==2 && iteration<friction_start) {{if(lane==0)lam[row]=0.0f;__syncwarp();continue;}}
            const float denom=norm[row];if(!(denom>0.0f))continue;
            const int tpl=templates[row],length=p.support_count.data[tpl],node=lane<length?p.support_nodes.data[tpl*18+lane]:-1;
            const float z=lane<length?zrows[row*18+lane]:0.0f;
            float dot=lane<length?z*du[node]:0.0f;
            for(int s=16;s>0;s>>=1)dot+=__shfl_down_sync(0xffffffff,dot,s);
            dot=__shfl_sync(0xffffffff,dot,0);
            float delta=0.0f,sibling_delta=0.0f;int sibling=-1;
            if(lane==0) {{
                const float old=lam[row],residual=dot+bias[row];
                float next=old+omega*(-residual/denom);
                if(type==0 || type==3)next=fmaxf(next,0.0f);
                else if(type==2) {{
                    const int par=parent.data[base+row];const float radius=fmaxf(mu.data[base+row]*lam[par],0.0f);
                    if(radius<=0.0f)next=0.0f;
                    else {{
                        sibling=row==par+1?par+2:par+1;
                        const float other=lam[sibling],mag=sqrtf(next*next+other*other);
                        if(mag>radius) {{const float scale=radius/mag;next*=scale;const float adjusted=other*scale;sibling_delta=adjusted-other;lam[sibling]=adjusted;}}
                    }}
                }}
                delta=next-old;lam[row]=next;
            }}
            delta=__shfl_sync(0xffffffff,delta,0);sibling_delta=__shfl_sync(0xffffffff,sibling_delta,0);sibling=__shfl_sync(0xffffffff,sibling,0);
            __syncwarp();
            if(sibling_delta!=0.0f) {{const int st=templates[sibling],sn=p.support_count.data[st];if(lane<sn)du[p.support_nodes.data[st*18+lane]]+=zrows[sibling*18+lane]*sibling_delta;changed=1;}}
            __syncwarp();
            if(delta!=0.0f) {{if(lane<length)du[node]+=z*delta;changed=1;}}
            __syncwarp();
        }}
        if(iteration>=friction_start && __ballot_sync(0xffffffff,changed)!=0u)continue;
        if(iteration>=friction_start)break;
    }}
    for(int col=lane;col<43;col+=32) {{
        float value=0.0f;
        for(int k=0;k<p.inverse_count.data[col];++k) {{const int row=p.inverse_nodes.data[col*18+k];value+=d.W.data[group*434+p.index.data[row*43+col]]*du[row];}}
        vout.data[start+42-col]=vhat.data[start+42-col]+value;
    }}
    for(int r=lane;r<count;r+=32)impulses.data[base+r]=lam[r];
#endif
"""


def cpu_source(capacity):
    """Exercise actual packet geometry, sparse rows and applied-delta law on CPU."""
    return f"""
#if !defined(__CUDA_ARCH__)
    const int art=p.group_to_art.data[group], world=p.art_to_world.data[art];
    const int start=p.art_dof_start.data[art], count=counts.data[world], base=world*{capacity};
    if(!d.valid.data[world] || d.status.data[world] || count>{capacity})return;
    float zrows[{capacity}][18]={{}},du[43]={{}},lam[{capacity}]={{}},norm[{capacity}]={{}},bias[{capacity}]={{}};
    int templates[{capacity}]={{}};
    for(int r=0;r<count;++r) {{
        const int type=row_type.data[base+r],par=parent.data[base+r];
        bool bad=type!=0 && type!=2 && type!=3;
        if(type==2) {{
            if(par<0 || par+2>=count || (r!=par+1 && r!=par+2))bad=true;
            else if(row_type.data[base+par]!=0 || row_type.data[base+par+1]!=2 || row_type.data[base+par+2]!=2 || parent.data[base+par+1]!=par || parent.data[base+par+2]!=par)bad=true;
        }}
        if(bad) {{d.status.data[world]|=4;return;}}
        lam[r]=impulses.data[base+r];
    }}
    for(int r=0;r<count;) {{
        const int key=d.support.data[base+r];
        if(key<0) {{
            const int candidate=-key-1,local=candidate/2;
            if(candidate>=86 || row_type.data[base+r]!=3) {{d.status.data[world]|=4;return;}}
            const float sign=(candidate&1)?-1.0f:1.0f;
            const int tpl=p.limit_support.data[local],length=p.support_count.data[tpl];
            float square=0.0f;
            for(int k=0;k<length;++k) {{
                const int node=p.support_nodes.data[tpl*18+k],e=p.index.data[node*43+42-local];
                const float z=e>=0?sign*d.W.data[group*434+e]:0.0f;
                zrows[r][k]=z;square+=z*z;
            }}
            templates[r]=tpl;norm[r]=square+x.cfm.data[base+r];bias[r]=sign*vhat.data[start+local]+rhs.data[base+r];
            ++r;continue;
        }}
        const int c=key/3,direction=key%3;
        if(direction!=0 || row_type.data[base+r]!=0) {{d.status.data[world]|=4;return;}}
        const int aa=x.art_a.data[c],ab=x.art_b.data[c];
        if((aa>=0 && ab>=0 && aa!=ab) || (aa!=art && ab!=art)) {{d.status.data[world]|=2;return;}}
        const int sa=x.shape0.data[c],sb=x.shape1.data[c];
        const int ba=sa>=0?x.shape_body.data[sa]:-1,bb=sb>=0?x.shape_body.data[sb]:-1;
        const int ta=ba>=0?p.body_tag.data[ba]:0,tb=bb>=0?p.body_tag.data[bb]:0;
        if(ta<0 || tb<0) {{d.status.data[world]|=3;return;}}
        const int tpl=p.pair_support.data[ta*4+tb],length=p.support_count.data[tpl];
        const wp::vec3 normal=-x.normal.data[c];
        const wp::vec3 pa=(ba>=0?wp::transform_point(x.body_q.data[ba],x.point0.data[c]):x.point0.data[c])-x.thickness0.data[c]*normal;
        const wp::vec3 pb=(bb>=0?wp::transform_point(x.body_q.data[bb],x.point1.data[c]):x.point1.data[c])+x.thickness1.data[c]*normal;
        const wp::vec3 anchor=0.5f*(pa+pb),pan=x.shared_anchor?anchor:pa,pbn=x.shared_anchor?anchor:pb;
        const wp::vec3 pat=(x.shared_anchor||x.friction_anchor)?anchor:pa,pbt=(x.shared_anchor||x.friction_anchor)?anchor:pb;
        wp::vec3 t0=wp::cross(normal,wp::vec3(1.0f,0.0f,0.0f));
        if(wp::length_sq(t0)<1.0e-12f)t0=wp::cross(normal,wp::vec3(0.0f,1.0f,0.0f));
        t0=wp::normalize(t0);const wp::vec3 t1=wp::normalize(wp::cross(normal,t0));
        float j[3][18]={{}};
        for(int k=0;k<length;++k) {{
            const int dof=42-p.support_nodes.data[tpl*18+k];
            const wp::spatial_vector screw=x.screw.data[start+dof];
            const wp::vec3 lin(screw[0],screw[1],screw[2]),ang(screw[3],screw[4],screw[5]);
            const unsigned long long bit=1ull<<dof;
            if(ba>=0 && aa==art && (p.body_mask.data[ba]&bit)) {{
                const wp::vec3 vn=lin+wp::cross(ang,pan-x.origin.data[art]),vt=lin+wp::cross(ang,pat-x.origin.data[art]);
                j[0][k]+=wp::dot(normal,vn);j[1][k]+=wp::dot(t0,vt);j[2][k]+=wp::dot(t1,vt);
            }}
            if(bb>=0 && ab==art && (p.body_mask.data[bb]&bit)) {{
                const wp::vec3 vn=lin+wp::cross(ang,pbn-x.origin.data[art]),vt=lin+wp::cross(ang,pbt-x.origin.data[art]);
                j[0][k]-=wp::dot(normal,vn);j[1][k]-=wp::dot(t0,vt);j[2][k]-=wp::dot(t1,vt);
            }}
        }}
        const int directions=(r+1<count && row_type.data[base+r+1]==2 && parent.data[base+r+1]==r)?3:1;
        for(int a=0;a<directions;++a) {{
            const int row=r+a;
            if(d.support.data[base+row]!=3*c+a) {{d.status.data[world]|=4;return;}}
            float square=0.0f,incident=0.0f;
            for(int k=0;k<length;++k) {{
                const int node=p.support_nodes.data[tpl*18+k];float z=0.0f;
                for(int t=0;t<length;++t) {{const int col=p.support_nodes.data[tpl*18+t],e=p.index.data[node*43+col];if(e>=0)z+=d.W.data[group*434+e]*j[a][t];}}
                zrows[row][k]=z;square+=z*z;incident+=j[a][k]*vhat.data[start+42-node];
            }}
            templates[row]=tpl;norm[row]=square+x.cfm.data[base+row];float b=rhs.data[base+row];
            if(a==0) {{
                const float velocity=incident-x.target.data[base+row],e=x.restitution.data[base+row],phi=x.phi.data[base+row];
                if(e>0.0f && velocity<-x.threshold && (phi<={float(_FPGS_CONTACT_END_GAP_SLOP):.9g}f || phi+x.dt*velocity<={float(_FPGS_CONTACT_END_GAP_SLOP):.9g}f))b=-x.target.data[base+row]+e*velocity;
            }}
            bias[row]=incident+b;
        }}
        r+=directions;
    }}
    for(int iteration=0;iteration<iterations;++iteration) {{
        bool changed=false;
        for(int row=0;row<count;++row) {{
            const int type=row_type.data[base+row];
            if(type==2 && iteration<friction_start) {{lam[row]=0.0f;continue;}}
            const float denom=norm[row];if(!(denom>0.0f))continue;
            const int tpl=templates[row],length=p.support_count.data[tpl];float dot=0.0f;
            for(int k=0;k<length;++k)dot+=zrows[row][k]*du[p.support_nodes.data[tpl*18+k]];
            const float old=lam[row],residual=dot+bias[row];float next=old+omega*(-residual/denom),sibling_delta=0.0f;int sibling=-1;
            if(type!=2)next=wp::max(next,0.0f);
            else {{
                const int par=parent.data[base+row];const float radius=wp::max(mu.data[base+row]*lam[par],0.0f);
                if(radius<=0.0f)next=0.0f;
                else {{
                    sibling=row==par+1?par+2:par+1;const float other=lam[sibling],mag=sqrtf(next*next+other*other);
                    if(mag>radius) {{const float scale=radius/mag;next*=scale;const float adjusted=other*scale;sibling_delta=adjusted-other;lam[sibling]=adjusted;}}
                }}
            }}
            const float delta=next-old;lam[row]=next;
            if(sibling_delta!=0.0f) {{const int st=templates[sibling],n=p.support_count.data[st];for(int k=0;k<n;++k)du[p.support_nodes.data[st*18+k]]+=zrows[sibling][k]*sibling_delta;changed=true;}}
            if(delta!=0.0f) {{for(int k=0;k<length;++k)du[p.support_nodes.data[tpl*18+k]]+=zrows[row][k]*delta;changed=true;}}
        }}
        if(iteration>=friction_start && !changed)break;
    }}
    for(int col=0;col<43;++col) {{
        float value=0.0f;for(int k=0;k<p.inverse_count.data[col];++k) {{const int row=p.inverse_nodes.data[col*18+k];value+=d.W.data[group*434+p.index.data[row*43+col]]*du[row];}}
        vout.data[start+42-col]=vhat.data[start+42-col]+value;
    }}
    for(int r=0;r<count;++r)impulses.data[base+r]=lam[r];
#endif
"""


@cache
def get_solve_kernel(capacity):
    """Return the complete packet-to-velocity owner for the current capacity."""
    source = cuda_source(capacity) + cpu_source(capacity)

    @wp.func_native(source)
    def native(
        group: int,
        p: SparsePlan,
        d: SparseData,
        x: PacketInput,
        counts: wp.array[int],
        rhs: wp.array2d[float],
        impulses: wp.array2d[float],
        row_type: wp.array2d[int],
        parent: wp.array2d[int],
        mu: wp.array2d[float],
        iterations: int,
        omega: float,
        friction_start: int,
        vhat: wp.array[float],
        vout: wp.array[float],
    ): ...

    def solve(
        p: SparsePlan,
        d: SparseData,
        x: PacketInput,
        counts: wp.array[int],
        rhs: wp.array2d[float],
        impulses: wp.array2d[float],
        row_type: wp.array2d[int],
        parent: wp.array2d[int],
        mu: wp.array2d[float],
        iterations: int,
        omega: float,
        friction_start: int,
        vhat: wp.array[float],
        vout: wp.array[float],
    ):
        group, _ = wp.tid()
        native(
            group, p, d, x, counts, rhs, impulses, row_type, parent, mu, iterations, omega, friction_start, vhat, vout
        )

    solve.__name__ = solve.__qualname__ = f"sparse_packet_gs43_s18_c{capacity}"
    return wp.kernel(enable_backward=False, module="unique")(solve)


def install(owner):
    """Bind packet producers and the complete local-row solve after admission."""
    owner.packet_rows = True
    owner.packet_input = PacketInput()
    owner.kernels.prefix = get_prefix_kernel()
    owner.kernels.contacts = packet_contacts
    owner.kernels.solve = get_solve_kernel(owner.solver.dense_max_constraints)


def bind_current(owner, state_in, state_aug, contacts, dt):
    """Bind actual current pointers; no response cache survives the solve."""
    s, x = owner.solver, owner.packet_input
    if contacts is not None:
        for name, source in (
            ("shape0", "shape0"),
            ("shape1", "shape1"),
            ("point0", "point0"),
            ("point1", "point1"),
            ("normal", "normal"),
            ("thickness0", "margin0"),
            ("thickness1", "margin1"),
        ):
            setattr(x, name, getattr(contacts, "rigid_contact_" + source))
    x.shape_body = s.model.shape_body
    x.body_q = state_in.body_q
    x.screw = state_aug.joint_S_s
    x.origin = s.articulation_origin
    x.art_a, x.art_b = s.contact_art_a, s.contact_art_b
    x.cfm, x.phi, x.target, x.restitution = s.row_cfm, s.phi, s.target_velocity, s.row_restitution
    x.dt, x.threshold = dt, s._effective_restitution_velocity_threshold
    x.shared_anchor, x.friction_anchor = int(s.contact_shared_anchor), int(s.contact_friction_shared_anchor)


def solve(owner, rhs, iterations, omega, friction_start):
    """Launch the sole current-row and finite-eight consumer."""
    s = owner.solver
    wp.launch_tiled(
        owner.kernels.solve,
        dim=[s.world_count],
        inputs=[
            owner.plan,
            owner.data,
            owner.packet_input,
            s.constraint_count,
            rhs,
            s.impulses,
            s.row_type,
            s.row_parent,
            s.row_mu,
            iterations,
            omega,
            friction_start,
            s.v_hat,
            s.v_out,
        ],
        block_dim=32,
        device=s.model.device,
    )
