# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Current sparse rows and original finite-eight law for the checked G1 owner."""

from functools import cache

import warp as wp

from .kernels import contact_restitution_fires
from .sparse_factor import SparseData, SparsePlan


@cache
def get_parallel_limit_kernel(skip_selected=False):
    """Emit original global rows with three stable active-candidate ballots."""
    source = r"""
    const int art=p.group_to_art.data[group], world=p.art_to_world.data[art];
    const int start=p.art_dof_start.data[art], capacity=row_type.shape[1], base=world*capacity;
    auto emit=[&](int candidate,int row,float value) {
        if(row>=capacity)return;
        const int local=candidate/2, tpl=p.limit_support.data[local], at=base+row;
        const float sign=(candidate&1)?-1.0f:1.0f;
        float norm=0.0f;
        for(int k=0;k<18;++k) {
            const int node=p.support_nodes.data[tpl*18+k];
            float z=0.0f;
            if(node>=0) {
                const int entry=p.index.data[node*43+42-local];
                if(entry>=0)z=sign*d.W.data[group*434+entry];
            }
            d.Z.data[at*18+k]=z;norm+=z*z;
        }
        d.support.data[at]=tpl; d.incident.data[at]=sign*vhat.data[start+local];
        diagonal.data[at]=norm+cfm;
        row_type.data[at]=3;parent.data[at]=-1;mu.data[at]=0.0f;
        row_beta.data[at]=beta;row_cfm.data[at]=cfm;
        phi.data[at]=value;target.data[at]=0.0f;
    };
    int count=0;
#if defined(__CUDA_ARCH__)
    const int lane=threadIdx.x&31;
    for(int batch=0;batch<3;++batch) {
        const int candidate=batch*32+lane, local=candidate/2, side=candidate&1;
        bool active=false;float value=0.0f;
        if(enabled && candidate<86) {
            const int dof=start+local, qi=limit_q.data[dof];
            if(qi>=0) {
                const float bound=side?upper.data[dof]:lower.data[dof], position=q.data[qi];
                value=(side?-1.0f:1.0f)*(position-bound);
                active=isfinite(bound) && (side?position>=bound-gap:position<=bound+gap);
            }
        }
        const unsigned mask=__ballot_sync(0xffffffff,active);
        const int row=count+__popc(mask&((1u<<lane)-1u));count+=__popc(mask);
        if(active)emit(candidate,row,value);
    }
    if(lane==0) {
        counter.data[world]=count;
        phase.data[world*phase.shape[1]]=count;phase.data[world*phase.shape[1]+1]=count;
    }
#else
    if(enabled)for(int candidate=0;candidate<86;++candidate) {
        const int local=candidate/2,side=candidate&1,dof=start+local,qi=limit_q.data[dof];
        if(qi<0)continue;
        const float bound=side?upper.data[dof]:lower.data[dof],position=q.data[qi];
        if(!isfinite(bound) || !(side?position>=bound-gap:position<=bound+gap))continue;
        emit(candidate,count++,(side?-1.0f:1.0f)*(position-bound));
    }
    counter.data[world]=count;
    phase.data[world*phase.shape[1]]=count;phase.data[world*phase.shape[1]+1]=count;
#endif
"""

    if skip_selected:
        anchor = "    auto emit=[&](int candidate,int row,float value) {"
        assert source.count(anchor) == 1
        source = source.replace(anchor, "    if(d.selected.data[world]) return;\n" + anchor)

    @wp.func_native(source)
    def native(
        group: int,
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
            vhat,
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
            diagonal,
            phase,
        )

    prefix.__name__ = prefix.__qualname__ = "sparse_factor_parallel_limit_prefix43" + (
        "_fallback" if skip_selected else ""
    )
    return wp.kernel(enable_backward=False, module="unique")(prefix)


@wp.kernel(enable_backward=False)
def build_limit_prefix(
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
    """Emit the original lower/upper candidate order directly as W columns."""
    group = wp.tid()
    art = p.group_to_art[group]
    world = p.art_to_world[art]
    start = p.art_dof_start[art]
    count = int(0)
    if enabled != 0:
        for candidate in range(86):
            local = candidate // 2
            side = candidate % 2
            dof = start + local
            qi = limit_q[dof]
            if qi < 0:
                continue
            bound = lower[dof]
            sign = float(1.0)
            if side != 0:
                bound = upper[dof]
                sign = -1.0
            value = sign * (q[qi] - bound)
            active = wp.isfinite(bound) and (
                (side == 0 and q[qi] <= bound + gap) or (side != 0 and q[qi] >= bound - gap)
            )
            if not active:
                continue
            row = count
            count += 1
            if row >= row_type.shape[1]:
                continue
            template = p.limit_support[local]
            d.support[world, row] = template
            norm = float(0.0)
            for k in range(18):
                node = p.support_nodes[template, k]
                z = float(0.0)
                if node >= 0:
                    entry = p.index[node, 42 - local]
                    if entry >= 0:
                        z = sign * d.W[group, entry]
                d.Z[world, row, k] = z
                norm += z * z
            diagonal[world, row] = norm + cfm
            d.incident[world, row] = sign * vhat[dof]
            row_type[world, row] = 3
            parent[world, row] = -1
            mu[world, row] = 0.0
            row_beta[world, row] = beta
            row_cfm[world, row] = cfm
            phi[world, row] = value
            target[world, row] = 0.0
    counter[world] = count
    phase[world, 0] = count
    phase[world, 1] = count


@wp.kernel(enable_backward=False)
def apply_restitution(
    d: SparseData,
    counts: wp.array[int],
    row_type: wp.array2d[int],
    phi: wp.array2d[float],
    target: wp.array2d[float],
    restitution: wp.array2d[float],
    dt: float,
    threshold: float,
    rhs: wp.array2d[float],
):
    """Retain the one-shot original incident restitution target without dense J."""
    world, row = wp.tid()
    if row >= counts[world] or row_type[world, row] != 0:
        return
    e = restitution[world, row]
    incident = d.incident[world, row] - target[world, row]
    if e > 0.0 and contact_restitution_fires(phi[world, row], incident, dt, threshold):
        rhs[world, row] = -target[world, row] + e * incident


@cache
def get_contact_kernel(skip_selected=False):
    """Hoist current contact geometry and apply W to three sparse directions."""
    source = r"""
#if defined(__CUDA_ARCH__)
    const int lane = threadIdx.x & 31;
    const int capacity = d.Z.shape[1];
    for (int c = worker; c < count.data[0] && c < path.shape[0]; c += workers) {
        if (path.data[c] != 0 || slot.data[c] < 0) continue;
        const int w = world.data[c], row0 = slot.data[c];
        if (row0 >= capacity || !d.valid.data[w]) continue;
        const int aa = art_a.data[c], ab = art_b.data[c];
        if (aa >= 0 && ab >= 0 && aa != ab) {
            if (lane == 0) d.status.data[w] = 2;
            continue;
        }
        const int art = aa >= 0 ? aa : ab;
        if (art < 0) continue;
        const int group = group_of_art.data[art], start = p.art_dof_start.data[art];
        const int sa = shape0.data[c], sb = shape1.data[c];
        const int ba = sa >= 0 ? shape_body.data[sa] : -1;
        const int bb = sb >= 0 ? shape_body.data[sb] : -1;
        const int ta = ba >= 0 ? p.body_tag.data[ba] : 0;
        const int tb = bb >= 0 ? p.body_tag.data[bb] : 0;
        if (ta < 0 || tb < 0) { if (lane == 0) d.status.data[w] = 3; continue; }
        const int tpl = p.pair_support.data[ta*4+tb], length = p.support_count.data[tpl];
        const wp::vec3 normal = -normals.data[c];
        const wp::vec3 pa = (ba >= 0 ? wp::transform_point(body_q.data[ba], point0.data[c]) : point0.data[c]) - thickness0.data[c]*normal;
        const wp::vec3 pb = (bb >= 0 ? wp::transform_point(body_q.data[bb], point1.data[c]) : point1.data[c]) + thickness1.data[c]*normal;
        const wp::vec3 anchor = 0.5f*(pa+pb);
        const wp::vec3 pan = shared_anchor ? anchor : pa, pbn = shared_anchor ? anchor : pb;
        const wp::vec3 pat = (shared_anchor || friction_anchor) ? anchor : pa;
        const wp::vec3 pbt = (shared_anchor || friction_anchor) ? anchor : pb;
        wp::vec3 t0 = wp::cross(normal, wp::vec3(1.0f,0.0f,0.0f));
        if (wp::length_sq(t0) < 1.0e-12f) t0 = wp::cross(normal, wp::vec3(0.0f,1.0f,0.0f));
        t0 = wp::normalize(t0);
        const wp::vec3 t1 = wp::normalize(wp::cross(normal,t0));
        float jn=0.0f, jt0=0.0f, jt1=0.0f;
        const int node = lane < length ? p.support_nodes.data[tpl*18+lane] : -1;
        const int dof = 42-node;
        if (lane < length) {
            const wp::spatial_vector screw = S.data[start+dof];
            const wp::vec3 lin(screw[0],screw[1],screw[2]), ang(screw[3],screw[4],screw[5]);
            const unsigned long long bit = 1ull << dof;
            if (ba >= 0 && aa == art && (p.body_mask.data[ba] & bit)) {
                const wp::vec3 vn = lin + wp::cross(ang,pan-origin.data[art]);
                const wp::vec3 vt = lin + wp::cross(ang,pat-origin.data[art]);
                jn += wp::dot(normal,vn); jt0 += wp::dot(t0,vt); jt1 += wp::dot(t1,vt);
            }
            if (bb >= 0 && ab == art && (p.body_mask.data[bb] & bit)) {
                const wp::vec3 vn = lin + wp::cross(ang,pbn-origin.data[art]);
                const wp::vec3 vt = lin + wp::cross(ang,pbt-origin.data[art]);
                jn -= wp::dot(normal,vn); jt0 -= wp::dot(t0,vt); jt1 -= wp::dot(t1,vt);
            }
        }
        float zn=0.0f, zt0=0.0f, zt1=0.0f;
        for (int k=0; k<length; ++k) {
            const float x=__shfl_sync(0xffffffff,jn,k), y=__shfl_sync(0xffffffff,jt0,k), z=__shfl_sync(0xffffffff,jt1,k);
            if (lane < length) {
                const int col=p.support_nodes.data[tpl*18+k], e=p.index.data[node*43+col];
                if (e >= 0) { const float a=d.W.data[group*434+e]; zn+=a*x; zt0+=a*y; zt1+=a*z; }
            }
        }
        const int directions=needed.data[c];
        for (int r=0; r<directions && r<3; ++r) {
            const int row=row0+r;
            if (row>=capacity) break;
            const float z=r==0?zn:(r==1?zt0:zt1), j=r==0?jn:(r==1?jt0:jt1);
            if (lane<18) d.Z.data[(w*capacity+row)*18+lane]=z;
            float norm=z*z, incident=lane<length?j*vhat.data[start+dof]:0.0f;
            for (int shift=16;shift>0;shift>>=1) { norm+=__shfl_down_sync(0xffffffff,norm,shift); incident+=__shfl_down_sync(0xffffffff,incident,shift); }
            if (lane==0) { d.support.data[w*capacity+row]=tpl; d.incident.data[w*capacity+row]=incident; diagonal.data[w*capacity+row]=norm+cfm.data[w*capacity+row]; }
        }
    }
#endif
"""

    if skip_selected:
        anchor = "        const int w = world.data[c], row0 = slot.data[c];"
        assert source.count(anchor) == 1
        source = source.replace(anchor, anchor + "\n        if(d.selected.data[w]) continue;")

    @wp.func_native(source)
    def native(
        worker: int,
        workers: int,
        p: SparsePlan,
        d: SparseData,
        count: wp.array[int],
        path: wp.array[int],
        slot: wp.array[int],
        world: wp.array[int],
        art_a: wp.array[int],
        art_b: wp.array[int],
        needed: wp.array[int],
        shape0: wp.array[int],
        shape1: wp.array[int],
        point0: wp.array[wp.vec3],
        point1: wp.array[wp.vec3],
        normals: wp.array[wp.vec3],
        thickness0: wp.array[float],
        thickness1: wp.array[float],
        shape_body: wp.array[int],
        body_q: wp.array[wp.transform],
        S: wp.array[wp.spatial_vector],
        origin: wp.array[wp.vec3],
        group_of_art: wp.array[int],
        vhat: wp.array[float],
        shared_anchor: int,
        friction_anchor: int,
        cfm: wp.array2d[float],
        diagonal: wp.array2d[float],
    ): ...

    def contacts(
        workers: int,
        p: SparsePlan,
        d: SparseData,
        count: wp.array[int],
        path: wp.array[int],
        slot: wp.array[int],
        world: wp.array[int],
        art_a: wp.array[int],
        art_b: wp.array[int],
        needed: wp.array[int],
        shape0: wp.array[int],
        shape1: wp.array[int],
        point0: wp.array[wp.vec3],
        point1: wp.array[wp.vec3],
        normals: wp.array[wp.vec3],
        thickness0: wp.array[float],
        thickness1: wp.array[float],
        shape_body: wp.array[int],
        body_q: wp.array[wp.transform],
        S: wp.array[wp.spatial_vector],
        origin: wp.array[wp.vec3],
        group_of_art: wp.array[int],
        vhat: wp.array[float],
        shared_anchor: int,
        friction_anchor: int,
        cfm: wp.array2d[float],
        diagonal: wp.array2d[float],
    ):
        worker, _ = wp.tid()
        native(
            worker,
            workers,
            p,
            d,
            count,
            path,
            slot,
            world,
            art_a,
            art_b,
            needed,
            shape0,
            shape1,
            point0,
            point1,
            normals,
            thickness0,
            thickness1,
            shape_body,
            body_q,
            S,
            origin,
            group_of_art,
            vhat,
            shared_anchor,
            friction_anchor,
            cfm,
            diagonal,
        )

    contacts.__name__ = contacts.__qualname__ = "sparse_factor_contact_triplet18" + (
        "_fallback" if skip_selected else ""
    )
    return wp.kernel(enable_backward=False, module="unique")(contacts)


def _contact_block_fragments(capacity: int):
    """Local sticking corrections; retain original scalar transactions on rejection."""
    setup = r"""
    __shared__ float contact_cross[BLOCK_CAPACITY];
    __shared__ unsigned int contact_ready[(BLOCK_CAPACITY+31)/32];
    for(int word=lane;word<(BLOCK_CAPACITY+31)/32;word+=32)contact_ready[word]=0u;
    __syncwarp();
""".replace("BLOCK_CAPACITY", str(capacity))
    update = r"""
            if(type==0 && row+2<count && iteration>=friction_start &&
               isfinite(omega) && omega>0.0f && omega<=1.0f &&
               row_type.data[base+row+1]==2 && row_type.data[base+row+2]==2 &&
               parent.data[base+row+1]==row && parent.data[base+row+2]==row &&
               d.support.data[base+row]==d.support.data[base+row+1] &&
               d.support.data[base+row]==d.support.data[base+row+2]) {
                const float friction=mu.data[base+row+1];
                if(isfinite(friction) && friction>=0.0f && friction==mu.data[base+row+2]) {
                    const int tpl=d.support.data[base+row], length=p.support_count.data[tpl];
                    const int node=lane<length?p.support_nodes.data[tpl*18+lane]:-1;
                    const float z0=lane<length?d.Z.data[(base+row)*18+lane]:0.0f;
                    float r0=lane<length?z0*du[node]:0.0f;
                    for(int shift=16;shift>0;shift>>=1)r0+=__shfl_down_sync(0xffffffff,r0,shift);
                    r0=__shfl_sync(0xffffffff,r0,0)+d.incident.data[base+row]+rhs.data[base+row];
                    // At zero normal radius, zero tangents cannot change the
                    // state. This is the exact original open-contact branch.
                    if(lam[row]==0.0f && lam[row+1]==0.0f && lam[row+2]==0.0f && r0>=0.0f) {
                        row+=2;
                        continue;
                    }
                    const float z1=lane<length?d.Z.data[(base+row+1)*18+lane]:0.0f;
                    const float z2=lane<length?d.Z.data[(base+row+2)*18+lane]:0.0f;
                    // An open contact never consumes the self block. Build it
                    // only on its first non-open visit in this solver call.
                    const int word=row/32;
                    const unsigned int bit=1u<<(row%32);
                    int needs_block=lane==0 && (contact_ready[word]&bit)==0u;
                    needs_block=__shfl_sync(0xffffffff,needs_block,0);
                    if(needs_block) {
                        float a01=z0*z1,a02=z0*z2,a12=z1*z2;
                        for(int shift=16;shift>0;shift>>=1) {
                            a01+=__shfl_down_sync(0xffffffff,a01,shift);
                            a02+=__shfl_down_sync(0xffffffff,a02,shift);
                            a12+=__shfl_down_sync(0xffffffff,a12,shift);
                        }
                        if(lane==0) {
                            contact_cross[row]=a01;contact_cross[row+1]=a02;contact_cross[row+2]=a12;
                            contact_ready[word]|=bit;
                        }
                        __syncwarp();
                    }
                    float r1=lane<length?z1*du[node]:0.0f, r2=lane<length?z2*du[node]:0.0f;
                    for(int shift=16;shift>0;shift>>=1) {
                        r1+=__shfl_down_sync(0xffffffff,r1,shift);
                        r2+=__shfl_down_sync(0xffffffff,r2,shift);
                    }
                    r1=__shfl_sync(0xffffffff,r1,0)+d.incident.data[base+row+1]+rhs.data[base+row+1];
                    r2=__shfl_sync(0xffffffff,r2,0)+d.incident.data[base+row+2]+rhs.data[base+row+2];
                    int accepted=0;
                    float change0=0.0f,change1=0.0f,change2=0.0f;
                    if(lane==0) {
                        const float a00=diagonal.data[base+row],a11=diagonal.data[base+row+1],a22=diagonal.data[base+row+2];
                        const float a01=contact_cross[row],a02=contact_cross[row+1],a12=contact_cross[row+2];
                        const float scale=fmaxf(a00,fmaxf(a11,a22));
                        const float floor=1.0e-7f*scale;
                        if(isfinite(scale) && a00>floor) {
                            const float l10=a01/a00,l20=a02/a00,d1=a11-a01*l10;
                            if(isfinite(d1) && d1>floor) {
                                const float u12=a12-a02*l10,l21=u12/d1,d2=a22-a02*l20-u12*l21;
                                if(isfinite(d2) && d2>floor) {
                                    // Existing diagonal includes denominator-only
                                    // CFM; the current residual does not contain CFM*lambda.
                                    const float y0=-r0,y1=-r1-l10*y0,y2=-r2-l20*y0-l21*y1;
                                    const float delta2=y2/d2,delta1=y1/d1-l21*delta2;
                                    const float delta0=y0/a00-l10*delta1-l20*delta2;
                                    const float n0=lam[row]+omega*delta0,n1=lam[row+1]+omega*delta1,n2=lam[row+2]+omega*delta2;
                                    const float radius=friction*n0;
                                    if(isfinite(n0) && isfinite(n1) && isfinite(n2) &&
                                       isfinite(radius) && n0>=0.0f && hypotf(n1,n2)<=radius) {
                                        change0=n0-lam[row];change1=n1-lam[row+1];change2=n2-lam[row+2];
                                        lam[row]=n0;lam[row+1]=n1;lam[row+2]=n2;accepted=1;
                                    }
                                }
                            }
                        }
                    }
                    accepted=__shfl_sync(0xffffffff,accepted,0);
                    if(accepted) {
                        change0=__shfl_sync(0xffffffff,change0,0);
                        change1=__shfl_sync(0xffffffff,change1,0);
                        change2=__shfl_sync(0xffffffff,change2,0);
                        if(change0!=0.0f || change1!=0.0f || change2!=0.0f) {
                            if(lane<length)du[node]+=z0*change0+z1*change1+z2*change2;
                            changed=1;
                        }
                        __syncwarp();
                        row+=2;
                        continue;
                    }
                    // Rejection has not touched lambda or kinetic velocity.
                    // The following original scalar rows own all slip/fallback work.
                }
            }
"""
    return setup, update


@cache
def get_solve_kernel(
    capacity: int, block_contacts: bool = False, *, metric_tangents: bool = False, skip_selected: bool = False
):
    """Apply original current-friction GS and decode the complete43 velocity."""
    source = f"""
#if defined(__CUDA_ARCH__)
    const int lane=threadIdx.x&31, art=p.group_to_art.data[group], world=p.art_to_world.data[art];
    const int start=p.art_dof_start.data[art], count=counts.data[world], base=world*{capacity};
    if (!d.valid.data[world] || d.status.data[world] || count>{capacity}) return;
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
    __shared__ float du[43], lam[{capacity}];
    for(int k=lane;k<43;k+=32) du[k]=0.0f;
    for(int r=lane;r<count;r+=32) lam[r]=impulses.data[base+r];
    __syncwarp();
    for(int iteration=0;iteration<iterations;++iteration) {{
        int changed=0;
        for(int row=0;row<count;++row) {{
            const int type=row_type.data[base+row];
            if(type==2 && iteration<friction_start) {{ if(lane==0)lam[row]=0.0f; __syncwarp(); continue; }}
            const float denom=diagonal.data[base+row];
            if(!(denom>0.0f))continue;
            const int tpl=d.support.data[base+row], length=p.support_count.data[tpl];
            const int node=lane<length?p.support_nodes.data[tpl*18+lane]:-1;
            const float z=lane<length?d.Z.data[(base+row)*18+lane]:0.0f;
            float dot=lane<length?z*du[node]:0.0f;
            for(int s=16;s>0;s>>=1)dot+=__shfl_down_sync(0xffffffff,dot,s);
            dot=__shfl_sync(0xffffffff,dot,0);
            float delta=0.0f,sibling_delta=0.0f;
            int sibling=-1;
            if(lane==0) {{
                const float old=lam[row], residual=dot+d.incident.data[base+row]+rhs.data[base+row];
                float next=old+omega*(-residual/denom);
                if(type==0 || type==3)next=fmaxf(next,0.0f);
                else if(type==2) {{
                    const int par=parent.data[base+row];
                    const float radius=fmaxf(mu.data[base+row]*lam[par],0.0f);
                    if(radius<=0.0f)next=0.0f;
                    else {{
                        sibling=row==par+1?par+2:par+1;
                        const float other=lam[sibling], mag=sqrtf(next*next+other*other);
                        if(mag>radius) {{ const float scale=radius/mag;next*=scale;const float adjusted=other*scale;sibling_delta=adjusted-other;lam[sibling]=adjusted; }}
                    }}
                }}
                delta=next-old;lam[row]=next;
            }}
            delta=__shfl_sync(0xffffffff,delta,0);sibling_delta=__shfl_sync(0xffffffff,sibling_delta,0);sibling=__shfl_sync(0xffffffff,sibling,0);
            __syncwarp();
            if(sibling_delta!=0.0f) {{
                const int st=d.support.data[base+sibling], sn=p.support_count.data[st];
                if(lane<sn)du[p.support_nodes.data[st*18+lane]]+=d.Z.data[(base+sibling)*18+lane]*sibling_delta;
                changed=1;
            }}
            __syncwarp();
            if(delta!=0.0f) {{ if(lane<length)du[node]+=z*delta;changed=1; }}
            __syncwarp();
        }}
        if(iteration>=friction_start && __ballot_sync(0xffffffff,changed)!=0u)continue;
        if(iteration>=friction_start)break;
    }}
    for(int col=lane;col<43;col+=32) {{
        float value=0.0f;
        for(int k=0;k<p.inverse_count.data[col];++k) {{ const int row=p.inverse_nodes.data[col*18+k];value+=d.W.data[group*434+p.index.data[row*43+col]]*du[row]; }}
        vout.data[start+42-col]=vhat.data[start+42-col]+value;
    }}
    for(int r=lane;r<count;r+=32)impulses.data[base+r]=lam[r];
#endif
"""
    if skip_selected:
        anchor = "    const int start=p.art_dof_start.data[art], count=counts.data[world], base=world*"
        position = source.index(anchor)
        source = source[:position] + "    if(d.selected.data[world])return;\n" + source[position:]
    if metric_tangents and (block_contacts or capacity != 100):
        raise ValueError("Sparse metric tangents require the exclusive capacity100 owner")
    if block_contacts or metric_tangents:
        if metric_tangents:
            from .sparse_metric_tangents import get_fragments  # noqa: PLC0415

            setup, update = get_fragments(capacity)
        else:
            setup, update = _contact_block_fragments(capacity)
        setup_anchor = "    for(int iteration=0;iteration<iterations;++iteration) {"
        update_anchor = "            if(type==2 && iteration<friction_start)"
        assert source.count(setup_anchor) == source.count(update_anchor) == 1
        source = source.replace(setup_anchor, setup + setup_anchor)
        source = source.replace(update_anchor, update + update_anchor)

    @wp.func_native(source)
    def native(
        group: int,
        p: SparsePlan,
        d: SparseData,
        counts: wp.array[int],
        rhs: wp.array2d[float],
        diagonal: wp.array2d[float],
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
        counts: wp.array[int],
        rhs: wp.array2d[float],
        diagonal: wp.array2d[float],
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
            group,
            p,
            d,
            counts,
            rhs,
            diagonal,
            impulses,
            row_type,
            parent,
            mu,
            iterations,
            omega,
            friction_start,
            vhat,
            vout,
        )

    name = (
        "sparse_metric_tangent" if metric_tangents else "sparse_contact_block" if block_contacts else "sparse_factor_gs"
    )
    solve.__name__ = solve.__qualname__ = f"{name}43_s18_c{capacity}" + ("_fallback" if skip_selected else "")
    return wp.kernel(enable_backward=False, module="unique")(solve)
