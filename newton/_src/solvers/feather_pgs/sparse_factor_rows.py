# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Current sparse rows and original finite-eight law for the checked G1 owner."""

from functools import cache

import warp as wp

from .kernels import contact_restitution_fires
from .sparse_factor import SparseData, SparsePlan


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
def get_contact_kernel():
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

    contacts.__name__ = contacts.__qualname__ = "sparse_factor_contact_triplet18"
    return wp.kernel(enable_backward=False, module="unique")(contacts)


@cache
def get_solve_kernel(capacity: int, *, minimum_count: int = -1):
    """Apply original current-friction GS and decode the complete43 velocity."""
    source = f"""
#if defined(__CUDA_ARCH__)
    const int lane=threadIdx.x&31, art=p.group_to_art.data[group], world=p.art_to_world.data[art];
    const int start=p.art_dof_start.data[art], count=counts.data[world], base=world*{capacity};
    if(count<={minimum_count})return;
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

    suffix = f"_gt{minimum_count}" if minimum_count >= 0 else ""
    solve.__name__ = solve.__qualname__ = f"sparse_factor_gs43_s18_c{capacity}{suffix}"
    return wp.kernel(enable_backward=False, module="unique")(solve)
