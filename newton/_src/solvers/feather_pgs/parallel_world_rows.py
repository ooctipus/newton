# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Four contact-worker warps followed by the original sparse metric recurrence.

The held operator, current screws and canonical raw IDs keep their original
meaning. Only prediction/row construction scratch is private to this CTA.
"""

from functools import cache

import warp as wp

from .g1_kinetic_state import _PREDICT, ForceInput, KineticData, KineticPlan
from .raw_world_contacts import RawWorldContacts
from .sparse_factor import SparseData, SparsePlan
from .sparse_metric_tangents import get_fragments
from .sparse_packet_rows import PacketInput, cuda_source


@wp.struct
class WorldInput:
    count: wp.array[int]
    body_art: wp.array[int]
    response_dofs: wp.array[int]
    body_response: wp.array[int]
    body_flags: wp.array[int]
    prescribed: wp.array[int]
    body_v: wp.array[wp.spatial_vector]
    material_mu: wp.array[float]
    material_restitution: wp.array[float]
    limit_q: wp.array[int]
    lower: wp.array[float]
    upper: wp.array[float]
    contact_world: wp.array[int]
    contact_slot: wp.array[int]
    contact_path: wp.array[int]
    contact_needed: wp.array[int]
    counter: wp.array[int]
    counts: wp.array[int]
    phase: wp.array2d[int]
    dense_flag: wp.array[int]
    dropped: wp.array[int]
    capacity_status: wp.array[int]
    status: wp.array[int]
    row_type: wp.array2d[int]
    parent: wp.array2d[int]
    mu: wp.array2d[float]
    beta_rows: wp.array2d[float]
    rhs: wp.array2d[float]
    diagonal: wp.array2d[float]
    row_w: wp.array2d[float]
    impulses: wp.array2d[float]
    limits: int
    friction: int
    anchor_limit: int
    pairs_only: int
    telemetry: int
    gap: float
    same_gap: float
    pair_gap: float
    friction_gap: float
    friction_scale: float
    limit_gap: float
    beta: float
    cfm: float
    speculative_scale: float
    contact_w: float


_PREFIX = r"""
    if(warp==0) {
        int total=0;
        for(int batch=0;batch<3;++batch) {
            const int candidate=batch*32+lane,local=candidate/2,side=candidate&1;
            bool active=false;float value=0.0f;
            if(r.limits && candidate<86) {
                const int dof=start+local,qi=r.limit_q.data[dof];
                if(qi>=0) {
                    const float bound=side?r.upper.data[dof]:r.lower.data[dof],position=f.joint_q.data[qi];
                    value=(side?-1.0f:1.0f)*(position-bound);
                    active=isfinite(bound)&&(side?position>=bound-r.limit_gap:position<=bound+r.limit_gap);
                }
            }
            const unsigned mask=__ballot_sync(0xffffffffu,active);
            const int row=total+__popc(mask&((1u<<lane)-1u));total+=__popc(mask);
            if(active) {
                limit_keys[row]=candidate;
                const int at=base+row;
                row_type.data[at]=3;parent.data[at]=-1;mu.data[at]=0.0f;
                r.beta_rows.data[at]=r.beta;x.cfm.data[at]=r.cfm;
                x.phi.data[at]=value;x.target.data[at]=0.0f;x.restitution.data[at]=0.0f;
                rhs.data[at]=(value<0.0f?r.beta*value:value)/x.dt;
            }
        }
        if(lane==0){row_count=total;limit_count=total;r.phase.data[world*r.phase.shape[1]]=total;r.phase.data[world*r.phase.shape[1]+1]=total;}
    }
    __syncthreads();
    // Every warp forms an independent limit response, preserving prefix order.
    for(int row=warp;row<limit_count;row+=4) {
        const int candidate=limit_keys[row],local=candidate/2;
        const float sign=(candidate&1)?-1.0f:1.0f;
        const int tpl=p.limit_support.data[local],length=p.support_count.data[tpl];
        float z=0.0f;
        if(lane<length){const int node=p.support_nodes.data[tpl*18+lane],e=p.index.data[node*43+42-local];if(e>=0)z=sign*d.W.data[group*434+e];}
        if(lane<18)zrows[row*18+lane]=z;
        float square=z*z;
        for(int shift=16;shift>0;shift>>=1)square+=__shfl_down_sync(0xffffffffu,square,shift);
        if(lane==0){templates[row]=tpl;norm[row]=square+r.cfm;bias[row]=sign*vh[local]+rhs.data[base+row];r.diagonal.data[base+row]=norm[row];d.support.data[base+row]=tpl;}
    }
"""


_CONTACTS = r"""
    const int raw_count=r.count.data[0];
    const int begin=buckets.offsets.data[world],end=buckets.offsets.data[world+1];
    for(int entry=begin+warp;entry<end;entry+=4) {
        const int c=buckets.ids.data[entry];
        if(c<0||c>=raw_count){if(lane==0)atomicOr(&r.status.data[0],1);continue;}
        const int sa=x.shape0.data[c],sb=x.shape1.data[c];
        if(sa< -1||sb< -1||sa>=x.shape_body.shape[0]||sb>=x.shape_body.shape[0]){if(lane==0)atomicOr(&r.status.data[0],1);continue;}
        const int ba=sa>=0?x.shape_body.data[sa]:-1,bb=sb>=0?x.shape_body.data[sb]:-1;
        if(ba< -1||bb< -1||ba>=r.body_art.shape[0]||bb>=r.body_art.shape[0]){if(lane==0)atomicOr(&r.status.data[0],1);continue;}
        const int aa=ba>=0?r.body_art.data[ba]:-1,ab=bb>=0?r.body_art.data[bb]:-1;
        if(aa< -1||ab< -1||aa>=p.art_to_world.shape[0]||ab>=p.art_to_world.shape[0]){if(lane==0)atomicOr(&r.status.data[0],1);continue;}
        const int wa=aa>=0?p.art_to_world.data[aa]:-1,wb=ab>=0?p.art_to_world.data[ab]:-1;
        if((wa>=0&&wb>=0&&wa!=wb)||(wa<0&&wb<0)) {
            // The original allocator discards these contacts. One world owns
            // their canonical invalidation even during the slow raw scan.
            if(world==0&&lane==0){r.contact_slot.data[c]=-1;r.contact_path.data[c]=-1;r.contact_needed.data[c]=0;}
            continue;
        }
        if(wp::max(wa,wb)!=world)continue;
        if(lane==0){r.contact_slot.data[c]=-1;r.contact_path.data[c]=-1;r.contact_needed.data[c]=0;}
        const bool responds_a=aa>=0&&r.response_dofs.data[aa]>0&&r.body_response.data[ba]!=0&&(r.body_flags.data[ba]&2)==0;
        const bool responds_b=ab>=0&&r.response_dofs.data[ab]>0&&r.body_response.data[bb]!=0&&(r.body_flags.data[bb]&2)==0;
        if(!responds_a&&!responds_b)continue;
        const wp::vec3 normal=-x.normal.data[c];
        const wp::vec3 pa=(ba>=0?wp::transform_point(x.body_q.data[ba],x.point0.data[c]):x.point0.data[c])-x.thickness0.data[c]*normal;
        const wp::vec3 pb=(bb>=0?wp::transform_point(x.body_q.data[bb],x.point1.data[c]):x.point1.data[c])+x.thickness1.data[c]*normal;
        const float phi=wp::dot(normal,pa-pb);
        if((r.gap>0.0f&&phi>r.gap)||(r.same_gap>0.0f&&aa>=0&&aa==ab&&phi>r.same_gap)||(r.pair_gap>0.0f&&aa>=0&&ab>=0&&phi>r.pair_gap))continue;
        const bool filtered=r.pairs_only==0||(aa>=0&&ab>=0);
        const int anchor_limit=filtered?r.anchor_limit:0;
        int rank=0;
        if(anchor_limit>0)for(int back=1;back<=8;++back){const int previous=c-back;if(previous<0||x.shape0.data[previous]!=sa||x.shape1.data[previous]!=sb)break;++rank;}
        const bool following=anchor_limit>0&&c+1<raw_count&&x.shape0.data[c+1]==sa&&x.shape1.data[c+1]==sb;
        const int directions=r.friction&&(!filtered||phi<=r.friction_gap)&&(anchor_limit==0||rank<anchor_limit)?3:1;
        const int ta=ba>=0?p.body_tag.data[ba]:0,tb=bb>=0?p.body_tag.data[bb]:0;
        if(ta<0||ta>=4||tb<0||tb>=4){if(lane==0)atomicOr(&r.status.data[0],2);continue;}
        int row=0;
        if(lane==0) {
            row=atomicAdd(&row_count,directions);
            if(row+directions>100){atomicAdd(&row_count,-directions);atomicMax(&r.capacity_status.data[0],1);if(r.telemetry)atomicAdd(&r.dropped.data[world],directions);row=-1;}
            r.contact_needed.data[c]=directions;
            if(row>=0){r.contact_world.data[c]=world;r.contact_slot.data[c]=row;r.contact_path.data[c]=0;x.art_a.data[c]=aa;x.art_b.data[c]=ab;r.dense_flag.data[world]=1;}
        }
        row=__shfl_sync(0xffffffffu,row,0);
        if(row<0)continue;
        const int tpl=p.pair_support.data[ta*4+tb],length=p.support_count.data[tpl];
        const wp::vec3 anchor=0.5f*(pa+pb),pan=x.shared_anchor?anchor:pa,pbn=x.shared_anchor?anchor:pb;
        const wp::vec3 pat=(x.shared_anchor||x.friction_anchor)?anchor:pa,pbt=(x.shared_anchor||x.friction_anchor)?anchor:pb;
        wp::vec3 t0=wp::cross(normal,wp::vec3(1.0f,0.0f,0.0f));
        if(wp::length_sq(t0)<1.0e-12f)t0=wp::cross(normal,wp::vec3(0.0f,1.0f,0.0f));
        t0=wp::normalize(t0);const wp::vec3 t1=wp::normalize(wp::cross(normal,t0));
        float jn=0.0f,jt0=0.0f,jt1=0.0f;
        const int node=lane<length?p.support_nodes.data[tpl*18+lane]:-1,dof=42-node;
        if(lane<length) {
            const auto screw=x.screw.data[start+dof];
            const wp::vec3 lin(screw[0],screw[1],screw[2]),ang(screw[3],screw[4],screw[5]);
            const unsigned long long bit=1ull<<dof;
            if(ba>=0&&aa==art&&(p.body_mask.data[ba]&bit)){const auto vn=lin+wp::cross(ang,pan-x.origin.data[art]),vt=lin+wp::cross(ang,pat-x.origin.data[art]);jn+=wp::dot(normal,vn);jt0+=wp::dot(t0,vt);jt1+=wp::dot(t1,vt);}
            if(bb>=0&&ab==art&&(p.body_mask.data[bb]&bit)){const auto vn=lin+wp::cross(ang,pbn-x.origin.data[art]),vt=lin+wp::cross(ang,pbt-x.origin.data[art]);jn-=wp::dot(normal,vn);jt0-=wp::dot(t0,vt);jt1-=wp::dot(t1,vt);}
        }
        float zn=0.0f,zt0=0.0f,zt1=0.0f;
        for(int k=0;k<length;++k){const float j0=__shfl_sync(0xffffffffu,jn,k),j1=__shfl_sync(0xffffffffu,jt0,k),j2=__shfl_sync(0xffffffffu,jt1,k);if(lane<length){const int col=p.support_nodes.data[tpl*18+k],e=p.index.data[node*43+col];if(e>=0){const float a=d.W.data[group*434+e];zn+=a*j0;zt0+=a*j1;zt1+=a*j2;}}}
        for(int direction=0;direction<directions;++direction) {
            const int at=base+row+direction;
            const float z=direction==0?zn:(direction==1?zt0:zt1),j=direction==0?jn:(direction==1?jt0:jt1);
            if(lane<18)zrows[(row+direction)*18+lane]=z;
            float square=z*z,incident=lane<length?j*vh[dof]:0.0f;
            for(int shift=16;shift>0;shift>>=1){square+=__shfl_down_sync(0xffffffffu,square,shift);incident+=__shfl_down_sync(0xffffffffu,incident,shift);}
            if(lane==0) {
                float material_mu=0.0f;int material_count=0;
                if(sa>=0){material_mu+=r.material_mu.data[sa];++material_count;}
                if(sb>=0){material_mu+=r.material_mu.data[sb];++material_count;}
                if(material_count)material_mu/=static_cast<float>(material_count);
                float restitution=0.0f;
                if(sa>=0){const float e=r.material_restitution.data[sa];if(isfinite(e))restitution+=wp::clamp(e,0.0f,1.0f);}
                if(sb>=0){const float e=r.material_restitution.data[sb];if(isfinite(e))restitution+=wp::clamp(e,0.0f,1.0f);}
                if(material_count)restitution/=static_cast<float>(material_count);
                const wp::vec3 axis=direction==0?normal:(direction==1?t0:t1);
                const wp::vec3 point_a=direction==0?pan:pat,point_b=direction==0?pbn:pbt;
                float known=0.0f;
                if(ba>=0&&aa>=0&&r.prescribed.data[aa]){const auto v=r.body_v.data[ba];known+=wp::dot(axis,wp::vec3(v[0],v[1],v[2])+wp::cross(wp::vec3(v[3],v[4],v[5]),point_a-x.origin.data[aa]));}
                if(bb>=0&&ab>=0&&r.prescribed.data[ab]){const auto v=r.body_v.data[bb];known-=wp::dot(axis,wp::vec3(v[0],v[1],v[2])+wp::cross(wp::vec3(v[3],v[4],v[5]),point_b-x.origin.data[ab]));}
                const float target=-known;
                float value=-target;
                if(direction==0)value+=(phi<=0.0f?r.beta*phi:r.speculative_scale*phi)/x.dt;
                const float velocity=incident-target;
                if(direction==0&&restitution>0.0f&&velocity< -x.threshold&&(phi<=END_SLOP||phi+x.dt*velocity<=END_SLOP))value=-target+restitution*velocity;
                row_type.data[at]=direction==0?0:2;parent.data[at]=direction==0?-1:row;
                mu.data[at]=direction==0?material_mu:material_mu*r.friction_scale*((anchor_limit>0&&(rank>0||following))?0.5f:1.0f);
                r.beta_rows.data[at]=direction==0?r.beta:0.0f;x.cfm.data[at]=r.cfm;
                x.phi.data[at]=direction==0?phi:0.0f;x.target.data[at]=target;x.restitution.data[at]=direction==0?restitution:0.0f;
                rhs.data[at]=value;r.diagonal.data[at]=square+r.cfm;
                templates[row+direction]=tpl;norm[row+direction]=square+r.cfm;bias[row+direction]=incident+value;d.support.data[at]=tpl;
            }
        }
    }
"""


def _source(chain_scan):
    """Compose current predictor and the unchanged qualified metric transaction."""
    from .kernels import _FPGS_CONTACT_END_GAP_SLOP  # noqa: PLC0415

    predictor = _PREDICT
    if chain_scan:
        from .g1_chain_scan import predictor_source  # noqa: PLC0415

        predictor = predictor_source(predictor)
        # Padding warps participate in every CTA fence but own no chain slot.
        predictor = predictor.replace(
            "body=plan.chain_meta.data[5*thread]", "body=thread<64?plan.chain_meta.data[5*thread]:-1"
        )
        for offset in (1, 2, 3):
            predictor = predictor.replace(
                f"plan.chain_meta.data[5*thread+{offset}]", f"(thread<64?plan.chain_meta.data[5*thread+{offset}]:0)"
            )
    predictor = predictor.replace("__shared__ float s[350];", "float* s=arena;")
    predictor = predictor.replace("__shared__ int nonzero;", "int& nonzero=nonzero_shared;")
    predictor = predictor.replace("f.vhat.data[dof]", "vh[dof-ds]").replace("f.vhat.data[ds+k]", "vh[k]")
    # Keep predictor qdd as a canonical diagnostic; private vhat has no global
    # publication or load until an explicit diagnostic asks for it.
    solve = cuda_source(100)
    solve = solve[solve.index("    for(int k=lane;k<43;k+=32)du[k]=0.0f;") : solve.rindex("#endif")]
    solve = solve.replace("lam[r]=impulses.data[base+r]", "lam[r]=0.0f")
    solve = solve.replace("vhat.data[start+42-col]", "vh[42-col]")
    setup, update = get_fragments(100)
    for suffix in ("", "+1", "+2"):
        update = update.replace(f"d.support.data[base+row{suffix}]", f"templates[row{suffix}]")
        update = update.replace(f"d.Z.data[(base+row{suffix})*18+lane]", f"zrows[(row{suffix})*18+lane]")
        update = update.replace(f"diagonal.data[base+row{suffix}]", f"norm[row{suffix}]")
        update = update.replace(f"d.incident.data[base+row{suffix}]+rhs.data[base+row{suffix}]", f"bias[row{suffix}]")
    solve = solve.replace(
        "    for(int iteration=0;iteration<iterations;++iteration) {",
        setup + "    for(int iteration=0;iteration<iterations;++iteration) {",
    )
    solve = solve.replace(
        "            if(type==2 && iteration<friction_start)",
        update + "            if(type==2 && iteration<friction_start)",
    )
    return (
        r"""
#if defined(__CUDA_ARCH__)
    const int thread=threadIdx.x,lane=thread&31,warp=thread>>5;
    const int art=p.group_to_art.data[group],world=p.art_to_world.data[art],start=p.art_dof_start.data[art],base=world*100;
    auto row_type=r.row_type;auto parent=r.parent;auto mu=r.mu;auto rhs=r.rhs;auto impulses=r.impulses;
    __shared__ float arena[1800],vh[43],du[43],lam[100],norm[100],bias[100];
    __shared__ int templates[100],limit_keys[86],row_count,limit_count,nonzero_shared;
    float* zrows=arena;
    if(r.count.data[0]<0||r.count.data[0]>x.shape0.shape[0]){if(thread==0){atomicMax(&r.capacity_status.data[3],1);atomicOr(&r.status.data[0],1);}return;}
    if(buckets.invalid.data[0]){if(thread==0)atomicOr(&r.status.data[0],1);return;}
    if(!d.valid.data[world]||d.status.data[world]||!cache.current_valid.data[art]||cache.status.data[group]){if(thread==0)atomicOr(&r.status.data[0],4);return;}
    if(thread==0){r.dense_flag.data[world]=0;if(r.telemetry)r.dropped.data[world]=0;}
    {
PREDICTOR
    }
    __syncthreads();
PREFIX
CONTACTS
    __syncthreads();
    const int count=row_count;
    if(thread==0){r.counter.data[world]=count;r.counts.data[world]=count;}
    if(warp==0) {
SOLVE
        if(lane==0&&world==0)atomicAdd(&r.status.data[1],1);
    }
#endif
""".replace("PREDICTOR", predictor)
        .replace("PREFIX", _PREFIX)
        .replace("CONTACTS", _CONTACTS.replace("END_SLOP", f"{float(_FPGS_CONTACT_END_GAP_SLOP):.9g}f"))
        .replace("SOLVE", solve)
    )


def _producer_source(chain_scan):
    """Keep the same four row workers, releasing their CTA before original GS."""
    source = _source(chain_scan)
    end = "    const int count=row_count;\n"
    assert source.count(end) == 1
    source = source[: source.index(end)]
    substitutions = (
        (
            "__shared__ float arena[1800],vh[43],du[43],lam[100],norm[100],bias[100];",
            "__shared__ float arena[350],vh[43];",
        ),
        (
            "__shared__ int templates[100],limit_keys[86],row_count,limit_count,nonzero_shared;",
            "__shared__ int limit_keys[86],row_count,limit_count,nonzero_shared;",
        ),
        ("    float* zrows=arena;\n", ""),
        ("zrows[row*18+lane]=z", "d.Z.data[(base+row)*18+lane]=z"),
        ("zrows[(row+direction)*18+lane]=z", "d.Z.data[at*18+lane]=z"),
        (
            "templates[row]=tpl;norm[row]=square+r.cfm;bias[row]=sign*vh[local]+rhs.data[base+row];"
            "r.diagonal.data[base+row]=norm[row];d.support.data[base+row]=tpl;",
            "r.diagonal.data[base+row]=square+r.cfm;d.support.data[base+row]=tpl;"
            "d.incident.data[base+row]=sign*vh[local];",
        ),
        (
            "templates[row+direction]=tpl;norm[row+direction]=square+r.cfm;"
            "bias[row+direction]=incident+value;d.support.data[at]=tpl;",
            "d.incident.data[at]=incident;d.support.data[at]=tpl;",
        ),
    )
    for old, new in substitutions:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    # The unchanged separate solver checks d.status before reading its inputs.
    # Every producer error must invalidate that world, not just our observer.
    for bit in (1, 2, 4):
        old = f"atomicOr(&r.status.data[0],{bit})"
        assert old in source
        source = source.replace(old, f"({old},atomicOr(&d.status.data[world],{bit}))")
    return (
        source
        + r"""
    const int count=row_count;
    if(thread==0){r.counter.data[world]=count;r.counts.data[world]=count;}
    for(int row=thread;row<count;row+=128)impulses.data[base+row]=0.0f;
    for(int k=thread;k<43;k+=128)f.vhat.data[start+k]=vh[k];
    if(thread==0&&world==0)atomicAdd(&r.status.data[1],1);
#endif
"""
    )


@cache
def _get_kernel(chain_scan: bool, producer: bool):
    """Build either explicit ownership boundary without changing its operands."""
    plan_type = KineticPlan
    if chain_scan:
        from .g1_chain_scan import ChainPlan  # noqa: PLC0415

        plan_type = ChainPlan

    @wp.func_native(_producer_source(chain_scan) if producer else _source(chain_scan))
    def native(
        group: int,
        p: SparsePlan,
        d: SparseData,
        plan: plan_type,
        cache: KineticData,
        f: ForceInput,
        x: PacketInput,
        r: WorldInput,
        buckets: RawWorldContacts,
        iterations: int,
        omega: float,
        friction_start: int,
        vout: wp.array[float],
    ): ...

    def solve(
        p: SparsePlan,
        d: SparseData,
        plan: plan_type,
        cache: KineticData,
        f: ForceInput,
        x: PacketInput,
        r: WorldInput,
        buckets: RawWorldContacts,
        iterations: int,
        omega: float,
        friction_start: int,
        vout: wp.array[float],
    ):
        group, _ = wp.tid()
        native(group, p, d, plan, cache, f, x, r, buckets, iterations, omega, friction_start, vout)

    solve.__name__ = solve.__qualname__ = (
        "sparse_parallel_world_rows43_s18_c100" if producer else "sparse_parallel_world43_s18_c100"
    )
    return wp.kernel(module="unique", enable_backward=False)(solve)


def get_kernel(chain_scan: bool = True):
    """Return the fixed128-thread complete prediction/rows/solve CUDA owner."""
    return _get_kernel(chain_scan, False)


def get_producer_kernel(chain_scan: bool = True):
    """Return the four-worker producer for the unchanged separate32-thread GS."""
    return _get_kernel(chain_scan, True)
