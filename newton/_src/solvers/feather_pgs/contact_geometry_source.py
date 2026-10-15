# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared contact traversal, material law and explicit row staging ownership."""

from string import Template

from .kinetic_source import checked_definitions

_TRAVERSAL = r"""
    const int count=raw.count.data[0];
    if(count<0 || count>settings.raw_capacity || count>raw.shape0.shape[0] || settings.workers<=0){
        if(worker==0&&lane==0)out.global_status.data[0]=1;return;}
    if(state.raw_invalid.data[0]!=0){if(worker==0&&lane==0)out.global_status.data[0]=1;return;}
    for(int c=worker;c<count;c+=settings.workers){
        if(raw.path.data[c]!=0 || raw.slot.data[c]<0)continue;
        const int world=raw.world.data[c],slot=raw.slot.data[c];
        if(world<0 || world>=out.status.shape[0]){if(lane==0)wp::atomic_max(&out.global_status.data[0],1);continue;}
        if(!all(state.resolved.data[world]==0 && out.status.data[world]==0))continue;
        const int aa=raw.art_a.data[c],ab=raw.art_b.data[c];
        if(aa>=raw.response_count.shape[0] || ab>=raw.response_count.shape[0]){if(lane==0)wp::atomic_max(&out.status.data[world],5);continue;}
        if((aa<0 || (raw.response_count.data[aa]!=23&&raw.response_count.data[aa]!=6)) &&
           (ab<0 || (raw.response_count.data[ab]!=23&&raw.response_count.data[ab]!=6)))continue;
        if(arm.valid.data[world]==0 || arm.state_generation.data[world]!=state.state_generation.data[world] ||
           arm.held_generation.data[world]!=state.held_generation.data[world]){if(lane==0)wp::atomic_max(&out.status.data[world],2);continue;}
        const int sa=raw.shape0.data[c],sb=raw.shape1.data[c];
        if(sa>=raw.shape_body.shape[0]||sb>=raw.shape_body.shape[0]){if(lane==0)wp::atomic_max(&out.status.data[world],5);continue;}
        ${bodies}
        if(ba>=raw.body_q.shape[0]||bb>=raw.body_q.shape[0]){if(lane==0)wp::atomic_max(&out.status.data[world],5);continue;}
        const int la=ba>=0?plan.body_lane.data[ba]:-1,lb=bb>=0?plan.body_lane.data[bb]:-1;
        bool good=(ba<0||(la>=0&&la<32&&plan.body_ids.data[world*32+la]==ba)) &&
                  (bb<0||(lb>=0&&lb<32&&plan.body_ids.data[world*32+lb]==bb));
        good=good&&(ba<0?aa<0:aa==raw.body_to_articulation.data[ba])&&(bb<0?ab<0:ab==raw.body_to_articulation.data[bb]);
        good=good&&(aa<0||raw.art_to_world.data[aa]==world)&&(ab<0||raw.art_to_world.data[ab]==world);
        if(!good){if(lane==0)wp::atomic_max(&out.status.data[world],5);continue;}
${geometry}        sync();const int nr=static_cast<int>(s[83]);
        if(slot<out.slot_counter.data[world]||slot+nr>state.dense_count.data[world]||slot+nr>settings.dense_capacity||nr!=raw.slots_needed.data[c]){
            if(lane==0)wp::atomic_max(&out.status.data[world],4);continue;}
        const unsigned ma=ba>=0?raw.body_response_mask.data[ba]:0u,mb=bb>=0?raw.body_response_mask.data[bb]:0u;
        const bool primary_a=la>=0&&la<30&&aa>=0&&raw.response_count.data[aa]==23;
        const bool primary_b=lb>=0&&lb<30&&ab>=0&&raw.response_count.data[ab]==23;
        const bool common=primary_a&&primary_b&&aa==ab&&(ma&mb&127u)==127u;
        const bool physical=state.mf_count.data[world]>0;
        const int po=state.primary_offset.data[world],so=state.secondary_offset.data[world];
        if(!((po==0&&so==23)||(po==6&&so==0))){if(lane==0)wp::atomic_max(&out.status.data[world],5);continue;}
"""


_GEOMETRY = r"""        ${entry}
            const auto n=-raw.normal.data[c];
            const auto xa=${point_a};
            const auto xb=${point_b};
            auto t0=wp::cross(n,wp::vec3(1.0f,0.0f,0.0f));if(wp::dot(t0,t0)<1.0e-12f)t0=wp::cross(n,wp::vec3(0.0f,1.0f,0.0f));
            t0=wp::normalize(t0);const auto t1=wp::normalize(wp::cross(n,t0));
            ${stores}
            const float phi=${phi};float mu=0.0f,e=0.0f;int nm=0;
            for(int endpoint=0;endpoint<2;++endpoint){const int shape=endpoint==0?sa:sb;if(shape>=0){mu+=raw.shape_mu.data[shape];const float ev=raw.shape_restitution.data[shape];if(isfinite(ev))e+=wp::clamp(ev,0.0f,1.0f);++nm;}}
            if(nm){mu/=static_cast<float>(nm);e/=static_cast<float>(nm);}
            const bool filter=settings.friction_articulation_pairs_only==0 ||
                (aa>=0&&ab>=0&&raw.is_free_rigid.data[aa]==0&&raw.is_free_rigid.data[ab]==0);
            const int limit=filter?settings.friction_anchor_limit:0;int rank=0,next=0;
            if(limit>0){for(int look=1;look<=8;++look){const int prev=c-look;if(prev<0)break;if(raw.shape0.data[prev]!=sa||raw.shape1.data[prev]!=sb)break;++rank;}
                if(c+1<count&&raw.shape0.data[c+1]==sa&&raw.shape1.data[c+1]==sb)next=1;}
            bool friction=settings.enable_friction!=0&&(!filter||phi<=settings.friction_gap_threshold);
            if(limit>0&&rank>=limit)friction=false;${patch_friction}
            s[${slot15}]=phi;s[${slot16}]=mu;s[${slot17}]=e;s[${slot18}]=mu*settings.friction_scale*((limit>0&&(rank>0||next))?0.5f:1.0f);s[${slot19}]=friction?3.0f:1.0f;
        }
"""


def _check():
    checked_definitions(
        "contact_geometry_source.py",
        ("_TRAVERSAL", "_GEOMETRY", "_check", "get_geometry_source", "get_prelude", "get_stage_source"),
    )


def get_geometry_source(*, staging=False, patches=False):
    """Retain material arithmetic; choose warp scratch or one admitted cache row."""
    _check()
    offset = 0 if staging else 64
    slots = {"slot" + str(i): str(offset + i) for i in range(20)}
    if staging:
        point_a = "wp::vec3(s[3],s[4],s[5])"
        point_b = "wp::vec3(s[6],s[7],s[8])"
        stores = "for(int k=0;k<3;++k){s[9+k]=t0[k];s[12+k]=t1[k];}"
    else:
        point_a = "(ba>=0?wp::transform_point(raw.body_q.data[ba],raw.point0.data[c]):raw.point0.data[c])-n*raw.margin0.data[c]"
        point_b = "(bb>=0?wp::transform_point(raw.body_q.data[bb],raw.point1.data[c]):raw.point1.data[c])+n*raw.margin1.data[c]"
        stores = "for(int k=0;k<3;++k){s[64+k]=n[k];s[67+k]=xa[k];s[70+k]=xb[k];s[73+k]=t0[k];s[76+k]=t1[k];}"
    return Template(_GEOMETRY).substitute(
        slots,
        entry="{" if staging else "if(lane==0){",
        point_a=point_a,
        point_b=point_b,
        stores=stores,
        phi="s[15]" if staging else "wp::dot(n,xa-xb)",
        patch_friction="\n            friction=friction&&patches.weight.data[c]>0.0f;" if patches else "",
    )


def get_prelude(common, *, cached_geometry=False, patches=False):
    """Bind original traversal to recomputed or already staged contact geometry."""
    _check()
    bodies = (
        "const int ba=cache.bodies.data[c*2],bb=cache.bodies.data[c*2+1];"
        if cached_geometry
        else "const int ba=sa>=0?raw.shape_body.data[sa]:-1,bb=sb>=0?raw.shape_body.data[sb]:-1;"
    )
    geometry = (
        "        for(int k=lane;k<20;k+=stride)s[64+k]=cache.values.data[c*20+k];\n"
        if cached_geometry
        else get_geometry_source(patches=patches)
    )
    return common + Template(_TRAVERSAL).substitute(bodies=bodies, geometry=geometry)


def get_stage_source():
    """Stage only material and tangent metadata into the caller's twenty floats."""
    return (
        "    float* s=cache.values.data+c*20;\n"
        "    const int sa=raw.shape0.data[c],sb=raw.shape1.data[c];\n"
        "    const int aa=raw.art_a.data[c],ab=raw.art_b.data[c];\n" + get_geometry_source(staging=True)
    )
