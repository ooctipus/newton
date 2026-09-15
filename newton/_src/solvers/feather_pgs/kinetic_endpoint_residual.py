# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Current contact packets and private endpoint increments for ordered PGS.

The current allocator owns minimal row metadata. Only responding rows form a
generalized response; other residuals use current endpoint delta twists.
"""

import functools
import inspect
import math
import re
import textwrap
from types import SimpleNamespace

import warp as wp

from . import kinetic_compact_coupled as compact
from . import kinetic_current_contact as contact
from . import kinetic_hybrid as hybrid
from . import kinetic_lazy_response as lazy
from . import kinetic_live_plan as live_plan
from . import kinetic_rows as row_source
from . import kinetic_rows_triplet as triplet
from . import kinetic_solve as original
from .kinetic_rows_types import ArmMap, DenseRowOutput, PrefixInput, RawRowInput, RowSettings, RowState
from .kinetic_solve_types import KineticMFData, KineticSolveData
from .kinetic_types import CurrentKineticCache, HeldKineticOperator, KineticPlan

replace = original.replace_once


@wp.struct
class EndpointData:
    route: wp.array[int]
    shared_anchor: int
    friction_shared_anchor: int
    cfm: float


_PACKET = r"""
    const int world=raw.world.data[c],slot=raw.slot.data[c];
    if(world<0||world>=out.status.shape[0]||slot<0)return;
    const int ba=cache.bodies.data[c*2],bb=cache.bodies.data[c*2+1];
    const int aa=raw.art_a.data[c],ab=raw.art_b.data[c];
    const int la=ba>=0?plan.body_lane.data[ba]:-1,lb=bb>=0?plan.body_lane.data[bb]:-1;
    bool good=(ba<0||(la>=0&&la<32&&plan.body_ids.data[world*32+la]==ba))&&
              (bb<0||(lb>=0&&lb<32&&plan.body_ids.data[world*32+lb]==bb));
    const float* s=cache.values.data+c*20;
    const int nr=static_cast<int>(s[19]);
    good=good&&(nr==1||nr==3)&&nr==raw.slots_needed.data[c]&&slot+nr<=settings.dense_capacity;
    good=good&&settings.contact_w==1.0f&&settings.dt>0.0f&&isfinite(settings.dt)&&isfinite(settings.cfm)&&settings.cfm>0.0f;
    if(!good){wp::atomic_max(&out.status.data[world],4);return;}
    const auto n=wp::vec3(s[0],s[1],s[2]);
    const auto t0=wp::vec3(s[9],s[10],s[11]),t1=wp::vec3(s[12],s[13],s[14]);
    const auto ra=wp::vec3(s[3],s[4],s[5]),rb=wp::vec3(s[6],s[7],s[8]);
    const bool shared=settings.shared_anchor!=0,shared_t=shared||settings.friction_shared_anchor!=0;
    const auto midpoint=(shared||shared_t)?(ra+rb)*0.5f:wp::vec3(0.0f);
    const auto xa=shared?midpoint:ra,xb=shared?midpoint:rb;
    const auto xta=shared_t?midpoint:ra,xtb=shared_t?midpoint:rb;
    const bool distinct_t=nr==3&&shared!=shared_t;
    auto relative=wp::vec3(0.0f);
    for(int endpoint=0;endpoint<2;++endpoint){
        const int body=endpoint==0?ba:bb,local=endpoint==0?la:lb;
        if(body>=0){
            const auto velocity=state.endpoint_twists.data[body];
            const auto lin=wp::vec3(velocity[0],velocity[1],velocity[2]);
            const auto ang=wp::vec3(velocity[3],velocity[4],velocity[5]);
            const int oi=local<30?0:(local==30?1:2);
            const auto origin=current.origin.data[world*3+oi];
            const auto an=endpoint==0?xa:xb,at=endpoint==0?xta:xtb;
            const auto vn=lin+wp::cross(ang,an-origin);
            const auto vt=distinct_t?lin+wp::cross(ang,at-origin):vn;
            const auto value=(endpoint==0?1.0f:-1.0f)*wp::vec3(wp::dot(n,vn),nr==3?wp::dot(t0,vt):0.0f,nr==3?wp::dot(t1,vt):0.0f);
            relative+=value;
        }
    }
    for(int component=0;component<nr;++component){
        const int id=world*192+slot+component;
        const float rv=component==0?relative[0]:(component==1?relative[1]:relative[2]);
        const float phi=component==0?s[15]:0.0f,e=component==0?s[17]:0.0f;
        float bias=component==0?(phi<=0.0f?settings.bias_scale*settings.beta:settings.contact_speculative_scale)*phi/settings.dt:0.0f;
        if(e>0.0f&&rv < -settings.restitution_velocity_threshold&&
           (phi<=1.0e-6f||phi+settings.dt*rv<=1.0e-6f))bias=e*rv;
        const float mu=component==0?s[16]:s[18];
        out.r0.data[id]=rv+bias;out.row_mu.data[id]=mu;
        out.row_type.data[id]=component==0?0:2;out.row_parent.data[id]=component==0?-1:slot;
        out.physical_J.data[id*29]=static_cast<float>(c*3+component+1);
        out.impulses.data[id]=0.0f;out.valid.data[id]=1;
        if(!isfinite(rv)||!isfinite(bias)||!isfinite(mu)||!isfinite(out.r0.data[id]))wp::atomic_max(&out.status.data[world],3);
    }
    const unsigned ma=ba>=0?raw.body_response_mask.data[ba]:0u,mb=bb>=0?raw.body_response_mask.data[bb]:0u;
    if((la==30&&(ma&63u))||(lb==30&&(mb&63u)))wp::atomic_max(&out.secondary_nonzero.data[world],1);
"""


@wp.func_native(_PACKET)
def _emit_packet(
    c: int,
    plan: KineticPlan,
    current: CurrentKineticCache,
    raw: RawRowInput,
    state: RowState,
    settings: RowSettings,
    out: DenseRowOutput,
    cache: contact.ContactGeometry,
): ...


@functools.cache
def get_allocate_kernel(arch):
    """Append packets to the existing allocator, after its exact slot/rank law."""
    source = textwrap.dedent(inspect.getsource(contact.get_allocate_kernel(arch).func))
    source = replace(source, "def kinetic_contact_allocate(", "def kinetic_endpoint_allocate(")
    source = replace(
        source,
        "    settings: RowSettings,",
        "    settings: RowSettings,\n    plan: KineticPlan,\n    current: CurrentKineticCache,\n    state: RowState,\n    out: DenseRowOutput,",
    )
    source = replace(
        source,
        "            _stage_row(c, total_contacts, cache, raw, settings)",
        "            _stage_row(c, total_contacts, cache, raw, settings)\n            _emit_packet(c, plan, current, raw, state, settings, out, cache)",
    )
    namespace = dict(contact.get_allocate_kernel(arch).func.__globals__)
    namespace.update(globals())
    return contact._compile(source, "kinetic_endpoint_allocate", namespace)


def prefix_source():
    """Keep the original46 bound/order tests, publishing only implicit row data."""
    source = row_source._PREFIX
    begin = source.index("        for(int row=lane;row<29;row+=stride){")
    end = source.index("        sync();", begin)
    source = source[:begin] + source[end:]
    source = replace(source, "float diag=settings.cfm;for(int k=0;k<29;++k)diag+=s[k];", "float diag=settings.cfm;")
    begin = source.index("            out.r0.data[id]=")
    end = source.index("            out.valid.data[id]=1;", begin)
    source = (
        source[:begin]
        + r"""
            out.r0.data[id]=sign*state.v_hat.data[g]+bias;
            out.row_type.data[id]=3;out.row_parent.data[id]=-1;out.row_mu.data[id]=0.0f;
            out.physical_J.data[id*29]=static_cast<float>(-code-1);out.impulses.data[id]=0.0f;
"""
        + source[end:]
    )
    return source


@functools.cache
def get_prefix_kernel(arch):
    """Retain the original prefix launch and descriptor ABI."""

    @wp.func_native(prefix_source())
    def emit(
        address: wp.uint64,
        index: int,
        plan: KineticPlan,
        held: HeldKineticOperator,
        prefix: PrefixInput,
        state: RowState,
        settings: RowSettings,
        out: DenseRowOutput,
    ): ...

    @wp.kernel(module="unique", enable_backward=False)
    def kinetic_endpoint_prefix(
        plan: KineticPlan,
        held: HeldKineticOperator,
        prefix: PrefixInput,
        state: RowState,
        settings: RowSettings,
        out: DenseRowOutput,
    ):
        index, logical = wp.tid()
        lanes = row_source._lane()
        if lanes[1] == 1 and logical != 0:
            return
        address = row_source._storage()
        emit(address, index, plan, held, prefix, state, settings, out)
        row_source._release(address)

    return kinetic_endpoint_prefix


def qualifier_source():
    """Retain current MF/held predicates, replacing dense arrays by incidence."""
    source = lazy.qualifier_source()
    source = replace(
        source,
        "hybrid_data.eligible.data[world]=0;",
        "hybrid_data.eligible.data[world]=0; packet.route.data[world]=mf_count==0?1:0;",
    )
    source = replace(source, "const float diag = rows.row_cfm.data[r];", "const float diag = packet.cfm;")
    source = replace(source, "wp::isfinite(rows.rhs.data[r])", "wp::isfinite(rows.r0.data[r])")
    begin = source.index("            for (int d = 0; d < 29; ++d) {")
    end = source.index("\n        }\n        for (int row = tid; row < mf_count", begin)
    source = source[:begin] + "            valid &= rows.secondary_nonzero.data[world]==0;" + source[end:]
    source = replace(
        source,
        "if (tid == 0) solve.selector.data[world] = -1;",
        "if (tid == 0) { solve.selector.data[world] = -1; packet.route.data[world]=1; }",
    )
    return source


@functools.cache
def get_qualify_kernel(arch):
    """Select packet ownership before any eager fallback response is built."""

    @wp.func_native(qualifier_source())
    def qualify(
        worker: int,
        logical_lane: int,
        workers: int,
        plan: KineticPlan,
        held: HeldKineticOperator,
        state: RowState,
        rows: DenseRowOutput,
        mf: KineticMFData,
        solve: KineticSolveData,
        hybrid_data: hybrid.HybridCoupledData,
        packet: EndpointData,
    ): ...

    @wp.kernel(module="unique", enable_backward=False)
    def kinetic_endpoint_qualify(
        plan: KineticPlan,
        held: HeldKineticOperator,
        state: RowState,
        rows: DenseRowOutput,
        mf: KineticMFData,
        solve: KineticSolveData,
        hybrid_data: hybrid.HybridCoupledData,
        workers: int,
        packet: EndpointData,
    ):
        worker, logical = wp.tid()
        qualify(worker, logical, workers, plan, held, state, rows, mf, solve, hybrid_data, packet)

    return kinetic_endpoint_qualify


@functools.cache
def get_fallback_contact_kernel(arch):
    """Keep the exact raw-striped eager triplet only for declined worlds."""
    source = replace(
        contact._contact_source(),
        "        const int aa=raw.art_a.data[c],ab=raw.art_b.data[c];",
        "        if(packet.route.data[world]!=0)continue;\n        const int aa=raw.art_a.data[c],ab=raw.art_b.data[c];",
    )

    @wp.func_native(source)
    def emit(
        address: wp.uint64,
        worker: int,
        plan: KineticPlan,
        current: CurrentKineticCache,
        held: HeldKineticOperator,
        raw: RawRowInput,
        state: RowState,
        settings: RowSettings,
        arm: ArmMap,
        out: DenseRowOutput,
        cache: contact.ContactGeometry,
        packet: EndpointData,
    ): ...

    @wp.kernel(module="unique", enable_backward=False)
    def kinetic_endpoint_fallback_rows(
        plan: KineticPlan,
        current: CurrentKineticCache,
        held: HeldKineticOperator,
        raw: RawRowInput,
        state: RowState,
        settings: RowSettings,
        arm: ArmMap,
        out: DenseRowOutput,
        cache: contact.ContactGeometry,
        packet: EndpointData,
    ):
        worker, logical = wp.tid()
        lanes = row_source._lane()
        if lanes[1] == 1 and logical != 0:
            return
        address = triplet._storage()
        emit(address, worker, plan, current, held, raw, state, settings, arm, out, cache, packet)
        row_source._release(address)

    return kinetic_endpoint_fallback_rows


@functools.cache
def get_begin_kernel(arch):
    """Original reset/C-map, preserving the already finalized prefix reservation."""
    kernel = row_source.get_arm_kernel(arch)
    source = textwrap.dedent(inspect.getsource(kernel.func))
    source = replace(source, "def kinetic_row_arm(", "def kinetic_endpoint_begin(")
    source = replace(source, "        out.slot_counter[world] = 0\n", "")
    return contact._compile(source, "kinetic_endpoint_begin", dict(row_source.__dict__))


_GEOMETRY = r"""
    auto geometry=[&](int row,int& c,int& component,int& la,int& lb,
                       wp::vec3& direction,wp::vec3& xa,wp::vec3& xb){
        const int encoded=static_cast<int>(rows.physical_J.data[(world*192+row)*29])-1;
        c=encoded/3;component=encoded%3;
        const float* values=cache.values.data+c*20;
        const int axis=component==0?0:(component==1?9:12);
        direction=wp::vec3(values[axis],values[axis+1],values[axis+2]);
        xa=wp::vec3(values[3],values[4],values[5]);xb=wp::vec3(values[6],values[7],values[8]);
        if(packet.shared_anchor||(component!=0&&packet.friction_shared_anchor))xa=xb=(xa+xb)*0.5f;
        const int ba=cache.bodies.data[c*2],bb=cache.bodies.data[c*2+1];
        la=ba<0?-1:plan.body_lane.data[ba];lb=bb<0?-1:plan.body_lane.data[bb];
    };
    auto build_J=[&](int row){
        const int key=static_cast<int>(rows.physical_J.data[(world*192+row)*29]);
        if(key<0){const int code=-key-1,col=code/2;const float sign=(code&1)?-1.0f:1.0f;
            for(int d=lane;d<29;d+=stride)js[d]=d==po+col?sign:0.0f;
        }else{
            int c,component,la,lb;wp::vec3 direction,xa,xb;
            geometry(row,c,component,la,lb,direction,xa,xb);
            const int ba=cache.bodies.data[c*2],bb=cache.bodies.data[c*2+1];
            const unsigned ma=ba<0?0u:raw.body_response_mask.data[ba],mb=bb<0?0u:raw.body_response_mask.data[bb];
            const int aa=raw.art_a.data[c],ab=raw.art_b.data[c];
            const bool primary_a=la>=0&&la<30&&aa>=0&&raw.response_count.data[aa]==23;
            const bool primary_b=lb>=0&&lb<30&&ab>=0&&raw.response_count.data[ab]==23;
            const bool common=primary_a&&primary_b&&aa==ab&&(ma&mb&127u)==127u;
            const unsigned support=(primary_a?ma:0u)|(primary_b?mb:0u);
            for(int d=lane;d<29;d+=stride){
                float j=0.0f;
                if(d<23){
                    if(support&(1u<<d)){
                        const auto axis=current.axes.data[world*23+d];
                        const auto lin=wp::vec3(axis[0],axis[1],axis[2]),ang=wp::vec3(axis[3],axis[4],axis[5]);
                        if(common&&d<7)j=wp::dot(ang,wp::cross(xa-xb,direction));
                        else{
                            if(primary_a&&(ma&(1u<<d)))j+=wp::dot(direction,lin+wp::cross(ang,xa-current.origin.data[world*3]));
                            if(primary_b&&(mb&(1u<<d)))j-=wp::dot(direction,lin+wp::cross(ang,xb-current.origin.data[world*3]));
                        }
                    }
                }else{
                    const int k=d-23;
                    if(((la==30?ma:0u)|(lb==30?mb:0u))&(1u<<k)){
                        const auto axis=current.free_axes.data[world*6+k];
                        const auto lin=wp::vec3(axis[0],axis[1],axis[2]),ang=wp::vec3(axis[3],axis[4],axis[5]);
                        if(la==30&&(ma&(1u<<k)))j+=wp::dot(direction,lin+wp::cross(ang,xa-current.origin.data[world*3+1]));
                        if(lb==30&&(mb&(1u<<k)))j-=wp::dot(direction,lin+wp::cross(ang,xb-current.origin.data[world*3+1]));
                    }
                }
                js[d<23?po+d:so+d-23]=j;
            }
        }
    };
"""


def residual_ancestry():
    """Emit fixed-child aliases from the same topology checked by live admission.

    All primary twists are represented about the same current origin. A fixed
    child and its nearest moving ancestor therefore have identical responsive
    twists, even when a parallel scan associates their sums differently.
    Actual contact anchors and first-use Jacobians remain those of each body.
    """
    moving = set(map(int, live_plan.SCALAR_BODIES))
    aliases = []
    for body in range(30):
        ancestor = body
        while ancestor not in moving and live_plan.PARENTS[ancestor] >= 0:
            ancestor = int(live_plan.PARENTS[ancestor])
        if ancestor != body:
            aliases.append(f"case {body}: return {ancestor};")
    return "\n    auto residual_lane=[](int body){switch(body){" + "".join(aliases) + "default:return body;}};\n"


def cache_source():
    source = replace(lazy._CACHE, "for(int d=lane;d<29;d+=stride)js[d]=rows.physical_J.data[id*29+d];", "build_J(row);")
    return (
        _GEOMETRY + residual_ancestry() + replace(source, "float diag=rows.row_cfm.data[id];", "float diag=packet.cfm;")
    )


def cuda_endpoints():
    """Six immutable axes and six private increments per body lane."""
    source = r"""
    const int body_dof=plan.body_local_dof.data[world*32+lane];
    const auto body_axis=body_dof>=0&&lane<30?current.axes.data[world*23+body_dof]:wp::spatial_vector(0.0f);
    bool motion_dirty=false;
"""
    source += "\n".join(f"    const float axis{k}=body_axis[{k}];float body{k}=0.0f;" for k in range(6))
    source += r"""
    auto physical_residual=[&](int row){
        const int key=static_cast<int>(rows.physical_J.data[(world*192+row)*29]);
        if(key<0){const int code=-key-1;return ((code&1)?-1.0f:1.0f)*__shfl_sync(MASK,factor_velocity,po+code/2);}
        if(motion_dirty){
            const float delta=__shfl_sync(MASK,factor_velocity,body_dof>=0&&lane<30?po+body_dof:0);
"""
    source += "\n".join(f"            body{k}=axis{k}*delta;" for k in range(6))
    source += "\n            int parent=plan.body_parent.data[world*32+lane];\n"
    source += "            for(int round=0;round<4;++round){const int from=parent>=0?parent:lane;\n"
    source += "\n".join(f"                const float incoming{k}=__shfl_sync(MASK,body{k},from);" for k in range(6))
    source += "\n                const int ancestor=__shfl_sync(MASK,parent,from);if(parent>=0){\n"
    source += "\n".join(f"                    body{k}+=incoming{k};" for k in range(6))
    source += "\n                    parent=ancestor;}}\n"
    source += "            for(int k=0;k<6;++k){const float free_delta=__shfl_sync(MASK,factor_velocity,so+k);if(lane==30){const auto free_axis=current.free_axes.data[world*6+k];\n"
    source += "\n".join(f"                body{k}+=free_axis[{k}]*free_delta;" for k in range(6))
    source += r"""
            }}
            motion_dirty=false;
        }
        int c,component,la,lb;wp::vec3 direction,xa,xb;
        geometry(row,c,component,la,lb,direction,xa,xb);
        la=residual_lane(la);lb=residual_lane(lb);
        const auto lin=wp::vec3(body0,body1,body2),ang=wp::vec3(body3,body4,body5);
        float va=0.0f,vb=0.0f;
        if(lane==la&&la<31)va=wp::dot(direction,lin+wp::cross(ang,xa-current.origin.data[world*3+(la<30?0:1)]));
        if(lane==lb&&lb<31)vb=wp::dot(direction,lin+wp::cross(ang,xb-current.origin.data[world*3+(lb<30?0:1)]));
        return __shfl_sync(MASK,va,la>=0?la:0)-__shfl_sync(MASK,vb,lb>=0?lb:0);
    };
"""
    return source


_CPU_ENDPOINTS = r"""
    float body_delta[32][6]={};bool motion_dirty=false;
    auto physical_residual=[&](int row){
        const int key=static_cast<int>(rows.physical_J.data[(world*192+row)*29]);
        if(key<0){const int code=-key-1;return ((code&1)?-1.0f:1.0f)*du[po+code/2];}
        if(motion_dirty){
            for(int b=0;b<32;++b){
                const int d=plan.body_local_dof.data[world*32+b],parent=plan.body_parent.data[world*32+b];
                for(int k=0;k<6;++k){float value=0.0f;
                    if(b<30&&d>=0)value=current.axes.data[world*23+d][k]*du[po+d];
                    if(b<30&&parent>=0)value+=body_delta[parent][k];
                    if(b==30)for(int q=0;q<6;++q)value+=current.free_axes.data[world*6+q][k]*du[so+q];
                    body_delta[b][k]=value;
                }
            }
            motion_dirty=false;
        }
        int c,component,la,lb;wp::vec3 direction,xa,xb;
        geometry(row,c,component,la,lb,direction,xa,xb);
        la=residual_lane(la);lb=residual_lane(lb);
        float result=0.0f;
        for(int endpoint=0;endpoint<2;++endpoint){const int b=endpoint==0?la:lb;
            if(b>=0&&b<31){const float* v=body_delta[b];
                const auto lin=wp::vec3(v[0],v[1],v[2]),ang=wp::vec3(v[3],v[4],v[5]);
                result+=(endpoint==0?1.0f:-1.0f)*wp::dot(direction,lin+wp::cross(ang,(endpoint==0?xa:xb)-current.origin.data[world*3+(b<30?0:1)]));
            }
        }
        return result;
    };
"""


def cuda_source():
    """Retain all original transactions and fences; replace residual evaluation."""
    source = replace(lazy.cuda_source(), lazy._CACHE, cache_source() + cuda_endpoints())
    # Remove only coefficient fetches. Response updates already use ready R.
    source, n = re.subn(
        r"\s*const float (?:row|normal|tangent1|tangent2|sibling)_factor = lane < 29\s*\? factor_rows\.data\[[^;]+;",
        "",
        source,
    )
    if n != 5:
        raise ValueError(f"Changed coefficient fetch sites: {n}")
    source, n = re.subn(r"\s*float prefetched_factor = lane < 29[^;]+;", "", source)
    if n != 1:
        raise ValueError("Changed generic prefetch declaration")
    source = replace(source, "            const float row_factor = prefetched_factor;", "")
    source, n = re.subn(r"\s*if \(i \+ 1 < m && lane < 29\)\s*prefetched_factor = [^;]+;", "", source)
    if n != 1:
        raise ValueError("Changed generic prefetch advance")
    for variable, row, factor in (
        ("sum", "i", "row_factor"),
        ("normal_sum", "normal", "normal_factor"),
        ("tangent1_sum", "tangent1", "tangent1_factor"),
        ("tangent2_sum", "tangent2", "tangent2_factor"),
    ):
        pattern = rf"float {variable} = lane < 29 \? {factor} \* factor_velocity : 0\.0f;(?:\s*{variable} \+= __shfl_down_sync\(MASK, {variable}, (?:16|8|4|2|1)\);){{5}}"
        source, n = re.subn(pattern, f"float {variable} = physical_residual({row});", source)
        if n != (2 if variable == "sum" else 1):
            raise ValueError(f"Changed residual sites {variable}: {n}")
        source = re.sub(rf"__shfl_sync\(MASK,\s*{variable},\s*0\)", variable, source)
    source, n = re.subn(r"(factor_velocity \+= response\([^;]+;)", r"\1 motion_dirty=true;", source)
    if n != 8:
        raise ValueError(f"Changed physical update sites: {n}")
    return source


def cpu_source():
    source = replace(lazy.cpu_source(), lazy._CACHE, cache_source())
    source = replace(source, "    float du[29]={},lambda[192];", "    float du[29]={},lambda[192];\n" + _CPU_ENDPOINTS)
    source = replace(
        source,
        "float residual=0.0f;for(int d=0;d<29;++d)residual+=rows.physical_J.data[id*29+d]*du[d];residual+=rows.r0.data[id];",
        "float residual=physical_residual(i)+rows.r0.data[id];",
    )
    source = source.replace(
        "{changed=true;for(int d=0;d<29;++d)du[d]+=", "{changed=true;motion_dirty=true;for(int d=0;d<29;++d)du[d]+="
    )
    return source


@functools.cache
def get_solve_kernel(arch):
    source = (
        original._GUARD
        + original._T
        + "\n#if defined(__CUDA_ARCH__)\n"
        + cuda_source()
        + "\n#else\n"
        + cpu_source()
        + "\n#endif\n"
    )

    @wp.func_native(source)
    def endpoint(
        world: int,
        plan: KineticPlan,
        held: HeldKineticOperator,
        state: RowState,
        rows: DenseRowOutput,
        solve: KineticSolveData,
        current: CurrentKineticCache,
        raw: RawRowInput,
        cache: contact.ContactGeometry,
        packet: EndpointData,
    ): ...

    eager = inspect.getclosurevars(original.get_solve_kernel(arch).func).nonlocals["solve_native"]

    @wp.kernel(module="unique", enable_backward=False)
    def kinetic_endpoint_physical_eight(
        plan: KineticPlan,
        held: HeldKineticOperator,
        state: RowState,
        rows: DenseRowOutput,
        solve: KineticSolveData,
        current: CurrentKineticCache,
        raw: RawRowInput,
        cache: contact.ContactGeometry,
        packet: EndpointData,
    ):
        block, logical = wp.tid()
        if state.active_count[0] < 0 or state.active_count[0] > state.active_worlds.shape[0]:
            return
        if original._is_cpu():
            if logical != 0:
                return
            for half in range(2):
                index = block * 2 + half
                if index < state.active_count[0]:
                    world = state.active_worlds[index]
                    if packet.route[world] != 0:
                        endpoint(world, plan, held, state, rows, solve, current, raw, cache, packet)
                    else:
                        eager(world, plan, held, state, rows, solve)
            return
        index = block * 2 + logical // 32
        if index < state.active_count[0]:
            world = state.active_worlds[index]
            if packet.route[world] != 0:
                endpoint(world, plan, held, state, rows, solve, current, raw, cache, packet)
            else:
                eager(world, plan, held, state, rows, solve)

    return kinetic_endpoint_physical_eight


@functools.cache
def get_materialize_kernel(arch):
    """Retain original eager prefix, numerical qualification and compact bridge."""
    qualify = inspect.getclosurevars(lazy.get_qualify_kernel(arch, True).func).nonlocals["qualify"]
    bridge = inspect.getclosurevars(compact.get_materialize_kernel(arch).func).nonlocals["materialize"]
    boundary_source = r"""
#if defined(__CUDA_ARCH__)
    __syncwarp(0xffffffffu);
#endif
    """

    @wp.func_native(boundary_source)
    def boundary(): ...

    @wp.kernel(module="unique", enable_backward=False)
    def kinetic_endpoint_positive_materialize(
        plan: KineticPlan,
        held: HeldKineticOperator,
        state: RowState,
        rows: DenseRowOutput,
        solve: KineticSolveData,
        hybrid_data: hybrid.HybridCoupledData,
        mf: KineticMFData,
        prefix: PrefixInput,
        settings: RowSettings,
        packet: EndpointData,
    ):
        index, logical = wp.tid()
        if state.active_count[0] < 0 or state.active_count[0] > state.active_worlds.shape[0]:
            return
        if index >= state.active_count[0]:
            return
        world = state.active_worlds[index]
        if world < 0 or world >= state.dense_count.shape[0]:
            return
        if packet.route[world] != 0:
            return
        lanes = row_source._lane()
        if lanes[1] == 1 and logical != 0:
            return
        address = row_source._storage()
        row_source._prefix(address, index, plan, held, prefix, state, settings, rows)
        boundary()
        qualify(world, logical, state.dense_count.shape[0], plan, held, state, rows, mf, solve, hybrid_data)
        boundary()
        bridge(world, logical, plan, held, state, rows, solve, hybrid_data)
        row_source._release(address)

    return kinetic_endpoint_positive_materialize


def install(binding, call):
    """Move existing phases, preserving original stream joins and failure guard."""
    from . import kernels  # noqa: PLC0415 -- optional live owner

    if binding.current_contact is None or binding.lazy_response:
        raise ValueError("Endpoint residual requires exclusive current-contact ownership")
    rows, owner = call.rows, call.solve
    if not math.isfinite(rows.settings.cfm) or rows.settings.cfm <= 0.0:
        call.endpoint_residual = SimpleNamespace(active=False, reason="nonpositive_or_nonfinite_cfm")
        return
    if rows.settings.raw_capacity < 0 or rows.settings.raw_capacity > 4_000_000:
        raise ValueError("Endpoint identity requires raw capacity at most 4,000,000")
    if rows.settings.dense_capacity != 192 or rows.settings.mf_capacity != 64:
        raise ValueError("Endpoint residual requires original192/64 capacities")
    arch, device, worlds = str(binding.device.arch), binding.device, binding.worlds
    if rows.kernels[2] is not contact.get_contact_kernel(arch):
        raise ValueError("Endpoint residual requires unchanged cached-triplet rows")
    packet = EndpointData()
    packet.route = binding.endpoint_route
    packet.shared_anchor = rows.settings.shared_anchor
    packet.friction_shared_anchor = rows.settings.friction_shared_anchor
    packet.cfm = rows.settings.cfm
    cache, allocation = binding.current_contact, owner.allocation
    begin, allocator = get_begin_kernel(arch), get_allocate_kernel(arch)
    prefix_kernel, fallback = get_prefix_kernel(arch), get_fallback_contact_kernel(arch)
    begin_args = rows.arguments[0]
    allocate_args = [*call.current_contact.arguments[3], rows.plan, rows.current, rows.state, rows.out]
    prefix_args = rows.arguments[1]
    fallback_args = [*rows.arguments[2], packet]
    validate, validate_args = rows.kernels[3], rows.arguments[3]
    owner.kernels["qualify"] = get_qualify_kernel(arch)
    owner.arguments["qualify"].append(packet)
    owner.kernels["offset_eight"] = get_solve_kernel(arch)
    owner.arguments["offset_eight"].extend([rows.current, rows.raw, cache, packet])
    owner.kernels["materialize"] = get_materialize_kernel(arch)
    owner.arguments["materialize"].extend([owner.mf_descriptor, rows.prefix, rows.settings, packet])
    old_prepare, old_qualify = owner.prepare_mf, owner.qualify

    def allocate_launch():
        # ZERO/finalize already owns capacity/raw failures and prefix reservation.
        # Begin clears row-local errors and builds C but must not erase reservation.
        rows.out.global_status.zero_()
        wp.launch_tiled(begin, dim=[worlds], block_dim=32, inputs=begin_args, device=device)
        wp.launch(allocator, dim=allocation.workers, inputs=allocate_args, device=device)
        wp.copy(allocation.mf.contact_end, allocation.data.mf_counter)
        wp.launch(
            kernels.allocate_rigid_velocity_limit_slots,
            dim=allocation.mf.free_bodies.size,
            inputs=allocation.limit_args,
            device=device,
        )
        for counter, cap, family, count in (
            (allocation.data.dense_counter, 192, 0, rows.state.dense_count),
            (allocation.data.mf_counter, 64, 1, rows.state.mf_count),
            (allocation.data.propagation_counter, 192, 2, allocation.data.propagation_count),
        ):
            wp.launch(
                kernels.finalize_constraint_counts_with_status,
                dim=worlds,
                inputs=[counter, cap, family, count, rows.state.capacity_status],
                device=device,
            )
        wp.launch_tiled(prefix_kernel, dim=[worlds], block_dim=32, inputs=prefix_args, device=device)

    def build_rows():
        old_prepare()
        old_qualify()
        wp.launch_tiled(fallback, dim=[rows.settings.workers], block_dim=32, inputs=fallback_args, device=device)
        wp.launch_tiled(validate, dim=[worlds], block_dim=32, inputs=validate_args, device=device)

    def already_completed():
        pass

    owner.allocate = allocation.launch = allocate_launch
    owner.build_rows = rows.launch = build_rows
    owner.prepare_mf = owner.qualify = already_completed
    call.endpoint_residual = SimpleNamespace(
        active=True,
        packet=packet,
        kernels=(
            begin,
            allocator,
            prefix_kernel,
            owner.kernels["qualify"],
            fallback,
            validate,
            owner.kernels["materialize"],
            owner.kernels["offset_eight"],
        ),
        arguments=(
            begin_args,
            allocate_args,
            prefix_args,
            owner.arguments["qualify"],
            fallback_args,
            validate_args,
            owner.arguments["materialize"],
            owner.arguments["offset_eight"],
        ),
        dimensions=(
            (worlds, 32),
            (allocation.workers, 0),
            (worlds, 32),
            owner.dimensions["qualify"],
            (rows.settings.workers, 32),
            (worlds, 32),
            owner.dimensions["materialize"],
            owner.dimensions["offset_eight"],
        ),
    )
