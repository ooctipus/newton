# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact-only successor: exact support gating and three current RHS at once.

No descriptor, allocation, prefix, C-map, row order, solver or output-capacity
change. The original checked raw traversal and geometry/material prelude are
reused verbatim. Only the direction loop is replaced. All public row fields
retain their original equations and all 29 coefficients remain published.
"""

import functools
import inspect

import warp as wp

from . import kinetic_rows as original
from .kinetic_rows_types import (
    ArmMap,
    DenseRowOutput,
    RawRowInput,
    RowSettings,
    RowState,
)
from .kinetic_source import checked_definitions
from .kinetic_types import CurrentKineticCache, HeldKineticOperator, KineticPlan

checked_definitions("kinetic_rows.py", ("_CONTACT", "_t", "_lane", "_release"))

_MARKER = "        for(int component=0;component<nr;++component){"
if original._CONTACT.count(_MARKER) != 1:
    raise RuntimeError("Original contact direction-loop seam changed")
_PRELUDE = original._CONTACT.split(_MARKER)[0]


# Existing prelude uses s[64:84]. J is component-major s[96:189], Z s[192:285].
# Staging remains CTA-private; no new global packet, queue or basis is allocated.
@wp.func_native(r"""
#if defined(__CUDA_ARCH__)
    __shared__ float values[288];
    return reinterpret_cast<uint64_t>(values);
#else
    return reinterpret_cast<uint64_t>(malloc(288*sizeof(float)));
#endif
""")
def _storage() -> wp.uint64: ...


_TRIPLET = r"""
        const auto normal=wp::vec3(s[64],s[65],s[66]);
        const auto tangent0=wp::vec3(s[73],s[74],s[75]);
        const auto tangent1=wp::vec3(s[76],s[77],s[78]);
        const auto raw_a=wp::vec3(s[67],s[68],s[69]);
        const auto raw_b=wp::vec3(s[70],s[71],s[72]);
        const bool shared=settings.shared_anchor!=0;
        const bool shared_t=shared||settings.friction_shared_anchor!=0;
        const auto midpoint=(shared||shared_t)?(raw_a+raw_b)*0.5f:wp::vec3(0.0f);
        const auto xa=shared?midpoint:raw_a,xb=shared?midpoint:raw_b;
        const auto xta=shared_t?midpoint:raw_a,xtb=shared_t?midpoint:raw_b;
        const auto moment=wp::cross(xa-xb,normal);
        const auto moment0=nr==3?wp::cross(xta-xtb,tangent0):wp::vec3(0.0f);
        const auto moment1=nr==3?wp::cross(xta-xtb,tangent1):wp::vec3(0.0f);
        const unsigned support=(primary_a?ma:0u)|(primary_b?mb:0u);
        const bool has_free=(la==30&&(ma&63u)!=0u)||(lb==30&&(mb&63u)!=0u);
        const bool distinct_t=nr==3&&shared!=shared_t;
        auto projected=[&](const wp::vec3& nvalue,const wp::vec3& tvalue){
            return wp::vec3(wp::dot(normal,nvalue),nr==3?wp::dot(tangent0,tvalue):0.0f,
                            nr==3?wp::dot(tangent1,tvalue):0.0f);
        };
        auto anchor_response=[&](const wp::vec3& lin,const wp::vec3& ang,
                                const wp::vec3& an,const wp::vec3& at,
                                const wp::vec3& origin){
            const auto vn=lin+wp::cross(ang,an-origin);
            const auto vt=distinct_t?lin+wp::cross(ang,at-origin):vn;
            return projected(vn,vt);
        };
        for(int d=lane;d<29;d+=stride){
            auto j=wp::vec3(0.0f);
            if(d<23){
                if((support&(1u<<d))!=0u){
                    const auto axis=current.axes.data[world*23+d];
                    const auto lin=wp::vec3(axis[0],axis[1],axis[2]);
                    const auto ang=wp::vec3(axis[3],axis[4],axis[5]);
                    if(common&&d<7){
                        if(physical)j=wp::vec3(wp::dot(ang,moment),nr==3?wp::dot(ang,moment0):0.0f,
                                             nr==3?wp::dot(ang,moment1):0.0f);
                    }else{
                        if(primary_a&&(ma&(1u<<d)))j+=anchor_response(lin,ang,xa,xta,current.origin.data[world*3]);
                        if(primary_b&&(mb&(1u<<d)))j-=anchor_response(lin,ang,xb,xtb,current.origin.data[world*3]);
                    }
                }
            }else{
                const int k=d-23;
                if(has_free&&(((la==30?ma:0u)|(lb==30?mb:0u))&(1u<<k))){
                    const auto axis=current.free_axes.data[world*6+k];
                    const auto lin=wp::vec3(axis[0],axis[1],axis[2]);
                    const auto ang=wp::vec3(axis[3],axis[4],axis[5]);
                    if(la==30&&(ma&(1u<<k)))j+=anchor_response(lin,ang,xa,xta,current.origin.data[world*3+1]);
                    if(lb==30&&(mb&(1u<<k)))j-=anchor_response(lin,ang,xb,xtb,current.origin.data[world*3+1]);
                }
            }
            s[96+d]=j[0];
            good=good&&isfinite(j[0]);
            if(nr==3){s[128+d]=j[1];s[160+d]=j[2];good=good&&isfinite(j[1])&&isfinite(j[2]);}
        }
        sync();
        for(int d=lane;d<29;d+=stride){
            auto z=wp::vec3(0.0f);
            auto add=[&](float coefficient,int col){
                z[0]+=coefficient*s[96+col];
                if(nr==3){z[1]+=coefficient*s[128+col];z[2]+=coefficient*s[160+col];}
            };
            if(d<7){
                if(common){const auto cvalue=arm.C.data[world*7+d];
                    z=wp::vec3(wp::dot(cvalue,moment),nr==3?wp::dot(cvalue,moment0):0.0f,
                               nr==3?wp::dot(cvalue,moment1):0.0f);
                }else for(int k=0;k<=d;++k)add(t(world,d,k),k);
                for(int k=7;k<23;++k)if(support&(1u<<k))add(t(world,d,k),k);
            }else if(d<23){
                const int begin=7+((d-7)/4)*4;
                if((support&(15u<<begin))!=0u)
                    for(int k=begin;k<=d;++k)add(t(world,d,k),k);
            }else if(has_free){
                const int row=d-23,group=plan.secondary_group.data[world];
                for(int k=0;k<=row;++k)add(held.inverse6.data[group*36+row*6+k],23+k);
            }
            s[192+d]=z[0];good=good&&isfinite(z[0]);
            if(nr==3){s[224+d]=z[1];s[256+d]=z[2];good=good&&isfinite(z[1])&&isfinite(z[2]);}
        }
        sync();good=all(good);
        if(!good){if(lane==0)wp::atomic_max(&out.status.data[world],3);continue;}
        for(int d=lane;d<29;d+=stride){
            const int coord=d<23?po+d:so+d-23;
            for(int component=0;component<nr;++component){
                const int id=world*settings.dense_capacity+slot+component;
                out.response.data[id*29+coord]=s[192+component*32+d];
                if(physical)out.physical_J.data[id*29+coord]=s[96+component*32+d];
            }
        }
        if(lane==0){
            auto relative=wp::vec3(0.0f),known=wp::vec3(0.0f);
            // Preserve original endpoint A THEN B summation per component.
            for(int endpoint=0;endpoint<2;++endpoint){
                const int body=endpoint==0?ba:bb,art=endpoint==0?aa:ab,local=endpoint==0?la:lb;
                if(body>=0){
                    const auto velocity=state.endpoint_twists.data[body];
                    const auto lin=wp::vec3(velocity[0],velocity[1],velocity[2]);
                    const auto ang=wp::vec3(velocity[3],velocity[4],velocity[5]);
                    const int oi=local<30?0:(local==30?1:2);
                    const auto value=(endpoint==0?1.0f:-1.0f)*anchor_response(
                        lin,ang,endpoint==0?xa:xb,endpoint==0?xta:xtb,current.origin.data[world*3+oi]);
                    relative+=value;if(art>=0&&raw.prescribed_articulation.data[art]!=0)known+=value;
                }
            }
            for(int component=0;component<nr;++component){
                const int id=world*settings.dense_capacity+slot+component;
                // Constant component accesses keep these six incident scalars
                // in registers instead of an addressable local vec3 array.
                const float relative_value=component==0?relative[0]:(component==1?relative[1]:relative[2]);
                const float known_value=component==0?known[0]:(component==1?known[1]:known[2]);
                float diag=settings.cfm;
                for(int d=0;d<29;++d){const float z=s[192+component*32+d];diag+=z*z;
                    if(d>=23&&s[96+component*32+d]!=0.0f)wp::atomic_max(&out.secondary_nonzero.data[world],1);}
                const float phi=component==0?s[79]:0.0f,e=component==0?s[81]:0.0f;
                float bias=component==0?(phi<=0.0f?settings.bias_scale*settings.beta:settings.contact_speculative_scale)*phi/settings.dt:0.0f;
                if(e>0.0f&&relative_value < -settings.restitution_velocity_threshold &&
                   (phi<=1.0e-6f||phi+settings.dt*relative_value<=1.0e-6f))bias=e*relative_value;
                out.r0.data[id]=relative_value+bias;out.rhs.data[id]=known_value+bias;out.diag.data[id]=diag;
                out.row_type.data[id]=component==0?0:2;out.row_parent.data[id]=component==0?-1:slot;
                out.row_mu.data[id]=component==0?s[80]:s[82];out.row_beta.data[id]=component==0?settings.beta:0.0f;
                out.row_cfm.data[id]=settings.cfm;out.phi.data[id]=phi;out.target_velocity.data[id]=-known_value;out.row_restitution.data[id]=e;
                if(out.row_w.shape[0]>1)out.row_w.data[id]=1.0f;
                if(!isfinite(relative_value)||!isfinite(bias)||!isfinite(diag)||!isfinite(out.row_mu.data[id]))wp::atomic_max(&out.status.data[world],3);
                out.valid.data[id]=1;
            }
        }
        sync();
    }
"""
_CONTACT = _PRELUDE + _TRIPLET


@wp.func_native(_CONTACT)
def _contact(
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
): ...


@functools.cache
def get_contact_kernel(arch):
    """Original eight-descriptor ABI, same raw worker grid and 32-thread CTA."""

    @wp.kernel(module="unique", enable_backward=False)
    def kinetic_contact_triplet(
        plan: KineticPlan,
        current: CurrentKineticCache,
        held: HeldKineticOperator,
        raw: RawRowInput,
        state: RowState,
        settings: RowSettings,
        arm: ArmMap,
        out: DenseRowOutput,
    ):
        worker, logical = wp.tid()
        lanes = original._lane()
        if lanes[1] == 1 and logical != 0:
            return
        address = _storage()
        _contact(address, worker, plan, current, held, raw, state, settings, arm, out)
        original._release(address)

    return kinetic_contact_triplet


def install(rows):
    """Replace only contact entry2 in the already-owned launch closure's list."""
    closure = inspect.getclosurevars(rows.launch).nonlocals
    kernels = closure.get("kernels")
    arguments = closure.get("arguments")
    if not isinstance(kernels, list) or len(kernels) != 4:
        raise ValueError("Expected original four-owner launch closure")
    if arguments is not rows.arguments or len(arguments) != 4 or len(arguments[2]) != 8:
        raise ValueError("Original contact launch arguments no longer match")
    if hasattr(rows, "kernels") and rows.kernels is not kernels:
        raise ValueError("Rows kernel list is not the actual launch list")
    arch = str(wp.get_device(rows.device).arch)
    candidate = get_contact_kernel(arch)
    if kernels[2] not in (original.get_contact_kernel(arch), candidate):
        raise ValueError("Refusing to overwrite a different contact successor")
    rows.kernels = kernels
    kernels[2] = candidate
    return rows
