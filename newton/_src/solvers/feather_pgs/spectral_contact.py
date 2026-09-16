# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental ordered spectral GS, retaining original row production/decode.

Translate frozen CPU control a39eb83e without EX1, Nesterov, local roots or
Schur preparation. Cold supported worlds receive at most 24 complete sweeps;
other configurations retain the original owner. This cost screen is unqualified.
"""

import ast
import functools
import hashlib
import inspect
import linecache
import textwrap

import warp as wp

from . import coupled_contact

_LOCAL_NATIVE = r"""
auto sg_project = [](const wp::vec3& old, const wp::vec3& residual,
                     const wp::vec4& coefficients, float friction) -> wp::vec3 {
    const float normal=wp::max(0.0f,old[0]-residual[0]/coefficients[0]);
    const float delta=normal-old[0];
    float x=old[1]-(residual[1]+coefficients[2]*delta)/coefficients[1];
    float y=old[2]-(residual[2]+coefficients[3]*delta)/coefficients[1];
    const float radius=wp::max(0.0f,friction*normal);
#if defined(__CUDA_ARCH__)
    const float length=hypotf(x,y);
#else
    const float length=wp::sqrt(x*x+y*y);
#endif
    if(length>radius){const float factor=radius/length;x*=factor;y*=factor;}
    return wp::vec3(normal,x,y);
};
"""


@wp.func_native(_LOCAL_NATIVE + "\nreturn sg_project(old, residual, coefficients, friction);\n")
def project_contact(old: wp.vec3, residual: wp.vec3, coefficients: wp.vec4, friction: float) -> wp.vec3:
    """Apply normal then one common spectral tangent action for CPU controls."""


_NATIVE = r"""
    __shared__ float sg_diag[AM],sg_physical[AM],sg_cross[AM],sg_spectral[AM];
    __shared__ int sg_changed;
    auto sg_any = [&](int value) {
        if(NT==32)return __any_sync(MASK,value);
        return __syncthreads_or(value);
    };
    auto sg_max = [&](float value) {
        for(int shift=16;shift>0;shift>>=1)value=fmaxf(value,__shfl_xor_sync(MASK,value,shift));
        if(NT>32){
            if((lane&31)==0)s_dot[lane>>5]=value;
            SYNC();value=fmaxf(s_dot[0],s_dot[1]);SYNC();
        }
        return value;
    };
    int sg_bad=row_phase!=0||freeze_drive_rows!=0||friction_start_iteration!=0||
        iteration_offset!=0||regularize!=0||omega!=1.0f;
    if(lane<n_rows){
        sg_bad|=!isfinite(s_lam0[lane])||s_lam0[lane]!=0.0f||!isfinite(s_rhs[lane])||
            s_kind[lane]<0||s_kind[lane]>1;
        float norm=0.0f;
        for(int d=0;d<18;++d){norm+=Jr[d]*Jr[d];sg_bad|=!isfinite(Jr[d]);}
        const float cfm=s_row_c[lane]>=0?wr_pgs_cfm:world_row_cfm.data[off_dense+lane];
        sg_diag[lane]=norm+cfm;sg_physical[lane]=norm;sg_spectral[lane]=norm;
        sg_bad|=!isfinite(cfm)||!isfinite(norm)||!(norm>0.0f)||
            !isfinite(sg_diag[lane])||!(sg_diag[lane]>0.0f);
        if(s_kind[lane]==1){
            const int p=s_parent[lane];
            if(p<0||p+2>=n_rows||lane<p+1||lane>p+2)sg_bad=1;
            else sg_bad|=s_kind[p]!=0||s_kind[p+1]!=1||s_kind[p+2]!=1||
                s_parent[p+1]!=p||s_parent[p+2]!=p||!isfinite(s_mu[p+1])||
                s_mu[p+1]<0.0f||s_mu[p+1]!=s_mu[p+2];
        }
    }
    sg_bad=sg_any(sg_bad);
    SYNC();
    if(!sg_bad){
        if(lane<n_rows&&s_kind[lane]==0&&lane+2<n_rows&&s_kind[lane+1]==1&&
           s_parent[lane+1]==lane&&s_parent[lane+2]==lane){
            float a01=0.0f,a02=0.0f,a12=0.0f;
            for(int d=0;d<18;++d){
                const float z1=ZAT(d,lane+1),z2=ZAT(d,lane+2);
                a01+=Jr[d]*z1;a02+=Jr[d]*z2;a12+=z1*z2;
            }
            sg_cross[lane]=a01;sg_cross[lane+1]=a02;
            const float a=sg_physical[lane+1],c=sg_physical[lane+2];
            const float spectral=0.5f*(a+c)+hypotf(0.5f*(a-c),a12);
            const float cfm1=s_row_c[lane+1]>=0?wr_pgs_cfm:world_row_cfm.data[off_dense+lane+1];
            const float cfm2=s_row_c[lane+2]>=0?wr_pgs_cfm:world_row_cfm.data[off_dense+lane+2];
            const float denominator=spectral+fmaxf(cfm1,cfm2);
            sg_spectral[lane+1]=spectral;sg_spectral[lane+2]=spectral;
            sg_cross[lane+2]=denominator;
            sg_bad|=!isfinite(a01)||!isfinite(a02)||!isfinite(spectral)||
                !(spectral>0.0f)||!isfinite(denominator)||!(denominator>0.0f);
        }
        sg_bad=sg_any(sg_bad);
    }
    SYNC();
    if(!sg_bad){
        if(lane<18)s_dv[lane]=0.0f;
        if(lane>=n_rows&&lane<((n_rows+3)&~3)){s_x[lane]=0.0f;s_y[lane]=0.0f;}
        SYNC();
        for(int pass=0;pass<24;++pass){
            if(lane==0)sg_changed=0;
            SYNC();
            // Only warp0 owns ordered transactions; the second warp retains
            // original row staging, end-pass row scans and physical decode.
            if(lane<32){
                int row=0;
                while(row<n_rows){
                    const bool triple=s_kind[row]==0&&row+2<n_rows&&s_kind[row+1]==1&&
                        s_parent[row+1]==row&&s_parent[row+2]==row;
                    const float z0=lane<18?ZAT(lane,row):0.0f;
                    const float z1=lane<18&&triple?ZAT(lane,row+1):0.0f;
                    const float z2=lane<18&&triple?ZAT(lane,row+2):0.0f;
                    const float du=lane<18?s_dv[lane]:0.0f;
                    float r0=z0*du,r1=z1*du,r2=z2*du;
                    for(int shift=16;shift>0;shift>>=1){
                        r0+=__shfl_down_sync(MASK,r0,shift);
                        r1+=__shfl_down_sync(MASK,r1,shift);
                        r2+=__shfl_down_sync(MASK,r2,shift);
                    }
                    float d0=0.0f,d1=0.0f,d2=0.0f;
                    if(lane==0){
                        r0+=s_rhs[row];
                        if(triple){
                            const wp::vec3 old(s_x[row],s_x[row+1],s_x[row+2]);
                            const wp::vec3 residual(r0,r1+s_rhs[row+1],r2+s_rhs[row+2]);
                            const wp::vec4 coefficients(sg_diag[row],sg_cross[row+2],sg_cross[row],sg_cross[row+1]);
                            const wp::vec3 trial=sg_project(old,residual,coefficients,s_mu[row+1]);
                            d0=trial[0]-old[0];d1=trial[1]-old[1];d2=trial[2]-old[2];
                            s_x[row]=trial[0];s_x[row+1]=trial[1];s_x[row+2]=trial[2];
                        }else{
                            const float old=s_x[row],value=fmaxf(0.0f,old-r0/sg_diag[row]);
                            s_x[row]=value;d0=value-old;
                        }
                        if(d0!=0.0f||d1!=0.0f||d2!=0.0f)sg_changed=1;
                    }
                    d0=__shfl_sync(MASK,d0,0);d1=__shfl_sync(MASK,d1,0);d2=__shfl_sync(MASK,d2,0);
                    if(lane<18)s_dv[lane]+=z0*d0+z1*d1+z2*d2;
                    __syncwarp(MASK);row+=triple?3:1;
                }
            }
            SYNC();
            float scale=1.0f;
            if(lane<n_rows){
                float r=s_rhs[lane];for(int d=0;d<18;++d)r+=Jr[d]*s_dv[d];
                s_y[lane]=r;scale=fmaxf(scale,fmaxf(fabsf(s_rhs[lane]),fabsf(r-s_rhs[lane])));
            }
            scale=sg_max(scale);
            SYNC();
            int failed=0;
            if(lane<n_rows){
                const float value=s_x[lane],r=s_y[lane];
                failed=!isfinite(value)||!isfinite(r);
                float natural=0.0f;
                if(s_kind[lane]==0){
                    natural=fabsf(sg_physical[lane]*(value-fmaxf(0.0f,value-r/sg_physical[lane])));
                    failed|=!(value> -1.0e-7f)||!(r> -3.0e-5f)||!(fabsf(value*r)<3.0e-5f);
                }else if(lane==s_parent[lane]+1){
                    const float other=s_x[lane+1],rr=s_y[lane+1];
                    const float radius=fmaxf(0.0f,s_mu[lane]*s_x[s_parent[lane]]);
                    const float spectral=sg_spectral[lane];
                    const float x=value-r/spectral,y=other-rr/spectral,length=hypotf(x,y);
                    const float factor=length>radius?radius/length:1.0f;
                    natural=spectral*fmaxf(fabsf(value-x*factor),fabsf(other-y*factor));
                    const float mdp=value*r+other*rr+radius*hypotf(r,rr);
                    failed|=!(hypotf(value,other)-radius<3.0e-5f)||!(mdp<3.0e-5f);
                }
                failed|=!(natural<3.0e-5f*scale);
            }
            if(sg_any(failed)==0||sg_changed==0)break;
        }
    }
"""


def _rewrite_native(source):
    """Retire EX1 and Nesterov only for the fully admitted cold configuration."""
    start = "    // b'_i = rhs_i + J_i . (v_in - Y^T lam0)"
    source = coupled_contact._replace_once(source, start, _LOCAL_NATIVE + _NATIVE + "\n    if (sg_bad) {\n" + start)
    end = "    SYNC();\n#else\n#if 1\n    // In-kernel response in whitened coordinates:"
    source = coupled_contact._replace_once(
        source, end, "    SYNC();\n    }\n#else\n#if 1\n    // In-kernel response in whitened coordinates:"
    )
    source = coupled_contact._replace_once(source, "    float t_k = 1.0f;", "    if (sg_bad) {\n    float t_k = 1.0f;")
    return coupled_contact._replace_once(
        source, "\n#if 1\n    // u = Z^T (x - lam0)", "\n    }\n#if 1\n    // u = Z^T (x - lam0)"
    )


@functools.cache
def get_parallel_factory(original):
    """Return an opt-in ABI-identical factory with original static fallbacks."""
    selected = coupled_contact.get_parallel_factory(original)
    successor = inspect.getclosurevars(selected).nonlocals["successor"]
    tree = ast.parse(textwrap.dedent(inspect.getsource(successor)))
    factory = tree.body[0]
    factory.name = "spectral_contact_parallel_factory"
    for node in ast.walk(factory):
        if isinstance(node, ast.Constant) and node.value == "_cc6":
            node.value = "_sgs24"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_fpgs_coupled_contact"):
            node.attr = node.attr.replace("_fpgs_coupled_contact", "_fpgs_spectral_contact")
    ast.fix_missing_locations(tree)
    generated = ast.unparse(tree) + "\n"
    filename = f"<spectral-contact-{hashlib.sha256(generated.encode()).hexdigest()}>"
    linecache.cache[filename] = (len(generated), None, generated.splitlines(keepends=True), filename)
    namespace = dict(successor.__globals__, _rewrite_native=_rewrite_native)
    exec(compile(generated, filename, "exec"), namespace)
    implementation = namespace[factory.name]
    signature = inspect.signature(original)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if not coupled_contact._supported(bound.arguments, original.__globals__):
            return original(*args, **kwargs)
        return implementation(*args, **kwargs)

    return wrapped
