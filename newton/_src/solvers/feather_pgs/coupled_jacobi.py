# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental simultaneous Coulomb blocks with one permanent half-step latch.

This factory is opt-in only and has no solver hook of its own. It retains the
original row producers, capacities, ABI and physical decode. Admitted worlds
replace the all-pair majorizer and Nesterov iteration with at most 24 concurrent
block passes. Unsupported configurations retain the original complete owner.

The local root and metric fallback are pinned to coupled_contact 3e972df7.
The native physical cone stop is explicitly 3e-5, matching qualification,
rather than the CPU-double control's former 1e-10. Other physical bounds and
local root tolerances are unchanged. This research path is not qualified.
"""

import ast
import functools
import hashlib
import inspect
import linecache
import textwrap
from pathlib import Path

from . import coupled_contact

_ROOT_SHA = "3e972df7b7e2659716718595a49164a3a7ed9f94c2d874c0e29855dca3384015"

# Reuse exactly the already-reviewed row admission, diagonal and local-block
# cache producer. No ordered sweep or its end-pass scan is retained.
_SEAM = "        while(cc_consumed < 6){"
if coupled_contact._NATIVE.count(_SEAM) != 1:
    raise RuntimeError("Coupled Jacobi initialization seam changed")
_INITIAL = coupled_contact._NATIVE.split(_SEAM)[0]
_INITIAL = _INITIAL.replace("    __shared__ int cc_changed;\n", "")
_INITIAL = _INITIAL.replace("    int cc_consumed=0;\n", "").replace("    bool cc_done=false;\n", "")

_JACOBI_NATIVE = r"""
        // One root-owning normal lane writes all three disjoint proposals.
        // All reads remain at the same old state until the following CTA fence.
        auto cj_propose = [&](int row) -> wp::vec3 {
            const wp::vec3 old(s_x[row],s_x[row+1],s_x[row+2]);
            const wp::vec3 residual(s_rhs[row],s_rhs[row+1],s_rhs[row+2]);
            wp::mat_t<3,3,float> H(0.0f);
            H.data[0][0]=cc_diag[row];H.data[1][1]=cc_diag[row+1];H.data[2][2]=cc_diag[row+2];
            H.data[0][1]=H.data[1][0]=cc_cross[row];
            H.data[0][2]=H.data[2][0]=cc_cross[row+1];
            H.data[1][2]=H.data[2][1]=cc_cross[row+2];
            wp::vec3 bias=residual;
            for(int i=0;i<3;++i)for(int j=0;j<3;++j)bias[i]-=H.data[i][j]*old[j];
            const wp::vec4 prepared(cc_schur[row*4],cc_schur[row*4+1],cc_schur[row*4+2],cc_schur[row*4+3]);
            wp::vec_t<5,float> trial=cc_block(H,bias,s_mu[row+1],prepared);
            if(trial[3]==0.0f){
                const wp::vec4 metric=cc_metric(H,residual,old,s_mu[row+1]);
                for(int k=0;k<4;++k)trial[k]=metric[k];
            }
            if(trial[3]==0.0f){
                // Complete original scalar triple, in physical local-residual
                // coordinates: CFM belongs only to each update denominator.
                wp::vec3 value=old;
                for(int k=0;k<3;++k){
                    float r=residual[k];
                    for(int j=0;j<3;++j){
                        const float g=k==j?cc_physical_diag[row+k]:H.data[k][j];
                        r+=g*(value[j]-old[j]);
                    }
                    float next=value[k]-r/cc_diag[row+k];
                    if(k==0)next=fmaxf(next,0.0f);
                    else{
                        const int sibling=k==1?2:1;
                        const float radius=fmaxf(s_mu[row+k]*value[0],0.0f);
                        const float length=hypotf(next,value[sibling]);
                        if(radius<=0.0f)next=0.0f;
                        else if(length>radius){const float scale=radius/length;next*=scale;value[sibling]*=scale;}
                    }
                    value[k]=next;
                }
                return value;
            }
            return wp::vec3(trial[0],trial[1],trial[2]);
        };
        const float cone_tolerance=3.0e-5f;
        auto cj_measure = [&](const float* impulse,const float* residual) {
            float energy=0.0f;
            if(lane<n_rows){
                const float value=impulse[lane],r=residual[lane];
                if(!isfinite(value)||!isfinite(r))energy=__int_as_float(0x7f800000);
                else if(s_kind[lane]==0){
                    const float correction=value-fmaxf(0.0f,value-r/cc_diag[lane]);
                    energy=fmaxf(energy,-value/cone_tolerance);
                    energy=fmaxf(energy,-r/3.0e-5f);
                    energy=fmaxf(energy,fabsf(value*r)/3.0e-5f);
                    energy=fmaxf(energy,sqrtf(cc_diag[lane])*fabsf(correction)/(1.0e-5f*cc_scale));
                }else if(lane==s_parent[lane]+1){
                    const float other=impulse[lane+1],rr=residual[lane+1];
                    const float radius=fmaxf(0.0f,s_mu[lane]*impulse[s_parent[lane]]);
                    const float diag=fmaxf(cc_diag[lane],cc_diag[lane+1]);
                    const float x=value-r/diag,y=other-rr/diag,length=hypotf(x,y);
                    const float factor=length>radius?radius/fmaxf(length,1.0e-30f):1.0f;
                    const float mdp=fabsf(value*r+other*rr+radius*hypotf(r,rr));
                    if(!isfinite(mdp))energy=__int_as_float(0x7f800000);
                    energy=fmaxf(energy,(hypotf(value,other)-radius)/cone_tolerance);
                    energy=fmaxf(energy,mdp/3.0e-5f);
                    energy=fmaxf(energy,sqrtf(cc_diag[lane])*fabsf(value-x*factor)/(1.0e-5f*cc_scale));
                    energy=fmaxf(energy,sqrtf(cc_diag[lane+1])*fabsf(other-y*factor)/(1.0e-5f*cc_scale));
                }
                if(!isfinite(energy))energy=__int_as_float(0x7f800000);
            }
            return cc_max(energy);
        };
        // The original float4 transpose reads the last padded row group.
        if(lane>=n_rows&&lane<((n_rows+3)&~3)){s_x[lane]=0.0f;s_y[lane]=0.0f;}
        SYNC();
        float cj_alpha=1.0f;
        float cj_energy=cj_measure(s_x,s_rhs);
        for(int pass=0;pass<24;++pass){
            if(lane<n_rows&&s_kind[lane]==0){
                const bool triple=lane+2<n_rows&&s_kind[lane+1]==1&&
                    s_parent[lane+1]==lane&&s_parent[lane+2]==lane;
                if(triple){
                    const wp::vec3 trial=cj_propose(lane);
                    for(int k=0;k<3;++k){
                        const float old=s_x[lane+k];
                        s_y[lane+k]=old+cj_alpha*(trial[k]-old);
                    }
                }else{
                    const float old=s_x[lane];
                    const float trial=fmaxf(0.0f,old-s_rhs[lane]/cc_diag[lane]);
                    s_y[lane]=old+cj_alpha*(trial-old);
                }
            }
            SYNC();
            // One complete Z^T delta, retaining the existing row-chunk layout.
            constexpr int NCH=(NT/18)>0?NT/18:1;
            const int n4=(n_rows+3)>>2,per=(n4+NCH-1)/NCH;
            if(lane<NCH*18){
                const int d=lane%18,ch=lane/18;
                const float4* proposal=reinterpret_cast<const float4*>(s_y);
                const float4* previous=reinterpret_cast<const float4*>(s_x);
                const float4* coefficients=reinterpret_cast<const float4*>(&s_Yt[d*YS]);
                float a0=0.0f,a1=0.0f,a2=0.0f,a3=0.0f;
                const int end=min(n4,(ch+1)*per);
                for(int j4=ch*per;j4<end;++j4){
                    const float4 y=proposal[j4],x=previous[j4],z=coefficients[j4];
                    a0+=z.x*(y.x-x.x);a1+=z.y*(y.y-x.y);
                    a2+=z.z*(y.z-x.z);a3+=z.w*(y.w-x.w);
                }
                if(NCH==1)s_dv[d]=(a0+a1)+(a2+a3);
                else s_dvp[ch*18+d]=(a0+a1)+(a2+a3);
            }
            SYNC();
            if(NCH>1){
                if(lane<18){
                    float value=0.0f;
                    for(int ch=0;ch<NCH;++ch)value+=s_dvp[ch*18+lane];
                    s_dv[lane]=value;
                }
                SYNC();
            }
            // One complete Z delta_u; the resulting residual also serves stop
            // and next-pass proposals, with no extra end-pass operator scan.
            if(lane<n_rows){
                float r=s_rhs[lane];
                for(int d=0;d<18;++d)r+=Jr[d]*s_dv[d];
                s_step[lane]=r;
            }
            SYNC();
            float candidate_energy=cj_measure(s_y,s_step);
            if(cj_alpha==1.0f&&candidate_energy>cj_energy){
                cj_alpha=0.5f;
                if(lane<n_rows){
                    s_y[lane]=0.5f*(s_x[lane]+s_y[lane]);
                    s_step[lane]=0.5f*(s_rhs[lane]+s_step[lane]);
                }
                SYNC();
                candidate_energy=cj_measure(s_y,s_step);
            }
            if(lane<n_rows){s_x[lane]=s_y[lane];s_rhs[lane]=s_step[lane];}
            SYNC();
            cj_energy=candidate_energy;
            if(cj_energy<=1.0f)break;
        }
    }
"""


def _rewrite_native(source):
    """Replace only the old solve region; keep unsupported runtime fallback."""
    start = "    // b'_i = rhs_i + J_i . (v_in - Y^T lam0)"
    replacement = coupled_contact._LOCAL_NATIVE + coupled_contact._METRIC_NATIVE + _INITIAL + _JACOBI_NATIVE
    source = coupled_contact._replace_once(source, start, replacement + "\n    if (cc_bad) {\n" + start)
    end = "    SYNC();\n#else\n#if 1\n    // In-kernel response in whitened coordinates:"
    source = coupled_contact._replace_once(
        source, end, "    SYNC();\n    }\n#else\n#if 1\n    // In-kernel response in whitened coordinates:"
    )
    source = coupled_contact._replace_once(source, "    float t_k = 1.0f;", "    if (cc_bad) {\n    float t_k = 1.0f;")
    return coupled_contact._replace_once(
        source, "\n#if 1\n    // u = Z^T (x - lam0)", "\n    }\n#if 1\n    // u = Z^T (x - lam0)"
    )


@functools.cache
def get_parallel_factory(original):
    """Return the explicitly selected ABI-identical Jacobi factory."""
    if hashlib.sha256(Path(coupled_contact.__file__).read_bytes()).hexdigest() != _ROOT_SHA:
        raise RuntimeError("Coupled Jacobi frozen local root changed")
    selected = coupled_contact.get_parallel_factory(original)
    successor = inspect.getclosurevars(selected).nonlocals["successor"]
    tree = ast.parse(textwrap.dedent(inspect.getsource(successor)))
    factory = tree.body[0]
    factory.name = "coupled_jacobi_parallel_factory"
    for node in ast.walk(factory):
        if isinstance(node, ast.Constant) and node.value == "_cc6":
            node.value = "_ccj24"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_fpgs_coupled_contact"):
            node.attr = node.attr.replace("_fpgs_coupled_contact", "_fpgs_coupled_jacobi")
    ast.fix_missing_locations(tree)
    generated = ast.unparse(tree) + "\n"
    filename = f"<coupled-jacobi-{hashlib.sha256(generated.encode()).hexdigest()}>"
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
