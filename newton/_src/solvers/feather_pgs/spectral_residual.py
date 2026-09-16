# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental pipelined spectral proposals with separate Q-or-N half latch.

Preserve current rows/held operator, cold admission, original CTA, at most 24
corrections and full physical stopping. Unrelaxed lookahead supplies both the
next proposal and a proved natural-residual bound. Absolute normal/comp merit
is a separate latch stream, so a loose Q cannot mask its increase. No EX1,
history, local root, extra operator or old per-pass full-merit scan survives.
"""

import ast
import functools
import hashlib
import inspect
import linecache
import textwrap
from pathlib import Path

import warp as wp

from . import coupled_contact, spectral_jacobi

SOURCE_PINS = dict(spectral_jacobi.SOURCE_PINS)
SOURCE_PINS[spectral_jacobi.__file__] = "16a112d3661f3470ddf2c581ed28f14d9255db2d32a696694af2502c69d7c32c"

_METRICS_NATIVE = r"""
    const float dn=fabsf(trial[0]-old[0]);
    float q=wp::max(weights[0]*dn,
        weights[1]*(fabsf(trial[1]-old[1])+fabsf(trial[2]-old[2]))+weights[2]*dn);
    float normal=wp::max(wp::max(-residual[0],0.0f),fabsf(old[0]*residual[0]))*(1.0f/3.0e-5f);
    for(int k=0;k<3;++k){
        if(!isfinite(old[k])||!isfinite(residual[k])||!isfinite(trial[k])){q=INFINITY;normal=INFINITY;}
    }
    if(!isfinite(q)||!isfinite(normal)){q=INFINITY;normal=INFINITY;}
    return wp::vec2(q,normal);
"""


@wp.func_native(_METRICS_NATIVE)
def proposal_metrics(old: wp.vec3, residual: wp.vec3, trial: wp.vec3, weights: wp.vec3) -> wp.vec2:
    """Return unrelaxed Q/1e-5 and current-state absolute normal/comp merit."""


_LOCAL_NATIVE = (
    spectral_jacobi._LOCAL_NATIVE
    + r"""
    auto sr_metrics = [&](const wp::vec3& old,const wp::vec3& residual,
                          const wp::vec3& trial,const wp::vec3& weights) -> wp::vec2 {
"""
    + _METRICS_NATIVE
    + "\n    };\n"
)

# Two extra normal-owner coefficient registers; no extra shared/global arrays.
# All original diagonal reads finish before the cached reciprocal overwrite.
_CACHE_PREP = spectral_jacobi._CACHE_PREP.replace(
    "    float sj_scale=1.0f;", "    float sj_scale=1.0f,sr_weight_t=0.0f,sr_weight_n=0.0f;"
)
_COEFFICIENT_SEAM = "            if(triple)spectral_inverse=1.0f/sg_cross[lane+2];"
if _CACHE_PREP.count(_COEFFICIENT_SEAM) != 1:
    raise RuntimeError("Frozen spectral coefficient seam changed")
_CACHE_PREP = _CACHE_PREP.replace(
    _COEFFICIENT_SEAM,
    r"""
            if(triple){
                const float s=sg_cross[lane+2],d=fmaxf(sg_diag[lane+1],sg_diag[lane+2]);
                const float weight_scale=1.0f/(sqrtf(d)*(1.0e-5f*sj_scale));
                sr_weight_t=s*weight_scale;
                sr_weight_n=(s_mu[lane+1]*s+fabsf(sg_cross[lane])+fabsf(sg_cross[lane+1]))*weight_scale;
                spectral_inverse=1.0f/s;
                sg_bad|=!(s>=d)||!isfinite(sr_weight_t)||!(sr_weight_t>0.0f)||
                    !isfinite(sr_weight_n)||!(sr_weight_n>=0.0f);
            }
""",
)

# Retain the literal, cached full physical gates only at Q permission/final.
_MEASURE_SEAM = "        // The original float4 transpose"
if spectral_jacobi._RECURRENCE.count(_MEASURE_SEAM) != 1:
    raise RuntimeError("Frozen full physical measure seam changed")
_MEASURE = spectral_jacobi._RECURRENCE.split(_MEASURE_SEAM)[0]
_OPERATOR_START = "            // One complete Z^T delta, retaining the existing row-chunk layout."
_OPERATOR_END = "            float candidate_energy=sj_measure(s_y,s_step);"
if spectral_jacobi._RECURRENCE.count(_OPERATOR_START) != 1 or spectral_jacobi._RECURRENCE.count(_OPERATOR_END) != 1:
    raise RuntimeError("Frozen physical delta-operator seam changed")
_OPERATORS = spectral_jacobi._RECURRENCE.split(_OPERATOR_START)[1].split(_OPERATOR_END)[0]
_OPERATORS = _OPERATOR_START + _OPERATORS

_LOOKAHEAD = r"""
        const bool sr_owner=lane<n_rows&&s_kind[lane]==0;
        const bool sr_triple=sr_owner&&lane+2<n_rows&&s_kind[lane+1]==1&&
            s_parent[lane+1]==lane&&s_parent[lane+2]==lane;
        // Three normal-owner registers retain T(x), never alpha*(T(x)-x).
        wp::vec3 sr_proposal(0.0f);
        auto sr_lookahead = [&](const float* impulse,const float* residual) -> wp::vec2 {
            float q=0.0f,normal=0.0f;
            if(sr_owner){
                const float old=impulse[lane],r=residual[lane];
                if(sr_triple){
                    const wp::vec3 x(old,impulse[lane+1],impulse[lane+2]);
                    const wp::vec3 v(r,residual[lane+1],residual[lane+2]);
                    const wp::vec4 coefficients(sg_diag[lane],sg_cross[lane+2],sg_cross[lane],sg_cross[lane+1]);
                    sr_proposal=sj_project_cached(x,v,coefficients,s_mu[lane+1]);
                    const wp::vec3 weights(sg_physical[lane],sr_weight_t,sr_weight_n);
                    const wp::vec2 score=sr_metrics(x,v,sr_proposal,weights);
                    q=score[0];normal=score[1];
                }else{
                    sr_proposal[0]=fmaxf(0.0f,old-r*sg_diag[lane]);
                    q=sg_physical[lane]*fabsf(sr_proposal[0]-old);
                    normal=fmaxf(fmaxf(-r,0.0f),fabsf(old*r))*sj_inverse_tolerance;
                    if(!isfinite(old)||!isfinite(r)||!isfinite(sr_proposal[0])||!isfinite(q)||!isfinite(normal)){
                        q=INFINITY;normal=INFINITY;
                    }
                }
            }
            // Two arithmetic reductions share the same cross-warp joins.
            // Delta_u is dead after the preceding row-action fence and is
            // overwritten by the next operator / original final decode.
            for(int shift=16;shift>0;shift>>=1){
                q=fmaxf(q,__shfl_xor_sync(MASK,q,shift));
                normal=fmaxf(normal,__shfl_xor_sync(MASK,normal,shift));
            }
            if(NT>32){
                if((lane&31)==0){s_dv[(lane>>5)*2]=q;s_dv[(lane>>5)*2+1]=normal;}
                SYNC();
                q=fmaxf(s_dv[0],s_dv[2]);normal=fmaxf(s_dv[1],s_dv[3]);
                SYNC();
            }
            return wp::vec2(q,normal);
        };
        if(lane>=n_rows&&lane<((n_rows+3)&~3)){s_x[lane]=0.0f;s_y[lane]=0.0f;}
        SYNC();
        float sr_alpha=1.0f;
        wp::vec2 sr_previous=sr_lookahead(s_x,s_rhs);
        for(int pass=0;pass<24;++pass){
            if(sr_owner){
                const int count=sr_triple?3:1;
                for(int k=0;k<count;++k){
                    const float old=s_x[lane+k];
                    s_y[lane+k]=old+sr_alpha*(sr_proposal[k]-old);
                }
            }
            SYNC();
"""

_COMMIT = r"""
            // This final-pass lookahead is intentional and fully charged.
            wp::vec2 sr_trial=sr_lookahead(s_y,s_step);
            if(sr_alpha==1.0f&&(sr_trial[0]>sr_previous[0]||sr_trial[1]>sr_previous[1])){
                sr_alpha=0.5f;
                if(lane<n_rows){
                    s_y[lane]=0.5f*(s_x[lane]+s_y[lane]);
                    s_step[lane]=0.5f*(s_rhs[lane]+s_step[lane]);
                }
                SYNC();
                sr_trial=sr_lookahead(s_y,s_step);
            }
            if(lane<n_rows){s_x[lane]=s_y[lane];s_rhs[lane]=s_step[lane];}
            SYNC();
            sr_previous=sr_trial;
            if(sr_previous[0]<=1.0f||pass==23){
                const float physical_merit=sj_measure(s_x,s_rhs);
                if(physical_merit<=1.0f)break;
            }
        }
    }
"""

_RECURRENCE = "    if(!sg_bad){\n" + _MEASURE + _LOOKAHEAD + _OPERATORS + _COMMIT


def _rewrite_native(source):
    """Replace admitted solve work; retain all original fallback/publication."""
    start = "    // b'_i = rhs_i + J_i . (v_in - Y^T lam0)"
    replacement = _LOCAL_NATIVE + spectral_jacobi._INITIAL + _CACHE_PREP + _RECURRENCE
    source = coupled_contact._replace_once(source, start, replacement + "\n    if (sg_bad) {\n" + start)
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
    """Return a distinct default-off owner preserving original ABI/admission."""
    for name, digest in SOURCE_PINS.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Frozen residual-workflow dependency changed")
    selected = coupled_contact.get_parallel_factory(original)
    successor = inspect.getclosurevars(selected).nonlocals["successor"]
    tree = ast.parse(textwrap.dedent(inspect.getsource(successor)))
    factory = tree.body[0]
    factory.name = "spectral_residual_parallel_factory"
    for node in ast.walk(factory):
        if isinstance(node, ast.Constant) and node.value == "_cc6":
            node.value = "_sr24"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_fpgs_coupled_contact"):
            node.attr = node.attr.replace("_fpgs_coupled_contact", "_fpgs_spectral_residual")
    ast.fix_missing_locations(tree)
    generated = ast.unparse(tree) + "\n"
    filename = f"<spectral-residual-{hashlib.sha256(generated.encode()).hexdigest()}>"
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
