# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental simultaneous limit prefix with physical descent admission.

Only joint limits change iteration order. Contact transactions, the maximum
outer allowance, incoming-impulse convention, and canonical W/Z publication
remain the original spectral owner's. Rejection runs its untouched prefix.
"""

import functools
import inspect
import linecache
import textwrap

from . import sparse_factor_rows, sparse_spectral_tangents

PREFIX_CAPACITY = 86
SCRATCH_BYTES = PREFIX_CAPACITY * 8
ENERGY_MARGIN = 64.0 * 2.0**-24

_SETUP = r"""
    // No global allocation: existing signed W columns describe every limit.
    __shared__ float limit_delta[86];
    __shared__ int limit_column[86];
    int limit_count=0;
    if(lane==0)while(limit_count<count && row_type.data[base+limit_count]==3)++limit_count;
    limit_count=__shfl_sync(0xffffffff,limit_count,0);
    int limit_bad=limit_count>86;
    if(limit_count<=86)for(int r=lane;r<limit_count;r+=32) {
        const int tpl=d.support.data[base+r];
        bool good=tpl>0 && tpl<p.support_nodes.shape[0];
        int column=0;float first=0.0f;
        if(good) {
            const int length=p.support_count.data[tpl];
            column=p.support_nodes.data[tpl*18];
            good=length>0 && length<=18 && column>=0 && column<37;
            if(good)good=p.limit_support.data[42-column]==tpl;
            if(good) {
                const int entry=p.index.data[column*43+column];
                first=d.Z.data[(base+r)*18];
                const float diagonal_w=entry>=0?d.W.data[group*434+entry]:0.0f;
                good=isfinite(first) && isfinite(diagonal_w) && diagonal_w>0.0f &&
                    fabsf(first)==diagonal_w;
            }
        }
        limit_column[r]=first<0.0f?~column:column;
        if(!good)limit_bad=1;
    }
    const bool limit_admitted=limit_count>0 && !__any_sync(0xffffffff,limit_bad);
    __syncwarp();
"""

_UPDATE = r"""
            if(row==0 && limit_admitted && omega==1.0f) {
                // All tiles read the same old du. Four lanes own one row.
                const int sublane=lane&3, row_group=lane>>2;
                int bad_proposal=0, any_delta=0;
                float linear=0.0f;
                for(int tile=0;tile<limit_count;tile+=8) {
                    const int r=tile+row_group;
                    const bool live=r<limit_count;
                    int length=0,tpl=0;
                    if(live) {tpl=d.support.data[base+r];length=p.support_count.data[tpl];}
                    float dot=0.0f;
                    for(int k=sublane;k<length;k+=4) {
                        const int node=p.support_nodes.data[tpl*18+k];
                        const float z=d.Z.data[(base+r)*18+k];
                        if(node<0 || node>=43 || !isfinite(z))bad_proposal=1;
                        else dot+=z*du[node];
                    }
                    dot+=__shfl_down_sync(0xffffffff,dot,2,4);
                    dot+=__shfl_down_sync(0xffffffff,dot,1,4);
                    if(live && sublane==0) {
                        const float old=lam[r],denominator=diagonal.data[base+r];
                        const float residual=dot+d.incident.data[base+r]+rhs.data[base+r];
                        const bool good=isfinite(old) && old>=0.0f && isfinite(denominator) &&
                            denominator>0.0f && isfinite(residual);
                        const float trial=good?old-residual/denominator:old;
                        const float next=fmaxf(trial,0.0f);
                        const float delta=next-old;
                        if(!good || !isfinite(trial) || !isfinite(next) || !isfinite(delta))bad_proposal=1;
                        limit_delta[r]=limit_column[r]<0?-delta:delta;
                        linear+=residual*delta;
                        any_delta|=delta!=0.0f;
                    }
                }
                __syncwarp();
                const bool proposal_ok=!__any_sync(0xffffffff,bad_proposal);
                const bool nonzero=__any_sync(0xffffffff,any_delta);
                bool accepted=proposal_ok && !nonzero;
                float step0=0.0f,step1=0.0f;
                if(proposal_ok && nonzero) {
                    // Node-owned deterministic accumulation; never mutate du
                    // or lambda until the complete physical energy is checked.
                    int bad_response=0;
                    for(int r=0;r<limit_count;++r) {
                        const float delta=limit_delta[r];
                        if(delta!=0.0f) {
                            const int code=limit_column[r],column=code<0?~code:code;
                            const int entry0=p.index.data[lane*43+column];
                            if(entry0>=0) {
                                const float value=d.W.data[group*434+entry0];
                                bad_response|=!isfinite(value);
                                step0+=value*delta;
                            }
                            if(lane+32<43) {
                                const int entry1=p.index.data[(lane+32)*43+column];
                                if(entry1>=0) {
                                    const float value=d.W.data[group*434+entry1];
                                    bad_response|=!isfinite(value);
                                    step1+=value*delta;
                                }
                            }
                        }
                    }
                    bad_response|=!isfinite(step0) || !isfinite(step1);
                    float norm=step0*step0+step1*step1;
                    for(int shift=16;shift>0;shift>>=1) {
                        linear+=__shfl_down_sync(0xffffffff,linear,shift);
                        norm+=__shfl_down_sync(0xffffffff,norm,shift);
                    }
                    const bool response_ok=!__any_sync(0xffffffff,bad_response);
                    if(lane==0) {
                        const float energy=linear+0.5f*norm,scale=fabsf(linear)+0.5f*norm;
                        // Scale-aware numerical guard, not a formal FP32 proof
                        // or a convergence/stop certificate. CFM is absent.
                        accepted=response_ok && isfinite(energy) && isfinite(scale) &&
                            energy<=-0x1p-18f*scale;
                    }
                    accepted=__shfl_sync(0xffffffff,static_cast<int>(accepted),0)!=0;
                }
                if(accepted) {
                    if(nonzero) {
                        du[lane]+=step0;
                        if(lane+32<43)du[lane+32]+=step1;
                        for(int r=lane;r<limit_count;r+=32) {
                            const float delta=limit_delta[r];
                            lam[r]+=limit_column[r]<0?-delta:delta;
                        }
                        changed=1;
                    }
                    __syncwarp();
                    row=limit_count-1;continue;
                }
                // Original row zero and remaining prefix execute below from
                // unchanged lam/du. No extra committed outer pass is added.
            }
"""


def get_fragments(capacity: int):
    """Add only a prefix transaction to the unchanged spectral contact body."""
    setup, update = sparse_spectral_tangents.get_fragments(capacity)
    return setup + _SETUP, _UPDATE + update


@functools.cache
def get_solve_kernel():
    """Keep the spectral owner's exact fifteen-argument, block32 solve ABI."""
    original = sparse_factor_rows.get_solve_kernel
    source = textwrap.dedent(inspect.getsource(original))
    source = source[source.index("def get_solve_kernel(") :]
    replace = sparse_spectral_tangents._replace
    source = replace(
        source, "from .sparse_metric_tangents import get_fragments", "from .sparse_limit_jacobi import get_fragments"
    )
    source = replace(
        source,
        "        diagonal: wp.array2d[float],",
        "        diagonal: wp.array2d[float],\n        cfm: wp.array2d[float],",
        2,
    )
    source = replace(
        source,
        "            diagonal,\n            impulses,",
        "            diagonal,\n            cfm,\n            impulses,",
    )
    source = replace(
        source, '"sparse_metric_tangent" if metric_tangents', '"sparse_spectral_limit_jacobi" if metric_tangents'
    )
    filename = "<sparse_limit_jacobi_factory>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = dict(original.__wrapped__.__globals__)
    exec(compile(source, filename, "exec"), namespace)
    return namespace["get_solve_kernel"](100, metric_tangents=True)
