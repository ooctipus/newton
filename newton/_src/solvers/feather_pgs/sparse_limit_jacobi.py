# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental simultaneous limit prefix with physical descent admission.

Only joint limits change iteration order. Contact transactions, the maximum
outer allowance, incoming-impulse convention, and canonical W/Z publication
remain the original spectral owner's. Rejection runs its untouched prefix.
"""

import functools

from . import sparse_factor_rows, sparse_spectral_tangents

ENERGY_MARGIN = 64.0 * 2.0**-24


def prefix_capacity(dof_count: int) -> int:
    """Lower and upper candidate rows for every coordinate."""
    return 2 * dof_count


def scratch_bytes(dof_count: int) -> int:
    """Shared scratch of the simultaneous limit step."""
    return (prefix_capacity(dof_count) + dof_count) * 4


_SETUP = r"""
    // No global allocation: actual packed Z is the physical response authority.
    __shared__ float limit_delta[PREFIX_CAPACITY];
    __shared__ float limit_step[DOF_COUNT];
    int limit_count=0;
    if(lane==0)while(limit_count<count && row_type.data[base+limit_count]==3)++limit_count;
    limit_count=__shfl_sync(0xffffffff,limit_count,0);
    int limit_bad=limit_count>PREFIX_CAPACITY;
    if(limit_count<=PREFIX_CAPACITY)for(int r=lane;r<limit_count;r+=32) {
        const int tpl=d.support.data[base+r];
        bool good=tpl>=0 && tpl<p.support_nodes.shape[0];
        if(good) {
            const int length=p.support_count.data[tpl];
            good=length>0 && length<=SUPPORT_WIDTH;
        }
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
                        const int node=p.support_nodes.data[tpl*SUPPORT_WIDTH+k];
                        const float z=d.Z.data[(base+r)*SUPPORT_WIDTH+k];
                        if(node<0 || node>=DOF_COUNT || !isfinite(z))bad_proposal=1;
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
                        limit_delta[r]=delta;
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
                    // Distinct support nodes make each row race-free. Preserve
                    // prefix order between rows, without W/index/sign gathers.
                    // Only scratch changes before the complete energy check.
                    limit_step[lane]=0.0f;
                    if(lane+32<DOF_COUNT)limit_step[lane+32]=0.0f;
                    __syncwarp();
                    for(int r=0;r<limit_count;++r) {
                        const float delta=limit_delta[r];
                        if(delta!=0.0f) {
                            const int tpl=d.support.data[base+r];
                            if(lane<p.support_count.data[tpl]) {
                                const int node=p.support_nodes.data[tpl*SUPPORT_WIDTH+lane];
                                limit_step[node]+=d.Z.data[(base+r)*SUPPORT_WIDTH+lane]*delta;
                            }
                            __syncwarp();
                        }
                    }
                    step0=limit_step[lane];
                    if(lane+32<DOF_COUNT)step1=limit_step[lane+32];
                    const int bad_response=!isfinite(step0) || !isfinite(step1);
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
                        if(lane+32<DOF_COUNT)du[lane+32]+=step1;
                        for(int r=lane;r<limit_count;r+=32) {
                            lam[r]+=limit_delta[r];
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


def get_fragments(capacity: int, *, support_width: int = 18, dof_count: int = 43):
    """Add only a prefix transaction to the unchanged spectral contact body."""
    if dof_count > 64:
        raise ValueError("The simultaneous limit step keeps two lanes of coordinates (at most 64)")
    setup, update = sparse_spectral_tangents.get_fragments(capacity, support_width=support_width, dof_count=dof_count)
    sized_setup, sized_update = _SETUP, _UPDATE
    for token, value in (
        ("PREFIX_CAPACITY", prefix_capacity(dof_count)),
        ("SUPPORT_WIDTH", support_width),
        ("DOF_COUNT", dof_count),
    ):
        sized_setup = sized_setup.replace(token, str(value))
        sized_update = sized_update.replace(token, str(value))
    return setup + sized_setup, sized_update + update


@functools.cache
def get_solve_kernel(*, dof_count: int, support_width: int, inverse_width: int, nonzero_count: int):
    """Keep the spectral owner's exact fifteen-argument, block32 solve ABI."""
    return sparse_factor_rows.get_solve_kernel(
        100,
        dof_count=dof_count,
        support_width=support_width,
        inverse_width=inverse_width,
        nonzero_count=nonzero_count,
        metric_tangents=True,
        spectral_tangents=True,
        limit_jacobi=True,
    )
