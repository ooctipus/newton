# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental small-world static-register Gram/residual representation.

The separate filtered fallback keeps the corrected limit-prefix and spectral
owner unchanged. No global Gram, cross-call response cache, or new iteration
policy is introduced. All small-world physics outputs commit transactionally.
"""

import functools
import inspect
import linecache
import textwrap

import warp as wp

from . import sparse_factor_rows, sparse_spectral_tangents
from .sparse_factor import SparseData, SparsePlan

CAPACITY = 100
SMALL_CAPACITY = 32
STAGING_BYTES = 32 * 44 * 4


def _prefix_source():
    """Keep simultaneous proposals and the actual-Z physical energy guard."""
    physical = []
    residual = []
    for row in range(32):
        physical.append(f"""
                    if({row}<limit_count) {{
                        const float delta=__shfl_sync(0xffffffff,proposal_delta,{row});
                        if(delta!=0.0f) {{
                            step0+=staged[{row}*44+lane]*delta;
                            if(lane+32<43)step1+=staged[{row}*44+lane+32]*delta;
                        }}
                    }}""")
        residual.append(f"""
                    if({row}<limit_count) {{
                        const float delta=__shfl_sync(0xffffffff,proposal_delta,{row});
                        if(delta!=0.0f)residual_row+=g{row}*delta;
                    }}""")
    return (
        r"""
        bool prefix_handled=false;
        if(limit_count>0 && omega==1.0f) {
            float proposal_delta=0.0f;
            int bad_proposal=0;
            if(lane<limit_count) {
                const float trial=lambda_row-residual_row/denominator_row;
                const float next=fmaxf(trial,0.0f);
                proposal_delta=next-lambda_row;
                bad_proposal=!(lambda_row>=0.0f) || !isfinite(trial) || !isfinite(next) ||
                    !isfinite(proposal_delta) || !isfinite(residual_row);
            }
            const bool proposal_ok=!__any_sync(0xffffffff,bad_proposal);
            const bool nonzero=__any_sync(0xffffffff,proposal_delta!=0.0f);
            bool accepted=proposal_ok && !nonzero;
            if(proposal_ok && nonzero) {
                // Coordinate owners reproduce each physical coefficient's
                // prefix-order sum directly from the retained actual-Z tile.
                // This charges43 rather than18 coordinates per changed row.
                float step0=0.0f,step1=0.0f;
    """
        + "".join(physical)
        + r"""
                float linear=lane<limit_count?residual_row*proposal_delta:0.0f;
                float norm=step0*step0+step1*step1;
                const bool response_ok=!__any_sync(0xffffffff,!isfinite(step0) || !isfinite(step1));
                for(int shift=16;shift>0;shift>>=1) {
                    linear+=__shfl_down_sync(0xffffffff,linear,shift);
                    norm+=__shfl_down_sync(0xffffffff,norm,shift);
                }
                if(lane==0) {
                    const float energy=linear+0.5f*norm,scale=fabsf(linear)+0.5f*norm;
                    accepted=response_ok && isfinite(energy) && isfinite(scale) &&
                        energy<=-0x1p-18f*scale;
                }
                accepted=__shfl_sync(0xffffffff,static_cast<int>(accepted),0)!=0;
            }
            if(accepted) {
                prefix_handled=true;
                if(nonzero) {
    """
        + "".join(residual)
        + r"""
                    if(lane<limit_count) {
                        lambda_row+=proposal_delta;
                        applied_row+=proposal_delta;
                    }
                    changed=1;
                }
            }
            // Rejection leaves all row registers untouched. The statically
            // ordered original prefix below then runs in this same pass.
        }
    """
    )


def _row_source():
    """Use one ordered body and a uniform selector of named Gram registers."""
    selection = []
    for row in range(32):
        previous = f"g{row - 1}" if row > 0 else "0.0f"
        following = f"g{row + 1}" if row < 31 else "0.0f"
        second = f"g{row + 2}" if row < 30 else "0.0f"
        selection.append(
            f"            case {row}: g_previous={previous}; g_current=g{row}; "
            f"g_next={following}; g_next2={second}; break;"
        )
    return (
        r"""
        // All lanes select the same column IDs. No register array, address,
        // or dynamically indexed local allocation is introduced.
        #pragma unroll 1
        for(int row=prefix_handled?limit_count:0;row<count;++row) {
            float g_previous=0.0f,g_current=0.0f,g_next=0.0f,g_next2=0.0f;
            switch(row) {
    """
        + "\n".join(selection)
        + r"""
            }
            const int first=row+1,second=row+2;
            const int type=__shfl_sync(0xffffffff,type_row,row);
            bool handled=false;
            if(type==0 && second<count && iteration>=friction_start && omega==1.0f &&
               __shfl_sync(0xffffffff,type_row,first)==2 &&
               __shfl_sync(0xffffffff,type_row,second)==2 &&
               __shfl_sync(0xffffffff,parent_row,first)==row &&
               __shfl_sync(0xffffffff,parent_row,second)==row &&
               __shfl_sync(0xffffffff,template_row,first)==__shfl_sync(0xffffffff,template_row,row) &&
               __shfl_sync(0xffffffff,template_row,second)==__shfl_sync(0xffffffff,template_row,row)) {
                const float friction=__shfl_sync(0xffffffff,friction_row,first);
                if(isfinite(friction) && friction>=0.0f &&
                   friction==__shfl_sync(0xffffffff,friction_row,second)) {
                    const float old0=__shfl_sync(0xffffffff,lambda_row,row);
                    const float old1=__shfl_sync(0xffffffff,lambda_row,first);
                    const float old2=__shfl_sync(0xffffffff,lambda_row,second);
                    const float r0=__shfl_sync(0xffffffff,residual_row,row);
                    const float d0=__shfl_sync(0xffffffff,denominator_row,row);
                    const float next0=fmaxf(old0-r0/d0,0.0f),delta0=next0-old0;
                    const float radius=friction*next0;
                    int accepted=isfinite(r0) && isfinite(next0) && isfinite(delta0) &&
                        isfinite(radius) && radius==0.0f;
                    float next1=0.0f,next2=0.0f;
                    if(isfinite(r0) && isfinite(next0) && isfinite(delta0) &&
                       isfinite(radius) && radius>0.0f) {
                        const int ready=__shfl_sync(0xffffffff,denominator_ready,row);
                        if(!ready) {
                            const float c1=__shfl_sync(0xffffffff,cfm_row,first);
                            const float c2=__shfl_sync(0xffffffff,cfm_row,second);
                            const float a=__shfl_sync(0xffffffff,denominator_row,first)-c1;
                            const float c=__shfl_sync(0xffffffff,denominator_row,second)-c2;
                            const float cross=__shfl_sync(0xffffffff,g_next2,first);
                            const float largest=0.5f*(a+c)+hypotf(0.5f*(a-c),cross);
                            const float den=largest+fmaxf(c1,c2);
                            if(lane==row) {
                                tangent_denominator=(isfinite(a) && isfinite(c) && a>0.0f && c>0.0f &&
                                    isfinite(c1) && isfinite(c2) && c1>=0.0f && c2>=0.0f &&
                                    isfinite(cross) && isfinite(den) && den>0.0f)?den:0.0f;
                                denominator_ready=1;
                            }
                        }
                        const float den=__shfl_sync(0xffffffff,tangent_denominator,row);
                        const float r1=__shfl_sync(0xffffffff,residual_row,first)+
                            __shfl_sync(0xffffffff,g_current,first)*delta0;
                        const float r2=__shfl_sync(0xffffffff,residual_row,second)+
                            __shfl_sync(0xffffffff,g_current,second)*delta0;
                        if(den>0.0f && isfinite(r1) && isfinite(r2)) {
                            const float x1=old1-r1/den,x2=old2-r2/den;
                            const float magnitude=hypotf(x1,x2);
                            if(isfinite(x1) && isfinite(x2) && isfinite(magnitude)) {
                                const float repair=magnitude>radius?radius/magnitude:1.0f;
                                next1=x1*repair;next2=x2*repair;accepted=1;
                            }
                        }
                    }
                    const float delta1=next1-old1,delta2=next2-old2;
                    if(accepted && isfinite(delta1) && isfinite(delta2)) {
                        if(lane==row) {lambda_row=next0;applied_row+=delta0;}
                        if(lane==first) {lambda_row=next1;applied_row+=delta1;}
                        if(lane==second) {lambda_row=next2;applied_row+=delta2;}
                        if(delta0!=0.0f || delta1!=0.0f || delta2!=0.0f) {
                            residual_row+=(g_current*delta0+g_next*delta1)+g_next2*delta2;
                            changed=1;
                        }
                        handled=true;
                    }
                }
            }
            if(!handled) {
                if(type==2 && iteration<friction_start) {
                    // Original delayed-friction clearing is not a physical
                    // impulse action. Preserve applied_row/residual here.
                    if(lane==row)lambda_row=0.0f;
                } else {
                    const float old=__shfl_sync(0xffffffff,lambda_row,row);
                    const float residual=__shfl_sync(0xffffffff,residual_row,row);
                    const float den=__shfl_sync(0xffffffff,denominator_row,row);
                    float next=old+omega*(-residual/den),sibling_delta=0.0f;
                    int sibling=-1;
                    if(type==0 || type==3)next=fmaxf(next,0.0f);
                    else {
                        const int par=__shfl_sync(0xffffffff,parent_row,row);
                        const float friction=__shfl_sync(0xffffffff,friction_row,row);
                        const float radius=fmaxf(friction*__shfl_sync(0xffffffff,lambda_row,par),0.0f);
                        if(radius<=0.0f)next=0.0f;
                        else {
                            sibling=row==par+1?par+2:par+1;
                            const float other=__shfl_sync(0xffffffff,lambda_row,sibling);
                            const float magnitude=sqrtf(next*next+other*other);
                            if(magnitude>radius) {
                                const float scale=radius/magnitude;next*=scale;
                                sibling_delta=other*scale-other;
                                if(lane==sibling)lambda_row=other*scale;
                            }
                        }
                    }
                    const float delta=next-old;
                    if(lane==row)lambda_row=next;
                    if(sibling_delta!=0.0f) {
                        residual_row+=(sibling==row-1?g_previous:g_next)*sibling_delta;
                        if(lane==sibling)applied_row+=sibling_delta;
                        changed=1;
                    }
                    if(delta!=0.0f) {
                        residual_row+=g_current*delta;
                        if(lane==row)applied_row+=delta;
                        changed=1;
                    }
                }
            }
            // Successful contact handling consumes this normal and both
            // tangents. The loop increment advances to the next original row.
            if(handled)row+=2;
        }
    """
    )


def native_source():
    """Emit named static Gram columns, never a dynamically indexed local array."""
    declarations = "\n".join(f"    float g{row}=0.0f;" for row in range(32))
    formation = "\n".join(f"        if({row}<count)g{row}+=value*staged[{row}*44+node];" for row in range(32))
    finite = " || ".join(f"!isfinite(g{row})" for row in range(32))
    return (
        r"""
    if(group<0 || group>=p.group_to_art.shape[0])return;
    const int art=p.group_to_art.data[group],world=p.art_to_world.data[art];
#if defined(__CUDA_ARCH__)
    const int lane=threadIdx.x&31;
    if(lane==0)routing.data[world]=0;
    const int count=counts.data[world],base=world*100,start=p.art_dof_start.data[art];
    if(!d.valid.data[world] || d.status.data[world] || count<0 || count>32)return;
    const bool live=lane<count;
    const int type_row=live?row_type.data[base+lane]:-1;
    const int parent_row=live?parent.data[base+lane]:-1;
    const int template_row=live?d.support.data[base+lane]:0;
    int bad=live && (template_row<0 || template_row>=p.support_nodes.shape[0]);
    if(__any_sync(0xffffffff,bad))return;
    const int length=live?p.support_count.data[template_row]:0;
    bad=live && (length<1 || length>18 || (type_row!=0 && type_row!=2 && type_row!=3));
    if(live && type_row==2) {
        const int par=parent_row;
        if(par<0 || par+2>=count || (lane!=par+1 && lane!=par+2))bad=1;
        else if(row_type.data[base+par]!=0 || row_type.data[base+par+1]!=2 ||
                row_type.data[base+par+2]!=2 || parent.data[base+par+1]!=par ||
                parent.data[base+par+2]!=par)bad=1;
    }
    float lambda_row=live?impulses.data[base+lane]:0.0f,applied_row=0.0f;
    float residual_row=live?d.incident.data[base+lane]+rhs.data[base+lane]:0.0f;
    const float denominator_row=live?diagonal.data[base+lane]:1.0f;
    const float cfm_row=live?cfm.data[base+lane]:0.0f;
    const float friction_row=live?mu.data[base+lane]:0.0f;
    bad|=live && (!isfinite(lambda_row) || !isfinite(residual_row) ||
                 !isfinite(denominator_row) || !(denominator_row>0.0f));
    if(__any_sync(0xffffffff,bad))return;
    if(count==0 || iterations<=0) {
        for(int col=lane;col<43;col+=32)vout.data[start+col]=vhat.data[start+col];
        if(lane==0)routing.data[world]=1;
        return;
    }
    __shared__ float staged[32*44];
    for(int item=lane;item<32*44;item+=32)staged[item]=0.0f;
    __syncwarp();
    for(int k=0;k<length;++k) {
        const int node=p.support_nodes.data[template_row*18+k];
        const float value=d.Z.data[(base+lane)*18+k];
        if(node<0 || node>=43 || !isfinite(value))bad=1;
        else staged[lane*44+node]=value;
    }
    if(__any_sync(0xffffffff,bad))return;
    __syncwarp();
    const bool zero_row=!live || (lambda_row==0.0f && (type_row==2 || residual_row>=0.0f));
    if(isfinite(omega) && omega>=0.0f && __all_sync(0xffffffff,zero_row)) {
        for(int col=lane;col<43;col+=32)vout.data[start+col]=vhat.data[start+col];
        if(lane==0)routing.data[world]=1;
        return;
    }
    """
        + declarations
        + r"""
    for(int k=0;k<length;++k) {
        const int node=p.support_nodes.data[template_row*18+k];
        const float value=staged[lane*44+node];
    """
        + formation
        + "\n    }\n    if(__any_sync(0xffffffff,"
        + finite
        + r"""))return;
    const unsigned not_limit=__ballot_sync(0xffffffff,lane<count && type_row!=3);
    const int limit_count=not_limit?__ffs(not_limit)-1:count;
    float tangent_denominator=0.0f;
    int denominator_ready=0;
    for(int iteration=0;iteration<iterations;++iteration) {
        int changed=0;
    """
        + _prefix_source()
        + _row_source()
        + r"""
        if(__any_sync(0xffffffff,!isfinite(lambda_row) || !isfinite(applied_row) ||
                      !isfinite(residual_row)))return;
        if(iteration>=friction_start && !__any_sync(0xffffffff,changed))break;
    }
    float du0=0.0f,du1=0.0f;
    for(int row=0;row<count;++row) {
        const float delta=__shfl_sync(0xffffffff,applied_row,row);
        du0+=staged[row*44+lane]*delta;
        if(lane+32<43)du1+=staged[row*44+lane+32]*delta;
    }
    // Reuse only the now-dead Z tile for final physical decode input.
    __syncwarp();
    staged[lane]=du0;
    if(lane+32<43)staged[lane+32]=du1;
    __syncwarp();
    float output0=0.0f,output1=0.0f;
    for(int pass=0;pass<2;++pass) {
        const int col=lane+32*pass;
        if(col<43) {
            float value=0.0f;
            for(int k=0;k<p.inverse_count.data[col];++k) {
                const int row=p.inverse_nodes.data[col*18+k];
                value+=d.W.data[group*434+p.index.data[row*43+col]]*staged[row];
            }
            const float output=vhat.data[start+42-col]+value;
            if(pass==0)output0=output;else output1=output;
        }
    }
    if(__any_sync(0xffffffff,!isfinite(du0) || !isfinite(du1) ||
                  !isfinite(output0) || !isfinite(output1)))return;
    vout.data[start+42-lane]=output0;
    if(lane+32<43)vout.data[start+42-(lane+32)]=output1;
    if(live)impulses.data[base+lane]=lambda_row;
    if(lane==0)routing.data[world]=1;
#else
    // CPU construction is supported; the original solve is CUDA-only.
    // Never leave a routing value uninitialized when a CPU caller probes it.
    routing.data[world]=0;
#endif
    """
    )


@functools.cache
def get_solve_kernel():
    """Return the small owner with the original15 arguments plus routing."""

    @wp.func_native(native_source())
    def native(
        group: int,
        p: SparsePlan,
        d: SparseData,
        counts: wp.array[int],
        rhs: wp.array2d[float],
        diagonal: wp.array2d[float],
        cfm: wp.array2d[float],
        impulses: wp.array2d[float],
        row_type: wp.array2d[int],
        parent: wp.array2d[int],
        mu: wp.array2d[float],
        iterations: int,
        omega: float,
        friction_start: int,
        vhat: wp.array[float],
        vout: wp.array[float],
        routing: wp.array[int],
    ): ...

    def solve(
        p: SparsePlan,
        d: SparseData,
        counts: wp.array[int],
        rhs: wp.array2d[float],
        diagonal: wp.array2d[float],
        cfm: wp.array2d[float],
        impulses: wp.array2d[float],
        row_type: wp.array2d[int],
        parent: wp.array2d[int],
        mu: wp.array2d[float],
        iterations: int,
        omega: float,
        friction_start: int,
        vhat: wp.array[float],
        vout: wp.array[float],
        routing: wp.array[int],
    ):
        group, _ = wp.tid()
        native(
            group,
            p,
            d,
            counts,
            rhs,
            diagonal,
            cfm,
            impulses,
            row_type,
            parent,
            mu,
            iterations,
            omega,
            friction_start,
            vhat,
            vout,
            routing,
        )

    solve.__name__ = solve.__qualname__ = "sparse_register_residual43_s18_c100"
    return wp.kernel(enable_backward=False, module="unique")(solve)


@functools.cache
def get_fallback_kernel():
    """Filter the untouched corrected-limit math using the current routing mask."""
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
        source, "        vout: wp.array[float],", "        vout: wp.array[float],\n        routing: wp.array[int],", 2
    )
    source = replace(source, "            vout,\n", "            vout,\n            routing,\n")
    source = replace(
        source,
        "    const int lane=threadIdx.x&31, art=",
        "    if(group<0 || group>=p.group_to_art.shape[0])return;\n    const int lane=threadIdx.x&31, art=",
    )
    source = replace(
        source,
        "    const int start=p.art_dof_start.data[art],",
        "    if(routing.data[world])return;\n    const int start=p.art_dof_start.data[art],",
    )
    source = replace(
        source, '"sparse_metric_tangent" if metric_tangents', '"sparse_register_residual_fallback" if metric_tangents'
    )
    filename = "<sparse_register_residual_fallback_factory>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = dict(original.__wrapped__.__globals__)
    exec(compile(source, filename, "exec"), namespace)
    return namespace["get_solve_kernel"](100, False, metric_tangents=True)
