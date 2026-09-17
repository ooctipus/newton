# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental row-owned local formation for the compact register solver.

Current packet keys become physical rows in the existing shared tile. Global
rows are materialized separately only for rejected worlds. The inherited
prefix, spectral iteration, applied-delta decode and budget are unchanged.
"""

from functools import cache

import warp as wp

from . import sparse_register_residual
from .kernels import _FPGS_CONTACT_END_GAP_SLOP
from .sparse_factor import SparseData, SparsePlan
from .sparse_packet_rows import PacketInput


def _formation_source():
    """Form independent rows without a second J tile or per-contact warp loop."""
    return r"""
    if(count==0) {
        for(int col=lane;col<43;col+=32)vout.data[start+col]=vhat.data[start+col];
        if(lane==0)routing.data[world]=1;
        return;
    }
    const bool live=lane<count;
    const int type_row=live?row_type.data[base+lane]:-1;
    const int parent_row=live?parent.data[base+lane]:-1;
    const int packet_key=live?d.support.data[base+lane]:0;
    const float cfm_row=live?cfm.data[base+lane]:0.0f;
    const float friction_row=live?mu.data[base+lane]:0.0f;
    float lambda_row=live?impulses.data[base+lane]:0.0f,applied_row=0.0f;
    float rhs_row=live?rhs.data[base+lane]:0.0f;
    float incident_row=0.0f,denominator_row=1.0f;
    int template_row=0,length=0;
    int bad=live && ((type_row!=0 && type_row!=2 && type_row!=3) ||
                    !isfinite(lambda_row) || !isfinite(rhs_row));
    if(live && type_row==2) {
        const int par=parent_row;
        if(par<0 || par+2>=count || (lane!=par+1 && lane!=par+2))bad=1;
        else if(row_type.data[base+par]!=0 || row_type.data[base+par+1]!=2 ||
                row_type.data[base+par+2]!=2 || parent.data[base+par+1]!=par ||
                parent.data[base+par+2]!=par)bad=1;
    }
    if(__any_sync(0xffffffff,bad))return;

    // Each lane owns this row throughout formation. Validate key/address
    // metadata before reading raw geometry; rejected worlds keep their keys.
    int local=0,direction=0,aa=-1,ab=-1,ba=-1,bb=-1;
    wp::vec3 normal(0.0f),pa(0.0f),pb(0.0f),dir(0.0f);
    if(live) {
        if(packet_key<0) {
            if(packet_key < -86 || type_row!=3)bad=1;
            else {
                const int candidate=-packet_key-1;
                local=candidate/2;
                template_row=p.limit_support.data[local];
            }
        } else {
            const int c=packet_key/3;
            direction=packet_key%3;
            if(type_row==3 || (type_row==0 && direction!=0) ||
               (type_row==2 && direction!=lane-parent_row) ||
               c>=x.shape0.shape[0] || c>=x.art_a.shape[0])bad=1;
            else {
                aa=x.art_a.data[c];ab=x.art_b.data[c];
                const int sa=x.shape0.data[c],sb=x.shape1.data[c];
                if((aa>=0 && ab>=0 && aa!=ab) || (aa!=art && ab!=art) ||
                   sa>=x.shape_body.shape[0] || sb>=x.shape_body.shape[0])bad=1;
                else {
                    ba=sa>=0?x.shape_body.data[sa]:-1;
                    bb=sb>=0?x.shape_body.data[sb]:-1;
                    if(ba>=p.body_tag.shape[0] || bb>=p.body_tag.shape[0])bad=1;
                    else {
                        const int ta=ba>=0?p.body_tag.data[ba]:0;
                        const int tb=bb>=0?p.body_tag.data[bb]:0;
                        if(ta<0 || ta>3 || tb<0 || tb>3)bad=1;
                        else template_row=p.pair_support.data[ta*4+tb];
                    }
                }
                if(type_row==2 && d.support.data[base+parent_row]!=3*c)bad=1;
            }
        }
    }
    bad|=live && (template_row<0 || template_row>=p.support_nodes.shape[0]);
    if(__any_sync(0xffffffff,bad))return;
    length=live?p.support_count.data[template_row]:0;
    bad=live && (length<1 || length>18);
    if(__any_sync(0xffffffff,bad))return;

    __shared__ float staged[32*44];
    for(int item=lane;item<32*44;item+=32)staged[item]=0.0f;
    __syncwarp();
    float norm=0.0f;
    if(live && packet_key<0) {
        const float sign=((-packet_key-1)&1)?-1.0f:1.0f;
        for(int k=0;k<length;++k) {
            const int node=p.support_nodes.data[template_row*18+k];
            if(node<0 || node>=43) {bad=1;continue;}
            const int entry=p.index.data[node*43+42-local];
            const float z=entry>=0?sign*d.W.data[group*434+entry]:0.0f;
            staged[lane*44+node]=z;
            norm+=z*z;
            if(!isfinite(z))bad=1;
        }
        incident_row=sign*vhat.data[start+local];
    } else if(live) {
        const int c=packet_key/3;
        normal=-x.normal.data[c];
        pa=(ba>=0?wp::transform_point(x.body_q.data[ba],x.point0.data[c]):x.point0.data[c])-
            x.thickness0.data[c]*normal;
        pb=(bb>=0?wp::transform_point(x.body_q.data[bb],x.point1.data[c]):x.point1.data[c])+
            x.thickness1.data[c]*normal;
        const wp::vec3 anchor=0.5f*(pa+pb);
        dir=normal;
        if(direction!=0) {
            wp::vec3 tangent=wp::cross(normal,wp::vec3(1.0f,0.0f,0.0f));
            if(wp::length_sq(tangent)<1.0e-12f)tangent=wp::cross(normal,wp::vec3(0.0f,1.0f,0.0f));
            tangent=wp::normalize(tangent);
            dir=direction==1?tangent:wp::normalize(wp::cross(normal,tangent));
        }
        if(x.shared_anchor || (direction!=0 && x.friction_anchor)) {pa=anchor;pb=anchor;}
        const wp::vec3 origin=x.origin.data[art];
        // Preserve the original support order for physical J and incident.
        // The original parallel incident reduction is now a scalar row sum.
        for(int k=0;k<length;++k) {
            const int node=p.support_nodes.data[template_row*18+k];
            if(node<0 || node>=43) {bad=1;continue;}
            const int dof=42-node;
            const wp::spatial_vector screw=x.screw.data[start+dof];
            const wp::vec3 lin(screw[0],screw[1],screw[2]),ang(screw[3],screw[4],screw[5]);
            const unsigned long long bit=1ull<<dof;
            float value=0.0f;
            if(ba>=0 && aa==art && (p.body_mask.data[ba]&bit))
                value+=wp::dot(dir,lin+wp::cross(ang,pa-origin));
            if(bb>=0 && ab==art && (p.body_mask.data[bb]&bit))
                value-=wp::dot(dir,lin+wp::cross(ang,pb-origin));
            staged[lane*44+node]=value;
            incident_row+=value*vhat.data[start+dof];
            if(!isfinite(value))bad=1;
        }
        // W is lower triangular and support nodes are sorted/closed. Descend
        // outputs so every k<=node input is still original J, not an output.
        // Keep each output's original ascending input summation order.
        for(int out=length-1;out>=0;--out) {
            const int node=p.support_nodes.data[template_row*18+out];
            if(node<0 || node>=43)continue;
            float z=0.0f;
            for(int k=0;k<=out;++k) {
                const int col=p.support_nodes.data[template_row*18+k];
                if(col<0 || col>=43)continue;
                const int entry=p.index.data[node*43+col];
                if(entry>=0)z+=d.W.data[group*434+entry]*staged[lane*44+col];
            }
            staged[lane*44+node]=z;
            norm+=z*z;
            if(!isfinite(z))bad=1;
        }
        if(type_row==0) {
            const float target=x.target.data[base+lane];
            const float velocity=incident_row-target;
            const float e=x.restitution.data[base+lane],phi=x.phi.data[base+lane];
            if(e>0.0f && !(velocity>=-x.threshold) &&
               (phi<=PACKET_END_GAP_SLOP || phi+x.dt*velocity<=PACKET_END_GAP_SLOP))
                rhs_row=-target+e*velocity;
        }
    }
    if(live)denominator_row=norm+cfm_row;
    float residual_row=incident_row+rhs_row;
    bad|=live && (!isfinite(denominator_row) || !(denominator_row>0.0f) ||
                 !isfinite(incident_row) || !isfinite(rhs_row) || !isfinite(residual_row));
    if(__any_sync(0xffffffff,bad))return;
    __syncwarp();
    const bool zero_row=!live || (lambda_row==0.0f && (type_row==2 || residual_row>=0.0f));
    if(iterations<=0 || (isfinite(omega) && omega>=0.0f && __all_sync(0xffffffff,zero_row))) {
        for(int col=lane;col<43;col+=32)vout.data[start+col]=vhat.data[start+col];
        PACKET_PUBLISH_ROWS
        if(lane==0)routing.data[world]=1;
        return;
    }
    """.replace("PACKET_END_GAP_SLOP", f"{float(_FPGS_CONTACT_END_GAP_SLOP):.9g}f")


def native_source():
    """Replace only compact input formation and successful scalar publication."""
    source = sparse_register_residual.native_source()
    first = "    const bool live=lane<count;"
    last = "    float g0=0.0f;"
    if source.count(first) != 1 or source.count(last) != 1:
        raise RuntimeError("Original compact formation boundary changed")
    begin, end = source.index(first), source.index(last)
    source = source[:begin] + _formation_source() + source[end:]
    publish = r"""
        if(live) {
            d.support.data[base+lane]=template_row;
            d.incident.data[base+lane]=incident_row;
            diagonal.data[base+lane]=denominator_row;
            rhs.data[base+lane]=rhs_row;
        }
    """
    source = source.replace("PACKET_PUBLISH_ROWS", publish)
    final = "    if(live)impulses.data[base+lane]=lambda_row;"
    if source.count(final) != 1:
        raise RuntimeError("Original compact publication boundary changed")
    source = source.replace(final, publish + final)
    if "d.Z.data" in source:
        raise RuntimeError("Local packet owner must not produce or consume global Z")
    return source


@cache
def get_solve_kernel():
    """Return the complete small owner with original16 arguments and current input."""

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
        x: PacketInput,
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
        x: PacketInput,
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
            x,
        )

    solve.__name__ = solve.__qualname__ = "sparse_register_packets43_s18_c100"
    return wp.kernel(enable_backward=False, module="unique")(solve)
