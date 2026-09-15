# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Current body/limit ports with complete original sparse fallback."""

import inspect
import linecache
import textwrap
from functools import cache

import numpy as np
import warp as wp

from . import kernels, sparse_factor_rows, sparse_packet_rows
from .sparse_factor import SparseData, SparsePlan
from .sparse_packet_rows import PacketInput


@wp.struct
class PresentData:
    geometry: wp.array3d[float]
    mode: wp.array[int]
    position: wp.array2d[int]


def _replace(source, old, new, count=1):
    if source.count(old) != count:
        raise ValueError("Present-port original source seam changed: " + old[:80])
    return source.replace(old, new)


def _compile(source, name, namespace):
    filename = "<sparse-present-ports-" + name + ">"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    exec(compile(source, filename, "exec"), namespace)
    return namespace[name]


@cache
def get_metadata_kernel():
    """Append wrench geometry to the existing complete metadata producer."""
    source = textwrap.dedent(inspect.getsource(kernels.prepare_world_contact_rows.func))
    source = _replace(source, "@wp.kernel\n", '@wp.kernel(module="unique", enable_backward=False)\n')
    source = _replace(source, "def prepare_world_contact_rows(", "def present_port_metadata(")
    source = _replace(
        source,
        "    world_row_restitution: wp.array2d[float],\n",
        "    world_row_restitution: wp.array2d[float],\n    ports: PresentData,\n",
    )
    source += """
        # Unsupported endpoint pairs retain all original metadata and use fallback.
        one_a = body_a >= 0 and body_b < 0
        one_b = body_b >= 0 and body_a < 0
        if one_a or one_b:
            art = art_a
            body = body_a
            sign = float(1.0)
            point_n = point_a_normal
            point_t = point_a_friction
            if one_b:
                art = art_b
                body = body_b
                sign = -1.0
                point_n = point_b_normal
                point_t = point_b_friction
            local_body = body % 44
            if art >= 0 and (local_body == 6 or local_body == 13 or local_body == 14):
                for direction in range(contact_slots_needed[c]):
                    axis = normal
                    point = point_n
                    if direction == 1:
                        axis = tangent0
                        point = point_t
                    elif direction == 2:
                        axis = tangent1
                        point = point_t
                    force = sign * axis
                    torque = wp.cross(point - articulation_origin[art], force)
                    for k in range(3):
                        ports.geometry[world, slot + direction, k] = force[k]
                        ports.geometry[world, slot + direction, k + 3] = torque[k]
"""
    namespace = dict(kernels.prepare_world_contact_rows.func.__globals__, PresentData=PresentData)
    return _compile(source, "present_port_metadata", namespace)


@cache
def get_fallback_contacts():
    """Guard original response formation before its first geometry/W read."""
    source = textwrap.dedent(inspect.getsource(sparse_factor_rows.get_contact_kernel))
    source = _replace(source, "def get_contact_kernel():", "def get_present_fallback_contacts():")
    source = _replace(
        source,
        "        if (row0 >= capacity || !d.valid.data[w]) continue;",
        "        if (row0 >= capacity || !d.valid.data[w]) continue;\n        if (ports.mode.data[w]) continue;",
    )
    source = _replace(
        source,
        "        diagonal: wp.array2d[float],",
        "        diagonal: wp.array2d[float],\n        ports: PresentData,",
        2,
    )
    source = _replace(
        source, "            diagonal,\n        )", "            diagonal,\n            ports,\n        )"
    )
    source = _replace(source, '"sparse_factor_contact_triplet18"', '"present_port_fallback_contact18"')
    namespace = dict(sparse_factor_rows.__dict__, PresentData=PresentData)
    return _compile(source, "get_present_fallback_contacts", namespace)()


@cache
def get_fallback_solve():
    """Keep the original complete metric kernel behind one world guard."""
    source = textwrap.dedent(inspect.getsource(sparse_factor_rows.get_solve_kernel))
    source = _replace(source, "def get_solve_kernel(", "def get_present_fallback_solve(")
    source = _replace(
        source, "    if (!d.valid.data[world]", "    if (ports.mode.data[world]) return;\n    if (!d.valid.data[world]"
    )
    source = _replace(
        source, "        vout: wp.array[float],", "        vout: wp.array[float],\n        ports: PresentData,", 2
    )
    source = _replace(source, "            vout,\n        )", "            vout,\n            ports,\n        )")
    source = _replace(source, 'f"{name}43_s18_c{capacity}"', 'f"present_port_fallback_{name}43_s18_c{capacity}"')
    namespace = dict(sparse_factor_rows.__dict__, PresentData=PresentData)
    return _compile(source, "get_present_fallback_solve", namespace)(100, metric_tangents=True)


@cache
def get_fallback_restitution():
    """Apply original incident restitution only to original-response worlds."""
    source = textwrap.dedent(inspect.getsource(sparse_factor_rows.apply_restitution.func))
    source = _replace(
        source,
        "@wp.kernel(enable_backward=False)\n",
        '@wp.kernel(module="unique", enable_backward=False)\n',
    )
    source = _replace(source, "def apply_restitution(", "def present_port_fallback_restitution(")
    source = _replace(source, "    rhs: wp.array2d[float],", "    rhs: wp.array2d[float],\n    ports: PresentData,")
    source = _replace(
        source, "    world, row = wp.tid()", "    world, row = wp.tid()\n    if ports.mode[world] != 0:\n        return"
    )
    namespace = dict(sparse_factor_rows.__dict__, PresentData=PresentData)
    return _compile(source, "present_port_fallback_restitution", namespace)


@wp.kernel(enable_backward=False)
def present_port_fallback_limits(
    p: SparsePlan,
    d: SparseData,
    ports: PresentData,
    prefix: wp.array2d[int],
    cfm: wp.array2d[float],
    diagonal: wp.array2d[float],
    vhat: wp.array[float],
):
    """Emit the recorded original limit responses without reallocation."""
    world, row = wp.tid()
    if ports.mode[world] != 0 or row >= wp.min(prefix[world, 0], d.support.shape[1]):
        return
    candidate = -d.support[world, row] - 1
    if candidate < 0 or candidate >= 86:
        wp.atomic_or(d.status, world, 4)
        return
    local = candidate // 2
    sign = float(1.0)
    if candidate % 2 != 0:
        sign = -1.0
    art = p.group_to_art[world]
    start = p.art_dof_start[art]
    tpl = p.limit_support[local]
    norm = float(0.0)
    for k in range(18):
        node = p.support_nodes[tpl, k]
        z = float(0.0)
        if node >= 0:
            entry = p.index[node, 42 - local]
            if entry >= 0:
                z = sign * d.W[world, entry]
        d.Z[world, row, k] = z
        norm += z * z
    d.support[world, row] = tpl
    d.incident[world, row] = sign * vhat[start + local]
    diagonal[world, row] = norm + cfm[world, row]


def _metric_update():
    from .sparse_metric_tangents import get_fragments  # noqa: PLC0415

    _, source = get_fragments(100)
    source = _replace(
        source,
        "const int tpl=d.support.data[base+row],length=p.support_count.data[tpl];",
        "const int tpl=d.support.data[base+row],length=6;",
    )
    source = _replace(
        source, "const int node=lane<length?p.support_nodes.data[tpl*18+lane]:-1;", "const int node=tpl+lane;"
    )
    for offset in ("", "+1", "+2"):
        source = _replace(
            source, f"d.Z.data[(base+row{offset})*18+lane]", f"ports.geometry.data[(base+row{offset})*6+lane]"
        )
    source = source.replace("du[node]", "h[node]").replace("shift=16", "shift=4")
    old = """                                float a01=z0*z1,a02=z0*z2,a12=z1*z2;
                                for(int shift=4;shift>0;shift>>=1) {
                                    a01+=__shfl_down_sync(0xffffffff,a01,shift);
                                    a02+=__shfl_down_sync(0xffffffff,a02,shift);
                                    a12+=__shfl_down_sync(0xffffffff,a12,shift);
                                }"""
    source = _replace(
        source,
        old,
        """                                float a01=bilinear(row,row+1),a02=bilinear(row,row+2),a12=bilinear(row+1,row+2);""",
    )
    source = _replace(
        source, "if(lane<length)h[node]+=z0*change0+z1*change1+z2*change2;", "apply_three(row,change0,change1,change2);"
    )
    return source


def cuda_source():
    """Use one current compact port operator, never a private response panel."""
    source = r"""
#if defined(__CUDA_ARCH__)
    const int lane=threadIdx.x&31,art=p.group_to_art.data[group],world=p.art_to_world.data[art];
    const int start=p.art_dof_start.data[art],count=counts.data[world],base=world*100;
    if(lane==0)ports.mode.data[world]=0;
    __syncwarp();
    if(!d.valid.data[world] || d.status.data[world])return;
    if(count<0 || count>100){if(lane==0)atomicOr(&d.status.data[world],4);return;}
    int invalid=0;
    for(int r=lane;r<count;r+=32){
        const int type=row_type.data[base+r],par=parent.data[base+r];
        if(type!=0 && type!=2 && type!=3)invalid=1;
        if(type==2){
            if(par<0 || par+2>=count || (r!=par+1 && r!=par+2))invalid=1;
            else if(row_type.data[base+par]!=0 || row_type.data[base+par+1]!=2 || row_type.data[base+par+2]!=2 || parent.data[base+par+1]!=par || parent.data[base+par+2]!=par)invalid=1;
        }
    }
    if(__ballot_sync(0xffffffff,invalid)){if(lane==0)atomicOr(&d.status.data[world],4);return;}
    __shared__ float F[576],C[528],lam[100],contact_cross[100],h[32],pi[32];
    __shared__ int keys[32],body_base[3],port_count,bad;
    __shared__ unsigned int contact_ready[4];
    if(lane==0){
        bad=0;port_count=0;unsigned bodies=0;unsigned long long limits=0ull;
        for(int r=0;r<count;++r){
            const int key=d.support.data[base+r],type=row_type.data[base+r];
            if(key<0){
                const int candidate=-key-1;
                if(candidate>=86 || type!=3){bad=1;break;}
                limits|=1ull<<(candidate/2);
            }else{
                const int c=key/3;
                if(c>=x.shape0.shape[0] || c>=x.art_a.shape[0] || type==3){bad=1;break;}
                const int aa=x.art_a.data[c],ab=x.art_b.data[c];
                const int sa=x.shape0.data[c],sb=x.shape1.data[c];
                if(sa>=x.shape_body.shape[0] || sb>=x.shape_body.shape[0]){bad=1;break;}
                const int ba=sa>=0?x.shape_body.data[sa]:-1,bb=sb>=0?x.shape_body.data[sb]:-1;
                if((ba>=0)==(bb>=0)){bad=1;break;}
                const int body=ba>=0?ba:bb,a=ba>=0?aa:ab;
                if(body>=p.body_tag.shape[0] || a!=art){bad=1;break;}
                const int tag=p.body_tag.data[body];
                if(tag<1 || tag>3){bad=1;break;}
                bodies|=1u<<(tag-1);
                if(type==0 && key%3!=0){bad=1;break;}
                if(type==2 && key!=d.support.data[base+parent.data[base+r]]+r-parent.data[base+r]){bad=1;break;}
            }
        }
        for(int tag=1;tag<=3;++tag){
            body_base[tag-1]=-1;
            if(bodies&(1u<<(tag-1))){
                body_base[tag-1]=port_count;
                for(int axis=0;axis<6;++axis)keys[port_count++]=-1-tag*6-axis;
            }
        }
        for(int dof=0;dof<43;++dof)if(limits&(1ull<<dof)){
            if(port_count<32)keys[port_count]=dof;
            ++port_count;
        }
        if(port_count>32)bad=1;
    }
    __syncwarp();
    if(bad)return;
    const int P=port_count;
    auto tpl_of=[&](int key){return key<0?p.pair_support.data[((-key-1)/6)*4]:p.limit_support.data[key];};
    auto c_at=[&](int i,int j){const int hi=i>j?i:j,lo=i>j?j:i;return C[hi*(hi+1)/2+lo];};
    for(int k=lane;k<576;k+=32)F[k]=0.0f;
    __syncwarp();
    for(int tag=1;tag<=3;++tag){
        const int first=body_base[tag-1];
        if(first<0)continue;
        const int tpl=p.pair_support.data[tag*4],n=p.support_count.data[tpl];
        const int node=lane<n?p.support_nodes.data[tpl*18+lane]:0;
        wp::spatial_vector screw;
        for(int a=0;a<6;++a)screw[a]=lane<n?x.screw.data[start+42-node][a]:0.0f;
        float f0=0.0f,f1=0.0f,f2=0.0f,f3=0.0f,f4=0.0f,f5=0.0f;
        for(int k=0;k<n;++k){
            const int col=p.support_nodes.data[tpl*18+k],e=lane<n?p.index.data[node*43+col]:-1;
            const float w=e>=0?d.W.data[group*434+e]:0.0f;
            f0+=w*__shfl_sync(0xffffffff,screw[0],k);f1+=w*__shfl_sync(0xffffffff,screw[1],k);
            f2+=w*__shfl_sync(0xffffffff,screw[2],k);f3+=w*__shfl_sync(0xffffffff,screw[3],k);
            f4+=w*__shfl_sync(0xffffffff,screw[4],k);f5+=w*__shfl_sync(0xffffffff,screw[5],k);
        }
        if(lane<n){F[(first+0)*18+lane]=f0;F[(first+1)*18+lane]=f1;F[(first+2)*18+lane]=f2;F[(first+3)*18+lane]=f3;F[(first+4)*18+lane]=f4;F[(first+5)*18+lane]=f5;}
        for(int a=0;a<6;++a){
            float v=lane<n?screw[a]*vhat.data[start+42-node]:0.0f;
            for(int s=16;s>0;s>>=1)v+=__shfl_down_sync(0xffffffff,v,s);
            if(lane==0)pi[first+a]=v;
        }
    }
    for(int i=lane;i<P;i+=32)if(keys[i]>=0){
        const int dof=keys[i],tpl=p.limit_support.data[dof],n=p.support_count.data[tpl];
        for(int k=0;k<n;++k){const int node=p.support_nodes.data[tpl*18+k],e=p.index.data[node*43+42-dof];F[i*18+k]=e>=0?d.W.data[group*434+e]:0.0f;}
        pi[i]=vhat.data[start+dof];
    }
    __syncwarp();
    for(int pair=lane;pair<P*(P+1)/2;pair+=32){
        int i=0;while((i+1)*(i+2)/2<=pair)++i;
        const int j=pair-i*(i+1)/2,ti=tpl_of(keys[i]),tj=tpl_of(keys[j]);
        float value=0.0f;
        for(int k=0;k<p.support_count.data[ti];++k){
            const int node=p.support_nodes.data[ti*18+k],at=ports.position.data[tj*43+node];
            if(at>=0)value+=F[i*18+k]*F[j*18+at];
        }
        C[pair]=value;
    }
    __syncwarp();
    for(int k=lane;k<P*18;k+=32)if(!isfinite(F[k]))invalid=1;
    for(int k=lane;k<P*(P+1)/2;k+=32)if(!isfinite(C[k]))invalid=1;
    if(__ballot_sync(0xffffffff,invalid))return;
    auto raw_port=[&](int key){
        if(key<0){const int dof=(-key-1)/2;for(int i=0;i<P;++i)if(keys[i]==dof)return i;return -1;}
        const int c=key/3,sa=x.shape0.data[c],sb=x.shape1.data[c];
        const int ba=sa>=0?x.shape_body.data[sa]:-1,bb=sb>=0?x.shape_body.data[sb]:-1;
        return body_base[p.body_tag.data[ba>=0?ba:bb]-1];
    };
    for(int r=lane;r<count;r+=32){
        const int key=d.support.data[base+r],first=raw_port(key);
        float norm=0.0f,incident=0.0f;
        if(key<0){norm=c_at(first,first);incident=(((-key-1)&1)?-1.0f:1.0f)*pi[first];}
        else{
            for(int i=0;i<6;++i){
                const float a=ports.geometry.data[(base+r)*6+i];float value=0.0f;
                if(!isfinite(a))invalid=1;
                for(int j=0;j<6;++j)value+=c_at(first+i,first+j)*ports.geometry.data[(base+r)*6+j];
                norm+=a*value;incident+=a*pi[first+i];
            }
        }
        const float denom=norm+x.cfm.data[base+r];
        if(!isfinite(denom) || !isfinite(incident) || !isfinite(rhs.data[base+r]))invalid=1;
        diagonal.data[base+r]=denom;d.incident.data[base+r]=incident;
    }
    if(__ballot_sync(0xffffffff,invalid))return;
    __syncwarp();
    for(int r=lane;r<count;r+=32){
        const int key=d.support.data[base+r],first=raw_port(key);
        d.support.data[base+r]=key<0?first+((((-key-1)&1)!=0)?32:0):first;
        if(row_type.data[base+r]==0){
            const float velocity=d.incident.data[base+r]-x.target.data[base+r],e=x.restitution.data[base+r],phi=x.phi.data[base+r];
            if(e>0.0f && velocity<-x.threshold && (phi<=END_GAP || phi+x.dt*velocity<=END_GAP))rhs.data[base+r]=-x.target.data[base+r]+e*velocity;
        }
        lam[r]=impulses.data[base+r];
    }
    if(lane<4)contact_ready[lane]=0;
    if(lane<P){h[lane]=0.0f;pi[lane]=0.0f;}
    if(lane==0)ports.mode.data[world]=1;
    __syncwarp();
    auto dot_row=[&](int r){
        const int code=d.support.data[base+r],type=row_type.data[base+r];float value=0.0f;
        if(type==3){if(lane==0)value=((code&32)?-1.0f:1.0f)*h[code&31];}
        else{
            if(lane<6)value=ports.geometry.data[(base+r)*6+lane]*h[code+lane];
            for(int s=4;s>0;s>>=1)value+=__shfl_down_sync(0xffffffff,value,s);
        }
        return __shfl_sync(0xffffffff,value,0);
    };
    auto bilinear=[&](int a,int b){
        const int pa=d.support.data[base+a],pb=d.support.data[base+b];float value=0.0f;
        if(lane<6){for(int j=0;j<6;++j)value+=c_at(pa+lane,pb+j)*ports.geometry.data[(base+b)*6+j];value*=ports.geometry.data[(base+a)*6+lane];}
        for(int s=4;s>0;s>>=1)value+=__shfl_down_sync(0xffffffff,value,s);
        return __shfl_sync(0xffffffff,value,0);
    };
    auto apply_three=[&](int r,float a,float b,float c){
        const int first=d.support.data[base+r];
        float w=0.0f;
        if(lane<6)w=ports.geometry.data[(base+r)*6+lane]*a+ports.geometry.data[(base+r+1)*6+lane]*b+ports.geometry.data[(base+r+2)*6+lane]*c;
        float change=0.0f;
        for(int k=0;k<6;++k){const float q=__shfl_sync(0xffffffff,w,k);if(lane<P)change+=c_at(lane,first+k)*q;}
        if(lane<P)h[lane]+=change;
        if(lane<6)pi[first+lane]+=w;
        __syncwarp();
    };
    auto apply_row=[&](int r,float delta){
        const int code=d.support.data[base+r];
        if(row_type.data[base+r]==3){
            const int at=code&31;const float q=((code&32)?-1.0f:1.0f)*delta;
            if(lane<P)h[lane]+=c_at(lane,at)*q;
            if(lane==at)pi[at]+=q;
        }else{
            float w=lane<6?ports.geometry.data[(base+r)*6+lane]*delta:0.0f,change=0.0f;
            for(int k=0;k<6;++k){const float q=__shfl_sync(0xffffffff,w,k);if(lane<P)change+=c_at(lane,code+k)*q;}
            if(lane<P)h[lane]+=change;
            if(lane<6)pi[code+lane]+=w;
        }
        __syncwarp();
    };
    for(int iteration=0;iteration<iterations;++iteration){
        int changed=0;
        for(int row=0;row<count;++row){
            const int type=row_type.data[base+row];
            METRIC_UPDATE
            if(type==2 && iteration<friction_start){if(lane==0)lam[row]=0.0f;__syncwarp();continue;}
            const float denom=diagonal.data[base+row];if(!(denom>0.0f))continue;
            const float dot=dot_row(row);float delta=0.0f,sibling_delta=0.0f;int sibling=-1;
            if(lane==0){
                const float old=lam[row],residual=dot+d.incident.data[base+row]+rhs.data[base+row];
                float next=old+omega*(-residual/denom);
                if(type==0 || type==3)next=fmaxf(next,0.0f);
                else if(type==2){
                    const int par=parent.data[base+row];const float radius=fmaxf(mu.data[base+row]*lam[par],0.0f);
                    if(radius<=0.0f)next=0.0f;
                    else{
                        sibling=row==par+1?par+2:par+1;const float other=lam[sibling],mag=sqrtf(next*next+other*other);
                        if(mag>radius){const float scale=radius/mag;next*=scale;const float adjusted=other*scale;sibling_delta=adjusted-other;lam[sibling]=adjusted;}
                    }
                }
                delta=next-old;lam[row]=next;
            }
            delta=__shfl_sync(0xffffffff,delta,0);sibling_delta=__shfl_sync(0xffffffff,sibling_delta,0);sibling=__shfl_sync(0xffffffff,sibling,0);
            __syncwarp();
            if(sibling_delta!=0.0f){apply_row(sibling,sibling_delta);changed=1;}
            if(delta!=0.0f){apply_row(row,delta);changed=1;}
            __syncwarp();
        }
        if(iteration>=friction_start && __ballot_sync(0xffffffff,changed)!=0u)continue;
        if(iteration>=friction_start)break;
    }
    __syncwarp();
    // C is dead; its first43 entries now own the decoded kinetic displacement.
    for(int node=lane;node<43;node+=32){
        float value=0.0f;
        for(int i=0;i<P;++i){const int tpl=tpl_of(keys[i]),at=ports.position.data[tpl*43+node];if(at>=0)value+=F[i*18+at]*pi[i];}
        C[node]=value;
    }
    __syncwarp();
    for(int col=lane;col<43;col+=32){
        float value=0.0f;
        for(int k=0;k<p.inverse_count.data[col];++k){const int row=p.inverse_nodes.data[col*18+k];value+=d.W.data[group*434+p.index.data[row*43+col]]*C[row];}
        vout.data[start+42-col]=vhat.data[start+42-col]+value;
    }
    for(int r=lane;r<count;r+=32)impulses.data[base+r]=lam[r];
#endif
"""
    source = _replace(source, "METRIC_UPDATE", _metric_update())
    return source.replace("END_GAP", f"{float(kernels._FPGS_CONTACT_END_GAP_SLOP):.9g}f")


@cache
def get_solve_kernel():
    """Construct the actual current-port finite-eight CUDA owner."""

    @wp.func_native(cuda_source())
    def native(
        group: int,
        p: SparsePlan,
        d: SparseData,
        counts: wp.array[int],
        rhs: wp.array2d[float],
        diagonal: wp.array2d[float],
        impulses: wp.array2d[float],
        row_type: wp.array2d[int],
        parent: wp.array2d[int],
        mu: wp.array2d[float],
        iterations: int,
        omega: float,
        friction_start: int,
        vhat: wp.array[float],
        vout: wp.array[float],
        x: PacketInput,
        ports: PresentData,
    ): ...

    def sparse_present_ports43_p32_c100(
        p: SparsePlan,
        d: SparseData,
        counts: wp.array[int],
        rhs: wp.array2d[float],
        diagonal: wp.array2d[float],
        impulses: wp.array2d[float],
        row_type: wp.array2d[int],
        parent: wp.array2d[int],
        mu: wp.array2d[float],
        iterations: int,
        omega: float,
        friction_start: int,
        vhat: wp.array[float],
        vout: wp.array[float],
        x: PacketInput,
        ports: PresentData,
    ):
        group, _ = wp.tid()
        native(
            group,
            p,
            d,
            counts,
            rhs,
            diagonal,
            impulses,
            row_type,
            parent,
            mu,
            iterations,
            omega,
            friction_start,
            vhat,
            vout,
            x,
            ports,
        )

    sparse_present_ports43_p32_c100.__qualname__ = "sparse_present_ports43_p32_c100"
    return wp.kernel(enable_backward=False, module="unique")(sparse_present_ports43_p32_c100)


class PresentPorts:
    """Own current port geometry and the complete guarded fallback tail."""

    def __init__(self, owner):
        self.owner = owner
        s = owner.solver
        self.data = PresentData()
        self.data.geometry = wp.empty((s.world_count, 100, 6), dtype=float, device=s.model.device)
        self.data.mode = wp.zeros(s.world_count, dtype=int, device=s.model.device)
        slots = np.full((len(owner.host["support_nodes"]), 43), -1, np.int32)
        for tpl, nodes in enumerate(owner.host["support_nodes"]):
            for k, node in enumerate(nodes):
                if node >= 0:
                    slots[tpl, node] = k
        self.data.position = wp.array(slots, dtype=int, device=s.model.device)
        self.packet_input = PacketInput()
        self.metadata = get_metadata_kernel()
        self.kernel = get_solve_kernel()
        self.fallback_contacts = get_fallback_contacts()
        self.fallback_solve = get_fallback_solve()
        self.fallback_restitution = get_fallback_restitution()
        owner.kernels.prefix = sparse_packet_rows.get_prefix_kernel()
        owner.kernels.contacts = sparse_packet_rows.packet_contacts
        owner.kernels.solve = self.kernel

    def bind_current(self, state_in, state_aug, contacts, dt):
        """Bind pointers anew; no factor or geometry cache crosses calls."""

        # Reuse the original packet pointer binding without changing its laws.
        class View:
            pass

        view = View()
        view.solver, view.packet_input = self.owner.solver, self.packet_input
        sparse_packet_rows.bind_current(view, state_in, state_aug, contacts, dt)
        self.contacts = contacts

    def solve(self, rhs, iterations, omega, friction_start):
        """Finish eligible worlds first, then run every original fallback owner."""
        o, x = self.owner, self.packet_input
        s, device = o.solver, o.solver.model.device
        original = [
            o.plan,
            o.data,
            s.constraint_count,
            rhs,
            s.diag,
            s.impulses,
            s.row_type,
            s.row_parent,
            s.row_mu,
            iterations,
            omega,
            friction_start,
            s.v_hat,
            s.v_out,
        ]
        wp.launch_tiled(self.kernel, dim=[s.world_count], inputs=[*original, x, self.data], block_dim=32, device=device)
        wp.launch(
            present_port_fallback_limits,
            dim=(s.world_count, 100),
            inputs=[o.plan, o.data, self.data, s.dense_phase_bounds, s.row_cfm, s.diag, s.v_hat],
            device=device,
        )
        c = self.contacts
        if c is not None and c.rigid_contact_max > 0:
            workers = min(c.rigid_contact_max, 16384)
            wp.launch_tiled(
                self.fallback_contacts,
                dim=[workers],
                inputs=[
                    workers,
                    o.plan,
                    o.data,
                    c.rigid_contact_count,
                    s.contact_path,
                    s.contact_slot,
                    s.contact_world,
                    s.contact_art_a,
                    s.contact_art_b,
                    s.contact_slots_needed,
                    c.rigid_contact_shape0,
                    c.rigid_contact_shape1,
                    c.rigid_contact_point0,
                    c.rigid_contact_point1,
                    c.rigid_contact_normal,
                    c.rigid_contact_margin0,
                    c.rigid_contact_margin1,
                    s.model.shape_body,
                    x.body_q,
                    x.screw,
                    x.origin,
                    s.art_group_idx,
                    s.v_hat,
                    x.shared_anchor,
                    x.friction_anchor,
                    s.row_cfm,
                    s.diag,
                    self.data,
                ],
                block_dim=32,
                device=device,
            )
        wp.launch(
            self.fallback_restitution,
            dim=(s.world_count, 100),
            inputs=[
                o.data,
                s.constraint_count,
                s.row_type,
                s.phi,
                s.target_velocity,
                s.row_restitution,
                x.dt,
                x.threshold,
                rhs,
                self.data,
            ],
            device=device,
        )
        wp.launch_tiled(
            self.fallback_solve, dim=[s.world_count], inputs=[*original, self.data], block_dim=32, device=device
        )
