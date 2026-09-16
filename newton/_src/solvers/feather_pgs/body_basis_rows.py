# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental world-local current-body maps; retain materialized sparse rows."""

import ast
import hashlib
import inspect
import linecache
import textwrap
from functools import cache
from types import MethodType

import numpy as np
import warp as wp

from . import kernels
from .sparse_factor import SparseData, SparsePlan

_PREPARE_SHA = "1274deefcf63d18d4666b16bf0a85485dccaaa093b5461e390ccfb0ae1cdcadc"
_BUILD_SHA = "1532798c2128c0b5753d461e8d9921f04fcac9d5833a67b3aa85374df9d61feb"


def _source(function, expected):
    source = textwrap.dedent(inspect.getsource(function))
    if hashlib.sha256(source.encode()).hexdigest() != expected:
        raise RuntimeError("Body-basis original source changed")
    return source


def _compile(tree, namespace, label):
    source = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    filename = f"<body-basis-{label}-{hashlib.sha256(source.encode()).hexdigest()}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    exec(compile(source, filename, "exec"), namespace)
    return namespace[tree.body[0].name]


@cache
def get_prepare_kernel():
    """Append a normal-row raw key without changing original metadata arithmetic."""
    original = kernels.prepare_world_contact_rows.func
    tree = ast.parse(_source(original, _PREPARE_SHA))
    function = tree.body[0]
    function.name = "prepare_body_basis_contact_rows"
    function.decorator_list = []
    function.args.args.append(ast.arg(arg="row_key", annotation=ast.parse("wp.array2d[int]", mode="eval").body))
    writes = 0
    for node in ast.walk(function):
        if isinstance(node, ast.For):
            for i, statement in enumerate(node.body):
                if (
                    isinstance(statement, ast.Assign)
                    and ast.unparse(statement.targets[0]) == "world_row_type[world, slot]"
                ):
                    node.body.insert(i + 1, ast.parse("row_key[world, slot] = c").body[0])
                    writes += 1
                    break
    if writes != 1:
        raise RuntimeError("Body-basis metadata key seam changed")
    result = _compile(tree, dict(original.__globals__), "metadata")
    return wp.kernel(result, enable_backward=False, module="unique")


@cache
def get_contact_kernel(cache_capacity=3):
    """Build the original sparse rows with four lanes per current contact."""
    if not isinstance(cache_capacity, int) or not 0 <= cache_capacity <= 3:
        raise ValueError("Body-basis cache capacity must be between zero and three")
    source = r"""
    const int art_world=p.group_to_art.data[group], w=p.art_to_world.data[art_world];
    const int capacity=d.Z.shape[1], start=p.art_dof_start.data[art_world];
    if(!d.valid.data[w])return;
#if defined(__CUDA_ARCH__)
    const int tid=threadIdx.x;
    __shared__ int keys[100], key_count, bodies[3], labels[3], body_count, direct;
    __shared__ float basis[__STORAGE__*43*6], incident_body[__STORAGE__*6];
#else
    const int tid=0;
    int keys[100], key_count=0, bodies[3], labels[3], body_count=0, direct=0;
    float basis[__STORAGE__*43*6], incident_body[__STORAGE__*6];
#endif
    // Snapshot every current raw key before any subgroup publishes support.
    if(tid==0) {
        key_count=0;body_count=0;direct=0;
        for(int t=0;t<3;++t){bodies[t]=-1;labels[t]=-1;}
        const int nr=wp::min(wp::max(row_count.data[w],0),capacity);
        for(int row=0;row<nr;++row) {
            if(row_type.data[w*capacity+row]!=0)continue;
            const int c=d.support.data[w*capacity+row];
            if(c<0 || c>=count.data[0] || c>=path.shape[0] || path.data[c]!=0 ||
               slot.data[c]!=row || world.data[c]!=w) {d.status.data[w]=4;continue;}
            keys[key_count++]=c;
            const int aa=art_a.data[c],ab=art_b.data[c];
            if(aa>=0 && ab>=0 && aa!=ab){direct=1;continue;}
            const int sa=shape0.data[c],sb=shape1.data[c];
            const int ba=sa>=0?shape_body.data[sa]:-1,bb=sb>=0?shape_body.data[sb]:-1;
            for(int side=0;side<2;++side) {
                const int b=side==0?ba:bb, a=side==0?aa:ab;
                if(b<0 || a<0)continue;
                const int tag=p.body_tag.data[b];
                if(a!=art_world || tag<1 || tag>3){direct=1;continue;}
                int found=-1;
                for(int k=0;k<body_count;++k)if(bodies[k]==b)found=k;
                if(found<0) {
                    if(body_count>=__CAPACITY__){direct=1;continue;}
                    labels[tag-1]=body_count;bodies[body_count++]=b;
                }
            }
        }
    }
#if defined(__CUDA_ARCH__)
    __syncthreads();
#endif
    if(key_count==0)return;
    if(!direct) {
#if defined(__CUDA_ARCH__)
        for(int e=tid;e<__STORAGE__*43*6;e+=128)basis[e]=0.0f;
#else
        for(int e=0;e<__STORAGE__*43*6;++e)basis[e]=0.0f;
#endif
#if defined(__CUDA_ARCH__)
        __syncthreads();
        for(int task=tid;task<body_count*6*18;task+=128) {
#else
        for(int task=0;task<body_count*6*18;++task) {
#endif
            const int bslot=task/(6*18), axis=(task/18)%6, k=task%18;
            const int body=bodies[bslot],tag=p.body_tag.data[body];
            const int tpl=p.pair_support.data[tag*4],length=p.support_count.data[tpl];
            if(k<length) {
                const int node=p.support_nodes.data[tpl*18+k];
                float value=0.0f;
                for(int j=0;j<length;++j) {
                    const int col=p.support_nodes.data[tpl*18+j],dof=42-col;
                    const int entry=p.index.data[node*43+col];
                    if(entry>=0 && (p.body_mask.data[body]&(1ull<<dof)))
                        value+=d.W.data[group*434+entry]*S.data[start+dof][axis];
                }
                basis[(bslot*43+node)*6+axis]=value;
            }
            if(k==0) {
                float value=0.0f;
                for(int j=0;j<length;++j) {
                    const int dof=42-p.support_nodes.data[tpl*18+j];
                    if(p.body_mask.data[body]&(1ull<<dof))value+=S.data[start+dof][axis]*vhat.data[start+dof];
                }
                incident_body[bslot*6+axis]=value;
            }
        }
    }
#if defined(__CUDA_ARCH__)
    __syncthreads();
#endif
    struct Geometry {
        int c,row,art,group,start,ba,bb,aa,ab,ta,tb,tpl,length,directions;
        wp::vec3 n,t0,t1,pan,pbn,pat,pbt,o;
    };
    auto geometry=[&](int c,Geometry& g)->bool {
        g.c=c;g.row=slot.data[c];g.aa=art_a.data[c];g.ab=art_b.data[c];
        if(g.aa>=0 && g.ab>=0 && g.aa!=g.ab){d.status.data[w]=2;return false;}
        g.art=g.aa>=0?g.aa:g.ab;if(g.art<0)return false;
        g.group=group_of_art.data[g.art];g.start=p.art_dof_start.data[g.art];
        const int sa=shape0.data[c],sb=shape1.data[c];
        g.ba=sa>=0?shape_body.data[sa]:-1;g.bb=sb>=0?shape_body.data[sb]:-1;
        g.ta=g.ba>=0?p.body_tag.data[g.ba]:0;g.tb=g.bb>=0?p.body_tag.data[g.bb]:0;
        if(g.ta<0 || g.tb<0 || g.ta>3 || g.tb>3){d.status.data[w]=3;return false;}
        g.tpl=p.pair_support.data[g.ta*4+g.tb];g.length=p.support_count.data[g.tpl];
        g.n=-normals.data[c];
        const wp::vec3 pa=(g.ba>=0?wp::transform_point(body_q.data[g.ba],point0.data[c]):point0.data[c])-thickness0.data[c]*g.n;
        const wp::vec3 pb=(g.bb>=0?wp::transform_point(body_q.data[g.bb],point1.data[c]):point1.data[c])+thickness1.data[c]*g.n;
        const wp::vec3 anchor=0.5f*(pa+pb);
        g.pan=shared_anchor?anchor:pa;g.pbn=shared_anchor?anchor:pb;
        g.pat=(shared_anchor||friction_anchor)?anchor:pa;g.pbt=(shared_anchor||friction_anchor)?anchor:pb;
        g.t0=wp::cross(g.n,wp::vec3(1.0f,0.0f,0.0f));
        if(wp::length_sq(g.t0)<1.0e-12f)g.t0=wp::cross(g.n,wp::vec3(0.0f,1.0f,0.0f));
        g.t0=wp::normalize(g.t0);g.t1=wp::normalize(wp::cross(g.n,g.t0));
        g.o=origin.data[g.art];g.directions=wp::min(needed.data[c],3);
        return true;
    };
    auto physical=[&](const Geometry& g,int k,float& jn,float& jt0,float& jt1) {
        jn=0.0f;jt0=0.0f;jt1=0.0f;
        if(k>=g.length)return;
        const int node=p.support_nodes.data[g.tpl*18+k],dof=42-node;
        const wp::spatial_vector screw=S.data[g.start+dof];
        const wp::vec3 lin(screw[0],screw[1],screw[2]),ang(screw[3],screw[4],screw[5]);
        const unsigned long long bit=1ull<<dof;
        if(g.ba>=0 && g.aa==g.art && (p.body_mask.data[g.ba]&bit)) {
            const wp::vec3 vn=lin+wp::cross(ang,g.pan-g.o),vt=lin+wp::cross(ang,g.pat-g.o);
            jn+=wp::dot(g.n,vn);jt0+=wp::dot(g.t0,vt);jt1+=wp::dot(g.t1,vt);
        }
        if(g.bb>=0 && g.ab==g.art && (p.body_mask.data[g.bb]&bit)) {
            const wp::vec3 vn=lin+wp::cross(ang,g.pbn-g.o),vt=lin+wp::cross(ang,g.pbt-g.o);
            jn-=wp::dot(g.n,vn);jt0-=wp::dot(g.t0,vt);jt1-=wp::dot(g.t1,vt);
        }
    };
    auto publish=[&](const Geometry& g,int k,float z0,float z1,float z2) {
        if(k>=18)return;
        if(g.directions>0 && g.row<capacity)d.Z.data[(w*capacity+g.row)*18+k]=z0;
        if(g.directions>1 && g.row+1<capacity)d.Z.data[(w*capacity+g.row+1)*18+k]=z1;
        if(g.directions>2 && g.row+2<capacity)d.Z.data[(w*capacity+g.row+2)*18+k]=z2;
    };
    auto metadata=[&](const Geometry& g,int r,float norm,float incident) {
        const int row=g.row+r;if(r>=g.directions || row>=capacity)return;
        const int at=w*capacity+row;
        d.support.data[at]=g.tpl;d.incident.data[at]=incident;diagonal.data[at]=norm+cfm.data[at];
    };
    if(direct) {
#if defined(__CUDA_ARCH__)
        const int lane=tid&31,warp=tid/32;
        for(int contact=warp;contact<key_count;contact+=4) {
            Geometry g;if(!geometry(keys[contact],g))continue;
            float j0,j1,j2;physical(g,lane,j0,j1,j2);
            float z0=0.0f,z1=0.0f,z2=0.0f;
            const int node=lane<g.length?p.support_nodes.data[g.tpl*18+lane]:-1;
            for(int k=0;k<g.length;++k) {
                const float x=__shfl_sync(0xffffffff,j0,k),y=__shfl_sync(0xffffffff,j1,k),z=__shfl_sync(0xffffffff,j2,k);
                if(lane<g.length) {
                    const int col=p.support_nodes.data[g.tpl*18+k],entry=p.index.data[node*43+col];
                    if(entry>=0){const float a=d.W.data[g.group*434+entry];z0+=a*x;z1+=a*y;z2+=a*z;}
                }
            }
            publish(g,lane,z0,z1,z2);
            for(int r=0;r<g.directions;++r) {
                const float z=r==0?z0:(r==1?z1:z2),j=r==0?j0:(r==1?j1:j2);
                float norm=z*z,inc=lane<g.length?j*vhat.data[g.start+42-node]:0.0f;
                for(int shift=16;shift>0;shift>>=1){norm+=__shfl_down_sync(0xffffffff,norm,shift);inc+=__shfl_down_sync(0xffffffff,inc,shift);}
                if(lane==0)metadata(g,r,norm,inc);
            }
        }
#else
        for(int contact=0;contact<key_count;++contact) {
            Geometry g;if(!geometry(keys[contact],g))continue;
            float j[3][18];for(int k=0;k<18;++k)physical(g,k,j[0][k],j[1][k],j[2][k]);
            float norms[3]={0,0,0},inc[3]={0,0,0};
            for(int k=0;k<18;++k) {
                float z[3]={0,0,0};
                if(k<g.length) {
                    const int node=p.support_nodes.data[g.tpl*18+k];
                    for(int q=0;q<g.length;++q) {
                        const int col=p.support_nodes.data[g.tpl*18+q],entry=p.index.data[node*43+col];
                        if(entry>=0)for(int r=0;r<3;++r)z[r]+=d.W.data[g.group*434+entry]*j[r][q];
                    }
                    for(int r=0;r<3;++r)inc[r]+=j[r][k]*vhat.data[g.start+42-node];
                }
                publish(g,k,z[0],z[1],z[2]);for(int r=0;r<3;++r)norms[r]+=z[r]*z[r];
            }
            for(int r=0;r<g.directions;++r)metadata(g,r,norms[r],inc[r]);
        }
#endif
        return;
    }
#if defined(__CUDA_ARCH__)
    const int sub=tid&3,contact_group=tid/4;
    for(int contact=contact_group;contact<key_count;contact+=32) {
#else
    const int sub=0;
    for(int contact=0;contact<key_count;++contact) {
#endif
        Geometry g;if(!geometry(keys[contact],g))continue;
        const int ia=g.ta>0?labels[g.ta-1]:-1,ib=g.tb>0?labels[g.tb-1]:-1;
        const wp::vec3 ma0=wp::cross(g.pan-g.o,g.n),ma1=wp::cross(g.pat-g.o,g.t0),ma2=wp::cross(g.pat-g.o,g.t1);
        const wp::vec3 mb0=wp::cross(g.pbn-g.o,g.n),mb1=wp::cross(g.pbt-g.o,g.t0),mb2=wp::cross(g.pbt-g.o,g.t1);
        float norm0=0.0f,norm1=0.0f,norm2=0.0f;
#if defined(__CUDA_ARCH__)
        for(int k=sub;k<18;k+=4) {
#else
        for(int k=0;k<18;++k) {
#endif
            float z0=0.0f,z1=0.0f,z2=0.0f;
            if(k<g.length) {
                const int node=p.support_nodes.data[g.tpl*18+k];
                for(int axis=0;axis<6;++axis) {
                    const float a=ia>=0?basis[(ia*43+node)*6+axis]:0.0f;
                    const float b=ib>=0?basis[(ib*43+node)*6+axis]:0.0f;
                    z0+=a*(axis<3?g.n[axis]:ma0[axis-3])-b*(axis<3?g.n[axis]:mb0[axis-3]);
                    z1+=a*(axis<3?g.t0[axis]:ma1[axis-3])-b*(axis<3?g.t0[axis]:mb1[axis-3]);
                    z2+=a*(axis<3?g.t1[axis]:ma2[axis-3])-b*(axis<3?g.t1[axis]:mb2[axis-3]);
                }
            }
            publish(g,k,z0,z1,z2);norm0+=z0*z0;norm1+=z1*z1;norm2+=z2*z2;
        }
#if defined(__CUDA_ARCH__)
        const unsigned mask=0xfu<<((tid&31)&~3);
        for(int shift=2;shift>0;shift>>=1) {
            norm0+=__shfl_down_sync(mask,norm0,shift,4);
            norm1+=__shfl_down_sync(mask,norm1,shift,4);
            norm2+=__shfl_down_sync(mask,norm2,shift,4);
        }
#endif
        if(sub==0) {
            float inc0=0.0f,inc1=0.0f,inc2=0.0f;
            for(int axis=0;axis<6;++axis) {
                const float a=ia>=0?incident_body[ia*6+axis]:0.0f,b=ib>=0?incident_body[ib*6+axis]:0.0f;
                inc0+=a*(axis<3?g.n[axis]:ma0[axis-3])-b*(axis<3?g.n[axis]:mb0[axis-3]);
                inc1+=a*(axis<3?g.t0[axis]:ma1[axis-3])-b*(axis<3?g.t0[axis]:mb1[axis-3]);
                inc2+=a*(axis<3?g.t1[axis]:ma2[axis-3])-b*(axis<3?g.t1[axis]:mb2[axis-3]);
            }
            metadata(g,0,norm0,inc0);metadata(g,1,norm1,inc1);metadata(g,2,norm2,inc2);
        }
    }
""".replace("__CAPACITY__", str(cache_capacity)).replace("__STORAGE__", str(max(cache_capacity, 1)))

    @wp.func_native(source)
    def native(
        group: int,
        workers: int,
        p: SparsePlan,
        d: SparseData,
        count: wp.array[int],
        path: wp.array[int],
        slot: wp.array[int],
        world: wp.array[int],
        art_a: wp.array[int],
        art_b: wp.array[int],
        needed: wp.array[int],
        shape0: wp.array[int],
        shape1: wp.array[int],
        point0: wp.array[wp.vec3],
        point1: wp.array[wp.vec3],
        normals: wp.array[wp.vec3],
        thickness0: wp.array[float],
        thickness1: wp.array[float],
        shape_body: wp.array[int],
        body_q: wp.array[wp.transform],
        S: wp.array[wp.spatial_vector],
        origin: wp.array[wp.vec3],
        group_of_art: wp.array[int],
        vhat: wp.array[float],
        shared_anchor: int,
        friction_anchor: int,
        cfm: wp.array2d[float],
        diagonal: wp.array2d[float],
        row_count: wp.array[int],
        row_type: wp.array2d[int],
    ): ...

    def contacts(
        workers: int,
        p: SparsePlan,
        d: SparseData,
        count: wp.array[int],
        path: wp.array[int],
        slot: wp.array[int],
        world: wp.array[int],
        art_a: wp.array[int],
        art_b: wp.array[int],
        needed: wp.array[int],
        shape0: wp.array[int],
        shape1: wp.array[int],
        point0: wp.array[wp.vec3],
        point1: wp.array[wp.vec3],
        normals: wp.array[wp.vec3],
        thickness0: wp.array[float],
        thickness1: wp.array[float],
        shape_body: wp.array[int],
        body_q: wp.array[wp.transform],
        S: wp.array[wp.spatial_vector],
        origin: wp.array[wp.vec3],
        group_of_art: wp.array[int],
        vhat: wp.array[float],
        shared_anchor: int,
        friction_anchor: int,
        cfm: wp.array2d[float],
        diagonal: wp.array2d[float],
        row_count: wp.array[int],
        row_type: wp.array2d[int],
    ):
        group, _ = wp.tid()
        native(
            group,
            workers,
            p,
            d,
            count,
            path,
            slot,
            world,
            art_a,
            art_b,
            needed,
            shape0,
            shape1,
            point0,
            point1,
            normals,
            thickness0,
            thickness1,
            shape_body,
            body_q,
            S,
            origin,
            group_of_art,
            vhat,
            shared_anchor,
            friction_anchor,
            cfm,
            diagonal,
            row_count,
            row_type,
        )

    contacts.__name__ = contacts.__qualname__ = f"body_basis_contact_rows43_c100_b{cache_capacity}"
    return wp.kernel(contacts, enable_backward=False, module="unique")


@cache
def _build_rows(original):
    tree = ast.parse(_source(original, _BUILD_SHA))
    function = tree.body[0]
    function.name = "build_body_basis_rows"
    patches = [0, 0]
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        target = ast.unparse(node.args[0])
        if target == "k.prepare_world_contact_rows":
            node.args[0] = ast.Name(id="body_basis_prepare", ctx=ast.Load())
            for keyword in node.keywords:
                if keyword.arg == "outputs":
                    keyword.value.elts.append(ast.parse("self.data.support", mode="eval").body)
            patches[0] += 1
        elif target == "self.kernels.contacts" and ast.unparse(node.func) == "wp.launch_tiled":
            for keyword in node.keywords:
                if keyword.arg == "dim":
                    keyword.value = ast.parse("[s.world_count]", mode="eval").body
                elif keyword.arg == "block_dim":
                    keyword.value = ast.Constant(value=128)
                elif keyword.arg == "inputs":
                    keyword.value.elts.extend(ast.parse("[s.slot_counter, s.row_type]", mode="eval").body.elts)
            patches[1] += 1
    if patches != [1, 1]:
        raise RuntimeError("Body-basis build_rows launch seam changed")
    namespace = dict(original.__globals__)
    namespace["body_basis_prepare"] = get_prepare_kernel()
    return _compile(tree, namespace, "build-rows")


def install(owner):
    """Replace only row production for an already admitted sparse owner."""
    if getattr(owner, "body_basis_rows", False):
        return True
    if owner.packet_rows or owner.solver.dense_max_constraints != 100:
        return False
    tags = np.asarray(owner.host["body_tag"])
    if set(np.unique(tags)) != {-1, 1, 2, 3} or np.max(owner.host["support_count"]) > 18:
        return False
    method = _build_rows(type(owner).build_rows)
    owner.kernels.contacts = get_contact_kernel(3)
    owner.build_rows = MethodType(method, owner)
    owner.body_basis_rows = True
    owner.body_basis_capacity = 3
    return True
