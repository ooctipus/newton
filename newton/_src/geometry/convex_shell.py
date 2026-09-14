# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bounded current depth-aware shell polygon for positive warm convex queries.

Nothing is published by this primitive. Refusal, overflow, unsafe planes and
nonconvergence return count zero, selecting the complete original cold path.
Original vertices and triangles are retained; only their immutable incidence is
stored. The existing BSP invalidation/graph lifetime contract owns these views.
"""

import numpy as np
import warp as wp

from .support_function import GenericShapeData, unpack_mesh_ptr
from .types import GeoType

_Points = wp.types.matrix(shape=(5, 3), dtype=wp.float32)
_PatchMatrix = wp.types.matrix(shape=(12, 3), dtype=wp.float32)


@wp.struct
class _ShellData:
    mesh_ids: wp.array[wp.uint64]
    point_ptr: wp.array[wp.uint64]
    point_count: wp.array[int]
    face_offsets: wp.array[int]
    faces: wp.array[wp.vec3i]
    edge_offsets: wp.array[int]
    edges: wp.array[wp.vec2i]
    valid: wp.array[int]


@wp.struct
class _ShellResult:
    count: int
    reason: int
    cuts: int
    polygon_count: int
    point_a: _Points
    point_b: _Points


# One warp owns one workspace. Bounds do not change public contact capacity.
# Overflow is checked before every local append and before any public write.
_WORKSPACE = r"""
struct ShellWorkspace {
    wp::vec3 points[2][64], planes[2][128];
    int plane_valid[2][128], vertex_count[2], face_count[2], mesh[2];
    wp::vec3 centroid[2], basis[3];
    wp::vec2 cap[2][64], sort[128], polygon[2][64];
    float gap[2][64]; int active_a[2][64], active_b[2][64];
    int cap_count[2], count, bank, failed, finished, cuts;
    float slab[2], guard, scale;
};
"""


@wp.func_native(
    _WORKSPACE
    + r"""
#if defined(__CUDA_ARCH__)
__shared__ ShellWorkspace workspace;
return (uint64_t)&workspace;
#else
return (uint64_t)malloc(sizeof(ShellWorkspace));
#endif
"""
)
def _storage() -> wp.uint64: ...


@wp.func_native(r"""
#if !defined(__CUDA_ARCH__)
free((void*)address);
#endif
""")
def _release(address: wp.uint64): ...


@wp.func
def _mesh_index(data: _ShellData, geom: GenericShapeData) -> int:
    if geom.shape_type == int(GeoType.BOX):
        return -2
    if geom.shape_type != int(GeoType.CONVEX_MESH):
        return -1
    pointer = unpack_mesh_ptr(geom.auxiliary)
    lo = int(0)
    hi = data.mesh_ids.shape[0]
    while lo < hi:
        middle = (lo + hi) // 2
        if data.mesh_ids[middle] < pointer:
            lo = middle + 1
        else:
            hi = middle
    if lo < data.mesh_ids.shape[0] and data.mesh_ids[lo] == pointer:
        return lo
    return -1


_NATIVE = (
    _WORKSPACE
    + r"""
auto &s = *(ShellWorkspace*)address;
auto result = seed;
int lane=0, stride=1;
#if defined(__CUDA_ARCH__)
lane=threadIdx.x & 31; stride=32;
#endif
auto sync = [&]() {
#if defined(__CUDA_ARCH__)
    __syncwarp();
#endif
};
auto finite3=[](wp::vec3 p) { return ::isfinite(p[0]) && ::isfinite(p[1]) && ::isfinite(p[2]); };
auto cross2=[](wp::vec2 a,wp::vec2 b) { return a[0]*b[1]-a[1]*b[0]; };
auto xy=[](wp::vec3 a) { return wp::vec2(a[0],a[1]); };
if(lane==0){
    s.failed=0; s.finished=0; s.cuts=0; s.bank=0; s.count=0;
    s.mesh[0]=mesh_a;s.mesh[1]=mesh_b;
    float nlen=wp::length(normal);
    if(data.valid.data[0]==0 || mesh_a==-1 || mesh_b==-1 || !finite3(normal) ||
       !finite3(scale_a) || !finite3(scale_b) || !finite3(position_b) ||
       !::isfinite(shell) || shell<=0.0f || nlen<0.5f) s.failed=1;
    for(int k=0;k<3;++k)if(scale_a[k]<=0.0f || scale_b[k]<=0.0f)s.failed=1;
    for(int k=0;k<4;++k)if(!::isfinite(orientation_b[k]))s.failed=1;
    s.basis[2]=normal/nlen;
    wp::vec3 reference=::fabsf(s.basis[2][0])<0.8f?wp::vec3(1.f,0.f,0.f):wp::vec3(0.f,1.f,0.f);
    s.basis[0]=wp::normalize(wp::cross(reference,s.basis[2]));
    s.basis[1]=wp::cross(s.basis[2],s.basis[0]);
    for(int side=0;side<2;++side){
        int m=s.mesh[side];
        s.vertex_count[side]=m==-2?8:(m>=0?data.point_count.data[m]:0);
        s.face_count[side]=m==-2?12:(m>=0?data.face_offsets.data[m+1]-data.face_offsets.data[m]:0);
        if(s.vertex_count[side]<4 || s.vertex_count[side]>64 || s.face_count[side]<4 || s.face_count[side]>128)s.failed=1;
    }
}
sync();
if(s.failed){result.data[10][1]=1;return result;}
// Transform original current vertices once. Coordinates are in A's current
// frame, resolved along the current warm normal; no cached world points.
for(int side=0;side<2;++side){
    const auto scale=side==0?scale_a:scale_b;
    int m=s.mesh[side];
    for(int i=lane;i<s.vertex_count[side];i+=stride){
        wp::vec3 p;
        if(m==-2)p=wp::vec3((i&1)?1.f:-1.f,(i&2)?1.f:-1.f,(i&4)?1.f:-1.f);
        else p=((wp::vec3*)data.point_ptr.data[m])[i];
        p=wp::cw_mul(p,scale);
        if(side==1)p=wp::quat_rotate(orientation_b,p)+position_b;
        s.points[side][i]=wp::vec3(wp::dot(p,s.basis[0]),wp::dot(p,s.basis[1]),wp::dot(p,s.basis[2]));
    }
}
sync();
if(lane==0){
    s.scale=1e-12f;
    float maximum=-INFINITY,minimum=INFINITY;
    for(int side=0;side<2;++side){
        wp::vec3 centroid(0.f);
        for(int i=0;i<s.vertex_count[side];++i){
            auto p=s.points[side][i];if(!finite3(p))s.failed=1;
            centroid+=p;
            s.scale=wp::max(s.scale,wp::max(::fabsf(p[0]),wp::max(::fabsf(p[1]),::fabsf(p[2]))));
            if(side==0)maximum=wp::max(maximum,p[2]);else minimum=wp::min(minimum,p[2]);
        }
        s.centroid[side]=centroid/(float)(s.vertex_count[side]);
    }
    const float slack=shell-(minimum-maximum);
    if(!::isfinite(slack)||slack<=0.f)s.failed=2;
    s.slab[0]=maximum-slack;s.slab[1]=minimum+slack;
    s.guard=32.f*1.1920928955078125e-7f*s.scale+1e-30f;
}
sync();if(s.failed){result.data[10][1]=s.failed;return result;}
// Original triangulation, including every non-coplanar face. Coplanar
// triangles may repeat a height plane; no approximate face grouping is used.
const int box_faces[12][3]={{0,2,3},{0,3,1},{4,5,7},{4,7,6},{0,1,5},{0,5,4},
                          {2,6,7},{2,7,3},{0,4,6},{0,6,2},{1,3,7},{1,7,5}};
for(int side=0;side<2;++side){
    int m=s.mesh[side];
    for(int f=lane;f<s.face_count[side];f+=stride){
        wp::vec3i ids;
        if(m==-2)ids=wp::vec3i(box_faces[f][0],box_faces[f][1],box_faces[f][2]);
        else ids=data.faces.data[data.face_offsets.data[m]+f];
        auto a=s.points[side][ids[0]],b=s.points[side][ids[1]],c=s.points[side][ids[2]];
        auto n=wp::cross(b-a,c-a);float length=wp::length(n);
        s.plane_valid[side][f]=0;s.planes[side][f]=wp::vec3(0.f);
        if(::isfinite(length)&&length>1e-20f){
            n=n/length;if(wp::dot(n,a-s.centroid[side])<0.f)n=-n;
            bool applicable=side==0?(n[2]>0.f && wp::max(a[2],wp::max(b[2],c[2]))>=s.slab[side]):
                                   (n[2]<0.f && wp::min(a[2],wp::min(b[2],c[2]))<=s.slab[side]);
            if(applicable){
                // Near-vertical planes are not divided. Vertical sides are
                // represented by the complete projected cap domain.
                if(::fabsf(n[2])<1e-6f)s.plane_valid[side][f]=-1;
                else {auto p=wp::vec3(-n[0]/n[2],-n[1]/n[2],wp::dot(n,a)/n[2]);
                      s.planes[side][f]=p;s.plane_valid[side][f]=finite3(p)?1:-1;}
            }
        }else s.plane_valid[side][f]=-1;
    }
}
sync();
if(lane==0){
    for(int side=0;side<2;++side){
        int useful=0;
        for(int f=0;f<s.face_count[side];++f){if(s.plane_valid[side][f]<0)s.failed=3;useful+=s.plane_valid[side][f]>0;}
        if(useful==0)s.failed=3;
    }
    // Collect the actual slab vertices and edge intersections, then form
    // each projected convex cap with a bounded monotone hull.
    for(int side=0;side<2 && !s.failed;++side){
        int count=0,m=s.mesh[side];
        auto append=[&](wp::vec2 p){
            for(int i=0;i<count;++i)if(p[0]==s.cap[side][i][0]&&p[1]==s.cap[side][i][1])return;
            if(count==64){s.failed=4;return;}s.cap[side][count++]=p;
        };
        for(int i=0;i<s.vertex_count[side];++i){auto p=s.points[side][i];
            if(side==0?p[2]>=s.slab[side]:p[2]<=s.slab[side])append(xy(p));}
        int edge_count=m==-2?12:data.edge_offsets.data[m+1]-data.edge_offsets.data[m];
        for(int e=0;e<edge_count && !s.failed;++e){
            int i=0,j=0;
            if(m==-2){int axis=e/4,ordinal=e%4;i=0;int bit=0;
                for(int k=0;k<3;++k)if(k!=axis){if(ordinal&(1<<bit))i|=1<<k;++bit;}j=i|(1<<axis);}
            else{auto ids=data.edges.data[data.edge_offsets.data[m]+e];i=ids[0];j=ids[1];}
            auto a=s.points[side][i],b=s.points[side][j];float da=a[2]-s.slab[side],db=b[2]-s.slab[side];
            if((da<0.f&&db>0.f)||(da>0.f&&db<0.f))append(xy(a+(b-a)*(da/(da-db))));
        }
        if(s.failed)break;
        if(count<3){s.failed=5;break;}
        for(int i=1;i<count;++i){auto p=s.cap[side][i];int j=i;
            while(j>0){auto q=s.cap[side][j-1];if(q[0]<p[0]||(q[0]==p[0]&&q[1]<=p[1]))break;
                s.cap[side][j]=q;--j;}s.cap[side][j]=p;}
        int k=0;
        for(int i=0;i<count;++i){auto p=s.cap[side][i];
            while(k>=2&&cross2(s.sort[k-1]-s.sort[k-2],p-s.sort[k-1])<=0.f)--k;s.sort[k++]=p;}
        int lower=k+1;
        for(int i=count-2;i>=0;--i){auto p=s.cap[side][i];
            while(k>=lower&&cross2(s.sort[k-1]-s.sort[k-2],p-s.sort[k-1])<=0.f)--k;s.sort[k++]=p;}
        --k;if(k<3||k>64){s.failed=5;break;}
        s.cap_count[side]=k;for(int i=0;i<k;++i)s.cap[side][i]=s.sort[i];
    }
    if(!s.failed){
        s.count=s.cap_count[0];for(int i=0;i<s.count;++i)s.polygon[0][i]=s.cap[0][i];
        for(int e=0;e<s.cap_count[1] && !s.failed;++e){
            auto a=s.cap[1][e],b=s.cap[1][(e+1)%s.cap_count[1]];auto edge=b-a;
            int old=s.bank,next=1-old,output=0;
            for(int i=0;i<s.count;++i){auto p=s.polygon[old][i],q=s.polygon[old][(i+1)%s.count];
                float dp=cross2(edge,p-a),dq=cross2(edge,q-a);
                if(dp>=0.f){if(output==64){s.failed=4;break;}s.polygon[next][output++]=p;}
                if((dp<0.f&&dq>0.f)||(dp>0.f&&dq<0.f)){
                    if(output==64){s.failed=4;break;}s.polygon[next][output++]=p+(q-p)*(dp/(dp-dq));}
            }
            s.count=output;s.bank=next;if(output<3)s.failed=5;
        }
        for(int i=0;i<s.count;++i)s.active_a[s.bank][i]=-1;
    }
}
sync();if(s.failed){result.data[10][1]=s.failed;return result;}
// Each new polygon vertex evaluates A-min and B-max separately. Reused
// vertices retain their gap and active pair; no 128x128 plane product exists.
auto evaluate=[&](wp::vec2 u,float &ha,float &hb,int &ia,int &ib){
    ha=INFINITY;hb=-INFINITY;ia=-1;ib=-1;
    for(int f=lane;f<s.face_count[0];f+=stride)if(s.plane_valid[0][f]>0){auto p=s.planes[0][f];
        float h=p[0]*u[0]+p[1]*u[1]+p[2];if(h<ha){ha=h;ia=f;}}
    for(int f=lane;f<s.face_count[1];f+=stride)if(s.plane_valid[1][f]>0){auto p=s.planes[1][f];
        float h=p[0]*u[0]+p[1]*u[1]+p[2];if(h>hb){hb=h;ib=f;}}
#if defined(__CUDA_ARCH__)
    for(int offset=16;offset>0;offset>>=1){
        float a=__shfl_down_sync(0xffffffff,ha,offset),b=__shfl_down_sync(0xffffffff,hb,offset);
        int ai=__shfl_down_sync(0xffffffff,ia,offset),bi=__shfl_down_sync(0xffffffff,ib,offset);
        if(lane+offset<32){if(a<ha||(a==ha&&ai>=0&&(ia<0||ai<ia))){ha=a;ia=ai;}
                           if(b>hb||(b==hb&&bi>=0&&(ib<0||bi<ib))){hb=b;ib=bi;}}
    }
#endif
};
for(int iteration=0;iteration<=64;++iteration){
    // Keep every lane on the same polygon epoch. Lane zero clips and changes
    // count/bank only after all peer lanes have left the evaluation loop.
    const int bank=s.bank,vertex_count=s.count;
    sync();
    for(int v=0;v<vertex_count;++v){
        if(s.active_a[bank][v]<0){float ha,hb;int ia,ib;evaluate(s.polygon[bank][v],ha,hb,ia,ib);
            if(lane==0){s.gap[bank][v]=hb-ha;s.active_a[bank][v]=ia;s.active_b[bank][v]=ib;
                if(ia<0||ib<0||!::isfinite(ha)||!::isfinite(hb))s.failed=6;}}
        sync();
    }
    sync();
    if(lane==0 && !s.failed){
        int violated=-1;
        for(int i=0;i<s.count;++i)if(s.gap[bank][i]>shell+s.guard){violated=i;break;}
        if(violated<0)s.finished=1;
        else if(iteration==64)s.failed=7;
        else{
            auto cut=s.planes[1][s.active_b[bank][violated]]-s.planes[0][s.active_a[bank][violated]];
            cut[2]-=shell;int next=1-bank,output=0;
            for(int i=0;i<s.count;++i){int j=(i+1)%s.count;auto p=s.polygon[bank][i],q=s.polygon[bank][j];
                float dp=cut[0]*p[0]+cut[1]*p[1]+cut[2],dq=cut[0]*q[0]+cut[1]*q[1]+cut[2];
                if(dp<=0.f){if(output==64){s.failed=4;break;}
                    s.polygon[next][output]=p;s.gap[next][output]=s.gap[bank][i];
                    s.active_a[next][output]=s.active_a[bank][i];s.active_b[next][output]=s.active_b[bank][i];++output;}
                if((dp<0.f&&dq>0.f)||(dp>0.f&&dq<0.f)){if(output==64){s.failed=4;break;}
                    s.polygon[next][output]=p+(q-p)*(dp/(dp-dq));s.active_a[next][output]=-1;++output;}
            }
            ++s.cuts;s.count=output;s.bank=next;if(output<3)s.failed=5;
        }
    }
    sync();if(s.failed||s.finished)break;
}
if(s.failed||!s.finished){result.data[10][1]=s.failed?s.failed:7;return result;}
// Select a wide quadrilateral using the original diameter/area policy. The
// contact representatives are tiny convex combinations toward the centroid,
// NOT widened-shell points or clamped depths. Both current surfaces are then
// reconstructed and their actual strict gap is checked again before return.
if(lane==0){
    int bank=s.bank,n=s.count;auto *p=s.polygon[bank];
    wp::vec2 center(0.f);for(int i=0;i<n;++i)center+=p[i];center=center/(float)(n);
    int p1=0,p3=1,j=1;float longest=wp::length_sq(p[0]-p[1]);
    for(int i=0;i<n;++i){int next=(i+1)%n;
        for(int steps=0;steps<n;++steps){int nj=(j+1)%n;
            if(cross2(p[next]-p[i],p[nj]-p[i])>cross2(p[next]-p[i],p[j]-p[i]))j=nj;else break;}
        float d=wp::length_sq(p[i]-p[j]);if(d>longest*1.001f){longest=d;p1=i;p3=j;}
        d=wp::length_sq(p[next]-p[j]);if(d>longest*1.001f){longest=d;p1=next;p3=j;}
    }
    int p2=p1,p4=p1;float positive=0.f,negative=0.f;
    for(int i=0;i<n;++i){float a=cross2(p[p3]-p[p1],p[i]-p[p1]);
        if(a>positive){positive=a;p2=i;}if(a<negative){negative=a;p4=i;}}
    int selected[4]={p1,p2,p3,p4},output=0;
    float diameter=::sqrtf(longest);
    // At least a few FP32 ulps, scaled by actual geometry/patch extent.
    float spatial_weight=wp::max(8.f*1.1920928955078125e-7f,4.f*s.guard/wp::max(diameter,1e-12f));
    auto heights=[&](wp::vec2 u){
        float ha=INFINITY,hb=-INFINITY;
        for(int f=0;f<s.face_count[0];++f)if(s.plane_valid[0][f]>0){auto plane=s.planes[0][f];
            ha=wp::min(ha,plane[0]*u[0]+plane[1]*u[1]+plane[2]);}
        for(int f=0;f<s.face_count[1];++f)if(s.plane_valid[1][f]>0){auto plane=s.planes[1][f];
            hb=wp::max(hb,plane[0]*u[0]+plane[1]*u[1]+plane[2]);}
        return wp::vec2(ha,hb);
    };
    auto center_heights=heights(center);
    float center_gap=center_heights[1]-center_heights[0],safe_gap=shell-4.f*s.guard;
    if(spatial_weight>0.01f || !::isfinite(spatial_weight) || !::isfinite(center_gap) ||
       safe_gap<=0.f || center_gap<=0.f || center_gap>safe_gap)s.failed=8;
    for(int k=0;k<4 && !s.failed;++k){bool duplicate=false;
        for(int before=0;before<k;++before)if(selected[k]==selected[before])duplicate=true;
        if(duplicate)continue;
        float vertex_gap=s.gap[bank][selected[k]],weight=spatial_weight;
        // g=max(B planes)-min(A planes) is convex. This inward move must
        // gain GAP headroom, not merely a few ulps of projected distance.
        // Already-safe vertices avoid division, including equal gap planes.
        if(vertex_gap>safe_gap){float denominator=vertex_gap-center_gap;
            if(!(denominator>0.f) || !::isfinite(denominator)){s.failed=8;break;}
            weight=wp::max(weight,(vertex_gap-safe_gap)/denominator+8.f*1.1920928955078125e-7f);}
        if(!::isfinite(vertex_gap)||!::isfinite(weight)||weight>0.01f){s.failed=8;break;}
        auto u=(1.f-weight)*p[selected[k]]+weight*center;
        auto current_heights=heights(u);float ha=current_heights[0],hb=current_heights[1];
        if(!::isfinite(ha)||!::isfinite(hb)||hb-ha>=shell||hb<=ha){s.failed=8;break;}
        auto pa=s.basis[0]*u[0]+s.basis[1]*u[1]+s.basis[2]*ha;
        auto pb=s.basis[0]*u[0]+s.basis[1]*u[1]+s.basis[2]*hb;
        for(int axis=0;axis<3;++axis){result.data[output][axis]=pa[axis];result.data[output+5][axis]=pb[axis];}++output;
    }
    if(!s.failed){result.data[10][0]=output;result.data[10][2]=s.cuts;result.data[11][0]=n;}
    result.data[10][1]=s.failed;
}
sync();return result;
"""
)


@wp.func_native(_NATIVE)
def _shell_native(
    address: wp.uint64,
    data: _ShellData,
    mesh_a: int,
    mesh_b: int,
    scale_a: wp.vec3,
    scale_b: wp.vec3,
    orientation_b: wp.quat,
    position_b: wp.vec3,
    normal: wp.vec3,
    shell: float,
    seed: _PatchMatrix,
) -> _PatchMatrix: ...


@wp.func
def _shell_patch(
    address: wp.uint64,
    data: _ShellData,
    mesh_a: int,
    mesh_b: int,
    scale_a: wp.vec3,
    scale_b: wp.vec3,
    orientation_b: wp.quat,
    position_b: wp.vec3,
    normal: wp.vec3,
    shell: float,
    seed: _ShellResult,
) -> _ShellResult:
    raw = _shell_native(
        address, data, mesh_a, mesh_b, scale_a, scale_b, orientation_b, position_b, normal, shell, _PatchMatrix()
    )
    result = _ShellResult()
    result.count = int(raw[10, 0])
    result.reason = int(raw[10, 1])
    result.cuts = int(raw[10, 2])
    result.polygon_count = int(raw[11, 0])
    for i in range(5):
        for j in range(3):
            result.point_a[i, j] = raw[i, j]
            result.point_b[i, j] = raw[i + 5, j]
    return result


class _ShellOwner:
    """Exact-size immutable incidence, sharing BSP's explicit invalidation flag."""

    def __init__(self, model, bsp):
        meshes = {int(mesh.id): mesh for mesh in model._mesh_keep_alive}
        admitted = {entry["mesh_id"] for entry in bsp.metadata["admitted_meshes"]}
        ids, pointers, counts, faces, edges = [], [], [], [], []
        face_offsets, edge_offsets = [0], [0]
        for pointer in sorted(admitted):
            mesh = meshes[pointer]
            count = mesh.points.shape[0]
            triangles = mesh.indices.numpy().reshape(-1, 3)
            if count > 64 or len(triangles) > 128:
                continue
            incidence = sorted({tuple(sorted((int(t[i]), int(t[(i + 1) % 3])))) for t in triangles for i in range(3)})
            ids.append(pointer)
            pointers.append(mesh.points.ptr)
            counts.append(count)
            faces.extend(triangles)
            edges.extend(incidence)
            face_offsets.append(len(faces))
            edge_offsets.append(len(edges))
        data = _ShellData()
        device = model.device
        data.mesh_ids = wp.array(ids, dtype=wp.uint64, device=device)
        data.point_ptr = wp.array(pointers, dtype=wp.uint64, device=device)
        data.point_count = wp.array(counts, dtype=int, device=device)
        data.face_offsets = wp.array(face_offsets, dtype=int, device=device)
        data.edge_offsets = wp.array(edge_offsets, dtype=int, device=device)
        data.faces = wp.array(np.asarray(faces).reshape(-1, 3), dtype=wp.vec3i, device=device)
        data.edges = wp.array(np.asarray(edges).reshape(-1, 2), dtype=wp.vec2i, device=device)
        data.valid = bsp.data.valid
        self.data = data
        self.model = model
        self.bsp = bsp
        self.metadata = {"mesh_count": len(ids), "vertices": sum(counts), "triangles": len(faces), "edges": len(edges)}
