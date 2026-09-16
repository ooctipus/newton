# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental finite cuboid/heightfield contacts on the original triangle stream.

The immutable convex hull is recognized at pipeline construction. Current scale,
pose, heights, margins and gaps remain live inputs. Refit/replaced hull geometry
requires rebuilding this experimental pipeline; unsupported queries retain the
original generic query, writer and reducer.
"""

import functools
import inspect
import linecache
import textwrap
from typing import Any

import numpy as np
import warp as wp

from ..utils.heightfield import HeightfieldData, get_triangle_shape_from_heightfield
from .contact_data import ContactData
from .heightfield_features import WELD_FLAT_SEAMS, HeightfieldFeatureContext, flat_seam_query_allowed
from .types import GeoType

QueryResult = wp.types.vector(length=24, dtype=wp.float32)

_QUERY = r"""
using V=wp::vec3;
auto fminf=[](float a,float b){return wp::min(a,b);};
auto fmaxf=[](float a,float b){return wp::max(a,b);};
wp::vec_t<24,float> result;
for(int i=0;i<24;++i)result[i]=0.0f;
result[0]=-1.0f;
auto finite3=[](V v){return isfinite(v[0])&&isfinite(v[1])&&isfinite(v[2]);};
if(!finite3(e1)||!finite3(e2)||!finite3(center)||!finite3(half)||
   half[0]<=0.0f||half[1]<=0.0f||half[2]<=0.0f||!isfinite(threshold)||threshold<0.0f)return result;
float q2=0.0f;for(int i=0;i<4;++i)q2+=rotation[i]*rotation[i];
if(!isfinite(q2)||q2<1.0e-12f)return result;
const float qi=1.0f/sqrtf(q2);
const wp::quat q(rotation[0]*qi,rotation[1]*qi,rotation[2]*qi,rotation[3]*qi);
const V axes[3]={wp::quat_rotate(q,V(1.0f,0.0f,0.0f)),wp::quat_rotate(q,V(0.0f,1.0f,0.0f)),wp::quat_rotate(q,V(0.0f,0.0f,1.0f))};
auto local=[&](V v){return V(wp::dot(v,axes[0]),wp::dot(v,axes[1]),wp::dot(v,axes[2]));};
auto terrain=[&](V v){return axes[0]*v[0]+axes[1]*v[1]+axes[2]*v[2];};
const V top[3]={V(0.0f),e1,e2};
V normal=wp::cross(e1,e2);const float n2=wp::dot(normal,normal);
if(!isfinite(n2)||n2<1.0e-20f||normal[2]<=0.0f)return result;
normal=normal*(1.0f/sqrtf(n2));
const V down(0.0f,0.0f,-1.0f);
const V edge[3]={e1,e2-e1,-e2};
const V nl=local(normal);
V support=center;
for(int k=0;k<3;++k)support=support-axes[k]*(nl[k]>0.0f?half[k]:(nl[k]<0.0f?-half[k]:0.0f));
const float plane_gap=wp::dot(support,normal);
if(!isfinite(plane_gap))return result;
auto inside=[&](V point){
    for(int k=0;k<3;++k)if(wp::dot(wp::cross(edge[k],point-top[k]),normal)<-1.0e-9f)return false;
    return true;
};
bool overlap=false,face=false;
V point_a(0.0f),point_b(0.0f),contact_normal=normal;
float distance=0.0f;
if(plane_gap>=0.0f&&inside(support-normal*plane_gap)){
    point_b=support;point_a=support-normal*plane_gap;distance=plane_gap;face=true;
}else{
    if(plane_gap<0.0f){
        overlap=true;
        auto separates=[&](V axis){
            if(wp::dot(axis,axis)<1.0e-24f)return false;
            const float a=wp::dot(e1,axis),b=wp::dot(e2,axis),extrude=wp::dot(down,axis);
            const float lo=fminf(0.0f,fminf(a,b))+fminf(0.0f,extrude);
            const float hi=fmaxf(0.0f,fmaxf(a,b))+fmaxf(0.0f,extrude);
            const V axis_local=local(axis);
            const float radius=half[0]*fabsf(axis_local[0])+half[1]*fabsf(axis_local[1])+half[2]*fabsf(axis_local[2]);
            const float mid=wp::dot(center,axis);
            return mid+radius<lo||mid-radius>hi;
        };
        if(separates(normal))overlap=false;
        for(int k=0;k<3&&overlap;++k)if(separates(axes[k]))overlap=false;
        for(int k=0;k<3&&overlap;++k)if(separates(wp::cross(edge[k],down)))overlap=false;
        for(int k=0;k<3&&overlap;++k){
            for(int j=0;j<3&&overlap;++j)if(separates(wp::cross(axes[k],edge[j])))overlap=false;
            if(overlap&&separates(wp::cross(axes[k],down)))overlap=false;
        }
    }
    if(overlap){face=true;}
    else{
        const V tri[3]={local(-center),local(e1-center),local(e2-center)};
        V best_a(0.0f),best_b(0.0f);float best=3.402823466e+38f;
        auto consider=[&](V a,V b){const V delta=b-a;const float d2=wp::dot(delta,delta);
            if(d2<best){best=d2;best_a=a;best_b=b;}};
        auto clamped=[&](V v){return V(fminf(half[0],fmaxf(-half[0],v[0])),fminf(half[1],fmaxf(-half[1],v[1])),fminf(half[2],fmaxf(-half[2],v[2])));};
        for(int k=0;k<3;++k){
            const V a=tri[k],d=tri[(k+1)%3]-a;
            float cuts[8];int count=2;cuts[0]=0.0f;cuts[1]=1.0f;
            for(int j=0;j<3;++j)if(d[j]!=0.0f){
                for(int s=-1;s<=1;s+=2){const float t=(static_cast<float>(s)*half[j]-a[j])/d[j];
                    if(t>0.0f&&t<1.0f){int at=count;while(at>0&&cuts[at-1]>t){cuts[at]=cuts[at-1];--at;}cuts[at]=t;++count;}}
            }
            for(int j=0;j<count;++j){const V value=a+d*cuts[j];consider(value,clamped(value));}
            for(int j=0;j<count-1;++j){
                const float mid=0.5f*(cuts[j]+cuts[j+1]);const V value=a+d*mid;
                float denom=0.0f,numerator=0.0f;
                for(int axis=0;axis<3;++axis)if(fabsf(value[axis])>half[axis]){
                    const float boundary=value[axis]>0.0f?half[axis]:-half[axis];
                    denom+=d[axis]*d[axis];numerator+=d[axis]*(a[axis]-boundary);}
                if(denom>0.0f){const float t=fminf(cuts[j+1],fmaxf(cuts[j],-numerator/denom));const V point=a+d*t;consider(point,clamped(point));}
            }
        }
        const V local_cross=wp::cross(tri[1]-tri[0],tri[2]-tri[0]);
        const float inverse_cross=1.0f/wp::dot(local_cross,local_cross);
        for(int bits=0;bits<8;++bits){
            const V vertex((bits&1)?half[0]:-half[0],(bits&2)?half[1]:-half[1],(bits&4)?half[2]:-half[2]);
            const V projection=vertex-local_cross*(wp::dot(vertex-tri[0],local_cross)*inverse_cross);
            bool in=true;for(int k=0;k<3;++k)if(wp::dot(wp::cross(tri[(k+1)%3]-tri[k],projection-tri[k]),local_cross)<0.0f)in=false;
            if(in)consider(projection,vertex);
        }
        if(!isfinite(best)||best==3.402823466e+38f||best<=1.0e-16f)return result;
        const V delta=terrain(best_b-best_a);
        if(delta[2]<0.0f)return result;
        distance=sqrtf(best);contact_normal=delta*(1.0f/distance);
        point_a=center+terrain(best_a);point_b=center+terrain(best_b);
        face=wp::dot(contact_normal,normal)>1.0f-1.0e-6f;
    }
}
auto emit=[&](int index,V a,V b,float depth){
    const V middle=(a+b)*0.5f;
    result[4+4*index]=middle[0];result[5+4*index]=middle[1];result[6+4*index]=middle[2];result[7+4*index]=depth;
};
if(!overlap&&distance>threshold){
    if(!finite3(point_a)||!finite3(point_b)||!finite3(contact_normal)||!isfinite(distance))return result;
    result[0]=1.0f;for(int k=0;k<3;++k)result[k+1]=contact_normal[k];emit(0,point_a,point_b,distance);return result;
}
if(face){
    V seed(1.0f,0.0f,0.0f);if(fabsf(normal[1])<fabsf(normal[0]))seed=V(0.0f,1.0f,0.0f);
    if(fabsf(normal[2])<fminf(fabsf(normal[0]),fabsf(normal[1])))seed=V(0.0f,0.0f,1.0f);
    V u=wp::cross(normal,seed);u=u*(1.0f/sqrtf(wp::dot(u,u)));const V v=wp::cross(normal,u);
    V chosen[5];float depths[5],scores[4];bool have=false;
    auto collect=[&](V b){
        const float depth=wp::dot(b,normal);const V a=b-normal*depth;
        const float score[4]={wp::dot(a,u),-wp::dot(a,u),wp::dot(a,v),-wp::dot(a,v)};
        if(!have){for(int k=0;k<5;++k){chosen[k]=b;depths[k]=depth;}for(int k=0;k<4;++k)scores[k]=score[k];have=true;}
        else{if(depth<depths[0]){chosen[0]=b;depths[0]=depth;}for(int k=0;k<4;++k)if(score[k]>scores[k]){scores[k]=score[k];chosen[k+1]=b;depths[k+1]=depth;}}
    };
    for(int axis=0;axis<3;++axis){
        if(nl[axis]==0.0f)continue;
        const int other0=(axis+1)%3,other1=(axis+2)%3;
        V poly[8],next[8];int count=4;
        for(int k=0;k<4;++k){V vertex(0.0f);vertex[axis]=nl[axis]>0.0f?-half[axis]:half[axis];
            vertex[other0]=(k==0||k==3)?-half[other0]:half[other0];vertex[other1]=k<2?-half[other1]:half[other1];poly[k]=center+terrain(vertex);}
        for(int side=0;side<3&&count>0;++side){
            const V inward=wp::cross(normal,edge[side]);const float offset=wp::dot(inward,top[side]);
            int output=0;V previous=poly[count-1];float dp=wp::dot(previous,inward)-offset;
            for(int k=0;k<count;++k){const V current=poly[k];const float dc=wp::dot(current,inward)-offset;
                if((dc>=0.0f)!=(dp>=0.0f)){if(output>=8)return result;next[output++]=previous+(current-previous)*(dp/(dp-dc));}
                if(dc>=0.0f){if(output>=8)return result;next[output++]=current;}previous=current;dp=dc;}
            count=output;for(int k=0;k<count;++k)poly[k]=next[k];
        }
        for(int k=0;k<count;++k)collect(poly[k]);
    }
    if(!have||(overlap&&depths[0]>0.0f))return result;
    int count=0;
    for(int k=0;k<5;++k){bool duplicate=false;for(int j=0;j<k;++j){const V d=chosen[k]-chosen[j];if(wp::dot(d,d)<1.0e-12f&&fabsf(depths[k]-depths[j])<1.0e-6f)duplicate=true;}
        if(!duplicate){emit(count,chosen[k]-normal*depths[k],chosen[k],depths[k]);++count;}}
    result[0]=static_cast<float>(count);for(int k=0;k<3;++k)result[k+1]=normal[k];
}else{
    bool interval=false;
    const float scores[3]={0.0f,wp::dot(e1,contact_normal),wp::dot(e2,contact_normal)};
    const float maximum=fmaxf(scores[0],fmaxf(scores[1],scores[2]));
    const float tolerance=2.0e-6f*fmaxf(1.0f,fmaxf(sqrtf(wp::dot(e1,e1)),sqrtf(wp::dot(e2,e2))));
    for(int side=0;side<3&&!interval;++side){const int end=(side+1)%3;
        if(fabsf(scores[side]-maximum)>tolerance||fabsf(scores[end]-maximum)>tolerance)continue;
        const V a=local(top[side]+contact_normal*distance-center),d=local(edge[side]);float lo=0.0f,hi=1.0f;
        for(int k=0;k<3;++k){if(fabsf(d[k])>1.0e-12f){const float t0=(-half[k]-a[k])/d[k],t1=(half[k]-a[k])/d[k];lo=fmaxf(lo,fminf(t0,t1));hi=fminf(hi,fmaxf(t0,t1));}
            else if(fabsf(a[k])>half[k]+1.0e-6f){lo=1.0f;hi=0.0f;}}
        if(hi>=lo){const V a0=top[side]+edge[side]*lo;emit(0,a0,a0+contact_normal*distance,distance);result[0]=1.0f;
            if(hi-lo>1.0e-6f){const V a1=top[side]+edge[side]*hi;emit(1,a1,a1+contact_normal*distance,distance);result[0]=2.0f;}interval=true;}
    }
    if(!interval){emit(0,point_a,point_b,distance);result[0]=1.0f;}
    for(int k=0;k<3;++k)result[k+1]=contact_normal[k];
}
for(int i=0;i<24;++i)if(!isfinite(result[i])){result[0]=-1.0f;return result;}
return result;
"""


@wp.func_native(_QUERY)
def query(e1: wp.vec3, e2: wp.vec3, center: wp.vec3, rotation: wp.quat, half: wp.vec3, threshold: float) -> QueryResult:
    """Return count, normal and at most five midpoint/distance records, or -1 fallback."""
    ...


@wp.func
def query_contacts(
    e1: wp.vec3,
    e2: wp.vec3,
    center: wp.vec3,
    rotation: wp.quat,
    half: wp.vec3,
    threshold: float,
    margin_sum: float,
) -> QueryResult:
    """Retain original speculative face manifolds at the current finite-sweep law."""
    result = query(e1, e2, center, rotation, half, threshold)
    count = int(result[0])
    if count > 0:
        normal = wp.vec3(result[1], result[2], result[3])
        top_normal = wp.normalize(wp.cross(e1, e2))
        if wp.dot(normal, top_normal) > 1.0 - 1.0e-6:
            minimum = result[7]
            for k in range(1, count):
                minimum = wp.min(minimum, result[7 + 4 * k])
            # Positive-clearance rows do not enter the retained reducer's
            # spatial support slots. Keep their original manifold instead of
            # relying on depth ties to preserve finite-iteration support.
            if minimum > margin_sum and minimum <= threshold:
                result[0] = -1.0
    return result


def bind_model(narrow, model):
    """Bind immutable cuboid hull bounds once; all dynamic geometry stays current."""
    bounds = np.zeros((model.shape_count, 2, 3), dtype=np.float32)
    bounds[:, 1] = -1.0
    types = model.shape_type.numpy()
    cache = {}
    for index, kind in enumerate(types):
        if int(kind) == int(GeoType.BOX):
            bounds[index, 1] = 1.0
        elif int(kind) == int(GeoType.CONVEX_MESH):
            source = model.shape_source[index]
            key = id(source)
            if key not in cache:
                vertices = np.asarray(getattr(source, "vertices", np.empty((0, 3))), dtype=np.float32)
                answer = None
                if vertices.shape == (8, 3) and np.isfinite(vertices).all():
                    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
                    corners = {tuple(vertex) for vertex in vertices}
                    expected = {(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])}
                    if len(corners) == 8 and corners == expected and np.all(hi > lo):
                        answer = np.stack(((lo + hi) * 0.5, (hi - lo) * 0.5))
                cache[key] = answer
            if cache[key] is not None:
                bounds[index] = cache[key]
    narrow._finite_bounds = wp.array(bounds, dtype=wp.vec3, ndim=2, device=model.device)
    narrow._finite_source = wp.clone(model.shape_source_ptr)


@functools.cache
def create_query_kernel(writer_func, shell_support: bool = False):
    """Create the analytical writer for the same global triangle stream."""

    def heightfield_finite_contacts(
        shape_types: wp.array[int],
        shape_data: wp.array[wp.vec4],
        shape_transform: wp.array[wp.transform],
        shape_source: wp.array[wp.uint64],
        shape_gap: wp.array[float],
        shape_heightfield_index: wp.array[int],
        heightfield_data: wp.array[HeightfieldData],
        heightfield_elevations: wp.array[float],
        triangle_pairs: wp.array[wp.vec3i],
        triangle_pairs_count: wp.array[int],
        writer_data: Any,
        total_num_threads: int,
        bounds: wp.array2d[wp.vec3],
        bound_source: wp.array[wp.uint64],
    ):
        for i in range(wp.tid(), triangle_pairs_count[0], total_num_threads):
            if i >= triangle_pairs.shape[0]:
                break
            triple = triangle_pairs[i]
            a, b, tri_idx = triple[0], triple[1], triple[2]
            if tri_idx < 0 or shape_types[a] != GeoType.HFIELD:
                continue
            if shape_types[b] != GeoType.BOX and shape_types[b] != GeoType.CONVEX_MESH:
                continue
            if bounds[b, 1][0] <= 0.0 or bound_source[b] != shape_source[b]:
                continue
            margin_a, margin_b = shape_data[a][3], shape_data[b][3]
            if margin_a + margin_b < 1.0e-4:
                continue
            data = shape_data[b]
            scale = wp.vec3(data[0], data[1], data[2])
            if scale[0] <= 0.0 or scale[1] <= 0.0 or scale[2] <= 0.0:
                continue
            xa, xb = shape_transform[a], shape_transform[b]
            qa = wp.normalize(wp.transform_get_rotation(xa))
            qb = wp.normalize(wp.transform_get_rotation(xb))
            geom, origin = get_triangle_shape_from_heightfield(
                heightfield_data[shape_heightfield_index[a]], heightfield_elevations, xa, tri_idx
            )
            center_world = wp.transform_get_translation(xb) + wp.quat_rotate(qb, wp.cw_mul(bounds[b, 0], scale))
            center = wp.quat_rotate_inv(qa, center_world - origin)
            gap = shape_gap[a] + shape_gap[b]
            value = QueryResult()
            if wp.static(shell_support):
                # Only the paired buffered reducer supports these spatial
                # extrema; the direct writer retains its original admission.
                value = query(
                    geom.scale,
                    geom.auxiliary,
                    center,
                    wp.quat_inverse(qa) * qb,
                    wp.cw_mul(bounds[b, 1], scale),
                    gap + margin_a + margin_b,
                )
            else:
                value = query_contacts(
                    geom.scale,
                    geom.auxiliary,
                    center,
                    wp.quat_inverse(qa) * qb,
                    wp.cw_mul(bounds[b, 1], scale),
                    gap + margin_a + margin_b,
                    margin_a + margin_b,
                )
            count = int(value[0])
            if count < 0:
                continue
            local_normal = wp.vec3(value[1], value[2], value[3])
            feature_context = HeightfieldFeatureContext()
            if wp.static(WELD_FLAT_SEAMS):
                feature_context.heightfield = heightfield_data[shape_heightfield_index[a]]
                feature_context.elevations = heightfield_elevations
                feature_context.triangle = tri_idx
            normal = wp.quat_rotate(qa, wp.vec3(value[1], value[2], value[3]))
            for k in range(count):
                if wp.static(WELD_FLAT_SEAMS):
                    terrain_point = wp.vec3(value[4 + 4 * k], value[5 + 4 * k], value[6 + 4 * k])
                    terrain_point -= 0.5 * value[7 + 4 * k] * local_normal
                    if value[7 + 4 * k] >= 0.0 and not flat_seam_query_allowed(
                        geom, terrain_point, local_normal, feature_context
                    ):
                        continue
                contact = ContactData()
                contact.shape_a = a
                contact.shape_b = b
                contact.margin_a = margin_a
                contact.margin_b = margin_b
                contact.gap_sum = gap
                contact.contact_point_center = origin + wp.quat_rotate(
                    qa, wp.vec3(value[4 + 4 * k], value[5 + 4 * k], value[6 + 4 * k])
                )
                contact.contact_normal_a_to_b = normal
                contact.contact_distance = value[7 + 4 * k]
                contact.sort_sub_key = (((tri_idx << 1) | 1) << 3) | k
                # Axial/Minkowski postprocessing is identity for this admitted
                # HFIELD + BOX/CONVEX family; both effective radii remain zero.
                wp.static(writer_func)(contact, writer_data, -1)
            triangle_pairs[i] = wp.vec3i(a, b, ~tri_idx)

    if shell_support:
        heightfield_finite_contacts.__name__ = heightfield_finite_contacts.__qualname__ = (
            "heightfield_finite_shell_contacts"
        )
    return wp.kernel(heightfield_finite_contacts, enable_backward=False, module="unique")


@functools.cache
def marked_fallback(kernel):
    """Clone the original complete query with only a handled-entry skip inserted."""
    source = textwrap.dedent(inspect.getsource(kernel.func))
    lines = source.splitlines()
    while lines[0].startswith("@"):
        lines.pop(0)
    source = "\n".join(lines) + "\n"
    old = "        tri_idx = triple[2]"
    if source.count(old) != 1:
        raise RuntimeError("Original triangle query entry changed")
    source = source.replace(old, old + "\n        if tri_idx < 0:\n            continue", 1)
    name = kernel.func.__name__ + "_finite_fallback"
    source = source.replace("def " + kernel.func.__name__ + "(", "def " + name + "(", 1)
    filename = f"<heightfield_finite_{name}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = dict(kernel.func.__globals__)
    namespace.update(inspect.getclosurevars(kernel.func).nonlocals)
    exec(compile(source, filename, "exec"), namespace)
    return wp.kernel(namespace[name], enable_backward=False, module="unique")
