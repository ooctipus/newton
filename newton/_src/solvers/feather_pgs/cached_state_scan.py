# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Generate parent-jump publication scans with explicit lane and scratch operands."""

from string import Template

_POSE_SCAN_TEMPLATE = r"""
    float* s=reinterpret_cast<float*>(address);
#if defined(__CUDA_ARCH__)
    $LANE
    wp::transform value;
    for(int k=0;k<7;++k)value[k]=s[$POSE_STRIDE*lane+k];
    int parent=plan.body_parent.data[world*32+lane];
    for(int round=0;round<4;++round) {
        const int source=parent>=0?parent:lane;
        wp::transform ancestor;
        for(int k=0;k<7;++k)ancestor[k]=__shfl_sync($MASK,value[k],source$WIDTH);
        const int next=__shfl_sync($MASK,parent,source$WIDTH);
        if(parent>=0)value=wp::transform_multiply(ancestor,value);
        parent=parent>=0?next:-1;
    }
    for(int k=0;k<7;++k)s[$POSE_STRIDE*lane+k]=value[k];
    __syncwarp($MASK);
#else
    wp::transform value[$CPU_WIDTH],previous[$CPU_WIDTH];
    int parent[$CPU_WIDTH],next[$CPU_WIDTH];
    for(int lane=0;lane<$CPU_WIDTH;++lane) {
        parent[lane]=plan.body_parent.data[world*32+lane];
        for(int k=0;k<7;++k)value[lane][k]=s[$POSE_STRIDE*lane+k];
    }
    for(int round=0;round<4;++round) {
        for(int lane=0;lane<$CPU_WIDTH;++lane) {
            previous[lane]=value[lane];
            next[lane]=parent[lane]>=0?parent[parent[lane]]:-1;
        }
        for(int lane=0;lane<$CPU_WIDTH;++lane) {
            if(parent[lane]>=0)value[lane]=wp::transform_multiply(previous[parent[lane]],previous[lane]);
            parent[lane]=next[lane];
        }
    }
    for(int lane=0;lane<$CPU_WIDTH;++lane)for(int k=0;k<7;++k)s[$POSE_STRIDE*lane+k]=value[lane][k];
#endif
"""

_MOTION_SCAN_TEMPLATE = r"""
    float* s=reinterpret_cast<float*>(address)+$MOTION_OFFSET;
#if defined(__CUDA_ARCH__)
    $LANE
    float v[6],a[6];
    for(int k=0;k<6;++k){v[k]=s[$MOTION_STRIDE*lane+k];a[k]=0.0f;}
    int parent=plan.body_parent.data[world*32+lane];
    for(int round=0;round<4;++round) {
        const int source=parent>=0?parent:lane;
        float pv[6],pa[6];
        for(int k=0;k<6;++k){pv[k]=__shfl_sync($MASK,v[k],source$WIDTH);pa[k]=__shfl_sync($MASK,a[k],source$WIDTH);}
        const int next=__shfl_sync($MASK,parent,source$WIDTH);
        if(parent>=0) {
            const wp::vec3 p_lin(pv[0],pv[1],pv[2]),p_ang(pv[3],pv[4],pv[5]);
            const wp::vec3 c_lin(v[0],v[1],v[2]),c_ang(v[3],v[4],v[5]);
            const wp::vec3 c0=wp::cross(p_ang,c_lin),c1=wp::cross(p_lin,c_ang),c2=wp::cross(p_ang,c_ang);
            for(int k=0;k<3;++k){a[k]=pa[k]+a[k]+(c0[k]+c1[k]);a[k+3]=pa[k+3]+a[k+3]+c2[k];}
            for(int k=0;k<6;++k)v[k]=pv[k]+v[k];
        }
        parent=parent>=0?next:-1;
    }
    for(int k=0;k<6;++k){s[$MOTION_STRIDE*lane+k]=v[k];s[$ACCELERATION_OFFSET+$MOTION_STRIDE*lane+k]=a[k];}
    __syncwarp($MASK);
#else
    float v[$CPU_WIDTH][6],a[$CPU_WIDTH][6]={},old_v[$CPU_WIDTH][6],old_a[$CPU_WIDTH][6];
    int parent[$CPU_WIDTH],next[$CPU_WIDTH];
    for(int lane=0;lane<$CPU_WIDTH;++lane){parent[lane]=plan.body_parent.data[world*32+lane];for(int k=0;k<6;++k)v[lane][k]=s[$MOTION_STRIDE*lane+k];}
    for(int round=0;round<4;++round) {
        for(int lane=0;lane<$CPU_WIDTH;++lane) {
            next[lane]=parent[lane]>=0?parent[parent[lane]]:-1;
            for(int k=0;k<6;++k){old_v[lane][k]=v[lane][k];old_a[lane][k]=a[lane][k];}
        }
        for(int lane=0;lane<$CPU_WIDTH;++lane) {
            if(parent[lane]>=0) {
                const float* pv=old_v[parent[lane]];const float* pa=old_a[parent[lane]];const float* cv=old_v[lane];
                const wp::vec3 p_lin(pv[0],pv[1],pv[2]),p_ang(pv[3],pv[4],pv[5]);
                const wp::vec3 c_lin(cv[0],cv[1],cv[2]),c_ang(cv[3],cv[4],cv[5]);
                const wp::vec3 c0=wp::cross(p_ang,c_lin),c1=wp::cross(p_lin,c_ang),c2=wp::cross(p_ang,c_ang);
                for(int k=0;k<3;++k){a[lane][k]=pa[k]+old_a[lane][k]+(c0[k]+c1[k]);a[lane][k+3]=pa[k+3]+old_a[lane][k+3]+c2[k];}
                for(int k=0;k<6;++k)v[lane][k]=pv[k]+cv[k];
            }
            parent[lane]=next[lane];
        }
    }
    for(int lane=0;lane<$CPU_WIDTH;++lane)for(int k=0;k<6;++k){s[$MOTION_STRIDE*lane+k]=v[lane][k];s[$ACCELERATION_OFFSET+$MOTION_STRIDE*lane+k]=a[lane][k];}
#endif
"""


def scan_source(
    motion, *, cuda_width=32, cpu_width=32, pose_stride=7, motion_offset=224, motion_stride=6, acceleration_offset=192
):
    """Preserve four parent-jump rounds in independent 16/32-lane CUDA groups.

    The caller initializes every scan slot and provides 32-stride parent rows.
    CPU padding may differ from CUDA packing; scratch addressing is independent
    of the logical subgroup width. No live producer source is inspected/edited.
    """
    if cuda_width not in (16, 32) or cpu_width not in (16, 32):
        raise ValueError("Publication scans require 16 or 32 initialized slots")
    if min(pose_stride, motion_stride) <= 0 or min(motion_offset, acceleration_offset) < 0:
        raise ValueError("Publication scan addressing must be nonnegative")
    paired = cuda_width == 16
    return Template(_MOTION_SCAN_TEMPLATE if motion else _POSE_SCAN_TEMPLATE).substitute(
        LANE="const int lane=threadIdx.x&15; const unsigned mask=0xffffu<<(threadIdx.x&16);"
        if paired
        else "const int lane=threadIdx.x&31;",
        MASK="mask" if paired else "0xffffffffu",
        WIDTH=",16" if paired else "",
        CPU_WIDTH=cpu_width,
        POSE_STRIDE=pose_stride,
        MOTION_OFFSET=motion_offset,
        MOTION_STRIDE=motion_stride,
        ACCELERATION_OFFSET=acceleration_offset,
    )
