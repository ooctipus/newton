# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Common body validity checks with explicit endpoint twist production."""

from string import Template

from .kinetic_source import checked_definitions

_PRESCRIBED = r"""for(int k=0;k<6;++k) s[96+b*6+k]=data.body_v_s.data[body][k];"""


_FREE = r"""            if(b>=30) {
                const int first=data.articulation_dof_start.data[art];
                for(int k=0;k<6;++k) {
                    float value=0.0f;
                    for(int d=0;d<6;++d) value+=data.joint_S_s.data[first+d][k]*data.v_hat.data[first+d];
                    s[96+b*6+k]=value;
                }
            }
"""


_VALIDATION = r"""    for(int b=lane;b<32;b+=stride) {
        const int body=plan.body_ids.data[world*32+b], art=data.body_to_articulation.data[body];
        const auto pose=data.body_q.data[body];
        const auto p=wp::transform_get_translation(pose);
        const auto q=wp::transform_get_rotation(pose);
        bool valid=true; float norm=0.0f;
        for(int k=0;k<3;++k) valid=valid&&isfinite(p[k])&&isfinite(data.articulation_origin.data[art][k]);
        for(int k=0;k<4;++k) {valid=valid&&isfinite(q[k]);norm+=q[k]*q[k];}
        valid=valid&&isfinite(norm)&&norm>0.0f;
        const bool responding=data.body_has_response_dofs.data[body]!=0&&(data.body_flags.data[body]&2)==0;
        if(data.prescribed_articulation.data[art]!=0) {
            ${prescribed}
        } else if(responding) {
${free}            if(data.is_free_rigid.data[art]!=0) {
                const int first=data.articulation_dof_start.data[art];
                for(int k=0;k<6;++k) {
                    const float limit=k<3?data.max_linear_velocity.data[body]:data.max_angular_velocity.data[body];
                    valid=valid&&!isnan(limit);
                    if(isfinite(limit)&&limit>0.0f) valid=valid&&fabsf(data.v_hat.data[first+k])<=limit;
                }
            }
        } else {
            for(int k=0;k<6;++k) s[96+b*6+k]=0.0f;
            if(data.articulation_response_dof_count.data[art]>0&&(data.body_flags.data[body]&2)!=0) valid=false;
        }
        for(int k=0;k<6;++k) valid=valid&&isfinite(s[96+b*6+k]);
        s[288+b]=valid?1.0f:0.0f;
    }
#if defined(__CUDA_ARCH__)
    __syncwarp(0xffffffffu);
#endif
"""


def get_source(*, produce_twists):
    """Validate all 32 endpoints, preserving free-speed and prescribed rules."""
    checked_definitions("body_validation.py", ("_PRESCRIBED", "_FREE", "_VALIDATION", "get_source"))
    return Template(_VALIDATION).substitute(
        prescribed=_PRESCRIBED if produce_twists else "/* prescribed twist already produced */",
        free=_FREE if produce_twists else "",
    )
