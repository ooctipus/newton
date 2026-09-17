# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental coupled Krylov/chord cost screen; finite-budget tails remain.

Keep original current-row production, held whitening, cold admission, original
24-correction continuation and publication. The active linear solve is warp
local; all-row physical work retains the original 32/64-thread owner.
"""

import ast
import functools
import hashlib
import inspect
import linecache
import textwrap
from pathlib import Path

from . import coupled_contact, krylov_linear

SOURCE_PINS = {
    coupled_contact.__file__: "3e972df7b7e2659716718595a49164a3a7ed9f94c2d874c0e29855dca3384015",
    krylov_linear.__file__: "d68534d5a8abe82901f63707d3d27a661296aec5344dcf75fa3b1debab964249",
}

_NATIVE = (
    krylov_linear.NATIVE
    + r"""
    __shared__ float kc_q[AM],kc_pg[AM],kc_aux[AM];
    __shared__ int kc_ids[18],kc_old_ids[18],kc_map[AM];
    __shared__ int kc_count,kc_old_count,kc_rebuild;
    int kc_consumed=0;
    bool kc_done=false;
    const bool kc_row=lane<n_rows;
    float kc_diagonal=0.0f,kc_spectral=0.0f,kc_eta=0.0f;
    auto kc_any = [&](int value) {
        if(NT==32)return __any_sync(MASK,value);
        return __syncthreads_or(value);
    };
    auto kc_sum = [&](float value) {
        for(int shift=16;shift>0;shift>>=1)value+=__shfl_xor_sync(MASK,value,shift);
        if(NT>32){
            if((lane&31)==0)s_dot[lane>>5]=value;
            SYNC();value=s_dot[0]+s_dot[1];SYNC();
        }
        return value;
    };
    auto kc_max = [&](float value) {
        for(int shift=16;shift>0;shift>>=1)value=fmaxf(value,__shfl_xor_sync(MASK,value,shift));
        if(NT>32){
            if((lane&31)==0)s_dot[lane>>5]=value;
            SYNC();value=fmaxf(s_dot[0],s_dot[1]);SYNC();
        }
        return value;
    };
    auto kc_action = [&](const float* point,bool with_rhs) {
        constexpr int NCH=NT/18;
        const int n4=(n_rows+3)>>2,per=(n4+NCH-1)/NCH;
        if(lane<NCH*18){
            const int d=lane%18,ch=lane/18;
            const float4* x4=reinterpret_cast<const float4*>(point);
            const float4* z4=reinterpret_cast<const float4*>(&s_Yt[d*YS]);
            float a0=0.0f,a1=0.0f,a2=0.0f,a3=0.0f;
            for(int j=ch*per;j<min(n4,(ch+1)*per);++j){
                const float4 x=x4[j],z=z4[j];
                a0+=x.x*z.x;a1+=x.y*z.y;a2+=x.z*z.z;a3+=x.w*z.w;
            }
            if(NCH==1)s_dv[d]=(a0+a1)+(a2+a3);
            else s_dvp[ch*18+d]=(a0+a1)+(a2+a3);
        }
        SYNC();
        if(NCH>1){
            if(lane<18){float value=0.0f;for(int ch=0;ch<NCH;++ch)value+=s_dvp[ch*18+lane];s_dv[lane]=value;}
            SYNC();
        }
        float value=kc_row&&with_rhs?s_rhs[lane]:0.0f;
        if(kc_row){for(int d=0;d<18;++d)value+=Jr[d]*s_dv[d];}
        SYNC();
        return value;
    };
    auto kc_project = [&](float* point) {
        float value=kc_row?point[lane]:0.0f;
        if(kc_row&&s_kind[lane]==0)value=fmaxf(value,0.0f);
        if(kc_row)point[lane]=value;
        SYNC();
        if(kc_row&&s_kind[lane]==1){
            const int p=s_parent[lane],sibling=lane==p+1?p+2:p+1;
            const float radius=fmaxf(0.0f,s_mu[lane]*point[p]);
            const float other=point[sibling],length=hypotf(value,other);
            if(radius<=0.0f)value=0.0f;
            else if(length>radius)value*=radius/length;
        }
        SYNC();
        if(kc_row)point[lane]=value;
        SYNC();
    };
    auto kc_map_value = [&](const float* point,float residual) {
        if(kc_row)kc_q[lane]=point[lane]-kc_eta*residual;
        SYNC();
        float value=kc_row?fmaxf(kc_q[lane],0.0f):0.0f;
        if(kc_row&&s_kind[lane]==1){
            const int p=s_parent[lane],sibling=lane==p+1?p+2:p+1;
            const float radius=s_mu[lane]*fmaxf(kc_q[p],0.0f);
            const float q=kc_q[lane],other=kc_q[sibling],length=hypotf(q,other);
            value=q;
            if(radius<=0.0f)value=0.0f;
            else if(length>radius)value*=radius/length;
        }
        if(kc_row)kc_pg[lane]=value;
        SYNC();
        if(kc_row&&(!isfinite(point[lane])||!isfinite(residual)||!isfinite(kc_q[lane])||!isfinite(value)))
            return __int_as_float(0x7fffffff);
        return kc_row?(point[lane]-value)/kc_eta:0.0f;
    };
    auto kc_physical = [&](const float* point,float residual) {
        const float scale=fmaxf(1.0f,kc_max(kc_row?fmaxf(fabsf(s_rhs[lane]),fabsf(residual-s_rhs[lane])):0.0f));
        bool bad=false;
        if(kc_row){
            const float x=point[lane];
            bad=!isfinite(x)||!isfinite(residual);
            if(s_kind[lane]==0){
                const float defect=kc_diagonal*(x-fmaxf(0.0f,x-residual/kc_diagonal));
                bad|=!isfinite(defect)||fabsf(defect)>=3.0e-5f*scale||-residual>=3.0e-5f||
                    fabsf(x*residual)>=3.0e-5f||-x>=1.0e-7f;
            }
            kc_aux[lane]=residual;
        }
        SYNC();
        if(kc_row&&s_kind[lane]==1&&lane==s_parent[lane]+1){
            const int p=s_parent[lane];
            const float x=point[lane],y=point[lane+1],r=residual,t=kc_aux[lane+1];
            const float radius=fmaxf(0.0f,s_mu[lane]*point[p]);
            float qx=x-r/kc_spectral,qy=y-t/kc_spectral;
            const float length=hypotf(qx,qy);
            if(radius<=0.0f){qx=0.0f;qy=0.0f;}
            else if(length>radius){const float factor=radius/length;qx*=factor;qy*=factor;}
            const float defect=kc_spectral*fmaxf(fabsf(x-qx),fabsf(y-qy));
            const float mdp=x*r+y*t+radius*hypotf(r,t),cone=hypotf(x,y)-radius;
            bad|=!isfinite(defect)||!isfinite(mdp)||!isfinite(cone)||
                defect>=3.0e-5f*scale||mdp>=3.0e-5f||cone>=3.0e-5f;
        }
        return !kc_any(bad);
    };

    int kc_bad=row_phase!=0||freeze_drive_rows!=0||friction_start_iteration!=0||
        iteration_offset!=0||regularize!=0||omega!=1.0f;
    if(kc_row){
        kc_bad|=!isfinite(s_lam0[lane])||s_lam0[lane]!=0.0f||!isfinite(s_rhs[lane])||
            s_kind[lane]<0||s_kind[lane]>1;
        for(int d=0;d<18;++d){kc_diagonal+=Jr[d]*Jr[d];kc_bad|=!isfinite(Jr[d]);}
        const float cfm=s_row_c[lane]>=0?wr_pgs_cfm:world_row_cfm.data[off_dense+lane];
        kc_bad|=!isfinite(cfm)||!(kc_diagonal>0.0f)||!isfinite(kc_diagonal)||!(kc_diagonal+cfm>0.0f);
        kc_aux[lane]=kc_diagonal;
        kc_eta=1.0f/(kc_diagonal+cfm);
        if(s_row_c[lane]>=0){
            const int c=s_row_c[lane]>>2,a=contact_art_a.data[c],b=contact_art_b.data[c];
            kc_bad|=a>=0&&a==b;
        }
        if(s_kind[lane]==1){
            const int p=s_parent[lane];
            const bool pair=p>=0&&p+2<n_rows&&(lane==p+1||lane==p+2);
            kc_bad|=!pair;
            if(pair)kc_bad|=s_kind[p]!=0||s_kind[p+1]!=1||s_kind[p+2]!=1||
                s_parent[p+1]!=p||s_parent[p+2]!=p||!isfinite(s_mu[lane])||
                s_mu[lane]<0.0f||s_mu[p+1]!=s_mu[p+2];
        }
    }
    kc_bad=kc_any(kc_bad);
    SYNC();
    if(!kc_bad){
        if(kc_row&&s_kind[lane]==1&&lane==s_parent[lane]+1){
            float cross=0.0f;
            for(int d=0;d<18;++d)cross+=s_Yt[d*YS+lane]*s_Yt[d*YS+lane+1];
            const float a=kc_aux[lane],c=kc_aux[lane+1];
            const float spectral=0.5f*(a+c)+hypotf(0.5f*(a-c),cross);
            const float c0=s_row_c[lane]>=0?wr_pgs_cfm:world_row_cfm.data[off_dense+lane];
            const float c1=s_row_c[lane+1]>=0?wr_pgs_cfm:world_row_cfm.data[off_dense+lane+1];
            const float denominator=spectral+fmaxf(c0,c1);
            kc_bad|=!(spectral>0.0f)||!isfinite(spectral)||!(denominator>0.0f)||!isfinite(denominator);
            kc_q[lane]=kc_q[lane+1]=spectral;
            kc_pg[lane]=kc_pg[lane+1]=1.0f/denominator;
        }
        kc_bad=kc_any(kc_bad);
        SYNC();
        if(kc_row&&s_kind[lane]==1){kc_spectral=kc_q[lane];kc_eta=kc_pg[lane];}
        kc_bad=kc_any(kc_bad||(kc_row&&(!(kc_eta>0.0f)||!isfinite(kc_eta))));
    }
    if(!kc_bad){
        if(lane==0)kc_old_count=-1;
        if(kc_row)s_step[lane]=kc_eta;
        for(int j=n_rows+lane;j<((n_rows+3)&~3);j+=NT){s_x[j]=0.0f;s_y[j]=0.0f;}
        SYNC();
        float kc_residual=kc_row?s_rhs[lane]:0.0f;
        float kc_f=kc_map_value(s_x,kc_residual);
        const float kc_initial_norm=sqrtf(kc_sum(kc_f*kc_f));
        bool kc_stopped=kc_physical(s_x,kc_residual);
        while(kc_consumed<24){
            if(kc_stopped){
                kc_residual=kc_action(s_x,true);
                kc_f=kc_map_value(s_x,kc_residual);
                kc_stopped=kc_physical(s_x,kc_residual);
                if(kc_stopped){kc_done=true;break;}
            }
            const float kc_old_norm=sqrtf(kc_sum(kc_f*kc_f));
            if(!(kc_old_norm>0.0f)||!isfinite(kc_old_norm)||!(kc_initial_norm>0.0f))break;
            const float kc_projected=kc_row?kc_pg[lane]:0.0f;
            float p0=0.0f,p1=0.0f,p2=0.0f;
            int n0=lane,n1=-1,n2=-1;
            if(kc_row&&s_kind[lane]==0)p0=kc_q[lane]>0.0f?1.0f:0.0f;
            if(kc_row&&s_kind[lane]==1){
                n0=s_parent[lane];n1=n0+1;n2=n0+2;
                const float radius=s_mu[lane]*fmaxf(kc_q[n0],0.0f);
                const float q0=kc_q[n1],q1=kc_q[n2],length=hypotf(q0,q1);
                if(radius>0.0f){
                    p1=lane==n1?1.0f:0.0f;p2=lane==n2?1.0f:0.0f;
                    if(length>radius){
                        const float u0=q0/length,u1=q1/length,ui=lane==n1?u0:u1,ratio=radius/length;
                        p1=ratio*(p1-ui*u0);p2=ratio*(p2-ui*u1);
                        p0=s_mu[lane]*ui*(kc_q[n0]>0.0f?1.0f:0.0f);
                    }
                }
            }
            const bool active=kc_row&&(p0!=0.0f||p1!=0.0f||p2!=0.0f);
            if(kc_row)kc_map[lane]=active?1:0;
            SYNC();
            if(lane==0){
                int count=0;
                for(int i=0;i<n_rows;++i){
                    if(s_kind[i]!=0)continue;
                    const bool triple=i+2<n_rows&&s_kind[i+1]==1&&s_parent[i+1]==i&&s_parent[i+2]==i;
                    const int group_start=count,group_size=triple?3:1;
                    for(int j=0;j<group_size;++j){
                        const int row=i+j;const bool selected=kc_map[row]!=0;kc_map[row]=-1;
                        if(selected){if(count<18)kc_ids[count]=row;kc_map[row]=count++;}
                    }
                    for(int a=group_start;a<count&&a<18;++a){
                        kc_linear.block_start[a]=group_start;kc_linear.block_size[a]=count-group_start;
                    }
                }
                kc_count=count;kc_rebuild=count!=kc_old_count;
                if(count<=18){
                    if(!kc_rebuild)for(int a=0;a<count;++a)kc_rebuild|=kc_ids[a]!=kc_old_ids[a];
                    if(kc_rebuild)for(int a=0;a<count;++a)kc_old_ids[a]=kc_ids[a];
                    kc_old_count=count;
                }
            }
            SYNC();
            if(kc_count>18)break;
            const int k=kc_count;
            if(kc_rebuild){
                for(int e=lane;e<k*(k+1)/2;e+=NT){
                    const int a=static_cast<int>((sqrtf(static_cast<float>(8*e+1))-1.0f)*0.5f),b=e-a*(a+1)/2;
                    float value=0.0f;
                    for(int d=0;d<18;++d)value+=s_Yt[d*YS+kc_ids[a]]*s_Yt[d*YS+kc_ids[b]];
                    kc_linear.gram[e]=value;
                }
            }
            if(kc_row)s_y[lane]=active?0.0f:kc_projected-s_x[lane];
            SYNC();
            float shift=0.0f;
            if(kc_any(kc_row&&s_y[lane]!=0.0f))shift=kc_action(s_y,false);
            if(kc_row)kc_aux[lane]=shift;
            SYNC();
            int linear_bad=0;
            if(active){
                const int a=kc_map[lane];
                const int columns[3]={n0,n1,n2};const float coeff[3]={p0,p1,p2};
                float projected_shift=0.0f;
                for(int j=0;j<3;++j){
                    const int row=columns[j];const float value=coeff[j];
                    const int column=value!=0.0f&&row>=0?kc_map[row]:-1;
                    linear_bad|=value!=0.0f&&(column<0||column>=k);
                    kc_linear.columns[3*a+j]=column;kc_linear.projection[3*a+j]=value;
                    if(value!=0.0f&&row>=0)projected_shift+=value*(s_y[row]-s_step[row]*kc_aux[row]);
                }
                kc_linear.eta[a]=kc_eta;
                kc_linear.rhs[a]=-kc_f-(s_y[lane]-projected_shift)/kc_eta;
            }
            if(kc_any(linear_bad))break;
            const float target=fminf(0.5f,sqrtf(kc_old_norm/kc_initial_norm))*kc_old_norm;
            SYNC();
            if(lane<32)kc_solve_warp(kc_linear,k,target,lane&31);
            SYNC();
            if(!kc_linear.ok)break;
            if(kc_row){
                const float delta=active?kc_linear.delta[kc_map[lane]]:kc_projected-s_x[lane];
                s_y[lane]=s_x[lane]+delta;
            }
            SYNC();
            kc_project(s_y);
            const float chord=kc_row?s_y[lane]-s_x[lane]:0.0f;
            if(kc_any(kc_row&&!isfinite(chord))||!kc_any(kc_row&&chord!=0.0f))break;
            if(kc_row)s_y[lane]=chord;
            SYNC();
            float direction=chord,response=kc_action(s_y,false);
            bool accepted=false,finite_newton=true;
            float accepted_residual=kc_residual,accepted_f=kc_f;
            for(int family=0;family<2&&!accepted;++family){
                if(family==1){
                    if(!finite_newton)break;
                    direction=kc_row?kc_projected - s_x[lane]:0.0f;
                    if(kc_row)s_y[lane]=direction;
                    SYNC();response=kc_action(s_y,false);
                }
                for(int trial=0;trial<8;++trial){
                    const float alpha=ldexpf(1.0f,-trial);
                    if(kc_row)s_y[lane]=s_x[lane]+alpha*direction;
                    const float trial_residual=kc_residual+alpha*response;
                    SYNC();
                    const float trial_f=kc_map_value(s_y,trial_residual);
                    const float merit=sqrtf(kc_sum(trial_f*trial_f));
                    if(family==0&&!isfinite(merit))finite_newton=false;
                    if(isfinite(merit)&&merit<(1.0f-1.0e-4f*alpha)*kc_old_norm){
                        accepted=true;accepted_residual=trial_residual;accepted_f=trial_f;break;
                    }
                }
            }
            if(!accepted)break;
            if(kc_row)s_x[lane]=s_y[lane];
            kc_residual=accepted_residual;kc_f=accepted_f;++kc_consumed;
            SYNC();
            kc_stopped=kc_physical(s_x,kc_residual);
        }
        if(kc_consumed==24){
            kc_residual=kc_action(s_x,true);
            kc_f=kc_map_value(s_x,kc_residual);
            kc_stopped=kc_physical(s_x,kc_residual);
            kc_done=true;
        }
        if(!kc_done&&kc_row)s_y[lane]=s_x[lane];
        SYNC();
    }
"""
)


def _rewrite_native(source):
    """Replace admitted solve work without changing original stage or decode."""
    start = "    // b'_i = rhs_i + J_i . (v_in - Y^T lam0)"
    source = coupled_contact._replace_once(source, start, _NATIVE + "\n    if (!kc_done) {\n" + start)
    end = "    SYNC();\n#else\n#if 1\n    // In-kernel response in whitened coordinates:"
    source = coupled_contact._replace_once(
        source, end, "    SYNC();\n    }\n#else\n#if 1\n    // In-kernel response in whitened coordinates:"
    )
    source = coupled_contact._replace_once(
        source, "    float t_k = 1.0f;", "    if (!kc_done) {\n    float t_k = 1.0f;"
    )
    source = coupled_contact._replace_once(source, "sweep < 24;", "sweep < 24 - kc_consumed;")
    source = coupled_contact._replace_once(
        source,
        "const int global_iter = iteration_offset + sweep;",
        "const int global_iter = iteration_offset + kc_consumed + sweep;",
    )
    return coupled_contact._replace_once(
        source, "\n#if 1\n    // u = Z^T (x - lam0)", "\n    }\n#if 1\n    // u = Z^T (x - lam0)"
    )


@functools.cache
def get_parallel_factory(original):
    """Return the distinct default-off original-ABI experimental owner."""
    for name, digest in SOURCE_PINS.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Frozen Krylov native dependency changed")
    selected = coupled_contact.get_parallel_factory(original)
    successor = inspect.getclosurevars(selected).nonlocals["successor"]
    tree = ast.parse(textwrap.dedent(inspect.getsource(successor)))
    factory = tree.body[0]
    factory.name = "krylov_chord_parallel_factory"
    for node in ast.walk(factory):
        if isinstance(node, ast.Constant) and node.value == "_cc6":
            node.value = "_kc24"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_fpgs_coupled_contact"):
            node.attr = node.attr.replace("_fpgs_coupled_contact", "_fpgs_krylov_chord")
    ast.fix_missing_locations(tree)
    generated = ast.unparse(tree) + "\n"
    filename = f"<krylov-chord-{hashlib.sha256(generated.encode()).hexdigest()}>"
    linecache.cache[filename] = (len(generated), None, generated.splitlines(keepends=True), filename)
    namespace = dict(successor.__globals__, _rewrite_native=_rewrite_native)
    exec(compile(generated, filename, "exec"), namespace)
    implementation = namespace[factory.name]
    signature = inspect.signature(original)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if not coupled_contact._supported(bound.arguments, original.__globals__):
            return original(*args, **kwargs)
        return implementation(*args, **kwargs)

    return wrapped
