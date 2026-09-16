# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental coupled-contact prefix with the original remaining-budget tail.

The opt-in solver hook selects this factory; unsupported configurations retain
the original factory. No global scratch, new launch, or persistent row cache is
introduced. Local CFM remains a proximal impulse-change term, not compliance.
"""

import ast
import functools
import hashlib
import inspect
import linecache
import os
import textwrap

import warp as wp

_ORIGINAL_FACTORY_SHA = "2d195e7d37f5b9e127a75d2dd2be6185f78faac4f2d5bd29f1603f03d9745a48"
BlockResult = wp.types.vector(length=5, dtype=wp.float32)

# The analytic bracket, sixteen-step allowance, relative root tolerance,
# Schur floor and coupled repair follow CPU control 0fc44df2. The native law
# guard uses the Frobenius norm (a proven upper bound for its spectral norm),
# a disclosed numerical realization difference, not a physical gate change.
_LOCAL_NATIVE = r"""
auto cc_norm2 = [](float x, float y) {
    const float scale=wp::max(fabsf(x),fabsf(y));
    if(scale==0.0f)return 0.0f;
    x/=scale;y/=scale;return scale*sqrtf(x*x+y*y);
};
auto cc_prepare = [&](const wp::mat_t<3,3,float>& H) -> wp::vec4 {
    const float a=H.data[0][0], h0=H.data[0][1], h1=H.data[0][2];
    if (!(a>0.0f) || !isfinite(a)) return wp::vec4(0.0f);
    const float q0=H.data[1][1]-h0*h0/a;
    const float q1=H.data[1][2]-h0*h1/a;
    const float q2=H.data[2][2]-h1*h1/a;
    const float low=0.5f*(q0+q2)-cc_norm2(0.5f*(q0-q2),q1);
    const float scale=wp::max(wp::max(a,H.data[1][1]),wp::max(H.data[2][2],1.0e-30f));
    if (!isfinite(low) || !(low>1.0e-7f*scale)) return wp::vec4(0.0f);
    return wp::vec4(q0,q1,q2,low);
};
auto cc_block = [&](const wp::mat_t<3,3,float>& H, const wp::vec3& b,
                   float mu, const wp::vec4& Q) -> wp::vec_t<5,float> {
    wp::vec_t<5,float> out(0.0f);
    if (!isfinite(mu) || mu<0.0f) return out;
    for (int k=0;k<3;++k) if (!isfinite(b[k])) return out;
    if (b[0]>=0.0f) {out[3]=1.0f;return out;}
    const float a=H.data[0][0], h0=H.data[0][1], h1=H.data[0][2];
    if (!(a>0.0f) || !isfinite(a)) return out;
    if (mu==0.0f) {out[0]=-b[0]/a;out[3]=isfinite(out[0])?1.0f:0.0f;return out;}
    if (!(Q[3]>0.0f)) return out;
    const float p0=b[1]-h0*b[0]/a, p1=b[2]-h1*b[0]/a;
    auto eval = [&](float gamma, float& n, float& x, float& y, float& g) {
        const float aa=Q[0]+gamma, bb=Q[1], dd=Q[2]+gamma;
        const float det=aa*dd-bb*bb;
        if (!(det>0.0f) || !isfinite(det)) return false;
        x=(-dd*p0+bb*p1)/det;y=(bb*p0-aa*p1)/det;
        n=(-b[0]-h0*x-h1*y)/a;g=cc_norm2(x,y)-mu*n;
        return isfinite(n)&&isfinite(x)&&isfinite(y)&&isfinite(g);
    };
    float n,x,y,g;
    if (!eval(0.0f,n,x,y,g)) return out;
    if (n>=0.0f && g<=0.0f) {
        out[0]=n;out[1]=x;out[2]=y;out[3]=1.0f;return out;
    }
    const float rmin=mu*(-b[0])/(a+mu*cc_norm2(h0,h1));
    float lower=0.0f,upper=wp::max(0.0f,cc_norm2(p0,p1)/rmin-Q[3]);
    if (!(upper>0.0f) || !isfinite(upper)) return out;
    float un,ux,uy,ug;
    if (!eval(upper,un,ux,uy,ug) || ug>1.0e-12f*wp::max(rmin,1.0f)) return out;
    float gamma=0.0f;
    for (int probe=0;probe<16;++probe) {
        const float radius=mu*n,length=cc_norm2(x,y);
        if (n>=0.0f && radius>0.0f && fabsf(g/radius)<=2.0e-6f) {
            if (g>0.0f) {
                const float repair=mu*(-b[0])/(a*length+mu*(h0*x+h1*y));
                x*=repair;y*=repair;n=(-b[0]-h0*x-h1*y)/a;
            }
            const float e0=a*n+h0*x+h1*y+b[0];
            const float e1=h0*n+H.data[1][1]*x+H.data[1][2]*y+b[1]+gamma*x;
            const float e2=h1*n+H.data[1][2]*x+H.data[2][2]*y+b[2]+gamma*y;
            float hn=0.0f;for(int i=0;i<3;++i)for(int j=0;j<3;++j)hn+=H.data[i][j]*H.data[i][j];
            const float norm=cc_norm2(n,cc_norm2(x,y));
            const float bound=8.0e-6f*(1.0f+cc_norm2(b[0],cc_norm2(b[1],b[2]))+(sqrtf(hn)+gamma)*norm);
            if (isfinite(n)&&isfinite(x)&&isfinite(y)&&isfinite(bound)&&n>=0.0f&&
                cc_norm2(x,y)<=mu*n*(1.0f+4.0e-7f)&&cc_norm2(e0,cc_norm2(e1,e2))<=bound) {
                out[0]=n;out[1]=x;out[2]=y;out[3]=1.0f;out[4]=gamma;
            }
            return out;
        }
        if(g>0.0f)lower=gamma;else upper=gamma;
        const float aa=Q[0]+gamma,bb=Q[1],dd=Q[2]+gamma,det=aa*dd-bb*bb;
        const float dx=(-dd*x+bb*y)/det,dy=(bb*x-aa*y)/det;
        const float dn=-(h0*dx+h1*dy)/a;
        const float derivative=(x*dx+y*dy)/wp::max(length,1.0e-30f)-mu*dn;
        float proposal=gamma-g/derivative;
        if(!isfinite(proposal)||proposal<=lower||proposal>=upper)proposal=0.5f*(lower+upper);
        gamma=proposal;if(!eval(gamma,n,x,y,g))return out;
    }
    return out;
};
"""


@wp.func_native(_LOCAL_NATIVE + "\nreturn cc_block(matrix, bias, friction, cc_prepare(matrix));\n")
def project_contact_block(matrix: wp.mat33, bias: wp.vec3, friction: float) -> BlockResult:
    """Return impulse, acceptance status and multiplier for CPU/device controls."""


_METRIC_NATIVE = r"""
auto cc_metric = [](const wp::mat_t<3,3,float>& H, const wp::vec3& residual,
                    const wp::vec3& old, float friction) -> wp::vec4 {
    wp::vec4 result(0.0f);
    const float normal=fmaxf(old[0]-residual[0]/H.data[0][0],0.0f);
    const float radius=friction*normal;
    if(!isfinite(normal)||!isfinite(radius))return result;
    if(radius==0.0f){result[0]=normal;result[3]=1.0f;return result;}
    const float a=H.data[1][1],b=H.data[1][2],c=H.data[2][2];
    const float det=a*c-b*b,largest=0.5f*(a+c)+hypotf(0.5f*(a-c),b),smallest=det/largest;
    const float r1=residual[1]+H.data[1][0]*(normal-old[0]);
    const float r2=residual[2]+H.data[2][0]*(normal-old[0]);
    const float p1=r1-a*old[1]-b*old[2],p2=r2-b*old[1]-c*old[2];
    if(!(a>0.0f)||!(c>0.0f)||!isfinite(p1)||!isfinite(p2)||!isfinite(smallest)||
       !(smallest>1.0e-7f*fmaxf(a,c)))return result;
    float x=(-c*p1+b*p2)/det,y=(b*p1-a*p2)/det,alpha=0.0f;
    float magnitude=hypotf(x,y),pnorm=hypotf(p1,p2);
    bool ok=isfinite(magnitude)&&magnitude<=radius;
    if(isfinite(magnitude)&&magnitude>radius&&isfinite(pnorm)){
        float lower=fmaxf(pnorm/radius-largest,0.0f),upper=fmaxf(pnorm/radius-smallest,0.0f);
        alpha=lower;
        if(isfinite(lower)&&isfinite(upper))for(int probe=0;probe<16;++probe){
            const float aa=a+alpha,cc=c+alpha,dd=aa*cc-b*b;
            x=(-cc*p1+b*p2)/dd;y=(b*p1-aa*p2)/dd;magnitude=hypotf(x,y);
            if(!isfinite(dd)||!(dd>0.0f)||!isfinite(magnitude)||!(magnitude>0.0f))break;
            if(fabsf(magnitude/radius-1.0f)<=2.0e-6f){
                const float repair=fminf(1.0f,radius/magnitude);x*=repair;y*=repair;ok=true;break;
            }
            if(magnitude>radius)lower=fmaxf(lower,alpha);else upper=fminf(upper,alpha);
            const float denominator=(cc*x*x-2.0f*b*x*y+aa*y*y)/dd;
            if(!isfinite(denominator)||!(denominator>0.0f))break;
            float proposal=alpha+magnitude*magnitude*(magnitude/radius-1.0f)/denominator;
            if(!isfinite(proposal)||proposal<=lower||proposal>upper)proposal=0.5f*(lower+upper);
            alpha=proposal;
        }
    }
    const float error=hypotf((a+alpha)*x+b*y+p1,b*x+(c+alpha)*y+p2);
    const float bound=8.0e-6f*(1.0f+pnorm+(largest+alpha)*hypotf(x,y));
    if(ok&&isfinite(x)&&isfinite(y)&&isfinite(error)&&isfinite(bound)&&error<=bound&&
       hypotf(x,y)<=radius*(1.0f+4.0e-7f)){
        result[0]=normal;result[1]=x;result[2]=y;result[3]=1.0f;
    }
    return result;
};
"""


_NATIVE = r"""
    __shared__ float cc_diag[AM],cc_physical_diag[AM],cc_cross[AM];
    __shared__ float cc_schur[AM*4];
    __shared__ int cc_changed;
    int cc_consumed=0;
    bool cc_done=false;
    auto cc_any = [&](int value) {
        if(NT==32)return __any_sync(MASK,value);
        return __syncthreads_or(value);
    };
    auto cc_max = [&](float value) {
        for(int shift=16;shift>0;shift>>=1)value=fmaxf(value,__shfl_xor_sync(MASK,value,shift));
        if(NT>32){
            if((lane&31)==0)s_dot[lane>>5]=value;
            SYNC();value=fmaxf(s_dot[0],s_dot[1]);SYNC();
        }
        return value;
    };
    int cc_bad=row_phase!=0||freeze_drive_rows!=0||friction_start_iteration!=0||
        iteration_offset!=0||regularize!=0||omega!=1.0f;
    if(lane<n_rows){
        cc_bad|=!isfinite(s_lam0[lane])||s_lam0[lane]!=0.0f||!isfinite(s_rhs[lane])||
            s_kind[lane]<0||s_kind[lane]>1;
        float norm=0.0f;
        for(int d=0;d<18;++d){norm+=Jr[d]*Jr[d];cc_bad|=!isfinite(Jr[d]);}
        const float cfm=s_row_c[lane]>=0?wr_pgs_cfm:world_row_cfm.data[off_dense+lane];
        cc_diag[lane]=norm+cfm;cc_physical_diag[lane]=norm;
        cc_bad|=!isfinite(cfm)||!isfinite(cc_diag[lane])||!(cc_diag[lane]>0.0f);
        if(s_kind[lane]==1){
            const int p=s_parent[lane];
            if(p<0||p+2>=n_rows||lane<p+1||lane>p+2)cc_bad=1;
            else cc_bad|=s_kind[p]!=0||s_kind[p+1]!=1||s_kind[p+2]!=1||
                s_parent[p+1]!=p||s_parent[p+2]!=p||!isfinite(s_mu[p+1])||
                s_mu[p+1]<0.0f||s_mu[p+1]!=s_mu[p+2];
        }
    }
    cc_bad=cc_any(cc_bad);
    SYNC();
    if(!cc_bad){
        if(lane<n_rows&&s_kind[lane]==0&&lane+2<n_rows&&s_kind[lane+1]==1&&
           s_parent[lane+1]==lane&&s_parent[lane+2]==lane){
            float a01=0.0f,a02=0.0f,a12=0.0f;
            for(int d=0;d<18;++d){
                const float z1=ZAT(d,lane+1),z2=ZAT(d,lane+2);
                a01+=Jr[d]*z1;a02+=Jr[d]*z2;a12+=z1*z2;
            }
            cc_cross[lane]=a01;cc_cross[lane+1]=a02;cc_cross[lane+2]=a12;
            wp::mat_t<3,3,float> H(0.0f);
            H.data[0][0]=cc_diag[lane];H.data[1][1]=cc_diag[lane+1];H.data[2][2]=cc_diag[lane+2];
            H.data[0][1]=H.data[1][0]=a01;H.data[0][2]=H.data[2][0]=a02;
            H.data[1][2]=H.data[2][1]=a12;
            const wp::vec4 prepared=cc_prepare(H);
            for(int k=0;k<4;++k)cc_schur[lane*4+k]=prepared[k];
        }
        const float cold=lane<n_rows?fabsf(s_rhs[lane])/sqrtf(cc_diag[lane]):0.0f;
        const float cc_scale=1.0f+cc_max(cold);
        if(lane<18)s_dv[lane]=0.0f;
        SYNC();
        while(cc_consumed < 6){
            if(lane==0)cc_changed=0;
            SYNC();
            if(lane<32){
                int row=0;
                while(row<n_rows){
                    const bool triple=s_kind[row]==0&&row+2<n_rows&&s_kind[row+1]==1&&
                        s_parent[row+1]==row&&s_parent[row+2]==row;
                    const float z0=lane<18?ZAT(lane,row):0.0f;
                    const float z1=lane<18&&triple?ZAT(lane,row+1):0.0f;
                    const float z2=lane<18&&triple?ZAT(lane,row+2):0.0f;
                    const float du=lane<18?s_dv[lane]:0.0f;
                    float r0=z0*du,r1=z1*du,r2=z2*du;
                    for(int shift=16;shift>0;shift>>=1){
                        r0+=__shfl_down_sync(MASK,r0,shift);
                        r1+=__shfl_down_sync(MASK,r1,shift);
                        r2+=__shfl_down_sync(MASK,r2,shift);
                    }
                    float d0=0.0f,d1=0.0f,d2=0.0f;
                    if(lane==0){
                        r0+=s_rhs[row];
                        if(triple){
                            r1+=s_rhs[row+1];r2+=s_rhs[row+2];
                            const wp::vec3 old(s_x[row],s_x[row+1],s_x[row+2]);
                            const wp::vec3 residual(r0,r1,r2);
                            wp::mat_t<3,3,float> H(0.0f);
                            H.data[0][0]=cc_diag[row];H.data[1][1]=cc_diag[row+1];H.data[2][2]=cc_diag[row+2];
                            H.data[0][1]=H.data[1][0]=cc_cross[row];
                            H.data[0][2]=H.data[2][0]=cc_cross[row+1];
                            H.data[1][2]=H.data[2][1]=cc_cross[row+2];
                            wp::vec3 bias=residual;
                            for(int i=0;i<3;++i)for(int j=0;j<3;++j)bias[i]-=H.data[i][j]*old[j];
                            const wp::vec4 prepared(cc_schur[row*4],cc_schur[row*4+1],cc_schur[row*4+2],cc_schur[row*4+3]);
                            wp::vec_t<5,float> trial=cc_block(H,bias,s_mu[row+1],prepared);
                            if(trial[3]==0.0f){
                                const wp::vec4 metric=cc_metric(H,residual,old,s_mu[row+1]);
                                for(int k=0;k<4;++k)trial[k]=metric[k];
                            }
                            if(trial[3]==0.0f){
                                // Original scalar triplet after metric rejection.
                                wp::vec3 value=old;
                                for(int k=0;k<3;++k){
                                    float r=residual[k];
                                    for(int j=0;j<3;++j){
                                        const float g=k==j?cc_physical_diag[row+k]:H.data[k][j];
                                        r+=g*(value[j]-old[j]);
                                    }
                                    float next=value[k]-r/cc_diag[row+k];
                                    if(k==0)next=fmaxf(next,0.0f);
                                    else{
                                        const int sibling=k==1?2:1;
                                        const float radius=fmaxf(s_mu[row+k]*value[0],0.0f);
                                        const float length=hypotf(next,value[sibling]);
                                        if(radius<=0.0f)next=0.0f;
                                        else if(length>radius){const float scale=radius/length;next*=scale;value[sibling]*=scale;}
                                    }
                                    value[k]=next;
                                }
                                for(int k=0;k<3;++k)trial[k]=value[k];
                            }
                            d0=trial[0]-old[0];d1=trial[1]-old[1];d2=trial[2]-old[2];
                            s_x[row]=trial[0];s_x[row+1]=trial[1];s_x[row+2]=trial[2];
                        }else{
                            const float old=s_x[row],next=fmaxf(old-r0/cc_diag[row],0.0f);
                            s_x[row]=next;d0=next-old;
                        }
                        if(d0!=0.0f||d1!=0.0f||d2!=0.0f)cc_changed=1;
                    }
                    d0=__shfl_sync(MASK,d0,0);d1=__shfl_sync(MASK,d1,0);d2=__shfl_sync(MASK,d2,0);
                    if(lane<18)s_dv[lane]+=z0*d0+z1*d1+z2*d2;
                    __syncwarp();row+=triple?3:1;
                }
            }
            SYNC();
            ++cc_consumed;
            if(lane<n_rows){
                float r=s_rhs[lane];for(int d=0;d<18;++d)r+=Jr[d]*s_dv[d];s_y[lane]=r;
            }
            SYNC();
            int failed=0;
            if(lane<n_rows){
                const float r=s_y[lane],value=s_x[lane];
                failed=!isfinite(r)||!isfinite(value);
                if(s_kind[lane]==0){
                    const float correction=value-fmaxf(0.0f,value-r/cc_diag[lane]);
                    failed|=value< -1.0e-10f||r< -3.0e-5f||fabsf(value*r)>3.0e-5f||
                        sqrtf(cc_diag[lane])*fabsf(correction)>1.0e-5f*cc_scale;
                }else if(lane==s_parent[lane]+1){
                    const float radius=fmaxf(0.0f,s_mu[lane]*s_x[s_parent[lane]]);
                    const float other=s_x[lane+1],rr=s_y[lane+1];
                    const float diag=fmaxf(cc_diag[lane],cc_diag[lane+1]);
                    const float x=value-r/diag,y=other-rr/diag,length=hypotf(x,y);
                    const float factor=length>radius?radius/fmaxf(length,1.0e-30f):1.0f;
                    const float mdp=fabsf(value*r+other*rr+radius*hypotf(r,rr));
                    failed|=!isfinite(mdp)||hypotf(value,other)-radius>1.0e-10f||mdp>3.0e-5f||
                        sqrtf(cc_diag[lane])*fabsf(value-x*factor)>1.0e-5f*cc_scale||
                        sqrtf(cc_diag[lane+1])*fabsf(other-y*factor)>1.0e-5f*cc_scale;
                }
            }
            cc_done=cc_any(failed)==0;
            if(cc_done||cc_changed==0)break;
        }
        if(lane<n_rows)s_y[lane] = s_x[lane];
        SYNC();
    }
"""


def _replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError("Coupled-contact original native seam changed")
    return source.replace(old, new)


def _rewrite_native(source):
    """Retain the original producer, fallback recurrence and final decode."""
    start = "    // b'_i = rhs_i + J_i . (v_in - Y^T lam0)"
    source = _replace_once(source, start, _LOCAL_NATIVE + _METRIC_NATIVE + _NATIVE + "\n    if (!cc_done) {\n" + start)
    end = "    SYNC();\n#else\n#if 1\n    // In-kernel response in whitened coordinates:"
    source = _replace_once(
        source, end, "    SYNC();\n    }\n#else\n#if 1\n    // In-kernel response in whitened coordinates:"
    )
    source = _replace_once(source, "    float t_k = 1.0f;", "    if (!cc_done) {\n    float t_k = 1.0f;")
    source = _replace_once(source, "sweep < 24;", "sweep < 24 - cc_consumed;")
    source = _replace_once(
        source,
        "const int global_iter = iteration_offset + sweep;",
        "const int global_iter = iteration_offset + cc_consumed + sweep;",
    )
    return _replace_once(source, "\n#if 1\n    // u = Z^T (x - lam0)", "\n    }\n#if 1\n    // u = Z^T (x - lam0)")


def _supported(arguments, globals_):
    return (
        arguments["max_world_dofs"] == 18
        and arguments["rows"] in (32, 48)
        and arguments["min_rows"] == (0 if arguments["rows"] == 32 else 32)
        and arguments["sweeps"] == 24
        and arguments["nesterov"]
        and arguments["matrix_free"]
        and arguments["inkernel_response"] == (18, 0, 0, 0)
        and arguments["exact_row_sums"]
        and arguments["world_rows"]
        and not any(
            arguments[name]
            for name in (
                "has_drive_rows",
                "has_dense_velocity_limit_rows",
                "skip_local_internal_worlds",
                "lean_sweep",
                "bound_finger",
            )
        )
        and not any(globals_.get(name, False) for name in ("_INK_CHECK", "_WR_CHECK", "_WR_WARM", "_SHADOW_LEAN"))
        and int(os.environ.get("FEATHER_PGS_WR_STOP", "0")) == 0
    )


@functools.cache
def get_parallel_factory(original):
    """Return an ABI-identical factory for the explicitly opted-in solve owner."""
    original_source = inspect.getsource(original)
    if hashlib.sha256(original_source.encode()).hexdigest() != _ORIGINAL_FACTORY_SHA:
        raise RuntimeError("Coupled-contact original factory changed; review the complete seam")
    tree = ast.parse(textwrap.dedent(original_source))
    factory = tree.body[0]
    factory.name = "coupled_contact_parallel_factory"
    factory.decorator_list = []
    native = next(
        i
        for i, node in enumerate(factory.body)
        if isinstance(node, ast.FunctionDef) and node.name == "pgs_solve_parallel_native"
    )
    factory.body[native:native] = ast.parse("snippet = _rewrite_native(snippet)").body
    naming = next(
        i
        for i, node in enumerate(factory.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Attribute) and target.attr == "__name__" for target in node.targets)
    )
    factory.body[naming:naming] = ast.parse("name += '_cc6'").body
    final = next(i for i, node in enumerate(factory.body) if isinstance(node, ast.Return))
    factory.body[final:final] = ast.parse(
        "kernel._fpgs_coupled_contact = True\nkernel._fpgs_coupled_contact_native = snippet"
    ).body
    ast.fix_missing_locations(tree)
    generated = ast.unparse(tree) + "\n"
    filename = f"<coupled-contact-{hashlib.sha256(generated.encode()).hexdigest()}>"
    linecache.cache[filename] = (len(generated), None, generated.splitlines(keepends=True), filename)
    namespace = dict(original.__globals__)
    namespace["_rewrite_native"] = _rewrite_native
    exec(compile(generated, filename, "exec"), namespace)
    successor = namespace[factory.name]
    signature = inspect.signature(original)

    @functools.wraps(original)
    def selected(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if not _supported(bound.arguments, original.__globals__):
            return original(*args, **kwargs)
        return successor(*args, **kwargs)

    return selected
