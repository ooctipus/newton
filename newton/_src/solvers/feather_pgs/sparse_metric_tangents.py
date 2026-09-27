# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Native fused scalar-normal and proximal metric-disk tangent transactions."""


def get_fragments(capacity: int):
    """Return guarded contact updates for the existing sparse GS owner."""
    if capacity != 100:
        raise ValueError("Metric tangent shared storage is qualified only for capacity100")
    setup = r"""
    __shared__ float contact_cross[100];
    __shared__ unsigned int contact_ready[4];
    if(lane<4)contact_ready[lane]=0u;
    __syncwarp();
"""
    update = r"""
            if(type==0 && row+2<count && iteration>=friction_start && omega==1.0f &&
               row_type.data[base+row+1]==2 && row_type.data[base+row+2]==2 &&
               parent.data[base+row+1]==row && parent.data[base+row+2]==row &&
               d.support.data[base+row]==d.support.data[base+row+1] &&
               d.support.data[base+row]==d.support.data[base+row+2]) {
                const float friction=mu.data[base+row+1];
                if(isfinite(friction) && friction>=0.0f && friction==mu.data[base+row+2]) {
                    const int tpl=d.support.data[base+row],length=p.support_count.data[tpl];
                    const int node=lane<length?p.support_nodes.data[tpl*18+lane]:-1;
                    const float z0=lane<length?d.Z.data[(base+row)*18+lane]:0.0f;
                    float r0=lane<length?z0*du[node]:0.0f;
                    for(int shift=16;shift>0;shift>>=1)r0+=__shfl_down_sync(0xffffffff,r0,shift);
                    r0=__shfl_sync(0xffffffff,r0,0)+d.incident.data[base+row]+rhs.data[base+row];
                    const float old0=lam[row],old1=lam[row+1],old2=lam[row+2];
                    const float d0=diagonal.data[base+row];
                    const float next0=fmaxf(old0-r0/d0,0.0f),change0=next0-old0;
                    const float radius=friction*next0;
                    const int normal_ok=isfinite(r0) && isfinite(d0) && d0>0.0f &&
                        isfinite(old0) && isfinite(old1) && isfinite(old2) &&
                        isfinite(next0) && isfinite(change0) && isfinite(radius);
                    if(normal_ok) {
                        // Normal first: a zero disk needs neither tangential
                        // residuals nor the self-block cache. Old tangents still
                        // contribute their complete removal to the combined du.
                        const float z1=lane<length && (radius>0.0f || old1!=0.0f)
                            ?d.Z.data[(base+row+1)*18+lane]:0.0f;
                        const float z2=lane<length && (radius>0.0f || old2!=0.0f)
                            ?d.Z.data[(base+row+2)*18+lane]:0.0f;
                        int accepted=radius==0.0f;
                        float next1=0.0f,next2=0.0f;
                        if(radius>0.0f) {
                            const int word=row/32;
                            const unsigned int bit=1u<<(row%32);
                            int needs_block=lane==0 && (contact_ready[word]&bit)==0u;
                            needs_block=__shfl_sync(0xffffffff,needs_block,0);
                            if(needs_block) {
                                float a01=z0*z1,a02=z0*z2,a12=z1*z2;
                                for(int shift=16;shift>0;shift>>=1) {
                                    a01+=__shfl_down_sync(0xffffffff,a01,shift);
                                    a02+=__shfl_down_sync(0xffffffff,a02,shift);
                                    a12+=__shfl_down_sync(0xffffffff,a12,shift);
                                }
                                if(lane==0) {
                                    contact_cross[row]=a01;contact_cross[row+1]=a02;
                                    contact_cross[row+2]=a12;contact_ready[word]|=bit;
                                }
                                __syncwarp();
                            }
                            float r1=lane<length?z1*du[node]:0.0f,r2=lane<length?z2*du[node]:0.0f;
                            for(int shift=16;shift>0;shift>>=1) {
                                r1+=__shfl_down_sync(0xffffffff,r1,shift);
                                r2+=__shfl_down_sync(0xffffffff,r2,shift);
                            }
                            r1=__shfl_sync(0xffffffff,r1,0)+d.incident.data[base+row+1]+rhs.data[base+row+1];
                            r2=__shfl_sync(0xffffffff,r2,0)+d.incident.data[base+row+2]+rhs.data[base+row+2];
                            if(lane==0) {
                                r1+=contact_cross[row]*change0;r2+=contact_cross[row+1]*change0;
                                const float a=diagonal.data[base+row+1],c=diagonal.data[base+row+2];
                                const float b=contact_cross[row+2],scale=fmaxf(a,c);
                                const float det=a*c-b*b;
                                const float largest=0.5f*(a+c)+hypotf(0.5f*(a-c),b);
                                const float smallest=det/largest;
                                // Existing denominators enter a proximal delta,
                                // never the physical residual as CFM*lambda.
                                const float beta1=r1-a*old1-b*old2,beta2=r2-b*old1-c*old2;
                                if(isfinite(r1) && isfinite(r2) && isfinite(beta1) && isfinite(beta2) &&
                                   isfinite(scale) && isfinite(largest) && isfinite(smallest) &&
                                   a>0.0f && c>0.0f && smallest>1.0e-7f*scale) {
                                    float x1=(-c*beta1+b*beta2)/det,x2=(b*beta1-a*beta2)/det;
                                    float magnitude=hypotf(x1,x2),alpha=0.0f;
                                    const float beta_norm=hypotf(beta1,beta2);
                                    int root_ok=isfinite(magnitude) && magnitude<=radius;
                                    if(isfinite(magnitude) && magnitude>radius && isfinite(beta_norm)) {
                                        float lower=fmaxf(beta_norm/radius-largest,0.0f);
                                        float upper=fmaxf(beta_norm/radius-smallest,0.0f);
                                        alpha=lower;
                                        if(isfinite(lower) && isfinite(upper)) {
                                            for(int probe=0;probe<16;++probe) {
                                                const float aa=a+alpha,cc=c+alpha,dd=aa*cc-b*b;
                                                x1=(-cc*beta1+b*beta2)/dd;x2=(b*beta1-aa*beta2)/dd;
                                                magnitude=hypotf(x1,x2);
                                                if(!isfinite(dd) || !(dd>0.0f) || !isfinite(magnitude) || !(magnitude>0.0f))break;
                                                if(fabsf(magnitude/radius-1.0f)<=2.0e-6f) {
                                                    // A converged boundary root may
                                                    // need only an inward roundoff repair.
                                                    const float repair=fminf(1.0f,radius/magnitude);
                                                    x1*=repair;x2*=repair;root_ok=1;break;
                                                }
                                                if(magnitude>radius)lower=fmaxf(lower,alpha);
                                                else upper=fminf(upper,alpha);
                                                const float denominator=(cc*x1*x1-2.0f*b*x1*x2+aa*x2*x2)/dd;
                                                if(!isfinite(denominator) || !(denominator>0.0f))break;
                                                float proposal=alpha+magnitude*magnitude*(magnitude/radius-1.0f)/denominator;
                                                if(!isfinite(proposal) || proposal<=lower || proposal>upper)
                                                    proposal=0.5f*(lower+upper);
                                                alpha=proposal;
                                            }
                                        }
                                    }
                                    const float error=hypotf((a+alpha)*x1+b*x2+beta1,b*x1+(c+alpha)*x2+beta2);
                                    const float bound=8.0e-6f*(1.0f+beta_norm+(largest+alpha)*hypotf(x1,x2));
                                    if(root_ok && isfinite(x1) && isfinite(x2) && isfinite(error) && isfinite(bound) &&
                                       error<=bound && hypotf(x1,x2)<=radius*(1.0f+4.0e-7f)) {
                                        next1=x1;next2=x2;accepted=1;
                                    }
                                }
                            }
                        }
                        accepted=__shfl_sync(0xffffffff,accepted,0);
                        next1=__shfl_sync(0xffffffff,next1,0);next2=__shfl_sync(0xffffffff,next2,0);
                        // All physics output remains untouched until every
                        // participating coefficient and proposed delta is finite.
                        const float change1=next1-old1,change2=next2-old2;
                        const int bad_coeff=lane<length && (!isfinite(z0) || !isfinite(z1) || !isfinite(z2));
                        if(accepted && !__ballot_sync(0xffffffff,bad_coeff) &&
                           isfinite(change1) && isfinite(change2)) {
                            if(lane==0) {lam[row]=next0;lam[row+1]=next1;lam[row+2]=next2;}
                            if(change0!=0.0f || change1!=0.0f || change2!=0.0f) {
                                if(lane<length)du[node]+=z0*change0+z1*change1+z2*change2;
                                changed=1;
                            }
                            __syncwarp();row+=2;continue;
                        }
                    }
                    // Rejection retains the original complete scalar triplet.
                }
            }
"""
    return setup, update
