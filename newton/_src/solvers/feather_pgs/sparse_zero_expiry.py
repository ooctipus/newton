# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Omit certified zero transactions inside the unchanged ordered sparse solve."""


def apply_expiry(source: str, capacity: int):
    """Add solve-local row certificates without changing native inputs or row laws."""
    if capacity != 100:
        raise ValueError("Sparse zero expiry requires the metric capacity100 owner")

    setup = r"""
    // N bounds actual stored Z, independently of denominator-only compliance.
    // The additive floor covers underflow/FTZ in the eighteen positive terms.
    __shared__ float expiry_norm[100], expiry_at[100];
    __shared__ unsigned int expiry_triplets[4];
    if(lane<4)expiry_triplets[lane]=0u;
    constexpr float expiry_error=0x1p-16f; // 128 float32 epsilons.
    for(int r=lane;r<count;r+=32) {
        float sum=0.0f;
        for(int k=0;k<18;++k) {
            const float value=d.Z.data[(base+r)*18+k];
            sum+=value*value;
        }
        float norm=INFINITY;
        if(isfinite(sum) && sum>0.0f) {
            const float upper=__fadd_ru(__fmul_ru(sum,1.0f+expiry_error),0x1p-119f);
            norm=__fmul_ru(__fsqrt_ru(upper),1.0f+expiry_error);
        }
        expiry_norm[r]=norm;expiry_at[r]=-INFINITY;
    }
    __syncwarp();
    // Replicated warp-uniform clock bounds the total path length of rounded du.
    float expiry_clock=0.0f;
    auto expiry_advance=[&](float bound) {
        const float next=__fadd_ru(expiry_clock,bound);
        expiry_clock=__fadd_ru(next,__fmul_ru(expiry_error,__fadd_ru(1.0f,next)));
        if(!isfinite(expiry_clock))expiry_clock=INFINITY;
    };
    auto expiry_term=[&](int r,float delta) {
        return delta==0.0f?0.0f:__fmul_ru(fabsf(delta),expiry_norm[r]);
    };
    auto expiry_record=[&](int r,float residual,int triplet) {
        if(lane==0 && residual>0.0f && isfinite(residual) && isfinite(expiry_clock)) {
            const float norm=expiry_norm[r];
            const float constant=__fadd_ru(1.0f,__fadd_ru(fabsf(d.incident.data[base+r]),fabsf(rhs.data[base+r])));
            const float error=__fmul_ru(expiry_error,__fadd_ru(constant,__fmul_ru(norm,expiry_clock)));
            // Future dot error grows with clock. Solve that bound for its expiry;
            // both saved and future evaluations are charged, rounded inward.
            const float margin=__fsub_rd(residual,__fmul_ru(2.0f,error));
            const float room=__fdiv_rd(margin,__fmul_ru(norm,1.0f+expiry_error));
            const float at=__fadd_rd(expiry_clock,room);
            expiry_at[r]=isfinite(norm) && norm>0.0f && isfinite(at)?at:-INFINITY;
            const unsigned int bit=1u<<(r%32);
            if(triplet)expiry_triplets[r/32]|=bit;
            else expiry_triplets[r/32]&=~bit;
        }
    };
"""
    anchors = {
        "    for(int iteration=0;iteration<iterations;++iteration) {": setup
        + "    for(int iteration=0;iteration<iterations;++iteration) {",
        "            const int type=row_type.data[base+row];": r"""
            // A valid certificate retains its original zero transaction:
            // immutable row guards and exclusive lambda writers were proved
            // on issue. An expired monotone clock cannot revive stale state.
            if(expiry_clock<expiry_at[row]) {
                if(expiry_triplets[row/32]&(1u<<(row%32)))row+=2;
                continue;
            }
            const int type=row_type.data[base+row];
""",
        "                            if(lane==0) {lam[row]=next0;lam[row+1]=next1;lam[row+2]=next2;}": r"""
                            if(old0==0.0f && old1==0.0f && old2==0.0f && change0==0.0f && change1==0.0f && change2==0.0f)
                                expiry_record(row,r0,1);
                            if(lane==0) {lam[row]=next0;lam[row+1]=next1;lam[row+2]=next2;}
""",
        "                                if(lane<length)du[node]+=z0*change0+z1*change1+z2*change2;": r"""
                                if(lane<length)du[node]+=z0*change0+z1*change1+z2*change2;
                                const float bound=__fadd_ru(expiry_term(row,change0),
                                    __fadd_ru(expiry_term(row+1,change1),expiry_term(row+2,change2)));
                                expiry_advance(bound);
""",
        "            if(!(denom>0.0f))continue;": r"""
            if(!(denom>0.0f))continue;
            const int expiry_scalar=(type==0 || type==3) && isfinite(omega) && omega>=0.0f && isfinite(denom);
""",
        "                delta=next-old;lam[row]=next;": r"""
                delta=next-old;lam[row]=next;
                if(expiry_scalar && old==0.0f && delta==0.0f)expiry_record(row,residual,0);
""",
        "                changed=1;\n            }\n            __syncwarp();\n            if(delta!=0.0f)": r"""
                expiry_advance(expiry_term(sibling,sibling_delta));
                changed=1;
            }
            __syncwarp();
            if(delta!=0.0f)""",
        "            if(delta!=0.0f) { if(lane<length)du[node]+=z*delta;changed=1; }": r"""
            if(delta!=0.0f) { if(lane<length)du[node]+=z*delta;expiry_advance(expiry_term(row,delta));changed=1; }
""",
    }
    for anchor, replacement in anchors.items():
        if source.count(anchor) != 1:
            raise RuntimeError(f"Sparse zero-expiry source anchor changed: {anchor!r}")
        source = source.replace(anchor, replacement)
    return source
