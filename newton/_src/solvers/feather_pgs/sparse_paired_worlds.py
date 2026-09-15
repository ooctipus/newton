# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pack the retained metric solve into count-paired, independent half warps."""

import hashlib
import inspect
import re
from functools import cache

import warp as wp

from .sparse_factor import SparseData, SparsePlan

ORIGINAL_NATIVE_SHA = "eda6abf58c026c3abf9ebe6196c7790ed90444a0e5ce99f061d09fbd8a364352"
KERNEL_KEY = "sparse_metric_paired43_s18_c100_w16"
WORLDS_PER_BLOCK = 16
BLOCK_DIM = 256


def _replace(source, old, new, count=1):
    if source.count(old) != count:
        raise ValueError(f"Original paired solve seam changed: {old[:100]}")
    return source.replace(old, new)


def _rewrite_source(source):
    """Keep original projections while assigning every support element once."""
    if hashlib.sha256(source.encode()).hexdigest() != ORIGINAL_NATIVE_SHA:
        raise ValueError("Original sparse metric solve changed; review paired ownership")
    source = _replace(
        source,
        "    const int lane=threadIdx.x&31, art=p.group_to_art.data[group], world=p.art_to_world.data[art];",
        r"""
    const int thread=threadIdx.x,lane=thread&15,slot=thread>>4;
    __shared__ int permutation[16];
    // Only the first half warp sorts. Every CTA lane joins before any tail
    // return, so one incomplete world cannot strand another world's barrier.
    if(thread<16) {
        const int candidate=group*16+thread;
        int rows=101;
        if(candidate<p.group_to_art.shape[0]) {
            const int art=p.group_to_art.data[candidate];
            rows=counts.data[p.art_to_world.data[art]];
            rows=rows<0?0:(rows>100?100:rows);
        }
        unsigned int key=(static_cast<unsigned int>(rows)<<4)|static_cast<unsigned int>(thread);
        #pragma unroll
        for(int size=2;size<=16;size<<=1) {
            #pragma unroll
            for(int distance=size>>1;distance>0;distance>>=1) {
                const unsigned int other=__shfl_xor_sync(0x0000ffffu,key,distance,16);
                const bool low=((thread&distance)==0)==((thread&size)==0);
                key=low?(key<other?key:other):(key>other?key:other);
            }
        }
        permutation[thread]=static_cast<int>(key&15u);
    }
    __syncthreads();
    group=group*16+permutation[slot];
    if(group>=p.group_to_art.shape[0])return;
    const unsigned int mask=0xffffu<<((thread&31)>=16?16:0);
    const int art=p.group_to_art.data[group],world=p.art_to_world.data[art];
""",
    )
    source = _replace(
        source,
        "    __shared__ float du[43], lam[100];",
        "    __shared__ float du_storage[16][43],lam_storage[16][100];\n"
        "    float* du=du_storage[slot];float* lam=lam_storage[slot];",
    )
    source = _replace(
        source,
        "    __shared__ float contact_cross[100];\n    __shared__ unsigned int contact_ready[4];",
        "    __shared__ float cross_storage[16][100];\n"
        "    __shared__ unsigned int ready_storage[16][4];\n"
        "    float* contact_cross=cross_storage[slot];\n"
        "    unsigned int* contact_ready=ready_storage[slot];",
    )
    node = "const int node=lane<length?p.support_nodes.data[tpl*18+lane]:-1;"
    source = _replace(
        source,
        node,
        node + "\n            const int extra=lane+16;\n"
        "            const int node_extra=extra<length?p.support_nodes.data[tpl*18+extra]:-1;",
        2,
    )
    line = "const float z0=lane<length?d.Z.data[(base+row)*18+lane]:0.0f;"
    source = _replace(
        source,
        line,
        line + "\n                    const float z0_extra=extra<length?d.Z.data[(base+row)*18+extra]:0.0f;",
    )
    line = "float r0=lane<length?z0*du[node]:0.0f;"
    source = _replace(source, line, line + "\n                    if(extra<length)r0+=z0_extra*du[node_extra];")
    line = "const float z2=lane<length && (radius>0.0f || old2!=0.0f)\n                            ?d.Z.data[(base+row+2)*18+lane]:0.0f;"
    source = _replace(
        source,
        line,
        line + "\n"
        "                        const float z1_extra=extra<length && (radius>0.0f || old1!=0.0f)\n"
        "                            ?d.Z.data[(base+row+1)*18+extra]:0.0f;\n"
        "                        const float z2_extra=extra<length && (radius>0.0f || old2!=0.0f)\n"
        "                            ?d.Z.data[(base+row+2)*18+extra]:0.0f;",
    )
    line = "float a01=z0*z1,a02=z0*z2,a12=z1*z2;"
    source = _replace(
        source,
        line,
        line
        + "\n                                if(extra<length){a01+=z0_extra*z1_extra;a02+=z0_extra*z2_extra;a12+=z1_extra*z2_extra;}",
    )
    line = "float r1=lane<length?z1*du[node]:0.0f,r2=lane<length?z2*du[node]:0.0f;"
    source = _replace(
        source,
        line,
        line
        + "\n                            if(extra<length){r1+=z1_extra*du[node_extra];r2+=z2_extra*du[node_extra];}",
    )
    source = _replace(
        source,
        "const int bad_coeff=lane<length && (!isfinite(z0) || !isfinite(z1) || !isfinite(z2));",
        "const int bad_coeff=(lane<length && (!isfinite(z0) || !isfinite(z1) || !isfinite(z2))) ||\n"
        "                            (extra<length && (!isfinite(z0_extra) || !isfinite(z1_extra) || !isfinite(z2_extra)));",
    )
    line = "if(lane<length)du[node]+=z0*change0+z1*change1+z2*change2;"
    source = _replace(
        source,
        line,
        line
        + "\n                                if(extra<length)du[node_extra]+=z0_extra*change0+z1_extra*change1+z2_extra*change2;",
    )
    line = "const float z=lane<length?d.Z.data[(base+row)*18+lane]:0.0f;"
    source = _replace(
        source, line, line + "\n            const float z_extra=extra<length?d.Z.data[(base+row)*18+extra]:0.0f;"
    )
    line = "float dot=lane<length?z*du[node]:0.0f;"
    source = _replace(source, line, line + "\n            if(extra<length)dot+=z_extra*du[node_extra];")
    line = "if(lane<sn)du[p.support_nodes.data[st*18+lane]]+=d.Z.data[(base+sibling)*18+lane]*sibling_delta;"
    source = _replace(
        source,
        line,
        line
        + "\n                if(extra<sn)du[p.support_nodes.data[st*18+extra]]+=d.Z.data[(base+sibling)*18+extra]*sibling_delta;",
    )
    source = _replace(
        source,
        "if(delta!=0.0f) { if(lane<length)du[node]+=z*delta;changed=1; }",
        "if(delta!=0.0f) { if(lane<length)du[node]+=z*delta;\n"
        "                if(extra<length)du[node_extra]+=z_extra*delta;changed=1; }",
    )
    # The missing shift16 is exactly the secondary product addition above.
    source = source.replace("+=32", "+=16").replace("int shift=16;", "int shift=8;").replace("int s=16;", "int s=8;")
    source = re.sub(r"__shfl(_down)?_sync\(0xffffffff,([^()]*)\)", r"__shfl\1_sync(mask,\2,16)", source)
    source = source.replace("__ballot_sync(0xffffffff,", "__ballot_sync(mask,").replace(
        "__syncwarp()", "__syncwarp(mask)"
    )
    if "0xffffffff" in source or source.count("__syncthreads();") != 1:
        raise AssertionError("Paired solve retained a cross-world collective")
    return source


@cache
def native_source():
    """Return only the reviewed retained-native ownership rewrite."""
    from .sparse_factor_rows import get_solve_kernel as original_factory  # noqa: PLC0415

    original = original_factory(100, metric_tangents=True)
    native = inspect.getclosurevars(original.func).nonlocals["native"]
    return _rewrite_source(native.native_snippet)


@cache
def get_solve_kernel():
    """Return the fixed256-thread,16-world owner with unchanged input ABI."""

    @wp.func_native(native_source())
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
    ): ...

    def paired(
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
        )

    paired.__name__ = paired.__qualname__ = KERNEL_KEY
    return wp.kernel(enable_backward=False, module="unique")(paired)


def supported(owner):
    """Admit only the unchanged cold regular metric owner before substitution."""
    s = owner.solver
    return bool(
        s.model.device.is_cuda
        and s.dense_max_constraints == 100
        and owner.metric_tangents
        and not owner.packet_rows
        and not owner.block_contacts
        and not s.pgs_warmstart
        and not s._mf_warmstart_enabled
        and s.pgs_velocity_iterations == 0
        and s.pgs_iterations == 8
        and s.pgs_omega == 1.0
        and not any(
            getattr(owner, name, False)
            for name in ("lazy_compliance", "present_ports", "zero_expiry", "coordinate_register", "register_residual")
        )
    )


def solve(owner, rhs, iterations, omega, friction_start):
    """Replace one original launch, with no queue or intermediate output."""
    s = owner.solver
    wp.launch_tiled(
        owner.kernels.solve,
        dim=[(s.world_count + WORLDS_PER_BLOCK - 1) // WORLDS_PER_BLOCK],
        inputs=[
            owner.plan,
            owner.data,
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
        ],
        block_dim=BLOCK_DIM,
        device=s.model.device,
    )
