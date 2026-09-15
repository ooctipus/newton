# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Binary-region clock diagnostic with no per-row counters or phase dispatch.

Unsigned clock32 differences are wrap-safe provided each measured owner lasts
less than 2**32 SM cycles. Output fields are widened to uint64 after subtraction.
This is an observer-overhead experiment, not a replacement production owner.
"""

import functools
import hashlib
import types
from pathlib import Path

if __package__:
    from . import profile_sparse_metric_clock as clock
else:
    import profile_sparse_metric_clock as clock

FIELDS = ("total_cycles", "selected_cycles", "rows")


def instrument_native(source, phase):
    """Bracket only metric-disk (3) or complete scalar-row (5) work."""
    if phase not in (3, 5):
        raise ValueError("Binary diagnostic only supports metric-disk3 or scalar5")
    original = source
    declarations = r"""
    const bool bc_sample=lane==0 && sample_stride>0 && world%sample_stride==0;
    unsigned int bc_start=0,bc_region=0,bc_selected=0;
    if(bc_sample) {
        asm volatile("mov.u32 %0, %%clock;" : "=r"(bc_start) :: "memory");
    }
"""
    begin = r"""
    if(bc_sample) {
        asm volatile("mov.u32 %0, %%clock;" : "=r"(bc_region) :: "memory");
    }
"""
    end = r"""
    if(bc_sample) {
        unsigned int bc_now;
        asm volatile("mov.u32 %0, %%clock;" : "=r"(bc_now) :: "memory");
        bc_selected+=bc_now-bc_region;
    }
"""
    anchor = "    if (!d.valid.data[world] || d.status.data[world] || count>100) return;"
    source = clock._once(source, anchor, clock._block(declarations) + anchor)
    if phase == 3:
        anchor = (
            "                            if(lane==0) {\n                                r1+=contact_cross[row]*change0;"
        )
        source = clock._once(source, anchor, clock._block(begin) + anchor)
        anchor = "                            }\n                        }\n                        accepted=__shfl_sync(0xffffffff,accepted,0);"
        source = clock._once(
            source,
            anchor,
            "                            }\n"
            + clock._block(end)
            + "                        }\n                        accepted=__shfl_sync(0xffffffff,accepted,0);",
        )
    else:
        anchor = "            if(type==2 && iteration<friction_start)"
        source = clock._once(source, anchor, clock._block(begin) + anchor)
        anchor = "if(lane==0)lam[row]=0.0f; __syncwarp(); continue;"
        source = clock._once(
            source, anchor, "if(lane==0)lam[row]=0.0f; __syncwarp();" + clock._block(end) + " continue;"
        )
        anchor = "            if(!(denom>0.0f))continue;"
        source = clock._once(
            source,
            anchor,
            "            if(!(denom>0.0f))" + clock._block("{\n" + end) + "continue;" + clock._block("}\n"),
        )
        anchor = "            __syncwarp();\n        }\n        if(iteration>=friction_start &&"
        source = clock._once(
            source,
            anchor,
            "            __syncwarp();\n" + clock._block(end) + "        }\n        if(iteration>=friction_start &&",
        )
    publish = r"""
    if(bc_sample) {
        unsigned int bc_now;
        asm volatile("mov.u32 %0, %%clock;" : "=r"(bc_now) :: "memory");
        diagnostic.data[(size_t)world*3]=(unsigned int)(bc_now-bc_start);
        diagnostic.data[(size_t)world*3+1]=bc_selected;
        diagnostic.data[(size_t)world*3+2]=count;
    }
"""
    source = clock._once(source, "#endif", clock._block(publish) + "#endif")
    if clock.strip_instrumentation(source) != original:
        raise AssertionError("Binary instrumentation changed original numerical source")
    for barrier in ("__syncwarp(", "__syncthreads(", "__ballot_sync("):
        if source.count(barrier) != original.count(barrier):
            raise AssertionError("Binary instrumentation changed synchronization")
    return source


@functools.cache
def build_binary_clock_kernel(phase=5):
    """Return the pinned separate kernel and metadata with the same appended ABI."""
    if phase not in (3, 5):
        raise ValueError("Binary diagnostic only supports metric-disk3 or scalar5")
    builder = clock.build_clock_kernel.__wrapped__
    namespace = dict(builder.__globals__)
    namespace.update(FIELDS=FIELDS, instrument_native=lambda source: instrument_native(source, phase))
    # The generated wrapper name distinguishes the two fixed diagnostic owners;
    # its numerical native source still recovers byte-for-byte to the original.
    constants = tuple(
        f"name += '_binary{phase}'" if value == "name += '_clock18'" else value for value in builder.__code__.co_consts
    )
    if constants == builder.__code__.co_consts:
        raise ValueError("Pinned diagnostic wrapper naming seam changed")
    cloned = types.FunctionType(
        builder.__code__.replace(co_consts=constants),
        namespace,
        builder.__name__,
        builder.__defaults__,
        builder.__closure__,
    )
    kernel, metadata = cloned(100)
    metadata.update(
        selected_phase=phase,
        clock_bits=32,
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    return kernel, metadata
