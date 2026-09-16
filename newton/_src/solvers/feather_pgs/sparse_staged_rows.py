# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Stage a fixed small-row cohort for the unchanged ordered spectral solve.

The producer, physical rows and public solve ABI are unchanged. Worlds with
more than 32 rows retain the original spectral body. Both paths share one
allocation, so the fallback is charged the same maximum resource footprint.
"""

import functools
import inspect

import warp as wp

from . import sparse_spectral_tangents
from .sparse_factor import SparseData, SparsePlan


def _replace(source, before, after, count=1):
    if source.count(before) != count:
        raise RuntimeError(f"Sparse staged source seam changed: {before!r}")
    return source.replace(before, after)


_STORAGE = r"""
    union StagedSolveStorage {
        struct {
            float z[32][18];
            unsigned char nodes[32][18];
            float incident[32], rhs[32], diagonal[32], mu[32];
            unsigned int meta[32];
            float du[43], lam[32], cross[32];
            unsigned int ready;
        } fast;
        struct {
            float du[43], lam[100], cross[100];
            unsigned int ready[4];
        } full;
    };
    static_assert(sizeof(StagedSolveStorage)==3952,"staged solve storage changed");
    __shared__ StagedSolveStorage storage;
"""

_STAGE = r"""
        auto& staged=storage.fast;
        if(lane<count) {
            const int row=lane, type=row_type.data[base+row];
            const int tpl=d.support.data[base+row], length=p.support_count.data[tpl];
            // Only validated tangent parents are used by the solve. Other rows
            // do not need their (potentially stale) unused parent encoded.
            const int par=type==2?parent.data[base+row]:-1;
            int triplet=0;
            if(type==0 && row+2<count &&
               row_type.data[base+row+1]==2 && row_type.data[base+row+2]==2 &&
               parent.data[base+row+1]==row && parent.data[base+row+2]==row &&
               d.support.data[base+row]==d.support.data[base+row+1] &&
               d.support.data[base+row]==d.support.data[base+row+2]) {
                const float friction=mu.data[base+row+1];
                triplet=isfinite(friction) && friction>=0.0f && friction==mu.data[base+row+2];
            }
            staged.meta[row]=static_cast<unsigned int>(type) | (static_cast<unsigned int>(par+1)<<2) |
                (static_cast<unsigned int>(length)<<8) | (static_cast<unsigned int>(triplet)<<13);
            staged.incident[row]=d.incident.data[base+row];
            staged.rhs[row]=rhs.data[base+row];
            staged.diagonal[row]=diagonal.data[base+row];
            staged.mu[row]=mu.data[base+row];
        }
        __syncwarp();
        for(int item=lane;item<count*18;item+=32) {
            const int row=item/18, k=item%18;
            const int length=static_cast<int>((staged.meta[row]>>8)&31u);
            if(k<length) {
                const int tpl=d.support.data[base+row];
                staged.z[row][k]=d.Z.data[(base+row)*18+k];
                staged.nodes[row][k]=static_cast<unsigned char>(p.support_nodes.data[tpl*18+k]);
            } else {
                staged.z[row][k]=0.0f;staged.nodes[row][k]=255;
            }
        }
        __syncwarp();
"""


def _body_aliases(source, member, *, fast):
    source = _replace(
        source,
        "    __shared__ float du[43], lam[100];",
        f"    float* du=storage.{member}.du;float* lam=storage.{member}.lam;",
    )
    source = _replace(
        source,
        "    __shared__ float contact_cross[100];\n    __shared__ unsigned int contact_ready[4];",
        f"    float* contact_cross=storage.{member}.cross;"
        f"unsigned int* contact_ready={'&' if fast else ''}storage.{member}.ready;",
    )
    if fast:
        source = _replace(source, "if(lane<4)contact_ready[lane]=0u;", "if(lane==0)contact_ready[0]=0u;")
    return source


def _staged_source(original):
    """Retain source-checked transactions, changing only immutable row reads."""
    anchor = "    __shared__ float du[43], lam[100];"
    if original.count(anchor) != 1 or not original.endswith("#endif\n"):
        raise RuntimeError("Sparse staged solve body boundary changed")
    prefix, body = original.split(anchor)
    body = anchor + body[: -len("#endif\n")]
    full = _body_aliases(body, "full", fast=False)
    fast = _body_aliases(body, "fast", fast=True)
    fast = _replace(
        fast,
        """if(type==0 && row+2<count && iteration>=friction_start && omega==1.0f &&
               row_type.data[base+row+1]==2 && row_type.data[base+row+2]==2 &&
               parent.data[base+row+1]==row && parent.data[base+row+2]==row &&
               d.support.data[base+row]==d.support.data[base+row+1] &&
               d.support.data[base+row]==d.support.data[base+row+2])""",
        "if((staged.meta[row]&(1u<<13)) && iteration>=friction_start && omega==1.0f)",
    )
    fast = _replace(
        fast,
        "if(isfinite(friction) && friction>=0.0f && friction==mu.data[base+row+2]) {",
        "{ // Immutable eligibility was established before the first sweep.",
    )
    for declaration in (
        "const int tpl=d.support.data[base+row],length=p.support_count.data[tpl];",
        "const int tpl=d.support.data[base+row], length=p.support_count.data[tpl];",
    ):
        fast = _replace(fast, declaration, "const int length=static_cast<int>((staged.meta[row]>>8)&31u);")
    fast = _replace(fast, "p.support_nodes.data[tpl*18+lane]", "static_cast<int>(staged.nodes[row][lane])", 2)
    fast = _replace(
        fast,
        "const int st=d.support.data[base+sibling], sn=p.support_count.data[st];",
        "const int sn=static_cast<int>((staged.meta[sibling]>>8)&31u);",
    )
    fast = _replace(fast, "p.support_nodes.data[st*18+lane]", "static_cast<int>(staged.nodes[sibling][lane])")
    fast = _replace(fast, "row_type.data[base+row]", "static_cast<int>(staged.meta[row]&3u)")
    fast = _replace(fast, "parent.data[base+row]", "static_cast<int>((staged.meta[row]>>2)&63u)-1")
    for offset, occurrences in (("row", 2), ("row+1", 1), ("row+2", 1), ("sibling", 1)):
        fast = _replace(fast, f"d.Z.data[(base+{offset})*18+lane]", f"staged.z[{offset}][lane]", occurrences)
    for field, offsets in (
        ("d.incident", (("row", 2), ("row+1", 1), ("row+2", 1))),
        ("rhs", (("row", 2), ("row+1", 1), ("row+2", 1))),
        ("diagonal", (("row", 2), ("row+1", 1), ("row+2", 1))),
        ("mu", (("row", 1), ("row+1", 1))),
    ):
        target = field.split(".")[-1]
        for offset, occurrences in offsets:
            fast = _replace(fast, f"{field}.data[base+{offset}]", f"staged.{target}[{offset}]", occurrences)
    return prefix + _STORAGE + "    if(count<=32) {\n" + _STAGE + fast + "    } else {\n" + full + "    }\n#endif\n"


@functools.cache
def get_solve_kernel():
    """Return the original 15-argument spectral owner with fixed-cohort staging."""
    original = sparse_spectral_tangents.get_solve_kernel()
    source = inspect.getclosurevars(original.func).nonlocals["native"].native_snippet

    @wp.func_native(_staged_source(source))
    def native(
        group: int,
        p: SparsePlan,
        d: SparseData,
        counts: wp.array[int],
        rhs: wp.array2d[float],
        diagonal: wp.array2d[float],
        cfm: wp.array2d[float],
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

    def solve(
        p: SparsePlan,
        d: SparseData,
        counts: wp.array[int],
        rhs: wp.array2d[float],
        diagonal: wp.array2d[float],
        cfm: wp.array2d[float],
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
            cfm,
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

    solve.__name__ = solve.__qualname__ = "sparse_staged_spectral_tangent43_s18_c100"
    return wp.kernel(enable_backward=False, module="unique")(solve)
