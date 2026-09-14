# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bounded current-force/local-row owner with the held sparse kinetic factor."""

from functools import cache

import warp as wp

from .sparse_factor import SparseData, SparsePlan
from .sparse_metric_tangents import get_fragments
from .sparse_packet_rows import PacketInput
from .sparse_packet_rows import cuda_source as packet_cuda_source


def _replace(source, old, new):
    """Fail source construction when a reused implementation changes its contract."""
    if old not in source:
        raise RuntimeError(f"Small-step source anchor is absent: {old}")
    return source.replace(old, new)


def cuda_source(force_source: str):
    """Compose existing geometry and metric updates around private natural-order vhat.

    Canonical metadata and impulses retain their original stride; only local
    supported rows are bounded to32. The injected force fragment owns its
    private prediction and must not write the old global intermediate arrays.
    """
    source = packet_cuda_source(32)
    source = _replace(
        source,
        "count=counts.data[world], base=world*32;",
        "count=counts.data[world], base=world*row_type.shape[1], packet_base=world*packets.shape[1];\n"
        "    if(!selected.data[world])return;",
    )
    source = _replace(source, "count>32", "count<0 || count>32 || count>packets.shape[1]")
    source = _replace(source, "d.support.data[base+r]", "packets.data[packet_base+r]")
    source = _replace(source, "d.support.data[base+row]", "packets.data[packet_base+row]")
    source = _replace(source, "vhat.data[start+local]", "small_vhat[local]")
    source = _replace(source, "vhat.data[start+dof]", "small_vhat[dof]")
    source = _replace(source, "vhat.data[start+42-col]", "small_vhat[42-col]")
    source = _replace(source, "rhs.data[base+r]", "local_rhs[r]")
    source = _replace(source, "rhs.data[base+row]", "local_rhs[row]")
    # Admission excludes warm starts. Selected preparation is retired, so a
    # previous canonical impulse must never seed this generation's private du.
    source = _replace(source, "lam[r]=impulses.data[base+r]", "lam[r]=0.0f")
    anchor = "    __shared__ float zrows[32*18]"
    source = _replace(source, anchor, force_source + "\n" + anchor)
    source = _replace(
        source,
        "    __shared__ int templates[32];",
        """    __shared__ int templates[32];
    __shared__ float local_rhs[32];
    if(lane<count) {
        const int at=base+lane,type=row_type.data[at];
        const float phi=x.phi.data[at],beta=row_beta.data[at];
        float value=-x.target.data[at];
        if(type==0)value+=(phi<=0.0f?beta*phi:contact_speculative_scale*phi)/x.dt;
        else if(type==3)value+=(phi<0.0f?beta*phi:phi)/x.dt;
        local_rhs[lane]=value;
    }
    __syncwarp();""",
    )

    # Keep the qualified update arithmetic, bounded secular solve and fallback.
    # Only its storage references change; bias already includes current J*vhat.
    setup, update = get_fragments(100)
    setup = _replace(setup, "contact_cross[100]", "contact_cross[32]")
    setup = _replace(setup, "contact_ready[4]", "contact_ready[1]")
    setup = _replace(setup, "lane<4", "lane<1")
    for suffix in ("", "+1", "+2"):
        update = _replace(update, f"d.support.data[base+row{suffix}]", f"templates[row{suffix}]")
        update = _replace(update, f"d.Z.data[(base+row{suffix})*18+lane]", f"zrows[(row{suffix})*18+lane]")
        update = _replace(update, f"diagonal.data[base+row{suffix}]", f"norm[row{suffix}]")
        update = _replace(
            update,
            f"d.incident.data[base+row{suffix}]+rhs.data[base+row{suffix}]",
            f"bias[row{suffix}]",
        )
    source = _replace(
        source,
        "    for(int iteration=0;iteration<iterations;++iteration) {",
        setup + "    for(int iteration=0;iteration<iterations;++iteration) {",
    )
    source = _replace(
        source,
        "            if(type==2 && iteration<friction_start)",
        update + "            if(type==2 && iteration<friction_start)",
    )
    return source


@cache
def get_solve_kernel():
    """Return one32-thread selected-world force-to-final-velocity owner."""
    from .small_step_force import ForceInput, get_force_source  # noqa: PLC0415

    @wp.func_native(cuda_source(get_force_source()))
    def native(
        group: int,
        p: SparsePlan,
        d: SparseData,
        f: ForceInput,
        x: PacketInput,
        selected: wp.array[int],
        packets: wp.array2d[int],
        counts: wp.array[int],
        row_beta: wp.array2d[float],
        row_type: wp.array2d[int],
        parent: wp.array2d[int],
        mu: wp.array2d[float],
        iterations: int,
        omega: float,
        friction_start: int,
        contact_speculative_scale: float,
        impulses: wp.array2d[float],
        vout: wp.array[float],
    ): ...

    def solve(
        p: SparsePlan,
        d: SparseData,
        f: ForceInput,
        x: PacketInput,
        selected: wp.array[int],
        packets: wp.array2d[int],
        counts: wp.array[int],
        row_beta: wp.array2d[float],
        row_type: wp.array2d[int],
        parent: wp.array2d[int],
        mu: wp.array2d[float],
        iterations: int,
        omega: float,
        friction_start: int,
        contact_speculative_scale: float,
        impulses: wp.array2d[float],
        vout: wp.array[float],
    ):
        group, _ = wp.tid()
        native(
            group,
            p,
            d,
            f,
            x,
            selected,
            packets,
            counts,
            row_beta,
            row_type,
            parent,
            mu,
            iterations,
            omega,
            friction_start,
            contact_speculative_scale,
            impulses,
            vout,
        )

    solve.__name__ = solve.__qualname__ = "small_step_force_rows_metric43_s18_c32"
    return wp.kernel(enable_backward=False, module="unique")(solve)
