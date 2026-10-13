# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

# Preserve the checked original host/source-construction expressions.

"""One compact dense/current-MF recurrence with a shared fallback arena.

Eligible worlds retain primary kinetic rows, materialize six held physical
response columns, and own dense -> current MF -> rigid velocity limits in each
original sweep. Qualification, dispatch count, unsupported fallback arithmetic,
and the caller's stream joins are unchanged. Snapshot conversion is host-only.
"""

import functools

import warp as wp

from . import compact_recurrence_source as recurrence_source
from . import kinetic_guard as guards
from . import kinetic_hybrid as hybrid_source
from . import kinetic_solve as original
from .compact_recurrence_source import get_source as compact_recurrence
from .kinetic_coupled_cpu import get_source as coupled_cpu_source
from .kinetic_rows_types import DenseRowOutput, RowState
from .kinetic_solve_types import KineticMFData, KineticSolveData
from .kinetic_source import checked_definitions
from .kinetic_types import HeldKineticOperator, KineticPlan
from .mf_dispatch import make_dispatch, make_native

HybridCoupledData = hybrid_source.HybridCoupledData
allocate_hybrid = hybrid_source.allocate_hybrid
get_qualify_kernel = hybrid_source.get_qualify_kernel

ARENA_FLOATS = 1053
# Preserve the checked view alias; the shared factory owns its definition.
_VIEWS = recurrence_source._VIEWS


def _pin():
    """Check the retained predecessor definitions, not an edited module file."""
    hybrid_source._pin()
    checked_definitions(
        "kinetic_hybrid.py",
        (
            "HybridCoupledData",
            "_CPU",
            "_STAGE",
            "_STORE",
            "qualify_source",
            "general_source",
            "get_qualify_kernel",
            "get_materialize_kernel",
            "get_general_kernel",
        ),
    )


@wp.func_native(r"""
#if defined(__CUDA_ARCH__)
    __shared__ float compact_shared_arena[1053];
    return reinterpret_cast<uint64_t>(compact_shared_arena);
#else
    return 0;
#endif
""")
def _arena() -> wp.uint64:
    """Allocate one CTA arena shared by mutually exclusive native owners."""


_SIX_BRIDGE = r"""
#if defined(__CUDA_ARCH__)
    const int lane=threadIdx.x&31, stride=32;
#else
    if(logical!=0)return;
    const int lane=0,stride=1;
#endif
    if(rows.status.data[world]!=0 || rows.global_status.data[0]!=0 ||
       held.valid.data[world]==0 || held.generation.data[world]!=state.held_generation.data[world]) {
        if(lane==0)solve.status.data[world]=1;return;
    }
    const int count=state.dense_count.data[world], po=state.primary_offset.data[world];
    const int so=state.secondary_offset.data[world], group=plan.secondary_group.data[world];
    if(count<0 || count>192 || !((po==0&&so==23)||(po==6&&so==0))) {
        if(lane==0)solve.status.data[world]=1;return;
    }
    // Each lane owns a complete row: no in-place cross-lane load/write hazard.
    bool good=true;
    for(int row=lane;row<count;row+=stride) {
        const int start=(world*192+row)*29+so;
        float z[6]; for(int k=0;k<6;++k)z[k]=rows.response.data[start+k];
        for(int col=0;col<6;++col) {
            float y=0.0f;
            for(int k=col;k<6;++k)y+=held.inverse6.data[group*36+k*6+col]*z[k];
            rows.response.data[start+col]=y;good &= isfinite(y);
        }
    }
#if defined(__CUDA_ARCH__)
    good=__all_sync(0xffffffffu,good);
#endif
    if(lane==0)solve.status.data[world]=good?0:2;
"""


@functools.cache
def get_materialize_kernel(arch):
    """Replace the existing bridge: selected six columns, original fallback."""
    _pin()
    source = "if(hybrid.eligible.data[world]!=0){\n" + _SIX_BRIDGE + "\nreturn;}\n" + original._MATERIALIZE

    @wp.func_native(source)
    def materialize(
        world: int,
        logical: int,
        plan: KineticPlan,
        held: HeldKineticOperator,
        state: RowState,
        rows: DenseRowOutput,
        solve: KineticSolveData,
        hybrid: HybridCoupledData,
    ):
        """Perform only the selected representation conversion."""

    @wp.kernel(module="unique", enable_backward=False)
    def kinetic_compact_six_materialize(
        plan: KineticPlan,
        held: HeldKineticOperator,
        state: RowState,
        rows: DenseRowOutput,
        solve: KineticSolveData,
        hybrid: HybridCoupledData,
    ):
        index, logical = wp.tid()
        if state.active_count[0] < 0 or state.active_count[0] > state.active_worlds.shape[0]:
            return
        if index >= state.active_count[0]:
            return
        world = state.active_worlds[index]
        if world >= 0 and world < state.dense_count.shape[0]:
            materialize(world, logical, plan, held, state, rows, solve, hybrid)

    return kinetic_compact_six_materialize


def fallback_source(baseline):
    """Bind the original fallback law to exact nonoverlapping arena views."""
    if baseline != hybrid_source._fixed_source().native:
        raise ValueError("Compact fallback requires the retained physical recipe")
    source = hybrid_source._fixed_source(shared_arena=True)
    if source.arena_floats != ARENA_FLOATS:
        raise ValueError("Compact arena size differs from the physical fallback")
    return source.native, []


def compact_source(original_kernel):
    """Compose shared dense and current-MF source through explicit bindings."""
    _pin()
    _, baseline, _ = hybrid_source.general_source(original_kernel)
    mf_body = hybrid_source._fixed_source(held=True).mf_phase
    dense = compact_recurrence(mf_body)
    return (
        "#if defined(__CUDA_ARCH__)\n"
        + original.FRICTION_PAIR_CUDA
        + original._T
        + dense
        + "\n#else\n"
        + original._T
        + cpu_source()
        + "\n#endif\n",
        baseline,
        mf_body,
    )


def cpu_source():
    """Use the common scalar law with the six already converted columns."""
    return coupled_cpu_source(converted_secondary=True)


@functools.cache
def get_general_kernel(original_kernel):
    """Bind one guarded queue dispatch to mutually exclusive arena owners."""
    source, baseline, mf_body = compact_source(original_kernel)
    fallback, fallback_edits = fallback_source(baseline)
    extra = {
        "plan": KineticPlan,
        "held": HeldKineticOperator,
        "state": RowState,
        "rows": DenseRowOutput,
        "solve": KineticSolveData,
        "hybrid": HybridCoupledData,
        "arena_address": wp.uint64,
    }
    compact_native = make_native(source, name="kinetic_compact_native", extra=extra)
    fallback_native = make_native(fallback, name="kinetic_original_arena_native", extra={"arena_address": wp.uint64})
    function = make_dispatch(
        original_kernel.mf_native,
        policy="compact",
        extra={
            "guard": guards.BoundaryGuard,
            "plan": KineticPlan,
            "held": HeldKineticOperator,
            "state": RowState,
            "rows": DenseRowOutput,
            "mf": KineticMFData,
            "solve": KineticSolveData,
            "hybrid": HybridCoupledData,
        },
        bindings={
            "_guard_world_check": guards._guard_world_check,
            "kinetic_compact_native": compact_native,
            "kinetic_original_arena_native": fallback_native,
            "_arena": _arena,
            "_is_cpu": original._is_cpu,
        },
        native_extra=tuple(extra),
    )
    result = wp.kernel(function, module="unique", enable_backward=False)
    result.compact_source, result.compact_original_source = source, baseline
    result.compact_mf_source, result.compact_fallback_edits = mf_body, fallback_edits
    result.compact_dispatch_source = function.mf_dispatch_source
    return result


def install(owner):
    """Keep predecessor eligibility/alias checks and replace only two owners."""
    _pin()
    hybrid_source.install(owner)
    owner.kernels["materialize"] = get_materialize_kernel(str(owner.rows.device.arch))
    owner.kernels["general"] = get_general_kernel(owner.services.general)
    owner.compact_coupled = True
    return owner
