# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

# Preserve the checked original host/source-construction expressions.
# ruff: noqa: RUF005

"""One compact dense/current-MF recurrence with a shared fallback arena.

Eligible worlds retain primary kinetic rows, materialize six held physical
response columns, and own dense -> current MF -> rigid velocity limits in each
original sweep. Qualification, dispatch count, unsupported fallback arithmetic,
and the caller's stream joins are unchanged. Snapshot conversion is host-only.
"""

import ast
import copy
import functools
import inspect
import re

import warp as wp

from . import kinetic_guard as guards
from . import kinetic_hybrid as hybrid_source
from . import kinetic_solve as original
from .kinetic_rows_types import DenseRowOutput, RowState
from .kinetic_solve_types import KineticMFData, KineticSolveData
from .kinetic_source import checked_definitions
from .kinetic_types import HeldKineticOperator, KineticPlan

HybridCoupledData = hybrid_source.HybridCoupledData
allocate_hybrid = hybrid_source.allocate_hybrid
get_qualify_kernel = hybrid_source.get_qualify_kernel

ARENA_FLOATS = 1053


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
    """Replace only old shared declarations with exact nonoverlapping views."""
    source, changes, offset = baseline, [], 0
    declarations = list(re.finditer(r"__shared__\s+(float|int)\s+(\w+)\[(\d+)\];", baseline))
    if len(declarations) != 7:
        raise RuntimeError("Original shared inventory changed")
    for match in declarations:
        dtype, name, count = match.groups()
        replacement = f"{dtype}* {name}=reinterpret_cast<{dtype}*>(arena_address)+{offset};"
        source = original.replace_once(source, match.group(0), replacement)
        changes.append((match.group(0), replacement))
        offset += int(count)
    if offset != ARENA_FLOATS:
        raise RuntimeError("Original arena size changed")
    recovered = source
    for before, after in reversed(changes):
        recovered = original.replace_once(recovered, after, before)
    if recovered != baseline:
        raise RuntimeError("Fallback arithmetic recovery failed")
    return source, changes


_VIEWS = r"""
    float* arena=reinterpret_cast<float*>(arena_address);
    float* s_lam=arena; float* s_rhs=arena+192; float* s_diag=arena+384;
    unsigned char* s_type=reinterpret_cast<unsigned char*>(arena+576);
    float* s_contact_mu=arena+624; float* s_lam_mf=arena+688;
    float* s_v=arena+752; float* hybrid_base=arena+781;
"""


def compact_source(original_kernel):
    """Compose unchanged dense triple law and exact current MF/vlim blocks."""
    _pin()
    old_hybrid, baseline, _ = hybrid_source.general_source(original_kernel)
    mf_begin = old_hybrid.index("        // ── Phase 2: MF constraints")
    mf_end = old_hybrid.index("        if (global_iter >= friction_start_iteration)", mf_begin)
    mf_body = old_hybrid[mf_begin:mf_end]
    # The complete dense law comes from the already accepted offset owner.
    dense = original.cuda_recurrence()
    dense = dense[dense.index("    __shared__ float s_lam_storage") :]
    dense = re.sub(r"    __shared__ [^;]+;\n", "", dense)
    dense = original.replace_once(dense, "    const int world_slot = threadIdx.x >> 5;\n", "")
    begin = dense.index("    float* s_lam =")
    end = dense.index("    int contact_start =", begin)
    dense = dense[:begin] + _VIEWS + dense[end:]
    for name, value in {
        "primary_group_by_world": "plan.primary_group",
        "secondary_group_by_world": "plan.secondary_group",
        "primary_offset_by_world": "state.primary_offset",
        "secondary_offset_by_world": "state.secondary_offset",
        "factor_rows": "rows.response",
    }.items():
        dense = dense.replace(name + ".data", value + ".data")
    dense = dense.replace("rhs_bias.data[off + i]", "rows.r0.data[off + i]")
    # A response and a residual coincide on primary Z but not free physical Y.
    # Keep every update factor unchanged; replace only residual multiplication.
    for factor, row in (
        ("row_factor", "i"),
        ("normal_factor", "normal"),
        ("tangent1_factor", "tangent1"),
        ("tangent2_factor", "tangent2"),
    ):
        dense = dense.replace(f"{factor} * factor_velocity", f"dense_j({row}, {factor}) * factor_velocity")
    start = dense.index("    const int primary_lane=")
    dense = (
        dense[:start]
        + r"""
    const int po=primary_offset,so=secondary_offset;
    const bool split_component=false;
    const int m_mf=mf_constraint_count.data[world];
    int mf_contact_end=mf_contact_rows_end.data[world];
    const int off_mf=world*64,off_meta=off_mf*4,mf6_base=off_mf*6;
    if(m_mf<0||m_mf>64||mf_contact_end<0||mf_contact_end>m_mf){if(lane==0)solve.status.data[world]=1;return;}
    for(int i=lane;i<m_mf;i+=32)s_lam_mf[i]=mf_impulses.data[off_mf+i];
    if(lane<6)hybrid_base[lane]=state.v_hat.data[plan.dof_ids.data[world*35+23+lane]];
    auto dense_j=[&](int row,float y){return lane>=po&&lane<po+23?y:J_world.data[(world*192+row)*29+lane];};
"""
        + dense[start:]
    )
    seam = "        const unsigned changed = __ballot_sync(MASK, iteration_changed != 0);"
    dense = original.replace_once(
        dense,
        seam,
        r"""
        // Only six physical increments cross this phase boundary; primary du
        // stays in registers. The original MF and rigid-limit source is exact.
        if(lane>=so&&lane<so+6)s_v[lane]=factor_velocity;
        __syncwarp(MASK);
"""
        + mf_body
        + r"""
        __syncwarp(MASK);
        if(lane>=so&&lane<so+6)factor_velocity=s_v[lane];
"""
        + seam,
    )
    begin = dense.index("    for(int k=0;k<6;++k){const float du=")
    end = dense.index("    const bool finite=", begin)
    dense = (
        dense[:begin] + "    if(secondary_lane>=0&&secondary_lane<6)physical_output=factor_velocity;\n" + dense[end:]
    )
    dense = original.replace_once(dense, "    const bool finite=", "    bool finite=")
    dense = original.replace_once(
        dense,
        "    if(!__all_sync(MASK,finite))",
        "    for(int i=lane;i<m;i+=32)finite &= isfinite(s_lam[i]);\n    for(int i=lane;i<m_mf;i+=32)finite &= isfinite(s_lam_mf[i]);\n    if(!__all_sync(MASK,finite))",
    )
    dense += "\n    for(int i=lane;i<m_mf;i+=32)mf_impulses.data[off_mf+i]=s_lam_mf[i];\n"
    return (
        "#if defined(__CUDA_ARCH__)\n" + original._T + dense + "\n#else\n" + original._T + cpu_source() + "\n#endif\n",
        baseline,
        mf_body,
    )


def cpu_source():
    """Keep the predecessor scalar law, consume the six converted columns."""
    source = hybrid_source._CPU
    begin = source.index("        for (int col = 0; col < 6; ++col) {")
    end = source.index("    for (int r = 0; r < n;", begin)
    source = (
        source[:begin]
        + r"""
        for(int col=0;col<6;++col){const float y=rows.response.data[(world*192+r)*29+so+col];ys[r*6+col]=y;good &= isfinite(y);}
    }
"""
        + source[end:]
    )
    return source


def _native(old_native, source, name, extra):
    """Add private-only arguments to a checked native descriptor."""
    tree, function, namespace = guards._tree(old_native)
    function.name = name
    for field, annotation in extra.items():
        function.args.args.append(ast.arg(arg=field, annotation=ast.Name(id=annotation.__name__)))
        namespace[annotation.__name__] = annotation
    function, _ = hybrid_source._emit(tree, function, namespace, name)
    return wp.func_native(source)(function)


@functools.cache
def get_general_kernel(original_kernel):
    """Replace one existing dispatch with mutually exclusive arena owners."""
    source, baseline, mf_body = compact_source(original_kernel)
    fallback, fallback_edits = fallback_source(baseline)
    old_native = inspect.getclosurevars(original_kernel.func).nonlocals["pgs_solve_mf_gs_native"]
    extra = {
        "plan": KineticPlan,
        "held": HeldKineticOperator,
        "state": RowState,
        "rows": DenseRowOutput,
        "solve": KineticSolveData,
        "hybrid": HybridCoupledData,
        "arena_address": wp.uint64,
    }
    compact_native = _native(old_native, source, "kinetic_compact_native", extra)
    fallback_native = _native(
        old_native,
        fallback,
        "kinetic_original_arena_native",
        {"arena_address": wp.uint64},
    )
    guarded = guards.get_general_kernel(original_kernel)
    tree, function, namespace = guards._tree(guarded)
    before = ast.dump(function, include_attributes=False)
    old_name = function.name
    function.name += "_compact"
    public_extra = {
        "plan": KineticPlan,
        "held": HeldKineticOperator,
        "state": RowState,
        "rows": DenseRowOutput,
        "mf": KineticMFData,
        "solve": KineticSolveData,
        "hybrid": HybridCoupledData,
    }
    for name, annotation in public_extra.items():
        function.args.args.append(ast.arg(arg=name, annotation=ast.Name(id=annotation.__name__)))
        namespace[annotation.__name__] = annotation
    call = next(
        n
        for n in ast.walk(function)
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name)
        and n.value.func.id == "pgs_solve_mf_gs_native"
    )
    parent = next(n for n in ast.walk(function) if call in getattr(n, "body", []))
    old_call = copy.deepcopy(call)
    branch = ast.If(
        test=ast.parse("hybrid.eligible[world] != 0", mode="eval").body,
        body=[
            ast.Expr(
                value=ast.Call(
                    func=ast.Name(id="kinetic_compact_native"),
                    args=call.value.args + [ast.Name(id=n) for n in extra],
                    keywords=[],
                )
            )
        ],
        orelse=[
            ast.Expr(
                value=ast.Call(
                    func=ast.Name(id="kinetic_original_arena_native"),
                    args=call.value.args + [ast.Name(id="arena_address")],
                    keywords=[],
                )
            )
        ],
    )
    index = parent.body.index(call)
    assignment = ast.parse("arena_address = _arena()\n").body[0]
    parent.body[index : index + 1] = [assignment, branch]
    cpu_guard = ast.parse("if _is_cpu() and _lane != 0:\n    return\n").body[0]
    function.body.insert(1, cpu_guard)
    recovered = copy.deepcopy(function)
    del recovered.body[1]
    target = next(n for n in ast.walk(recovered) if isinstance(n, ast.If) and ast.dump(n.test) == ast.dump(branch.test))
    parent = next(n for n in ast.walk(recovered) if target in getattr(n, "body", []))
    index = parent.body.index(target)
    parent.body[index - 1 : index + 1] = [old_call]
    recovered.name = old_name
    del recovered.args.args[-len(public_extra) :]
    if ast.dump(recovered, include_attributes=False) != before:
        raise RuntimeError("Exact original dispatch recovery failed")
    namespace.update(
        kinetic_compact_native=compact_native,
        kinetic_original_arena_native=fallback_native,
        _arena=_arena,
        _is_cpu=original._is_cpu,
    )
    function, dispatch = hybrid_source._emit(tree, function, namespace, "compact-dispatch")
    result = wp.kernel(function, module="unique", enable_backward=False)
    result.compact_source, result.compact_original_source = source, baseline
    result.compact_mf_source, result.compact_fallback_edits = mf_body, fallback_edits
    result.compact_dispatch_source = dispatch
    return result


def install(owner):
    """Keep predecessor eligibility/alias checks and replace only two owners."""
    _pin()
    hybrid_source.install(owner)
    owner.kernels["materialize"] = get_materialize_kernel(str(owner.rows.device.arch))
    owner.kernels["general"] = get_general_kernel(owner.services.general)
    owner.compact_coupled = True
    return owner
