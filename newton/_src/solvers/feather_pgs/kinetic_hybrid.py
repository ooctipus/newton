# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

# Preserve the checked original host/source-construction expressions.
# ruff: noqa: RUF005

"""Coupled primary23 kinetic offsets with current physical free6 responses.

This isolated successor replaces existing qualification/materialization/general
dispatches. It does not add a row scan, solve launch, or response panel. Original
MF0/independent ownership and the complete caller stream joins remain unchanged.
"""

import ast
import copy
import functools
import hashlib
import inspect
import linecache
import re

import warp as wp

from . import kinetic_guard as guards
from . import kinetic_solve as original
from . import solver_feather_pgs as solver
from .kinetic_rows_types import DenseRowOutput, RowState
from .kinetic_solve_types import KineticMFData, KineticSolveData
from .kinetic_source import checked_definitions
from .kinetic_types import HeldKineticOperator, KineticPlan


@wp.struct
class HybridCoupledData:
    """Per-call ownership plus aliases to current MF impulse/phase outputs."""

    eligible: wp.array[int]
    mf_impulses: wp.array2d[float]
    mf_contact_end: wp.array[int]


def allocate_hybrid(worlds, mf_impulses, mf_contact_end, device):
    """Allocate only eligibility and alias current MF impulses/contact bounds."""
    device = wp.get_device(device)
    if worlds <= 0:
        raise ValueError("Require positive exact world capacity")
    for array, shape, dtype in (
        (mf_impulses, (worlds, 64), wp.float32),
        (mf_contact_end, (worlds,), wp.int32),
    ):
        if array.shape != shape or array.dtype != dtype or array.device != device or not array.is_contiguous:
            raise ValueError("Hybrid MF alias shape/dtype/device mismatch")
    data = HybridCoupledData()
    data.eligible = wp.zeros(worlds, dtype=int, device=device)
    data.mf_impulses, data.mf_contact_end = mf_impulses, mf_contact_end
    return data


def _pin():
    """Check only the retained representation and original kernel factories."""
    checked_definitions(
        "kinetic_solve.py",
        (
            "_T",
            "_GUARD",
            "_CPU",
            "_MATERIALIZE",
            "cuda_recurrence",
            "qualifier_source",
            "get_solve_kernel",
            "get_qualify_kernel",
            "get_materialize_kernel",
        ),
    )
    checked_definitions("solver_feather_pgs.py", ("_get_pgs_solve_paired_factor_kernel", "_get_pgs_solve_mf_gs_kernel"))
    checked_definitions("independent_components.py", ("get_prepare_kernel",))


def qualify_source():
    """Emit both ownership decisions during the existing predicate visits."""
    _pin()
    source = original.qualifier_source()
    replace = original.replace_once
    source = replace(
        source,
        "if (tid == 0) solve.selector.data[world] = mf_count == 0 ? 0 : wp::max(mf_count, 1);",
        "if (tid == 0) { solve.selector.data[world] = mf_count == 0 ? 0 : wp::max(mf_count, 1); hybrid.eligible.data[world] = 0; }",
    )
    source = replace(source, "if (tid == 0) accepted = 1;", "if (tid == 0) accepted = 3;")
    source = replace(source, "int valid = 1;", "int valid = 1, independent = 1, hybrid_route = 1;")
    source = replace(
        source,
        "if (d >= so && d < so + 6) valid &= j == 0.0f && y == 0.0f;",
        "if (d >= so && d < so + 6) independent &= j == 0.0f && y == 0.0f;",
    )
    source = replace(
        source,
        "const int tp = mf.meta.data[r * 4 + 3], type = tp & 65535;",
        "const int tp = mf.meta.data[r * 4 + 3], type = tp & 65535;\n            hybrid_route &= (a == so) != (b == so);",
    )
    source = replace(
        source,
        "if (!valid) atomicExch(&accepted, 0);",
        "const int keep = valid ? ((independent ? 1 : 0) | (hybrid_route ? 2 : 0)) : 0;\n        if (keep != 3) atomicAnd(&accepted, keep);",
    )
    source = replace(
        source,
        "accepted = valid;",
        "accepted = valid ? ((independent ? 1 : 0) | (hybrid_route ? 2 : 0)) : 0;",
    )
    source = replace(
        source,
        "if (accepted) {\n            if (tid == 0) solve.selector.data[world] = -1;\n        }",
        "if (tid == 0) {\n            if (accepted & 1) solve.selector.data[world] = -1;\n            else if ((accepted & 2) && m > 0 && state.resolved.data[world] == 0) hybrid.eligible.data[world] = 1;\n        }",
    )
    return source


@functools.cache
def get_qualify_kernel(arch):
    """Replace the existing workers×128 qualifier; no additional row pass."""  # noqa: RUF002

    @wp.func_native(qualify_source())
    def qualify(
        worker: int,
        logical_lane: int,
        workers: int,
        plan: KineticPlan,
        held: HeldKineticOperator,
        state: RowState,
        rows: DenseRowOutput,
        mf: KineticMFData,
        solve: KineticSolveData,
        hybrid: HybridCoupledData,
    ): ...

    @wp.kernel(module="unique", enable_backward=False)
    def kinetic_hybrid_qualify(
        plan: KineticPlan,
        held: HeldKineticOperator,
        state: RowState,
        rows: DenseRowOutput,
        mf: KineticMFData,
        solve: KineticSolveData,
        hybrid: HybridCoupledData,
        workers: int,
    ):
        worker, logical = wp.tid()
        qualify(worker, logical, workers, plan, held, state, rows, mf, solve, hybrid)

    return kinetic_hybrid_qualify


@functools.cache
def get_materialize_kernel(arch):
    """Retain exact old physical materialization only for unclaimed fallback."""
    _pin()
    source = "if (hybrid.eligible.data[world] != 0) return;\n" + original._MATERIALIZE

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
    ): ...

    @wp.kernel(module="unique", enable_backward=False)
    def kinetic_hybrid_fallback_materialize(
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

    return kinetic_hybrid_fallback_materialize


_STAGE = r"""
    const int po = state.primary_offset.data[world], so = state.secondary_offset.data[world];
    const int group = plan.secondary_group.data[world];
    __shared__ float hybrid_ys[192 * 6];
    __shared__ float hybrid_base[6];
    bool hybrid_good = true;
    for (int index = lane; index < m_dense * 6; index += 32) {
        const int row = index / 6, col = index % 6;
        float y = 0.0f;
        for (int k = col; k < 6; ++k)
            y += held.inverse6.data[group * 36 + k * 6 + col]
                * rows.response.data[(world * 192 + row) * 29 + so + k];
        hybrid_ys[index] = y;
        hybrid_good &= isfinite(y);
    }
    if (lane < 6) hybrid_base[lane] = state.v_hat.data[plan.dof_ids.data[world * 35 + 23 + lane]];
    if (!__all_sync(MASK, hybrid_good)) { if (lane == 0) solve.status.data[world] = 2; return; }
    __syncwarp(MASK);
    auto hybrid_j = [&](int index) {
        const int coord = index % 29;
        return coord >= po && coord < po + 23 ? rows.response.data[index] : J_world.data[index];
    };
    auto hybrid_y = [&](int index) {
        const int coord = index % 29, row = (index / 29) % 192;
        return coord >= po && coord < po + 23 ? rows.response.data[index] : hybrid_ys[row * 6 + coord - so];
    };
"""

_STORE = r"""
    float hybrid_output = 0.0f;
    int hybrid_global = -1;
    if (lane >= po && lane < po + 23) {
        const int col = lane - po;
        for (int k = 0; k < 23; ++k) hybrid_output += t(k, col) * s_v[po + k];
        hybrid_global = plan.dof_ids.data[world * 35 + col];
    } else if (lane >= so && lane < so + 6) {
        hybrid_output = s_v[lane];
        hybrid_global = plan.dof_ids.data[world * 35 + 23 + lane - so];
    }
    if (hybrid_global >= 0) hybrid_output += state.v_hat.data[hybrid_global];
    hybrid_good = hybrid_global < 0 || isfinite(hybrid_output);
    for (int i = lane; i < m_dense; i += 32) hybrid_good &= isfinite(s_lam_dense[i]);
    for (int i = lane; i < m_mf; i += 32) hybrid_good &= isfinite(s_lam_mf[i]);
    if (!__all_sync(MASK, hybrid_good)) { if (lane == 0) solve.status.data[world] = 2; return; }
    if (hybrid_global >= 0) v_out.data[hybrid_global] = hybrid_output;
    if (lane == 0) solve.status.data[world] = 0;
"""


def general_source(original_kernel):
    """Recover the exact generic recurrence and change only its representation."""
    _pin()
    native = inspect.getclosurevars(original_kernel.func).nonlocals["pgs_solve_mf_gs_native"]
    baseline = native.native_snippet
    expected = solver._get_pgs_solve_mf_gs_kernel(
        192,
        64,
        29,
        "cpu",
        has_drive_rows=False,
        has_dense_velocity_limit_rows=False,
        factor_coordinates=True,
        independent_components=True,
    )
    expected_native = inspect.getclosurevars(expected.func).nonlocals["pgs_solve_mf_gs_native"]
    if baseline != expected_native.native_snippet:
        raise ValueError("Hybrid requires the exact original fixed generic recipe")
    source, edits = baseline, []

    def change(old, new, count=1):
        nonlocal source
        if source.count(old) != count:
            raise RuntimeError("General source seam changed: " + old[:90])
        source = source.replace(old, new)
        edits.append((old, new, count))

    for array, function in (("J_world", "hybrid_j"), ("Y_world", "hybrid_y")):
        for expression in sorted(set(re.findall(r"\b" + array + r"\.data\[([^\]]+)\]", source))):
            old = array + ".data[" + expression + "]"
            change(old, function + "(" + expression + ")", source.count(old))
    change("    int lane = threadIdx.x;", "    int lane = threadIdx.x;\n" + original._T)
    change("    if (m_mf == 0) return;", "    if (m_mf == 0) return;\n" + _STAGE)
    change(
        "s_rhs_dense[i] = rhs_bias.data[off_dense + i];",
        "s_rhs_dense[i] = rows.r0.data[off_dense + i];",
    )
    old = "s_v[d] = global_dof >= 0 && (!split_component || (d >= component_offset && d < component_offset + 6)) ? v_out.data[global_dof] : 0.0f;"
    change(old, "s_v[d] = 0.0f;")
    # MF reads physical free velocities, never the held dense inverse action.
    for expression, base in (
        ("dof_a + lane", "lane"),
        ("dof_b + lane - 6", "lane - 6"),
    ):
        old = "* s_v[" + expression + "]"
        change(
            old,
            "* (s_v[" + expression + "] + hybrid_base[" + base + "])",
            source.count(old),
        )
    old = """    for (int d = lane; d < 29; d += 32) {
        int global_dof = world_dof_indices.data[dof_map_base + d];
        if (global_dof >= 0 && (!split_component || (d >= component_offset && d < component_offset + 6))) v_out.data[global_dof] = s_v[d];
    }"""
    change(old, _STORE)
    recovered = source
    for old, new, count in reversed(edits):
        if recovered.count(new) != count:
            raise RuntimeError("Ambiguous reverse source seam")
        recovered = recovered.replace(new, old)
    if recovered != baseline:
        raise RuntimeError("Representation removal did not recover exact original")
    return source, baseline, edits


_CPU = r"""
    const int m = world_constraint_count.data[world], n = mf_constraint_count.data[world];
    const int po = state.primary_offset.data[world], so = state.secondary_offset.data[world];
    const int group = plan.secondary_group.data[world], mend = mf_contact_rows_end.data[world];
    float du[29] = {}, lam[192], ml[64], ys[192 * 6], base[6];
    bool good = true;
    for (int k = 0; k < 6; ++k) base[k] = state.v_hat.data[plan.dof_ids.data[world * 35 + 23 + k]];
    for (int r = 0; r < m; ++r) {
        lam[r] = world_impulses.data[world * 192 + r];
        for (int col = 0; col < 6; ++col) {
            float y = 0.0f;
            for (int k = col; k < 6; ++k) y += held.inverse6.data[group * 36 + k * 6 + col]
                * rows.response.data[(world * 192 + r) * 29 + so + k];
            ys[r * 6 + col] = y; good &= isfinite(y);
        }
    }
    for (int r = 0; r < n; ++r) ml[r] = mf_impulses.data[world * 64 + r];
    if (!good) { solve.status.data[world] = 2; return; }
    auto dense_update = [&](int row, float delta) {
        if (delta != 0.0f) for (int d = 0; d < 29; ++d) {
            const float y = d >= po && d < po + 23 ? rows.response.data[(world * 192 + row) * 29 + d] : ys[row * 6 + d - so];
            du[d] += y * delta;
        }
    };
    auto mf_update = [&](int row, float delta) {
        if (delta == 0.0f) return;
        const int id = world * 64 + row, packed = mf_meta.data[id * 4];
        const int a = packed >> 16, b = static_cast<int>(static_cast<short>(packed & 65535));
        for (int k = 0; k < 6; ++k) {
            if (a >= 0) du[a + k] += mf_MiJt_a.data[id * 6 + k] * delta;
            if (b >= 0) du[b + k] += mf_MiJt_b.data[id * 6 + k] * delta;
        }
    };
    for (int iter = 0; iter < iterations; ++iter) {
        bool changed = false; const int global_iter = iter + iteration_offset;
        for (int r = 0; r < m; ++r) {
            const int id = world * 192 + r, type = world_row_type.data[id];
            if (type == 2 && global_iter < friction_start_iteration) { lam[r] = 0.0f; continue; }
            const float denom = world_diag.data[id]; if (denom <= 0.0f) continue;
            float dot = 0.0f;
            for (int d = 0; d < 29; ++d) {
                const float j = d >= po && d < po + 23 ? rows.response.data[id * 29 + d] : J_world.data[id * 29 + d];
                dot += j * du[d];
            }
            const float residual = dot + rows.r0.data[id], old = lam[r];
            float next = old - residual / denom, sibling_delta = 0.0f; int sibling = -1;
            if (type == 0 || type == 3) next = wp::max(next, 0.0f);
            else if (type == 2) {
                const int p = world_row_parent.data[id]; sibling = r == p + 1 ? p + 2 : p + 1;
                const float radius = wp::max(world_row_mu.data[id] * lam[p], 0.0f);
                if (radius <= 0.0f) next = 0.0f;
                else {
                    const float length = sqrtf(next * next + lam[sibling] * lam[sibling]);
                    if (length > radius) {
                        const float scale = radius / length; next *= scale;
                        const float other = lam[sibling] * scale; sibling_delta = other - lam[sibling]; lam[sibling] = other;
                    }
                }
            }
            if (sibling_delta != 0.0f) { changed = true; dense_update(sibling, sibling_delta); }
            lam[r] = next; const float delta = next - old;
            if (delta != 0.0f) { changed = true; dense_update(r, delta); }
        }
        for (int pass = 0; pass < 2; ++pass) {
            const int lo = pass == 0 ? 0 : mend, hi = n;
            for (int r = lo; r < hi; ++r) {
                const int id = world * 64 + r, packed = mf_meta.data[id * 4], tp = mf_meta.data[id * 4 + 3];
                const int a = packed >> 16, b = static_cast<int>(static_cast<short>(packed & 65535)), type = tp & 65535;
                if ((pass == 0 && type == 4) || (pass == 1 && type != 4)) continue;
                if (type == 2 && global_iter < friction_start_iteration) { ml[r] = 0.0f; continue; }
                union { int bits; float value; } inv, rhs;
                inv.bits = mf_meta.data[id * 4 + 1]; rhs.bits = mf_meta.data[id * 4 + 2];
                if (inv.value <= 0.0f) continue;
                float dot = 0.0f;
                for (int k = 0; k < 6; ++k) {
                    if (a >= 0) dot += mf_J_a.data[id * 6 + k] * (du[a + k] + base[k]);
                    if (b >= 0) dot += mf_J_b.data[id * 6 + k] * (du[b + k] + base[k]);
                }
                const float residual = dot + rhs.value, delta0 = -residual * inv.value, old = ml[r];
                float next = old + delta0, delta = 0.0f;
                if (type == 4) { delta = residual < 0.0f ? delta0 : 0.0f; next = delta; }
                else if (type == 0) next = wp::max(next, 0.0f);
                else if (type == 2) {
                    const int p = tp >> 16, sibling = r == p + 1 ? p + 2 : p + 1;
                    const float radius = wp::max(mf_row_mu.data[id] * ml[p], 0.0f);
                    if (radius <= 0.0f) next = 0.0f;
                    else {
                        const float length = sqrtf(next * next + ml[sibling] * ml[sibling]);
                        if (length > radius) {
                            const float scale = radius / length; next *= scale;
                            const float other = ml[sibling] * scale, sd = other - ml[sibling]; ml[sibling] = other;
                            if (sd != 0.0f) { changed = true; mf_update(sibling, sd); }
                        }
                    }
                }
                if (type != 4) delta = next - old;
                ml[r] = next;
                if (delta != 0.0f) { changed = true; mf_update(r, delta); }
            }
        }
        if (global_iter >= friction_start_iteration && !changed) break;
    }
    float output[29];
    for (int d = 0; d < 29; ++d) {
        float delta = 0.0f;
        if (d < 23) for (int k = 0; k < 23; ++k) delta += t(k, d) * du[po + k];
        else delta = du[so + d - 23];
        output[d] = state.v_hat.data[plan.dof_ids.data[world * 35 + d]] + delta;
        good &= isfinite(output[d]);
    }
    for (int r = 0; r < m; ++r) good &= isfinite(lam[r]);
    for (int r = 0; r < n; ++r) good &= isfinite(ml[r]);
    if (!good) { solve.status.data[world] = 2; return; }
    for (int d = 0; d < 29; ++d) v_out.data[plan.dof_ids.data[world * 35 + d]] = output[d];
    for (int r = 0; r < m; ++r) world_impulses.data[world * 192 + r] = lam[r];
    for (int r = 0; r < n; ++r) mf_impulses.data[world * 64 + r] = ml[r];
    solve.status.data[world] = 0;
"""


def _emit(tree, function, namespace, label):
    """Compile inspectable generated Python while keeping source files frozen."""
    ast.fix_missing_locations(tree)
    source = ast.unparse(tree) + "\n"
    filename = "<kinetic-hybrid-" + label + "-" + hashlib.sha256(source.encode()).hexdigest() + ">"
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace["__name__"] = __name__
    exec(compile(source, filename, "exec"), namespace)
    return namespace[function.name], source


@functools.cache
def get_general_kernel(original_kernel):
    """Replace only the existing guarded general dispatch with two owners."""
    source, baseline, edits = general_source(original_kernel)
    source = source.rsplit("#endif", 1)[0] + "#else\n" + original._T + _CPU + "\n#endif\n"
    old_native = inspect.getclosurevars(original_kernel.func).nonlocals["pgs_solve_mf_gs_native"]
    tree, function, namespace = guards._tree(old_native)
    function.name = "kinetic_hybrid_native"
    extra = {
        "plan": KineticPlan,
        "held": HeldKineticOperator,
        "state": RowState,
        "rows": DenseRowOutput,
        "solve": KineticSolveData,
        "hybrid": HybridCoupledData,
    }
    for name, annotation in extra.items():
        function.args.args.append(ast.arg(arg=name, annotation=ast.Name(id=annotation.__name__)))
        namespace[annotation.__name__] = annotation
    native_function, _ = _emit(tree, function, namespace, "native")
    new_native = wp.func_native(source)(native_function)
    guarded = guards.get_general_kernel(original_kernel)
    tree, function, namespace = guards._tree(guarded)
    before = ast.dump(function, include_attributes=False)
    old_name = function.name
    function.name = old_name + "_hybrid"
    public_extra = {
        **dict(list(extra.items())[:4]),
        "mf": KineticMFData,
        "solve": KineticSolveData,
        "hybrid": HybridCoupledData,
    }
    for name, annotation in public_extra.items():
        function.args.args.append(ast.arg(arg=name, annotation=ast.Name(id=annotation.__name__)))
        namespace[annotation.__name__] = annotation
    call = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "pgs_solve_mf_gs_native"
    )
    parent = next(node for node in ast.walk(function) if call in getattr(node, "body", []))
    branch = ast.If(
        test=ast.parse("hybrid.eligible[world] != 0", mode="eval").body,
        body=[
            ast.Expr(
                value=ast.Call(
                    func=ast.Name(id="kinetic_hybrid_native"),
                    args=call.value.args + [ast.Name(id=name) for name in extra],
                    keywords=[],
                )
            )
        ],
        orelse=[call],
    )
    parent.body[parent.body.index(call)] = branch
    cpu_guard = ast.parse("if _is_cpu() and _lane != 0:\n    return\n").body[0]
    function.body.insert(1, cpu_guard)
    recovered = copy.deepcopy(function)
    del recovered.body[1]
    target = next(
        node
        for node in ast.walk(recovered)
        if isinstance(node, ast.If) and ast.dump(node.test) == ast.dump(branch.test)
    )
    parent = next(node for node in ast.walk(recovered) if target in getattr(node, "body", []))
    parent.body[parent.body.index(target)] = target.orelse[0]
    recovered.name = old_name
    del recovered.args.args[-len(public_extra) :]
    if ast.dump(recovered, include_attributes=False) != before:
        raise RuntimeError("Dispatch removal failed exact guarded source recovery")
    namespace.update(kinetic_hybrid_native=new_native, _is_cpu=original._is_cpu)
    function, python_source = _emit(tree, function, namespace, "dispatch")
    result = wp.kernel(function, module="unique", enable_backward=False)
    result.hybrid_source, result.hybrid_original_source, result.hybrid_edits = (
        source,
        baseline,
        edits,
    )
    result.hybrid_dispatch_source = python_source
    result.hybrid_recovered_ast = before
    return result


def install(owner):
    """Replace existing owner dispatches and expose every new mutable array."""
    if hasattr(owner, "hybrid"):
        raise ValueError("Hybrid already installed")
    owner.join()
    rows, services, solve = owner.rows, owner.services, owner.solve_descriptor
    worlds, device = rows.worlds, rows.device
    if rows.plan.dof_ids.shape != (worlds, 35) or rows.plan.body_ids.shape != (
        worlds,
        32,
    ):
        raise ValueError("Require checked Kuka23 + respondingfree6 + prescribed6 plan")
    if rows.out.response.shape != (worlds, 192, 29) or rows.state.mf_count.shape != (worlds,):
        raise ValueError("Unsupported dense/MF capacity")
    if (solve.iterations, solve.omega, solve.iteration_offset) != (8, 1.0, 0):
        raise ValueError("Require original finite-eight recipe")
    values = dict(
        zip(
            inspect.signature(services.general.func).parameters,
            services.general_args,
            strict=True,
        )
    )
    for name, expected in (
        ("regularize", 0),
        ("row_phase", 0),
        ("freeze_drive_rows", 0),
        ("defer_dense_response", 0),
        ("iterations", 8),
        ("omega", 1.0),
        ("iteration_offset", 0),
        ("use_general_world_queue", 0),
    ):
        if values[name] != expected:
            raise ValueError("Unsupported original general setting: " + name)
    if solve.friction_start_iteration < 0 or values["friction_start_iteration"] != solve.friction_start_iteration:
        raise ValueError("Dense and generic friction schedules must match")
    aliases = {
        "world_constraint_count": rows.state.dense_count,
        "mf_constraint_count": rows.state.mf_count,
        "local_solve_owner": solve.selector,
        "v_out": solve.v_out,
        "J_world": rows.out.physical_J,
        "Y_world": rows.out.response,
        "rhs_bias": rows.out.rhs,
        "world_diag": rows.out.diag,
        "world_impulses": rows.out.impulses,
        "mf_impulses": services.impulses,
        "mf_contact_rows_end": services.contact_end,
    }
    for name, expected in aliases.items():
        value = values[name]
        if (
            value.ptr != expected.ptr
            or value.shape != expected.shape
            or value.dtype != expected.dtype
            or value.device != expected.device
            or not value.is_contiguous
        ):
            raise ValueError("Original general alias mismatch: " + name)
    for name in KineticMFData.vars:
        expected = getattr(services, "row_mu" if name == "mu" else name)
        value = getattr(owner.mf_descriptor, name)
        if value.ptr != expected.ptr or value.shape != expected.shape:
            raise ValueError("Qualifier and generic MF aliases differ: " + name)
    hybrid = allocate_hybrid(worlds, services.impulses, services.contact_end, device)
    owner.kernels["qualify"] = get_qualify_kernel(str(device.arch))
    owner.arguments["qualify"] = [
        rows.plan,
        rows.held,
        rows.state,
        rows.out,
        owner.mf_descriptor,
        solve,
        hybrid,
        min(worlds, 512),
    ]
    owner.kernels["materialize"] = get_materialize_kernel(str(device.arch))
    owner.arguments["materialize"] = [
        rows.plan,
        rows.held,
        rows.state,
        rows.out,
        solve,
        hybrid,
    ]
    owner.kernels["general"] = get_general_kernel(services.general)
    owner.arguments["general"] = services.general_args + [
        owner.guard,
        rows.plan,
        rows.held,
        rows.state,
        rows.out,
        owner.mf_descriptor,
        solve,
        hybrid,
    ]
    for name in ("qualify", "materialize", "general"):
        if len(inspect.signature(owner.kernels[name].func).parameters) != len(owner.arguments[name]):
            raise RuntimeError("Installed hybrid ABI mismatch: " + name)
    owner.hybrid = hybrid
    owner.hybrid_mutable_arrays = {"hybrid.eligible": hybrid.eligible}
    return owner
