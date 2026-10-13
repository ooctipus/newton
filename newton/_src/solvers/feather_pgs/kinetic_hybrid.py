# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

# Preserve the checked original host/source-construction expressions.
# ruff: noqa: RUF005

"""Coupled primary23 kinetic offsets with current physical free6 responses.

This isolated successor replaces existing qualification/materialization/general
dispatches. It does not add a row scan, solve launch, or response panel. Original
MF0/independent ownership and the complete caller stream joins remain unchanged.
"""

import functools
import inspect

import warp as wp

from . import kinetic_guard as guards
from . import kinetic_solve as original
from .kinetic_coupled_cpu import get_source as _coupled_cpu_source
from .kinetic_rows_types import DenseRowOutput, RowState
from .kinetic_solve_types import KineticMFData, KineticSolveData
from .kinetic_source import checked_definitions
from .kinetic_types import HeldKineticOperator, KineticPlan
from .mf_dispatch import make_dispatch, make_native


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
    checked_definitions("friction.py", ("_PAIR_METRIC", "_PAIR_STATE", "_PAIR_ENTRY", "get_pair_source"))
    checked_definitions("kinetic_coupled_cpu.py", ("_HELD_SECONDARY", "_CONVERTED_SECONDARY", "_BODY", "get_source"))
    checked_definitions(
        "compact_recurrence_source.py", ("_VIEWS", "_ENTRY", "_BEFORE_MF", "_AFTER_MF", "_STORE_MF", "get_source")
    )
    checked_definitions("mf_gs_source.py", ("GeneratedSource", "get_source"))
    checked_definitions(
        "row_metadata.py",
        (
            "_DENSE_META_ROW_TYPE_BITS",
            "_DENSE_META_ROW_TYPE_MASK",
            "_DENSE_META_MAX_PARENT",
            "_MF_META_MAX_CAPACITY",
            "_validate_dense_metadata_encoding",
            "_validate_mf_metadata_encoding",
            "_mf_parent_expression",
        ),
    )
    checked_definitions(
        "mf_dispatch.py",
        ("_NATIVE_ARGUMENTS", "_QUEUE_ARGUMENTS", "_compile", "_signature", "make_native", "make_dispatch"),
    )
    checked_definitions(
        "paired_factor_source.py",
        (
            "_KINETIC_ENTRY",
            "_KINETIC_ENTRY_BODY",
            "_KINETIC_EXIT",
            "_KINETIC_ALIASES",
            "_KINETIC_SECONDARY",
            "_KINETIC_EXIT_BODY",
            "_kinetic_exit",
            "_physical_entry",
            "_physical_exit",
            "get_source",
        ),
    )
    checked_definitions("component_qualification.py", ("_PHYSICAL_PUBLISH", "get_source"))
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
    """Emit independent and coupled ownership during the shared predicate visits."""
    from .component_qualification import get_source  # noqa: PLC0415

    return get_source("hybrid")


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


def _fixed_source(*, held=False, cpu=False, shared_arena=False):
    """Build the admitted recipe directly from common numerical source."""
    from .mf_gs_source import get_source  # noqa: PLC0415

    return get_source(
        192,
        64,
        29,
        "cpu",
        has_drive_rows=False,
        has_dense_velocity_limit_rows=False,
        factor_coordinates=True,
        independent_components=True,
        held_policy=(original._T, _STAGE, _STORE) if held else None,
        shared_arena=shared_arena,
        cpu_body=original._T + _CPU if cpu else "",
    )


def general_source(original_kernel):
    """Check the retained recipe and bind explicit held-coordinate policies."""
    _pin()
    expected = _fixed_source().native
    if original_kernel.mf_native.native_snippet != expected:
        raise ValueError("Hybrid requires the exact original fixed generic recipe")
    return _fixed_source(held=True).native, expected, []


_CPU = _coupled_cpu_source()


@functools.cache
def get_general_kernel(original_kernel):
    """Bind guarded general dispatch to mutually exclusive shared owners."""
    source, baseline, edits = general_source(original_kernel)
    source = _fixed_source(held=True, cpu=True).native
    extra = {
        "plan": KineticPlan,
        "held": HeldKineticOperator,
        "state": RowState,
        "rows": DenseRowOutput,
        "solve": KineticSolveData,
        "hybrid": HybridCoupledData,
    }
    new_native = make_native(source, name="kinetic_hybrid_native", extra=extra)
    function = make_dispatch(
        original_kernel.mf_native,
        policy="hybrid",
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
            "kinetic_hybrid_native": new_native,
            "_is_cpu": original._is_cpu,
        },
        native_extra=tuple(extra),
    )
    result = wp.kernel(function, module="unique", enable_backward=False)
    result.hybrid_source, result.hybrid_original_source, result.hybrid_edits = source, baseline, edits
    result.hybrid_dispatch_source = function.mf_dispatch_source
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
