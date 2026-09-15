# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Default-off finalized-MF0 private response and early eight-sweep owner.

The original prefix, cached triplet and offset recurrence are retained. Only
their coefficient storage and the MF0 scheduling boundary change. Positive-MF
worlds retain the original row/qualification/coupled/general implementation.
"""

import ast
import copy
import functools
import inspect
from types import SimpleNamespace

import warp as wp

from . import kinetic_compact_coupled as compact
from . import kinetic_current_contact as contact
from . import kinetic_guard as guards
from . import kinetic_hybrid as hybrid
from . import kinetic_rows as rows_source
from . import kinetic_solve as solve_source
from .kinetic_current_contact import ContactGeometry
from .kinetic_rows_types import ArmMap, DenseRowOutput, PrefixInput, RawRowInput, RowSettings, RowState
from .kinetic_solve_types import KineticSolveData
from .kinetic_types import CurrentKineticCache, HeldKineticOperator, KineticPlan

PANEL_FLOATS = 192 * 29
ARENA_FLOATS = 6276


@wp.func_native(r"""
#if defined(__CUDA_ARCH__)
    __shared__ float private_response_arena[6276];
    return reinterpret_cast<uint64_t>(private_response_arena);
#else
    return reinterpret_cast<uint64_t>(malloc(6276*sizeof(float)));
#endif
""")
def _storage() -> wp.uint64: ...


@wp.func_native(r"""
#if defined(__CUDA_ARCH__)
    __syncwarp(0xffffffffu);
    return __all_sync(0xffffffffu, out.status.data[world]==0)!=0;
#else
    return out.status.data[world]==0;
#endif
""")
def _good(world: int, out: DenseRowOutput) -> bool: ...


_VIEWS = r"""
    float* arena=reinterpret_cast<float*>(address);
    float* z=arena+288;
    const float* private_t=arena+5856;
    const float* private_inverse=arena+6036;
"""

_MOTION_VIEWS = r"""
    auto* private_axes=reinterpret_cast<decltype(current.axes.data)>(arena+6072);
    auto* private_free_axes=reinterpret_cast<decltype(current.free_axes.data)>(arena+6210);
    auto* private_origin=reinterpret_cast<decltype(current.origin.data)>(arena+6246);
    auto* private_c=reinterpret_cast<wp::vec3*>(arena+6255);
"""


def _operator_local(source):
    """Change only explicit held/current coefficient loads to owned views."""
    source = source.replace("held.T.data[world*180+", "private_t[")
    source = source.replace("held.inverse6.data[group*36+", "private_inverse[")
    source = source.replace("held.inverse6.data[secondary_group*36+", "private_inverse[")
    source = source.replace("current.axes.data[world*23+", "private_axes[")
    source = source.replace("current.free_axes.data[world*6+", "private_free_axes[")
    source = source.replace("current.origin.data[world*3+", "private_origin[")
    source = source.replace("current.origin.data[world*3]", "private_origin[0]")
    source = source.replace("arm.C.data[world*7+", "private_c[")
    return source


_BEGIN = (
    _VIEWS
    + _MOTION_VIEWS
    + rows_source._COMMON
    + r"""
    const int count=state.dense_count.data[world];
    bool good=state.resolved.data[world]==0 && state.mf_count.data[world]==0
        && count>=0 && count<=192 && settings.dense_capacity==192
        && state.predictor_status.data[world]==0 && current.valid.data[world]!=0
        && current.generation.data[world]==state.state_generation.data[world]
        && held.valid.data[world]!=0 && held.generation.data[world]==state.held_generation.data[world]
        && state.raw_invalid.data[0]==0 && solve.iterations==8 && solve.omega==1.0f
        && solve.iteration_offset==0 && solve.friction_start_iteration>=0;
    for(int k=0;k<4;++k)good=good && state.capacity_status.data[k]==0;
    if(lane==0){out.status.data[world]=good?0:2;out.secondary_nonzero.data[world]=0;
        out.slot_counter.data[world]=0;solve.status.data[world]=good?0:1;}
    if(!all(good))return false;
    const int group=plan.secondary_group.data[world];
    for(int k=lane;k<180;k+=stride)arena[5856+k]=held.T.data[world*180+k];
    for(int k=lane;k<36;k+=stride)arena[6036+k]=held.inverse6.data[group*36+k];
    for(int d=lane;d<23;d+=stride)private_axes[d]=current.axes.data[world*23+d];
    for(int d=lane;d<6;d+=stride)private_free_axes[d]=current.free_axes.data[world*6+d];
    for(int d=lane;d<3;d+=stride)private_origin[d]=current.origin.data[world*3+d];
    for(int row=lane;row<count;row+=stride){
        out.valid.data[world*192+row]=0;out.impulses.data[world*192+row]=0.0f;
    }
    sync();
    for(int row=lane;row<7;row+=stride){auto value=wp::vec3(0.0f);
        for(int col=0;col<=row;++col){const auto axis=private_axes[col];
            value+=t(world,row,col)*wp::vec3(axis[3],axis[4],axis[5]);}
        private_c[row]=value;
    }
    sync();return true;
"""
)
# Cache construction's original load expressions stay global; only its C-map
# reads use the cache that this same warp has just completed.
_BEGIN = contact._replace(_BEGIN, rows_source._COMMON, _operator_local(rows_source._COMMON))


@wp.func_native(_BEGIN)
def _begin(
    address: wp.uint64,
    world: int,
    plan: KineticPlan,
    current: CurrentKineticCache,
    held: HeldKineticOperator,
    state: RowState,
    settings: RowSettings,
    out: DenseRowOutput,
    solve: KineticSolveData,
) -> bool: ...


def prefix_source():
    """Retain prefix equations/order; change only local Z and status scope."""
    source = rows_source._PREFIX
    source = contact._replace(source, " || out.global_status.data[0]!=0", "")
    source = contact._replace(
        source,
        "out.response.data[(world*settings.dense_capacity+slot)*29+row]=z;",
        "private_z[slot*29+row]=z;",
    )
    source = contact._replace(
        source,
        "if(state.mf_count.data[world]>0)out.physical_J.data[(world*settings.dense_capacity+slot)*29+row]=(d==col?sign:0.0f);",
        "/* Finalized MF0 has no physical-J consumer. */",
    )
    # The native prefix's scalar z shadows the panel view name.
    views = _VIEWS.replace("float* z=", "float* private_z=")
    return views + _operator_local(source)


@wp.func_native(prefix_source())
def _prefix(
    address: wp.uint64,
    index: int,
    plan: KineticPlan,
    held: HeldKineticOperator,
    prefix: PrefixInput,
    state: RowState,
    settings: RowSettings,
    out: DenseRowOutput,
): ...


def contact_source():
    """Retain cached triplets; consume exact allocator slots, not a raw scan."""
    source = contact._contact_source()
    begin = source.index("    const int count=raw.count.data[0];")
    end = source.index("        const int aa=raw.art_a.data[c]", begin)
    source = (
        source[:begin]
        + r"""
    const int count=raw.count.data[0],m=state.dense_count.data[world];
    if(count<0||count>settings.raw_capacity||count>raw.shape0.shape[0]){
        if(lane==0)out.status.data[world]=4;return;}
    for(int slot=out.slot_counter.data[world];slot<m;++slot){
        const int c=slot_raw.data[world*192+slot];
        if(c==-1)continue;
        if(c<0||c>=count||raw.world.data[c]!=world||raw.slot.data[c]!=slot||raw.path.data[c]!=0){
            if(lane==0)out.status.data[world]=4;continue;}
        if(!all(out.status.data[world]==0))continue;
"""
        + source[end:]
    )
    begin = source.index("        if(arm.valid.data[world]==0")
    end = source.index("        const int sa=raw.shape0.data[c]", begin)
    source = source[:begin] + source[end:]
    source = contact._replace(
        source,
        "out.response.data[id*29+coord]=s[192+component*32+d];",
        "private_z[(slot+component)*29+coord]=s[192+component*32+d];",
    )
    source = contact._replace(source, "const bool physical=state.mf_count.data[world]>0;", "const bool physical=false;")
    source = contact._replace(
        source,
        "if(physical)out.physical_J.data[id*29+coord]=s[96+component*32+d];",
        "/* Finalized MF0 has no physical-J consumer. */",
    )
    views = _VIEWS.replace("float* z=", "float* private_z=")
    return views + _MOTION_VIEWS + _operator_local(source)


@wp.func_native(contact_source())
def _contact(
    address: wp.uint64,
    world: int,
    plan: KineticPlan,
    current: CurrentKineticCache,
    held: HeldKineticOperator,
    raw: RawRowInput,
    state: RowState,
    settings: RowSettings,
    arm: ArmMap,
    out: DenseRowOutput,
    cache: ContactGeometry,
    slot_raw: wp.array2d[int],
): ...


@wp.func_native(r"""
#if defined(__CUDA_ARCH__)
    const int lane=threadIdx.x&31,stride=32;
    __syncwarp(0xffffffffu);
#else
    const int lane=0,stride=1;
#endif
    bool good=out.status.data[world]==0;
    for(int row=lane;row<state.dense_count.data[world];row+=stride)
        good=good&&out.valid.data[world*192+row]==1;
#if defined(__CUDA_ARCH__)
    good=__all_sync(0xffffffffu,good)!=0;
#endif
    if(!good&&lane==0){out.status.data[world]=6;solve.status.data[world]=1;}
    return good;
""")
def _validate(world: int, state: RowState, out: DenseRowOutput, solve: KineticSolveData) -> bool: ...


def recurrence_source():
    """Recover the original eight law with local coefficient addressing."""
    cuda = solve_source.cuda_recurrence()
    edits = []

    def change(before, after, *, count=1):
        nonlocal cuda
        if cuda.count(before) != count:
            raise ValueError("Private recurrence seam changed: " + before)
        cuda = cuda.replace(before, after)
        edits.append((before, after))

    change("    auto factor_rows=rows.response;\n", "")
    change("factor_rows.data[row_base + ", "z[", count=7)
    for name, dtype, count in (
        ("s_lam_storage", "float", 384),
        ("s_rhs_storage", "float", 384),
        ("s_diag_storage", "float", 384),
        ("s_type_storage", "unsigned char", 384),
        ("s_contact_mu_storage", "float", 128),
    ):
        change(f"__shared__ {dtype} {name}[{count}];", f"__shared__ {dtype} {name}[{count // 2}];")
    recovered = cuda
    for before, after in reversed(edits):
        if after:
            recovered = recovered.replace(after, before)
        else:
            recovered = contact._replace(
                recovered, "    auto world_row_mu=rows.row_mu;\n", "    auto world_row_mu=rows.row_mu;\n" + before
            )
    if recovered != solve_source.cuda_recurrence():
        raise ValueError("Private recurrence changed undeclared arithmetic")
    cpu = solve_source._CPU
    cpu = cpu.replace("rows.response.data[id*29+d]", "z[i*29+d]")
    cpu = cpu.replace("rows.response.data[(world*192+sibling)*29+d]", "z[sibling*29+d]")
    if "rows.response" in cpu:
        raise ValueError("Unconverted CPU coefficient read")
    return (
        _VIEWS
        + _operator_local(solve_source._T)
        + "\nconst bool split_component=false;\n"
        + (
            "\n#if defined(__CUDA_ARCH__)\n"
            + _operator_local(cuda)
            + "\n#else\nconst int po=state.primary_offset.data[world],so=state.secondary_offset.data[world];\n"
            + _operator_local(cpu)
            + "\n#endif\n"
        )
    )


@wp.func_native(recurrence_source())
def _solve(
    address: wp.uint64,
    world: int,
    plan: KineticPlan,
    held: HeldKineticOperator,
    state: RowState,
    rows: DenseRowOutput,
    solve: KineticSolveData,
): ...


@functools.cache
def get_kernel(arch):
    """One warp owns one active MF0 world's complete row/solve lifetime."""

    @wp.kernel(module="unique", enable_backward=False)
    def kinetic_private_response_eight(
        plan: KineticPlan,
        current: CurrentKineticCache,
        held: HeldKineticOperator,
        raw: RawRowInput,
        state: RowState,
        prefix: PrefixInput,
        settings: RowSettings,
        arm: ArmMap,
        out: DenseRowOutput,
        cache: ContactGeometry,
        solve: KineticSolveData,
        slot_raw: wp.array2d[int],
    ):
        index, logical = wp.tid()
        lanes = rows_source._lane()
        if lanes[1] == 1 and logical != 0:
            return
        if state.active_count[0] < 0 or state.active_count[0] > state.active_worlds.shape[0]:
            return
        if index >= state.active_count[0]:
            return
        world = state.active_worlds[index]
        if world < 0 or world >= state.dense_count.shape[0]:
            return
        if state.resolved[world] != 0 or state.mf_count[world] != 0:
            return
        address = _storage()
        if _begin(address, world, plan, current, held, state, settings, out, solve):
            _prefix(address, index, plan, held, prefix, state, settings, out)
            if _good(world, out):
                _contact(address, world, plan, current, held, raw, state, settings, arm, out, cache, slot_raw)
            if _validate(world, state, out, solve):
                _solve(address, world, plan, held, state, out, solve)
        rows_source._release(address)

    return kinetic_private_response_eight


def _emit_kernel(kernel, tree, function, namespace):
    """Reuse the existing exact-source emitter with one distinct owner name."""
    function.name += "_mf_positive"
    function, source = hybrid._emit(tree, function, namespace, "private-response-complement")
    result = wp.kernel(function, module="unique", enable_backward=False)
    result.private_response_original = kernel
    result.private_response_source = source
    return result


@functools.cache
def get_allocate_kernel(arch):
    """Publish normal-slot identity during the original atomic reservation."""
    original = contact.get_allocate_kernel(arch)
    tree, function, namespace = guards._tree(original)
    before = ast.dump(function, include_attributes=False)
    function.args.args.append(ast.arg(arg="slot_raw", annotation=ast.parse("wp.array2d[int]", mode="eval").body))
    calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "_stage_row"
    ]
    if len(calls) != 1:
        raise ValueError("Allocator staging seam changed")
    parent = next(node for node in ast.walk(function) if calls[0] in getattr(node, "body", []))
    extra = ast.parse(
        "slot = contact_slot[c]\n"
        "world = contact_world[c]\n"
        "width = contact_slots_needed[c]\n"
        "if world >= 0 and world < slot_raw.shape[0] and slot >= 0 and slot + width <= 192:\n"
        "    for part in range(width):\n"
        "        value = int(-1)\n"
        "        if part == 0:\n"
        "            value = c\n"
        "        slot_raw[world, slot + part] = value\n"
        "else:\n"
        "    wp.atomic_max(capacity_status, 3, 1)\n"
    ).body
    index = parent.body.index(calls[0]) + 1
    parent.body[index:index] = extra
    recovery = copy.deepcopy(function)
    target = next(
        node
        for node in ast.walk(recovery)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "_stage_row"
    )
    owner = next(node for node in ast.walk(recovery) if target in getattr(node, "body", []))
    index = owner.body.index(target) + 1
    del owner.body[index : index + len(extra)]
    recovery.args.args.pop()
    if ast.dump(recovery, include_attributes=False) != before:
        raise ValueError("Allocator map changed the original reservation law")
    function.name += "_slot_map"
    function, source = hybrid._emit(tree, function, namespace, "private-response-slot-map")
    result = wp.kernel(function, module="unique", enable_backward=False)
    result.private_response_original_ast = before
    result.private_response_recovered_ast = ast.dump(recovery, include_attributes=False)
    result.private_response_source = source
    return result


@functools.cache
def get_masked_world_kernel(kernel):
    """Exclude MF0 only after the original active-world bounds check."""
    tree, function, namespace = guards._tree(kernel)
    before = ast.dump(function, include_attributes=False)
    candidates = [
        node
        for node in function.body
        if isinstance(node, ast.If) and ast.unparse(node.test).startswith("world < 0 or world >=")
    ]
    if len(candidates) != 1:
        raise ValueError("Row world-bounds seam changed")
    index = function.body.index(candidates[0]) + 1
    injected = ast.parse("if state.mf_count[world] == 0:\n    return\n").body[0]
    function.body.insert(index, injected)
    recovery = copy.deepcopy(function)
    del recovery.body[index]
    if ast.dump(recovery, include_attributes=False) != before:
        raise ValueError("Row complement changed original arithmetic")
    return _emit_kernel(kernel, tree, function, namespace)


@functools.cache
def get_masked_native_kernel(kernel, field, source):
    """Replace only a named native predicate while preserving the caller ABI."""
    tree, function, namespace = guards._tree(kernel)
    old = namespace[field]
    namespace[field] = compact._native(old, source, field + "_mf_positive", {})
    return _emit_kernel(kernel, tree, function, namespace)


@functools.cache
def get_masked_general_kernel(kernel):
    """Never read unfinished MF0 status from the main-stream general owner."""
    tree, function, namespace = guards._tree(kernel)
    before = ast.dump(function, include_attributes=False)
    candidates = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.If)
        and any(
            isinstance(item, ast.Call) and isinstance(item.func, ast.Name) and item.func.id == "_guard_world_check"
            for item in ast.walk(node.test)
        )
    ]
    if len(candidates) != 1:
        raise ValueError("General guard seam changed")
    guarded = candidates[0]
    owner = next(node for node in ast.walk(function) if guarded in getattr(node, "body", []))
    index = owner.body.index(guarded)
    # The original guard owns malformed queue IDs. Do not move an unchecked
    # mf_count load before its range check, even though live identity queues
    # normally make that branch unreachable.
    injected = ast.If(
        test=ast.parse("world >= 0 and world < state.dense_count.shape[0]", mode="eval").body,
        body=[ast.If(test=ast.parse("state.mf_count[world] != 0", mode="eval").body, body=[guarded], orelse=[])],
        orelse=[copy.deepcopy(guarded)],
    )
    owner.body[index] = injected
    recovery = copy.deepcopy(function)
    target = next(
        node
        for node in ast.walk(recovery)
        if isinstance(node, ast.If) and ast.dump(node.test) == ast.dump(injected.test)
    )
    parent = next(node for node in ast.walk(recovery) if target in getattr(node, "body", []))
    parent.body[parent.body.index(target)] = target.orelse[0]
    if ast.dump(recovery, include_attributes=False) != before:
        raise ValueError("General complement changed retained recurrence")
    return _emit_kernel(kernel, tree, function, namespace)


@wp.kernel
def _check_positive(guard: guards.BoundaryGuard, state: RowState):
    world = wp.tid()
    if state.mf_count[world] != 0 or state.resolved[world] != 0:
        guards._guard_check(world, guard)


def allocate(worlds, device):
    """Allocate only current normal-slot identity, with no response panel."""
    return wp.empty((worlds, 192), dtype=int, device=device)


def install(binding, call):
    """Move the finalized MF0 producer/solve before the retained MF chain."""
    rows, solve, allocation = call.rows, call.solve, call.solve.allocation
    device, worlds, arch = binding.device, binding.worlds, str(binding.device.arch)
    slot_raw = binding.private_response_slots
    if binding.current_contact is None or slot_raw is None or not solve.compact_coupled:
        raise ValueError("Private response requires current-contact and compact fallback")
    if rows.kernels[2] is not contact.get_contact_kernel(arch):
        raise ValueError("Private response requires the actual cached triplet")
    private_kernel = get_kernel(arch)
    private_args = [
        rows.plan,
        rows.current,
        rows.held,
        rows.raw,
        rows.state,
        rows.prefix,
        rows.settings,
        rows.arm,
        rows.out,
        binding.current_contact,
        solve.solve_descriptor,
        slot_raw,
    ]
    mapped_allocator = get_allocate_kernel(arch)
    mapped_args = [*call.current_contact.arguments[3], slot_raw]

    prefix = contact._replace(
        rows_source._PREFIX,
        "    if(world<0 || world>=out.status.shape[0])return;",
        "    if(world<0 || world>=out.status.shape[0])return;\n    if(state.mf_count.data[world]==0)return;",
    )
    contact_body = contact._replace(
        contact._contact_source(),
        "        if(!all(state.resolved.data[world]==0 && out.status.data[world]==0))continue;",
        "        if(state.mf_count.data[world]==0)continue;\n"
        "        if(!all(state.resolved.data[world]==0 && out.status.data[world]==0))continue;",
    )
    rows.kernels[0] = get_masked_world_kernel(rows.kernels[0])
    rows.kernels[1] = get_masked_native_kernel(rows.kernels[1], "_prefix", prefix)
    rows.kernels[2] = get_masked_native_kernel(rows.kernels[2], "_contact", contact_body)
    rows.kernels[3] = get_masked_world_kernel(rows.kernels[3])
    old_offset = solve.kernels["offset_eight"]
    native = inspect.getclosurevars(old_offset.func).nonlocals["solve_native"]
    masked = contact._replace(
        native.native_snippet,
        "    if(world<0||world>=state.dense_count.shape[0])return;",
        "    if(world<0||world>=state.dense_count.shape[0])return;\n    if(state.mf_count.data[world]==0)return;",
    )
    solve.kernels["offset_eight"] = get_masked_native_kernel(old_offset, "solve_native", masked)
    solve.kernels["general"] = get_masked_general_kernel(solve.kernels["general"])

    def dispatch(name):
        dim, block = solve.dimensions[name]
        wp.launch_tiled(solve.kernels[name], dim=[dim], block_dim=block, inputs=solve.arguments[name], device=device)

    def launch_private():
        wp.launch_tiled(private_kernel, dim=[worlds], block_dim=32, inputs=private_args, device=device)

    def allocate_launch():
        # The old row closure's clear moves BEFORE the fork. No other private
        # owner reads or writes row-global status while the MF chain runs.
        rows.out.global_status.zero_()
        wp.launch(mapped_allocator, dim=allocation.workers, inputs=mapped_args, device=device)
        wp.copy(allocation.mf.contact_end, allocation.data.mf_counter)
        wp.launch(
            contact.kernels.allocate_rigid_velocity_limit_slots,
            dim=allocation.mf.free_bodies.size,
            inputs=allocation.limit_args,
            device=device,
        )
        for counter, cap, family, count in (
            (allocation.data.dense_counter, 192, 0, rows.state.dense_count),
            (allocation.data.mf_counter, 64, 1, rows.state.mf_count),
            (allocation.data.propagation_counter, 192, 2, allocation.data.propagation_count),
        ):
            wp.launch(
                contact.kernels.finalize_constraint_counts_with_status,
                dim=worlds,
                inputs=[counter, cap, family, count, rows.state.capacity_status],
                device=device,
            )
        if device.is_cuda:
            wp.record_event(binding.ready)
            with wp.ScopedStream(binding.stream):
                wp.wait_event(binding.ready)
                try:
                    launch_private()
                finally:
                    wp.record_event(binding.done)
                    binding.pending = True
        else:
            launch_private()

    def build_rows():
        for kernel, args, dim in zip(
            rows.kernels, rows.arguments, (worlds, worlds, rows.settings.workers, worlds), strict=True
        ):
            wp.launch_tiled(kernel, dim=[dim], inputs=args, block_dim=32, device=device)

    def materialize():
        dispatch("materialize")
        wp.launch(_check_positive, dim=worlds, inputs=[solve.guard, rows.state], device=device)

    def solve_phase():
        if device.is_cuda:
            # The independent positive-MF offset owner queues after private
            # MF0 completion on the same stream, then overlaps original general.
            wp.record_event(binding.ready)
            with wp.ScopedStream(binding.stream):
                wp.wait_event(binding.ready)
                try:
                    dispatch("offset_eight")
                finally:
                    wp.record_event(binding.done)
                    binding.pending = True
            try:
                dispatch("general")
            finally:
                solve.join()
        else:
            dispatch("general")
            dispatch("offset_eight")
        guards.check_guard(solve.guard, device)

    solve.allocate = allocation.launch = allocate_launch
    solve.build_rows = rows.launch = build_rows
    solve.materialize, solve.solve = materialize, solve_phase
    call.private_response = SimpleNamespace(
        kernel=private_kernel,
        arguments=private_args,
        slot_raw=slot_raw,
        allocator=mapped_allocator,
        allocator_arguments=mapped_args,
        complementary_rows=tuple(rows.kernels),
    )
