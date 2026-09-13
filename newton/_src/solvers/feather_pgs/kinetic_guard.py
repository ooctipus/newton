# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Device-only call-error gates; original numerical sources remain unchanged.

Initialize once BEFORE qualification (after upstream preparation), check after
materialization and again AFTER the private/general stream join. Force export
must consume frame_status only after that join/check. The guarded finalizer
checks live statuses itself before original admission or generalized writes.
No numerical value, including a finite seeded v_out, authorizes publication.
"""

import ast
import copy
import functools
import hashlib
import inspect
import linecache
import textwrap

import warp as wp

from . import kinetic_state
from .kinetic_source import checked_definitions


@wp.struct
class BoundaryGuard:
    refresh_status: wp.array[int]
    predictor_status: wp.array[int]
    row_status: wp.array[int]
    solve_status: wp.array[int]
    resolved: wp.array[int]
    error: wp.array[int]
    row_global_status: wp.array[int]
    raw_invalid: wp.array[int]
    capacity_status: wp.array[int]
    frame_status: wp.array[int]


@wp.func
def _guard_check(world: int, guard: BoundaryGuard) -> int:
    error = int(0)
    if world < 0 or world >= guard.error.shape[0]:
        wp.atomic_or(guard.frame_status, 0, 128)
        return 128
    error = guard.error[world]
    if guard.refresh_status[world] != 0:
        error = error | 1
    if guard.predictor_status[world] != 0:
        error = error | 2
    if guard.resolved[world] != 0 and guard.resolved[world] != 1:
        error = error | 128
    if guard.resolved[world] == 0:
        if guard.row_status[world] != 0:
            error = error | 4
        if guard.solve_status[world] != 0:
            error = error | 8
    if guard.row_global_status[0] != 0:
        error = error | 16
    if guard.raw_invalid[0] != 0:
        error = error | 32
    for index in range(4):
        if guard.capacity_status[index] != 0:
            error = error | 64
    if error != 0:
        wp.atomic_or(guard.error, world, error)
        wp.atomic_or(guard.frame_status, 0, error)
    return error


@wp.func_native(r"""
#if defined(__CUDA_ARCH__)
    return __ballot_sync(0xffffffffu, error != 0) != 0 ? 1 : 0;
#else
    return error != 0 ? 1 : 0;
#endif
""")
def _uniform_error(error: int) -> int: ...


@wp.func
def _guard_world_check(world: int, guard: BoundaryGuard) -> int:
    # Both guarded owners execute one complete hardware warp per world. A
    # concurrent independent-component error must never split a native barrier.
    error = _guard_check(world, guard)
    return _uniform_error(error)


@wp.kernel
def _initialize(guard: BoundaryGuard):
    world = wp.tid()
    guard.error[world] = 0
    guard.solve_status[world] = 0
    if world == 0:
        guard.frame_status[0] = 0


@wp.kernel
def _check(guard: BoundaryGuard):
    world = wp.tid()
    _guard_check(world, guard)


def allocate_guard(worlds, device, **aliases):
    """Allocate call-owned errors and bind exact status aliases, never copy them.

    Unspecified inputs are zero fixtures, not a production completeness claim.
    Production callers must explicitly bind every upstream input status.
    """
    if worlds <= 0 or set(aliases) - set(BoundaryGuard.vars):
        raise ValueError("Positive world count and known guard fields required")
    result = BoundaryGuard()
    for name in BoundaryGuard.vars:
        size = worlds
        if name in ("row_global_status", "raw_invalid", "frame_status"):
            size = 1
        elif name == "capacity_status":
            size = 4
        value = aliases.get(name)
        if value is None:
            value = wp.zeros(size, dtype=int, device=device)
        if (
            value.shape != (size,)
            or value.dtype != wp.int32
            or not value.is_contiguous
            or value.device != wp.get_device(device)
        ):
            raise ValueError("Guard alias layout/device mismatch: " + name)
        setattr(result, name, value)
    reset_fields = ("error", "frame_status", "solve_status")
    for name in reset_fields:
        array = getattr(result, name)
        for other in BoundaryGuard.vars:
            if other == name:
                continue
            target = getattr(result, other)
            if array.ptr < target.ptr + target.capacity and target.ptr < array.ptr + array.capacity:
                raise ValueError("Reset-owned guard status aliases another field: " + name)
    return result


def initialize_guard(guard, device):
    """Reset only call-owned failures and solve status before qualification."""
    wp.launch(_initialize, dim=guard.error.shape[0], inputs=[guard], device=device)


def check_guard(guard, device):
    """Aggregate current errors without host reads or clearing earlier errors."""
    wp.launch(_check, dim=guard.error.shape[0], inputs=[guard], device=device)


def _tree(kernel):
    """Recover the original function body including its closure bindings."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(kernel.func)))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    function.decorator_list = []
    namespace = dict(kernel.func.__globals__)
    closure = inspect.getclosurevars(kernel.func)
    namespace.update(closure.globals)
    namespace.update(closure.nonlocals)
    return tree, function, namespace


def _compile(kernel, kind):
    """Inject one guard seam and prove its removal recovers original AST."""
    tree, function, namespace = _tree(kernel)
    original = ast.dump(function, include_attributes=False)
    name = function.name
    function.name = name + "_boundary_guarded"
    function.args.args.append(ast.arg(arg="guard", annotation=ast.Name(id="BoundaryGuard")))
    if kind == "general":
        calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "pgs_solve_mf_gs_native"
        ]
        if len(calls) != 1:
            raise RuntimeError("Original general native call seam changed")
        call = calls[0]
        owner = next(node for node in ast.walk(function) if call in getattr(node, "body", []))
        index = owner.body.index(call)
        injected = ast.If(
            test=ast.parse("_guard_world_check(world, guard) == 0", mode="eval").body,
            body=[call],
            orelse=[],
        )
        owner.body[index] = injected
        recovery = copy.deepcopy(function)
        target = next(
            node
            for node in ast.walk(recovery)
            if isinstance(node, ast.If) and ast.dump(node.test) == ast.dump(injected.test)
        )
        parent = next(node for node in ast.walk(recovery) if target in getattr(node, "body", []))
        parent.body[parent.body.index(target)] = target.body[0]
    elif kind == "finish":
        targets = [
            (index, node)
            for index, node in enumerate(function.body)
            if isinstance(node, ast.If)
            and any(isinstance(item, ast.Name) and item.id == "_admit" for item in ast.walk(node.test))
        ]
        if len(targets) != 1:
            raise RuntimeError("Original finish admission seam changed")
        index, _ = targets[0]
        injected = ast.parse(
            "if _guard_world_check(world, guard) != 0:\n"
            "    if lane == 0:\n"
            "        current.valid[world] = 0\n"
            "        if schedule.geometry_requested[world] != 0:\n"
            "            geometric.valid[world] = 0\n"
            "        schedule.status[world] = 3\n"
            "    return\n"
        ).body[0]
        function.body.insert(index, injected)
        recovery = copy.deepcopy(function)
        del recovery.body[index]
    else:
        raise ValueError("Unknown checked adapter kind")
    recovery.name = name
    recovery.args.args.pop()
    if ast.dump(recovery, include_attributes=False) != original:
        raise RuntimeError("Guard removal did not recover exact original arithmetic")
    ast.fix_missing_locations(tree)
    source = ast.unparse(tree) + "\n"
    filename = "<kinetic-guard-" + hashlib.sha256(source.encode()).hexdigest() + ">"
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace.update(BoundaryGuard=BoundaryGuard, _guard_world_check=_guard_world_check)
    exec(compile(source, filename, "exec"), namespace)
    result = wp.kernel(namespace[function.name], module="unique", enable_backward=False)
    result.guard_source = source
    result.guard_original_ast = original
    result.guard_recovered_ast = ast.dump(recovery, include_attributes=False)
    return result


@functools.cache
def get_general_kernel(original_kernel):
    """Original general ABI plus BoundaryGuard last; guard each queued world."""
    checked_definitions("solver_feather_pgs.py", ("_get_pgs_solve_mf_gs_kernel",))
    return _compile(original_kernel, "general")


@functools.cache
def get_finish_kernel(arch):
    """Original five-argument next-state finish plus BoundaryGuard last."""
    checked_definitions("kinetic_state.py", ("_get_kernel", "_admit", "_collect", "_COLLECT", "_primary_terms"))
    return _compile(kinetic_state.get_finish_kernel(arch), "finish")
