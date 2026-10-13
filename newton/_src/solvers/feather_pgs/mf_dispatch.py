# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Explicit native operands and shared matrix-free world queue dispatch."""

import hashlib
import linecache
from textwrap import indent

import warp as wp

_NATIVE_ARGUMENTS = (
    ("world", "int"),
    ("rb_class", "wp.array[int]"),
    ("world_constraint_count", "wp.array[int]"),
    ("dense_phase_bounds", "wp.array2d[int]"),
    ("local_solve_owner", "wp.array[int]"),
    ("world_dof_indices", "wp.array2d[int]"),
    ("world_deferred_dof_mask", "wp.array2d[int]"),
    ("rhs_bias", "wp.array2d[float]"),
    ("world_diag", "wp.array2d[float]"),
    ("world_row_w", "wp.array2d[float]"),
    ("world_impulses", "wp.array2d[float]"),
    ("J_world", "wp.array3d[float]"),
    ("Y_world", "wp.array3d[float]"),
    ("world_row_type", "wp.array2d[int]"),
    ("world_row_parent", "wp.array2d[int]"),
    ("world_row_mu", "wp.array2d[float]"),
    ("world_torsion_group", "wp.array2d[int]"),
    ("contact_torsion_radius", "float"),
    ("world_drive_target_vel_bias", "wp.array2d[float]"),
    ("world_drive_vel_multiplier", "wp.array2d[float]"),
    ("world_drive_impulse_multiplier", "wp.array2d[float]"),
    ("world_drive_max_impulse", "wp.array2d[float]"),
    ("world_drive_vel_limit", "wp.array2d[float]"),
    ("mf_constraint_count", "wp.array[int]"),
    ("mf_contact_rows_end", "wp.array[int]"),
    ("mf_meta", "wp.array2d[int]"),
    ("mf_impulses", "wp.array2d[float]"),
    ("mf_J_a", "wp.array3d[float]"),
    ("mf_J_b", "wp.array3d[float]"),
    ("mf_MiJt_a", "wp.array3d[float]"),
    ("mf_MiJt_b", "wp.array3d[float]"),
    ("mf_row_mu", "wp.array2d[float]"),
    ("mf_row_w", "wp.array2d[float]"),
    ("iterations", "int"),
    ("omega", "float"),
    ("regularize", "int"),
    ("row_phase", "int"),
    ("friction_start_iteration", "int"),
    ("iteration_offset", "int"),
    ("freeze_drive_rows", "int"),
    ("defer_dense_response", "int"),
    ("v_out", "wp.array[float]"),
)


_QUEUE_ARGUMENTS = (
    ("general_world_count", "wp.array[int]"),
    ("general_worlds", "wp.array[int]"),
    ("general_world_grid_stride", "int"),
    ("use_general_world_queue", "int"),
)


def _compile(source, namespace):
    """Compile an inspectable declaration from the explicit dispatch contract."""
    filename = "<mf-dispatch-" + hashlib.sha256(source.encode()).hexdigest() + ">"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    namespace = dict(namespace)
    exec(compile(source, filename, "exec"), namespace)
    return namespace


def _signature(arguments, extra):
    """Append distinct private operands while preserving the original ABI."""
    names = {name for name, _ in arguments}
    if names.intersection(extra) or any(not name.isidentifier() for name in extra):
        raise ValueError("Private operands must have distinct valid names")
    return ", ".join(
        [name + ": " + annotation for name, annotation in arguments]
        + [name + ": " + annotation.__name__ for name, annotation in extra.items()]
    )


def make_native(source, *, name="pgs_solve_mf_gs_native", extra=None):
    """Bind the original native argument contract to a complete numerical body."""
    extra = {} if extra is None else extra
    if not name.isidentifier():
        raise ValueError("Native declaration requires a valid name")
    signature = _signature(_NATIVE_ARGUMENTS, extra)
    namespace = dict(globals(), **{annotation.__name__: annotation for annotation in extra.values()})
    declaration = "def " + name + "(" + signature + "):\n    ...\n"
    return wp.func_native(source)(_compile(declaration, namespace)[name])


def make_dispatch(native, *, policy="plain", extra=None, bindings=None, native_extra=()):
    """Bind queue dispatch to one explicit boundary and mutually exclusive owner."""
    if policy not in ("plain", "guard", "hybrid", "compact"):
        raise ValueError("Unknown matrix-free dispatch policy")
    extra = {} if extra is None else extra
    bindings = {} if bindings is None else bindings
    if policy == "plain" and (extra or bindings or native_extra):
        raise ValueError("Plain dispatch has no private operands")
    if policy != "plain" and ("guard" not in extra or "_guard_world_check" not in bindings):
        raise ValueError("Guarded dispatch requires the boundary descriptor and predicate")
    required = {
        "plain": (),
        "guard": (),
        "hybrid": ("kinetic_hybrid_native", "_is_cpu"),
        "compact": ("kinetic_compact_native", "kinetic_original_arena_native", "_is_cpu", "_arena"),
    }[policy]
    if any(name not in bindings for name in required):
        raise ValueError("Dispatch is missing a required owner binding")
    name = "pgs_solve_mf_gs_template"
    if policy != "plain":
        name += "_boundary_guarded"
    if policy in ("hybrid", "compact"):
        name += "_" + policy
    signature = _signature(_QUEUE_ARGUMENTS + _NATIVE_ARGUMENTS[1:], extra)
    arguments = [name for name, _ in _NATIVE_ARGUMENTS]

    def call(owner, additional=()):
        return owner + "(" + ", ".join([*arguments, *additional]) + ")"

    dispatch = call("pgs_solve_mf_gs_native")
    if policy in ("hybrid", "compact"):
        selected = "kinetic_" + policy + "_native"
        fallback = dispatch if policy == "hybrid" else call("kinetic_original_arena_native", ("arena_address",))
        dispatch = (
            "if hybrid.eligible[world] != 0:\n"
            + indent(call(selected, native_extra), "    ")
            + "\nelse:\n"
            + indent(fallback, "    ")
        )
        if policy == "compact":
            dispatch = "arena_address = _arena()\n" + dispatch
    if policy != "plain":
        dispatch = "if _guard_world_check(world, guard) == 0:\n" + indent(dispatch, "    ")
    cpu_guard = "if _is_cpu() and _lane != 0:\n    return\n" if policy in ("hybrid", "compact") else ""
    body = (
        "candidate, _lane = wp.tid()\n"
        + cpu_guard
        + "general_index = candidate\n"
        + "general_count = general_world_count[0]\n"
        + "if use_general_world_queue == 0:\n    general_count = candidate + 1\n"
        + "while general_index < general_count:\n"
        + "    world = candidate\n"
        + "    if use_general_world_queue != 0:\n        world = general_worlds[general_index]\n"
        + indent(dispatch, "    ")
        + "\n    general_index += general_world_grid_stride\n"
    )
    closure = dict(pgs_solve_mf_gs_native=native, **bindings)
    declaration = "def " + name + "(" + signature + "):\n" + indent(body, "    ")
    source = "def _bind(" + ", ".join(closure) + "):\n" + indent(declaration, "    ") + "    return " + name + "\n"
    namespace = dict(globals(), **{annotation.__name__: annotation for annotation in extra.values()})
    result = _compile(source, namespace)["_bind"](**closure)
    result.mf_dispatch_source = declaration
    result.mf_dispatch_policy = policy
    return result
