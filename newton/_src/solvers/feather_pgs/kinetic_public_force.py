# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Guarded original public force conversion/packing; no new force law.

Frame status is completed by the upstream frame guard before these launches.
Nonzero status preserves every output byte. Current raw identities and original
row metadata/impulses remain unchanged inputs, including discarded path=-1.
"""

import ast
import functools
import hashlib
import inspect
import linecache
import textwrap
from types import SimpleNamespace

import warp as wp

from . import kernels as original
from .kinetic_source import checked_definitions

NAMES = (
    "compute_contact_linear_force_from_impulses",
    "pack_contact_linear_force_as_spatial",
)


def guarded_source(name):
    """Append frame flag and early return; all original statements remain AST-exact."""
    if name not in NAMES:
        raise ValueError("Only original public contact force kernels are allowed")
    checked_definitions("kernels.py", (name,))
    tree = ast.parse(textwrap.dedent(inspect.getsource(getattr(original, name).func)))
    function = tree.body[0]
    function.name = "guarded_" + name
    function.args.args.append(ast.arg(arg="frame_status", annotation=ast.parse("wp.array[int]", mode="eval").body))
    function.decorator_list = (
        ast.parse('@wp.kernel(module="unique", enable_backward=False)\ndef f():\n    pass\n').body[0].decorator_list
    )
    position = int(
        isinstance(function.body[0], ast.Expr)
        and isinstance(function.body[0].value, ast.Constant)
        and isinstance(function.body[0].value.value, str)
    )
    function.body.insert(position, ast.parse("if frame_status[0] != 0:\n    return\n").body[0])
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


@functools.cache
def get_kernel(name):
    """Compile source-pinned original conversion or packing with frame guard."""
    source = guarded_source(name)
    filename = f"<kinetic-public-force-{hashlib.sha256(source.encode()).hexdigest()}>"
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace = dict(original.__dict__)
    exec(compile(source, filename, "exec"), namespace)
    return namespace["guarded_" + name]


def _overlaps(a, b):
    """Byte-range comparison for admitted contiguous public buffers."""
    if a.size == 0 or b.size == 0:
        return False
    end_a = a.ptr + a.size * wp.types.type_size_in_bytes(a.dtype)
    end_b = b.ptr + b.size * wp.types.type_size_in_bytes(b.dtype)
    return a.ptr < end_b and b.ptr < end_a


def bind_public_force(rows, mf, frame_status, *, linear=None, spatial=None, propagation=None):
    """Bind current public force outputs at the exact declared raw capacity.

    `frame_status` is a completed shared int32[1] guard, zero only for a valid
    frame. The caller owns its reset/validation/join, not this public adapter.
    `propagation=None` is the admitted disabled family; active propagation must
    supply `.count`, `.impulses`, `.row_type`, `.row_parent` original buffers.
    """
    device, capacity = rows.device, rows.settings.raw_capacity
    if frame_status.shape != (1,) or frame_status.dtype != wp.int32 or frame_status.device != device:
        raise ValueError("Require completed same-device int32 frame status")
    if not (0 < capacity <= 4_000_000) or rows.raw.shape0.shape != (capacity,):
        raise ValueError("Exact unchanged raw capacity required")
    for name in ("count", "normal", "world", "slot", "path"):
        a = getattr(rows.raw, name)
        expected = (1,) if name == "count" else (capacity,)
        if a.shape != expected or a.device != device:
            raise ValueError("Raw public-force descriptor mismatch")
    if propagation is None:
        propagation = SimpleNamespace(
            count=wp.zeros(rows.worlds, dtype=int, device=device),
            impulses=wp.zeros((1, 1), dtype=float, device=device),
            row_type=wp.zeros((1, 1), dtype=int, device=device),
            row_parent=wp.zeros((1, 1), dtype=int, device=device),
        )
        propagation_enabled = False
    else:
        propagation_enabled = True
        if propagation.count.shape != (rows.worlds,):
            raise ValueError("Original propagation count ownership mismatch")
        for name in ("impulses", "row_type", "row_parent"):
            if getattr(propagation, name).shape != (rows.worlds, 192):
                raise ValueError("Original propagation capacity must remain192")
    linear = wp.empty(capacity, dtype=wp.vec3, device=device) if linear is None else linear
    spatial = wp.empty(capacity, dtype=wp.spatial_vector, device=device) if spatial is None else spatial
    for output, dtype in ((linear, wp.vec3), (spatial, wp.spatial_vector)):
        if output.shape != (capacity,) or output.dtype != dtype or output.device != device or not output.is_contiguous:
            raise ValueError("Public output must have exact raw capacity/dtype/device")
    r, state, out = rows.raw, rows.state, rows.out
    values = {
        "contact_count": r.count,
        "contact_normal": r.normal,
        "contact_world": r.world,
        "contact_slot": r.slot,
        "contact_path": r.path,
        "world_impulses": out.impulses,
        "mf_impulses": mf.impulses,
        "propagation_impulses": propagation.impulses,
        "world_constraint_count": state.dense_count,
        "mf_constraint_count": state.mf_count,
        "propagation_constraint_count": propagation.count,
        "world_row_type": out.row_type,
        "world_row_parent": out.row_parent,
        "mf_row_type": mf.row_type,
        "mf_row_parent": mf.row_parent,
        "propagation_row_type": propagation.row_type,
        "propagation_row_parent": propagation.row_parent,
        "enable_friction": rows.settings.enable_friction,
        "inv_dt": 1.0 / rows.settings.dt,
        "rigid_contact_force": linear,
    }
    inputs = [value for key, value in values.items() if key != "rigid_contact_force" and isinstance(value, wp.array)]
    inputs.append(frame_status)
    if _overlaps(linear, spatial) or any(_overlaps(output, value) for output in (linear, spatial) for value in inputs):
        raise ValueError("Public force outputs alias read-only inputs or each other")
    first, second = (get_kernel(name) for name in NAMES)
    parameters = inspect.signature(getattr(original, NAMES[0]).func).parameters
    if set(parameters) != set(values):
        raise ValueError("Original public-force ABI changed")
    first_args = [values[name] for name in parameters] + [frame_status]
    second_args = [r.count, linear, spatial, frame_status]

    def launch():
        """Charge both original services at full declared raw launch capacity."""
        wp.launch(first, dim=capacity, inputs=first_args, device=device)
        wp.launch(second, dim=capacity, inputs=second_args, device=device)

    return SimpleNamespace(
        launch=launch,
        linear=linear,
        spatial=spatial,
        frame_status=frame_status,
        values=values,
        kernels=(first, second),
        arguments=(first_args, second_args),
        metadata={
            "original_force_law": True,
            "guarded_no_write_on_invalid": True,
            "raw_capacity": capacity,
            "propagation_enabled": propagation_enabled,
            "retained_service_not_new_savings": True,
        },
    )
