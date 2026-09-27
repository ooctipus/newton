# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Checked typed-provider successors of all four rejection-route owners."""

import hashlib
import inspect
import linecache
from functools import cache
from typing import Any

from .convex_cells import (
    _bind_shape,
    _cell_support_feature,
    _CellShape,
    _create_postprocess,
    _create_support,
)


def _execute(source, namespace):
    filename = f"<newton_convex_cells_{hashlib.sha256(source.encode()).hexdigest()}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    exec(compile(source, filename, "exec"), namespace)


@cache
def _plane_bound():
    """Keep the original rejection arithmetic after producing typed support features."""
    from .coherent_convex import _plane_lower_bound  # noqa: PLC0415

    function = _plane_lower_bound.func
    source = inspect.getsource(function)
    changes = (
        ("def _plane_lower_bound(", "def _cell_plane_lower_bound(", 1),
        ("GenericShapeData", "_CellShape", 2),
        ("    direction: wp.vec3,\n", "    direction: wp.vec3,\n    support_data: Any,\n", 1),
        ("_support_feature(geom_a, direction)", "_cell_support_feature(geom_a, direction, support_data)", 1),
        (
            "_support_feature(geom_b, local_direction_b)",
            "_cell_support_feature(geom_b, local_direction_b, support_data)",
            1,
        ),
    )
    for old, new, count in changes:
        if source.count(old) != count:
            raise RuntimeError("Cell plane-bound source seam changed")
        source = source.replace(old, new)
    namespace = dict(function.__globals__)
    namespace.update(Any=Any, _CellShape=_CellShape, _cell_support_feature=_cell_support_feature)
    _execute(source, namespace)
    return namespace["_cell_plane_lower_bound"]


def source_successor(original, *, coherent=False):
    """Change only shape/provider plumbing, and classifier feature calls."""
    baseline = inspect.getsource(original)
    source = baseline

    def replace(old, new, count=1):
        nonlocal source
        if source.count(old) != count:
            raise RuntimeError(f"Cell factory source seam changed: {old!r}")
        source = source.replace(old, new)

    replace(f"def {original.__name__}(", f"def {original.__name__}_cells(")
    replace(
        "        manifold_work_count: wp.array[int],\n",
        "        manifold_work_count: wp.array[int],\n        support_data: Any,\n",
        3,
    )
    if coherent:
        replace("support_map_lean,", "_cell_support,", 3)
        replace(
            "                            bound = _plane_lower_bound(\n",
            "                            geom_a = _bind_shape(query.geom_a, support_data)\n"
            "                            geom_b = _bind_shape(query.geom_b, support_data)\n"
            "                            bound = _cell_plane_lower_bound(\n",
        )
        replace("                                axis,\n", "                                axis, support_data,\n")
    # Each original owner resolves one descriptor per endpoint. Its callbacks
    # retain immutable bound arrays; there are no per-pair hint writes.
    names = (
        ("cold_mpr", "cold_gjk")
        if coherent
        else ("narrow_phase_mpr_kernel", "narrow_phase_gjk_kernel", "narrow_phase_manifold_kernel")
    )
    for name in names:
        begin = source.index("    def " + name + "(")
        end = source.find("    @wp.kernel", begin)
        if end < 0:
            end = len(source)
        part = source[begin:end]
        indent = "                " if name != "narrow_phase_manifold_kernel" else "            "
        old = indent + "provider = SupportMapDataProvider()"
        if part.count(old) != 1:
            raise RuntimeError("Cell provider declaration changed")
        new = (
            indent
            + "provider = support_data\n"
            + indent
            + "geom_a = _bind_shape(query.geom_a, support_data)\n"
            + indent
            + "geom_b = _bind_shape(query.geom_b, support_data)"
        )
        part = part.replace(old, new)
        source = source[:begin] + part + source[end:]
    replace("query.geom_a,\n", "geom_a,\n", 3)
    replace("query.geom_b,\n", "geom_b,\n", 3)
    for name in ("classify_rejections", *names) if coherent else names:
        replace("def " + name + "(", "def " + name + "_cells(")
    if coherent:
        replace(
            "return classify_rejections, cold_mpr, cold_gjk",
            "return classify_rejections_cells, cold_mpr_cells, cold_gjk_cells",
        )
    else:
        replace(
            "return narrow_phase_mpr_kernel, narrow_phase_gjk_kernel, narrow_phase_manifold_kernel",
            "return narrow_phase_mpr_kernel_cells, narrow_phase_gjk_kernel_cells, narrow_phase_manifold_kernel_cells",
        )
    return source, baseline


@cache
def _factory(original, lean, coherent):
    source, _ = source_successor(original, coherent=coherent)
    namespace = dict(original.__globals__)
    namespace.update(
        Any=Any,
        _bind_shape=_bind_shape,
        _cell_support=_create_support(lean),
        _cell_plane_lower_bound=_plane_bound(),
    )
    _execute(source, namespace)
    return namespace[original.__name__ + "_cells"], namespace["_cell_support"]


def manifold_kernel(narrow_phase):
    """Use the unchanged original manifold with immutable support candidates."""
    from .narrow_phase import create_narrow_phase_kernels_gjk_mpr_split  # noqa: PLC0415

    factory, support = _factory(create_narrow_phase_kernels_gjk_mpr_split, narrow_phase._use_lean_gjk_mpr, False)
    return factory(
        narrow_phase.external_aabb,
        narrow_phase._convex_writer_func,
        support_func=support,
        post_process_contact=_create_postprocess(narrow_phase._use_lean_gjk_mpr),
        speculative=False,
    )[2]


def coherent_kernels(*, diagnostics=False):
    """Replace all classifier, cold MPR and cold GJK support scans under the same queues."""
    from .coherent_convex_rejection import _create_rejection_query_kernels  # noqa: PLC0415

    factory, _ = _factory(_create_rejection_query_kernels, True, True)
    return factory(diagnostics=diagnostics)
