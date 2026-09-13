# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Checked private-provider successors of the original split query factories."""

import hashlib
import inspect
import linecache
import re
from functools import cache
from typing import Any

from .convex_bsp import _bind_shape, _create_postprocess, _create_support


def source_successor(original, *, coherent=False):
    """Change only provider plumbing; preserve the original query arithmetic exactly."""
    baseline = inspect.getsource(original)

    def replace(before, after, count):
        nonlocal source
        if source.count(before) != count:
            raise RuntimeError(f"BSP factory source seam changed: {before!r}")
        source = source.replace(before, after)

    source = baseline
    replace(f"def {original.__name__}(", f"def {original.__name__}_bsp(", 1)
    replace(
        "        manifold_work_count: wp.array[int],\n",
        "        manifold_work_count: wp.array[int],\n        support_data: Any,\n",
        3,
    )
    if coherent:
        replace("support_map_lean,", "_bsp_support,", 3)
        # Only cold stages use support; the rejection classifier stays unchanged.
        split = source.index("    def cold_mpr(")
        head, source = source[:split], source[split:]
    source, count = re.subn(
        r"(?m)^( +)provider = SupportMapDataProvider\(\)$",
        lambda m: (
            m[1]
            + "provider = support_data\n"
            + m[1]
            + "geom_a = _bind_shape(query.geom_a, support_data)\n"
            + m[1]
            + "geom_b = _bind_shape(query.geom_b, support_data)"
        ),
        source,
    )
    if count != (2 if coherent else 3):
        raise RuntimeError("BSP provider source seam changed")
    # Restrict replacement to call arguments, never the binding expressions.
    replace("query.geom_a,\n", "geom_a,\n", 2 if coherent else 3)
    replace("query.geom_b,\n", "geom_b,\n", 2 if coherent else 3)
    if coherent:
        source = head + source
    return source, baseline


@cache
def _factory(original, lean, coherent):
    source, _ = source_successor(original, coherent=coherent)
    filename = f"<newton_convex_bsp_{hashlib.sha256(source.encode()).hexdigest()}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = dict(original.__globals__)
    namespace.update(Any=Any, _bind_shape=_bind_shape, _bsp_support=_create_support(lean))
    exec(compile(source, filename, "exec"), namespace)
    return namespace[original.__name__ + "_bsp"], namespace["_bsp_support"]


def split_kernels(narrow_phase):
    """Build all ordinary split stages using the same private static descriptor."""
    from .narrow_phase import create_narrow_phase_kernels_gjk_mpr_split  # noqa: PLC0415

    factory, support = _factory(create_narrow_phase_kernels_gjk_mpr_split, narrow_phase._use_lean_gjk_mpr, False)
    return factory(
        narrow_phase.external_aabb,
        narrow_phase._convex_writer_func,
        support_func=support,
        post_process_contact=_create_postprocess(narrow_phase._use_lean_gjk_mpr),
        speculative=False,
    )


def coherent_kernels(*, diagnostics=False):
    """Preserve classification and replace both retained cold support callbacks."""
    from .coherent_convex_rejection import _create_rejection_query_kernels  # noqa: PLC0415

    factory, _ = _factory(_create_rejection_query_kernels, True, True)
    return factory(diagnostics=diagnostics)
