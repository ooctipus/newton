# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Optional ordered normal-first/shared-spectral tangent projection.

This is the previously screened spectral-GS policy, not a new contact law.
It keeps the original sparse owner's geometry, physical response, scalar
fallback, iteration allowance and publication. CFM enters only denominators.
The per-contact scalar cache lives for one fixed-row solve, never across calls.
"""

import functools
import inspect
import linecache
import textwrap

from . import sparse_factor_rows, sparse_metric_tangents


def _replace(source, before, after, count=1):
    if source.count(before) != count:
        raise RuntimeError(f"Sparse spectral source seam changed: {before!r}")
    return source.replace(before, after)


def get_fragments(capacity: int):
    """Keep the original fused transaction around a root-free local proposal."""
    setup, update = sparse_metric_tangents.get_fragments(capacity)
    update = _replace(
        update,
        "                                    contact_cross[row+2]=a12;contact_ready[word]|=bit;",
        """                                    // The original tangent-cross slot becomes the spectral denominator.
                                    const float c1=cfm.data[base+row+1],c2=cfm.data[base+row+2];
                                    const float a=diagonal.data[base+row+1]-c1;
                                    const float c=diagonal.data[base+row+2]-c2;
                                    const float largest=0.5f*(a+c)+hypotf(0.5f*(a-c),a12);
                                    const float denominator=largest+fmaxf(c1,c2);
                                    contact_cross[row+2]=(isfinite(a) && isfinite(c) && a>0.0f && c>0.0f &&
                                        isfinite(a12) && isfinite(c1) && isfinite(c2) && c1>=0.0f && c2>=0.0f &&
                                        isfinite(denominator) && denominator>0.0f)?denominator:0.0f;
                                    contact_ready[word]|=bit;""",
    )
    begin = update.index(
        "                                const float a=diagonal.data[base+row+1],c=diagonal.data[base+row+2];"
    )
    end = update.index(
        "\n                            }\n                        }\n                        accepted=", begin
    )
    update = (
        update[:begin]
        + """                                const float denominator=contact_cross[row+2];
                                if(isfinite(r1) && isfinite(r2) && denominator>0.0f) {
                                    float x1=old1-r1/denominator,x2=old2-r2/denominator;
                                    const float magnitude=hypotf(x1,x2);
                                    if(isfinite(x1) && isfinite(x2) && isfinite(magnitude)) {
                                        const float repair=magnitude>radius?radius/magnitude:1.0f;
                                        next1=x1*repair;next2=x2*repair;accepted=1;
                                    }
                                }"""
        + update[end:]
    )
    if "probe<16" in update or "beta1" in update:
        raise RuntimeError("Metric root unexpectedly retained in spectral proposal")
    return setup, update


@functools.cache
def get_solve_kernel():
    """Clone the original owner, adding only current row CFM to its solve ABI."""
    original = sparse_factor_rows.get_solve_kernel
    source = textwrap.dedent(inspect.getsource(original))
    source = source[source.index("def get_solve_kernel(") :]
    source = _replace(
        source,
        "from .sparse_metric_tangents import get_fragments",
        "from .sparse_spectral_tangents import get_fragments",
    )
    source = _replace(
        source,
        "        diagonal: wp.array2d[float],",
        "        diagonal: wp.array2d[float],\n        cfm: wp.array2d[float],",
        2,
    )
    source = _replace(
        source,
        "            diagonal,\n            impulses,",
        "            diagonal,\n            cfm,\n            impulses,",
    )
    source = _replace(
        source, '"sparse_metric_tangent" if metric_tangents', '"sparse_spectral_tangent" if metric_tangents'
    )
    filename = "<sparse_spectral_tangents_factory>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = dict(original.__wrapped__.__globals__)
    exec(compile(source, filename, "exec"), namespace)
    return namespace["get_solve_kernel"](100, metric_tangents=True)
