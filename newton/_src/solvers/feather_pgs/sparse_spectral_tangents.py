# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Optional ordered normal-first/shared-spectral tangent projection.

This is the previously screened spectral-GS policy, not a new contact law.
It keeps the original sparse owner's geometry, physical response, scalar
fallback, iteration allowance and publication. CFM enters only denominators.
The per-contact scalar cache lives for one fixed-row solve, never across calls.
"""

import functools

from . import sparse_factor_rows, sparse_metric_tangents


def _replace(source, before, after, count=1):
    if source.count(before) != count:
        raise RuntimeError(f"Sparse spectral source seam changed: {before!r}")
    return source.replace(before, after)


def get_fragments(capacity: int, *, support_width: int = 18, dof_count: int = 43):
    """Keep the original fused transaction around a root-free local proposal."""
    setup, update = sparse_metric_tangents.get_fragments(capacity, support_width=support_width, dof_count=dof_count)
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
def get_solve_kernel(*, dof_count: int, support_width: int, inverse_width: int, nonzero_count: int):
    """Select the explicit CFM-aware spectral owner from the shared factory."""
    return sparse_factor_rows.get_solve_kernel(
        100,
        dof_count=dof_count,
        support_width=support_width,
        inverse_width=inverse_width,
        nonzero_count=nonzero_count,
        metric_tangents=True,
        spectral_tangents=True,
    )
