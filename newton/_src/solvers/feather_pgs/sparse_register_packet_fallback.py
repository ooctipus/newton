# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Materialize original sparse rows only after a packet-world rejection.

The key producer already finalized allocation and row metadata. These kernels
only publish physical coefficients and apply the original restitution rule;
they never allocate again or change counters, row order, or incoming impulses.
"""

import inspect
import linecache
import textwrap
from functools import cache

import warp as wp

from . import sparse_factor_rows
from .kernels import contact_restitution_fires
from .sparse_factor import SparseData, SparsePlan


@cache
def get_prefix_kernel():
    """Decode finalized negative keys without rerunning prefix allocation."""

    def prefix(
        p: SparsePlan,
        d: SparseData,
        counts: wp.array[int],
        vhat: wp.array[float],
        cfm: wp.array2d[float],
        diagonal: wp.array2d[float],
        routing: wp.array[int],
    ):
        group, row = wp.tid()
        art = p.group_to_art[group]
        world = p.art_to_world[art]
        if routing[world] != 0 or row >= counts[world] or row >= d.support.shape[1]:
            return
        key = d.support[world, row]
        if key >= 0:
            return
        if key < -86:
            wp.atomic_or(d.status, world, 4)
            return
        candidate = -key - 1
        local = candidate // 2
        sign = float(1.0)
        if candidate % 2 != 0:
            sign = -1.0
        template = p.limit_support[local]
        norm = float(0.0)
        # Keep the original parallel-prefix scalar accumulation, including
        # all zero padding. This is not the contact warp reduction.
        for k in range(18):
            node = p.support_nodes[template, k]
            z = float(0.0)
            if node >= 0:
                entry = p.index[node, 42 - local]
                if entry >= 0:
                    z = sign * d.W[group, entry]
            d.Z[world, row, k] = z
            norm += z * z
        diagonal[world, row] = norm + cfm[world, row]
        d.incident[world, row] = sign * vhat[p.art_dof_start[art] + local]
        d.support[world, row] = template

    prefix.__name__ = prefix.__qualname__ = "sparse_register_packet_prefix43"
    return wp.kernel(enable_backward=False, module="unique")(prefix)


@cache
def get_contact_kernel():
    """Retain the original triplet body with an early whole-world route guard."""
    source = textwrap.dedent(inspect.getsource(sparse_factor_rows.get_contact_kernel))

    def replace(old, new, count=1):
        if source.count(old) != count:
            raise RuntimeError(f"Original sparse contact source changed: {old!r}")
        return source.replace(old, new)

    source = replace("@cache\n", "")
    source = replace("def get_contact_kernel():", "def filtered_contact_factory():")
    source = replace(
        "        const int w = world.data[c], row0 = slot.data[c];",
        "        const int w = world.data[c], row0 = slot.data[c];\n        if (routing.data[w] != 0) continue;",
    )
    source = replace(
        "        diagonal: wp.array2d[float],\n",
        "        diagonal: wp.array2d[float],\n        routing: wp.array[int],\n",
        count=2,
    )
    source = replace("            diagonal,\n", "            diagonal,\n            routing,\n")
    source = replace("sparse_factor_contact_triplet18", "sparse_register_packet_contact_triplet18")
    filename = f"<{__name__}.filtered_contacts>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = dict(sparse_factor_rows.__dict__)
    namespace["__name__"] = __name__
    exec(compile(source, filename, "exec"), namespace)
    return namespace["filtered_contact_factory"]()


@cache
def get_restitution_kernel():
    """Apply the original one-shot restitution only to materialized worlds."""

    def restitution(
        d: SparseData,
        counts: wp.array[int],
        row_type: wp.array2d[int],
        phi: wp.array2d[float],
        target: wp.array2d[float],
        restitution: wp.array2d[float],
        dt: float,
        threshold: float,
        rhs: wp.array2d[float],
        routing: wp.array[int],
    ):
        world, row = wp.tid()
        if routing[world] != 0:
            return
        if row >= counts[world] or row_type[world, row] != 0:
            return
        e = restitution[world, row]
        incident = d.incident[world, row] - target[world, row]
        if e > 0.0 and contact_restitution_fires(phi[world, row], incident, dt, threshold):
            rhs[world, row] = -target[world, row] + e * incident

    restitution.__name__ = restitution.__qualname__ = "sparse_register_packet_restitution"
    return wp.kernel(enable_backward=False, module="unique")(restitution)
