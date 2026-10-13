# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared allocation output initialization with explicit prefix producers."""

import hashlib
import linecache
from string import Template

from .kinetic_source import checked_definitions

_SCAN = r"""    count = int(0)
    if state.resolved[world] == 0:
        for code in range(46):
            dof = plan.dof_ids[world, code // 2]
            qi = prefix.q_index[dof]
            if qi >= 0:
                q = prefix.q[qi]
                bound = prefix.lower[dof]
                active = wp.isfinite(bound) and q <= bound + settings.activation_gap
                if (code & 1) != 0:
                    bound = prefix.upper[dof]
                    active = wp.isfinite(bound) and q >= bound - settings.activation_gap
                if active:
                    count += 1
"""


_INITIALIZE = r'''${decorator}
def ${name}(
    plan: KineticPlan,
    prefix: PrefixInput,
    raw: RawRowInput,
    state: RowState,
    settings: RowSettings,
    allocation: AllocationData,
${extra}):
    """Charge current scalar-limit scan and initialize every allocation output."""
    world = wp.tid()
${prefix_count}    allocation.dense_counter[world] = count
    allocation.prefix_count[world] = count
    allocation.mf_counter[world] = 0
    allocation.propagation_counter[world] = 0
    allocation.propagation_count[world] = 0
    allocation.dense_contact_flag[world] = 0
    allocation.dropped_dense[world] = 0
    allocation.dropped_mf[world] = 0
    allocation.dropped_propagation[world] = 0
    allocation.first_rejected_dense[world] = 2147483647
    allocation.first_rejected_mf[world] = 2147483647
    allocation.first_rejected_propagation[world] = 2147483647
    state.dense_count[world] = 0
    state.mf_count[world] = 0
    if world == 0:
        for family in range(4):
            state.capacity_status[family] = 0
        if raw.count[0] < 0 or raw.count[0] > raw.shape0.shape[0]:
            state.raw_invalid[0] = 1
        if state.raw_invalid[0] != 0:
            state.capacity_status[3] = 1
'''


def get_source(*, cached_geometry=False):
    """Keep every allocation reset and capacity check under one authority."""
    checked_definitions("allocation_count_source.py", ("_SCAN", "_INITIALIZE", "get_source", "bind"))
    prefix = (
        "    selected = cache.rejected[world] == 0 and state.raw_invalid[0] == 0\n"
        "    state.resolved[world] = int(selected)\n    count = int(0)\n"
        "    if not selected:\n        index = wp.atomic_add(state.active_count, 0, 1)\n"
        "        state.active_worlds[index] = world\n        count = cache.prefix_count[world]\n"
        if cached_geometry
        else _SCAN
    )
    return Template(_INITIALIZE).substitute(
        decorator='@wp.kernel(module="unique", enable_backward=False)' if cached_geometry else "@wp.kernel",
        name="kinetic_contact_finalize" if cached_geometry else "initialize_count",
        extra="    cache: ContactGeometry,\n" if cached_geometry else "",
        prefix_count=prefix,
    )


def bind(namespace, *, cached_geometry=False):
    """Bind the selected producer with the common output-reset contract."""
    source = get_source(cached_geometry=cached_geometry)
    filename = "<allocation-count-" + hashlib.sha256(source.encode()).hexdigest() + ">"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    scope = dict(namespace)
    exec(compile(source, filename, "exec"), scope)
    return scope["kinetic_contact_finalize" if cached_geometry else "initialize_count"]
