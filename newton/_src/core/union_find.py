# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Union-find over an int array of parents, by lowest id: every root is its component's lowest id, whatever order
the unions run in, so the components' labels are a function of the edges alone.

Threads may unite concurrently (:func:`union`, a compare-and-swap hook of the higher root under the lower one). A later
launch reads the final roots by loads (:func:`root`); a thread of the uniting launch reads them by atomics
(:func:`root_now`) once every union is done, as a plain load may be served stale by a non-coherent cache on CUDA."""

import warp as wp


@wp.func
def find(parent: wp.array[wp.int32], x: int) -> int:
    """Root of ``x``'s tree as the union pass sees it, halving the path on the way (union pass only)."""
    while True:
        p = parent[x]
        if p == x:
            return x
        grand = parent[p]
        if grand != p:
            # Path halving stores into other nodes' slots. In the union pass that is safe: a node only ever points
            # further up its own tree (a store computed from an older parent still points at an ancestor), and nothing
            # reads a parent as final there. A pass that stores final parents must not halve.
            parent[x] = grand
        x = grand


@wp.func
def root(parent: wp.array[wp.int32], x: int) -> int:
    """Root of ``x``'s tree, by loads alone: parents only point to lower ids, so the walk ends at the root."""
    p = parent[x]
    while p != x:
        x = p
        p = parent[x]
    return x


@wp.func
def root_now(parent: wp.array[wp.int32], x: int) -> int:
    """Root of ``x``'s tree read by atomics, inside the launch whose threads united it (after they all did)."""
    p = wp.atomic_add(parent, x, 0)
    while p != x:
        x = p
        p = wp.atomic_add(parent, x, 0)
    return x


@wp.func
def union(parent: wp.array[wp.int32], a: int, b: int):
    """Join ``a``'s and ``b``'s trees: the higher root hooked under the lower one."""
    ra = find(parent, a)
    rb = find(parent, b)
    while ra != rb:
        lo = wp.min(ra, rb)
        hi = wp.max(ra, rb)
        old = wp.atomic_cas(parent, hi, hi, lo)
        if old == hi:
            return
        # Another thread hooked hi first. Continue from its parent as the atomic returned it (lower than hi, so the
        # loop ends) rather than from a plain reread, which a non-coherent cache may serve stale on CUDA.
        if ra == hi:
            ra = old
        else:
            rb = old
