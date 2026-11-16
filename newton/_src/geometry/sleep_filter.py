# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The sleep filter of a collision pass: which resting trees the pass holds, decided from its candidate pairs.

A solver that holds resting trees (sleeping) shares two arrays with the collision pipeline
(:meth:`~newton.CollisionPipeline.configure_sleep_filter`, the contract ``SolverMuJoCo.collision_sleep_filter``
names):

- ``shape_sleep_index[shape]``: ``(world, tree)`` for a shape on a tree of the solver, ``(world, -1)`` for a static
  shape and ``(world, -2)`` for a shape on a body the solver never holds (a kinematic body);
- ``tree_asleep[world, tree]``: the solver's live state of each tree. A value ``>= 0`` is an asleep tree (the next
  tree of its island's ring, its own index for an island of one); ``-1`` is an awake tree that is ready to sleep; a
  value below ``-1`` is an awake tree that is not (its count of quiet steps still running). :data:`ACTED` marks a tree
  a drive, a control or a wrench acts on (it never rests while it is); the filter writes :data:`LOADED` to every
  other tree of an island that holds an acted-on tree (it counts no quiet step: the island sleeps as a whole, and an
  acted-on member never rests), and :data:`RELEASED` to a loaded tree whose island no longer holds one (its count
  starts again).

At every pass, after the broad phase and before the narrow phase, :meth:`SleepFilter.apply`:

1. joins two trees whose shapes are a candidate pair with overlapping boxes into one island. The boxes are grown by
   the contact gap and the speculative margin, so they hold every contact that can arise before the next pass;
2. marks an island *held back* when one of its trees is awake and not ready, or touches a moving body the solver
   never holds (a kinematic body with a velocity);
3. puts every tree of a ready island to sleep, and wakes every asleep tree of a held-back island (its quiet count
   kept: it was held at rest), so a woken tree is awake in the pass that makes its first contact, with the contacts
   of its supports; every tree of an island that holds an acted-on tree is loaded;
4. drops every candidate pair whose two sides are each static, never held or asleep: no contact is made for them.

The islands are connected components of the overlap graph (a union-find, hooked lock-free and exact in one launch),
so the decision holds at any island size, in a fixed launch list that a CUDA graph replays.
"""

from __future__ import annotations

import warp as wp

from .contact_data import SHAPE_PAIR_INDEX_MASK

wp.set_module_options({"enable_backward": False})

STATIC = wp.constant(-1)
"""A static shape's tree in ``shape_sleep_index``."""
UNHELD = wp.constant(-2)
"""The tree of a shape on a body the solver never holds (a kinematic body) in ``shape_sleep_index``."""
READY = wp.constant(-1)
"""An awake tree ready to sleep in ``tree_asleep``."""
ACTED = wp.constant(-(1 << 30))
"""A tree a drive, a control or a wrench acts on, in ``tree_asleep`` (the solver writes it)."""
LOADED = wp.constant(-(1 << 29))
"""A tree whose island held an acted-on tree at the last pass, in ``tree_asleep`` (the filter writes it)."""
RELEASED = wp.constant(-(1 << 28))
"""A loaded tree whose island no longer holds an acted-on tree, in ``tree_asleep`` (the filter writes it)."""


@wp.func
def _find(parent: wp.array[wp.int32], node: int) -> int:
    """The root of ``node``'s island, halving its path (a parent always has the lower index)."""
    x = node
    p = parent[x]
    while p != x:
        g = parent[p]
        if g != p:
            parent[x] = g
        x = p
        p = parent[x]
    return x


@wp.func
def _unite(parent: wp.array[wp.int32], a: int, b: int):
    """Join the islands of ``a`` and ``b``: the higher root hooks onto the lower, retried when another thread moved
    it first (lock-free; every parent points to a lower index, so the forest stays acyclic)."""
    while True:
        ra = _find(parent, a)
        rb = _find(parent, b)
        if ra == rb:
            return
        high = wp.max(ra, rb)
        low = wp.min(ra, rb)
        if wp.atomic_cas(parent, high, high, low) == high:
            return


@wp.func
def _overlap(sa: int, sb: int, lower: wp.array[wp.vec3], upper: wp.array[wp.vec3]) -> bool:
    la = lower[sa]
    ua = upper[sa]
    lb = lower[sb]
    ub = upper[sb]
    return not (la[0] > ub[0] or lb[0] > ua[0] or la[1] > ub[1] or lb[1] > ua[1] or la[2] > ub[2] or lb[2] > ua[2])


@wp.func
def _moving(shape: int, shape_body: wp.array[wp.int32], body_qd: wp.array[wp.spatial_vector]) -> bool:
    body = shape_body[shape]
    if body < 0:
        return False
    v = body_qd[body]
    return v[0] != 0.0 or v[1] != 0.0 or v[2] != 0.0 or v[3] != 0.0 or v[4] != 0.0 or v[5] != 0.0


@wp.kernel
def _seed(
    tree_asleep: wp.array2d[wp.int32],
    trees: int,
    parent: wp.array[wp.int32],
    held_back: wp.array[wp.int32],
    acted: wp.array[wp.int32],
):
    """Each tree its own island, held back where it is awake and not ready, acted on where the solver says so."""
    i = wp.tid()
    parent[i] = i
    state = tree_asleep[i // trees, i % trees]
    held_back[i] = wp.where(state < READY, 1, 0)
    acted[i] = wp.where(state == ACTED, 1, 0)


@wp.kernel
def _join(
    pairs: wp.array[wp.vec2i],
    pair_count: wp.array[wp.int32],
    shape_sleep_index: wp.array[wp.vec2i],
    shape_body: wp.array[wp.int32],
    body_qd: wp.array[wp.spatial_vector],
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
    trees: int,
    threads: int,
    parent: wp.array[wp.int32],
    held_back: wp.array[wp.int32],
):
    """Join the trees of every candidate pair whose boxes overlap; hold back a tree a moving unheld body reaches."""
    n = wp.min(pair_count[0], pairs.shape[0])
    for k in range(wp.tid(), n, threads):
        pair = pairs[k]
        if pair[0] < 0 or pair[1] < 0:
            continue
        sa = pair[0] & SHAPE_PAIR_INDEX_MASK
        sb = pair[1] & SHAPE_PAIR_INDEX_MASK
        ia = shape_sleep_index[sa]
        ib = shape_sleep_index[sb]
        if ia[1] < 0 and ib[1] < 0:
            continue
        if not _overlap(sa, sb, lower, upper):
            continue
        if ia[1] >= 0 and ib[1] >= 0:
            a = ia[0] * trees + ia[1]
            b = ib[0] * trees + ib[1]
            if a != b:
                _unite(parent, a, b)
        elif ia[1] >= 0:
            if ib[1] == UNHELD and _moving(sb, shape_body, body_qd):
                held_back[ia[0] * trees + ia[1]] = 1
        elif ia[1] == UNHELD and _moving(sa, shape_body, body_qd):
            held_back[ib[0] * trees + ib[1]] = 1


@wp.kernel
def _close(
    parent: wp.array[wp.int32], root: wp.array[wp.int32], held_back: wp.array[wp.int32], acted: wp.array[wp.int32]
):
    """Each tree's island root; a held-back tree holds back its island, an acted-on tree loads it."""
    i = wp.tid()
    r = _find(parent, i)
    root[i] = r
    if held_back[i] != 0:
        held_back[r] = 1
    if acted[i] != 0:
        acted[r] = 1


@wp.kernel
def _decide(
    root: wp.array[wp.int32],
    held_back: wp.array[wp.int32],
    acted: wp.array[wp.int32],
    trees: int,
    tree_asleep: wp.array2d[wp.int32],
):
    """A ready island sleeps (each tree a ring of one); an asleep tree of a held-back island wakes, ready; the other
    trees of an island that holds an acted-on tree are loaded, and released when it no longer does."""
    i = wp.tid()
    w = i // trees
    t = i % trees
    state = tree_asleep[w, t]
    if state == ACTED:
        return
    back = held_back[root[i]] != 0
    if acted[root[i]] != 0:
        tree_asleep[w, t] = LOADED
    elif state == LOADED:
        tree_asleep[w, t] = RELEASED
    elif state >= 0 and back:
        tree_asleep[w, t] = READY
    elif state == READY and not back:
        tree_asleep[w, t] = t


@wp.func
def _held(index: wp.vec2i, tree_asleep: wp.array2d[wp.int32]) -> bool:
    if index[1] < 0:
        return True
    return tree_asleep[index[0], index[1]] >= 0


@wp.kernel
def _mask(
    pairs: wp.array[wp.vec2i],
    pair_count: wp.array[wp.int32],
    shape_sleep_index: wp.array[wp.vec2i],
    tree_asleep: wp.array2d[wp.int32],
    threads: int,
):
    """Drop a candidate pair whose two sides are each static, unheld or asleep: no contact is made for it (a moving
    unheld body woke its asleep partner in :func:`_decide`)."""
    n = wp.min(pair_count[0], pairs.shape[0])
    for k in range(wp.tid(), n, threads):
        pair = pairs[k]
        if pair[0] < 0 or pair[1] < 0:
            continue
        ia = shape_sleep_index[pair[0] & SHAPE_PAIR_INDEX_MASK]
        ib = shape_sleep_index[pair[1] & SHAPE_PAIR_INDEX_MASK]
        if ia[1] < 0 and ib[1] < 0:
            continue
        if _held(ia, tree_asleep) and _held(ib, tree_asleep):
            pairs[k] = wp.vec2i(-1, -1)


class SleepFilter:
    """A collision pipeline's sleep filter over one solver's sleep state (see the module docstring).

    Args:
        shape_sleep_index: Each shape's ``(world, tree)``, ``(world, -1)`` static or ``(world, -2)`` unheld.
        tree_asleep: The solver's live sleep state of each tree, shape ``(world_count, trees)``.
        shape_body: Each shape's body (the model's).
        device: The pipeline's device.
    """

    def __init__(self, shape_sleep_index: wp.array, tree_asleep: wp.array, shape_body: wp.array, device):
        self.shape_sleep_index = shape_sleep_index
        self.tree_asleep = tree_asleep
        self.shape_body = shape_body
        self.device = device
        self.trees = int(tree_asleep.shape[1])
        count = max(1, int(tree_asleep.shape[0]) * self.trees)
        self.parent = wp.zeros(count, dtype=wp.int32, device=device)
        self.root = wp.zeros(count, dtype=wp.int32, device=device)
        self.held_back = wp.zeros(count, dtype=wp.int32, device=device)
        self.acted = wp.zeros(count, dtype=wp.int32, device=device)

    def apply(self, pairs: wp.array, pair_count: wp.array, lower: wp.array, upper: wp.array, body_qd: wp.array):
        """Decide the pass's sleep and wake from the broad phase's candidate ``pairs`` and the grown boxes
        (``lower``, ``upper``), then drop the pairs of held sides (:func:`_mask`); ``body_qd`` tells a moving unheld
        body."""
        islands = int(self.tree_asleep.shape[0]) * self.trees
        if islands == 0:
            return
        threads = max(1, min(int(pairs.shape[0]), 1 << 16))
        device = self.device
        wp.launch(
            _seed,
            dim=islands,
            inputs=[self.tree_asleep, self.trees],
            outputs=[self.parent, self.held_back, self.acted],
            device=device,
            record_tape=False,
        )
        wp.launch(
            _join,
            dim=threads,
            inputs=[
                pairs,
                pair_count,
                self.shape_sleep_index,
                self.shape_body,
                body_qd,
                lower,
                upper,
                self.trees,
                threads,
            ],
            outputs=[self.parent, self.held_back],
            device=device,
            record_tape=False,
        )
        wp.launch(
            _close,
            dim=islands,
            inputs=[self.parent],
            outputs=[self.root, self.held_back, self.acted],
            device=device,
            record_tape=False,
        )
        wp.launch(
            _decide,
            dim=islands,
            inputs=[self.root, self.held_back, self.acted, self.trees],
            outputs=[self.tree_asleep],
            device=device,
            record_tape=False,
        )
        wp.launch(
            _mask,
            dim=threads,
            inputs=[pairs, pair_count, self.shape_sleep_index, self.tree_asleep, threads],
            device=device,
            record_tape=False,
        )
