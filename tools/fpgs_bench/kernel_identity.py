# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Record and compare the generated code of FeatherPGS kernels without a GPU.

For every kernel of a fixed inventory, ``record`` stores Warp's content hash and
digests of the CUDA and CPU code Warp generates for it; ``diff`` compares two
records. The content hash is the kernel's cache key and the value that
``newton/tests/test_feather_pgs_point_streams.py`` pins. The code digests ignore
comments (generated code quotes every Python line with its line number),
``#line`` directives, whitespace outside string literals and the hash suffix of
kernel names. A refactor is then exact:

* by hash when no hash changed: every kernel hits its existing cache entry, so
  the binaries are the ones already built;
* by generated code when hashes changed but no code digest did, as when only
  the Python text of a kernel changed (formatting, comments, its position).

A kernel recorded without a requested code digest (``--hash-only``) is identical
only by an unchanged hash; with a changed hash the diff reports it as not
compared and fails.

Warp's generated code also depends on the process that generates it: user
functions are numbered by how many functions of their name the process created
before, a function emits its adjoint once any earlier build used it backward,
and some loop-carried variables are ordered by Python's salted string hash. The
record removes all three: it renames functions by build order within the
kernel, builds each kernel as a fresh process would, and runs with
``PYTHONHASHSEED=0``.

The inventory is the point-stream pin inventory (every module-level kernel of
the FeatherPGS package, the held-tree owner's scalar-branch group included, and
the pinned factory calls), the factory variants those pins do not reach
(:func:`factory_calls`), a held-tree layout grid of the factory-built tree
kernels (:func:`held_tree_calls`) and a census of every flag combination of the
sweep factories that vote an exit (:func:`census_calls`), recorded by hash only, and the walk's readers over the
suite's class shapes (:func:`walk_calls`).
Nothing is compiled and no device is needed; a full record takes under three
minutes on the CPU, the census about two of them.

Run from the root of a Newton checkout with the import-time switches
(``FEATHER_PGS_*``) unset::

    python tools/fpgs_bench/kernel_identity.py record child.json
    python tools/fpgs_bench/kernel_identity.py record parent.json --root /path/to/parent
    python tools/fpgs_bench/kernel_identity.py diff parent.json child.json

``--root`` imports Newton from another checkout, for example a ``git archive``
of the parent commit, so one copy of this script records both sides.
"""

import argparse
import collections
import hashlib
import inspect
import itertools
import json
import os
import re
import sys
import time
from pathlib import Path

ARCH = "120"
FAMILIES = ("pins", "factories", "held_tree", "walk", "census")
COLUMNS = ("hash", "cuda", "cpu")
# Import-time switches select other generated kernels (environment_flags.PREFIXES).
SWITCH_PREFIXES = ("FEATHER_PGS_", "IL_NEWTON_FPGS_")

_TOKENS = re.compile(
    r"(?P<string>\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*')|//[^\n]*|/\*.*?\*/|[ \t\f\v]+",
    re.S,
)


def normalize(source: str, suffixes=(), functions=None) -> str:
    """Return generated code without comments, ``#line`` directives, layout whitespace or process-dependent names.

    Args:
        source: Generated CUDA or CPU code.
        suffixes: Hash prefixes that Warp appends to kernel names (eight hex
            digits of the content hash), replaced wherever they end an identifier
            segment, as in ``name_<hash>_cuda_kernel_forward`` and ``wp_args_name_<hash>``.
        functions: ``{native name: stable name}`` of the user functions built.
            Warp numbers a user function by how many functions of its name the
            process created before it, so ``native_26`` in one process is
            ``native_0`` in another; the forward, ``adj_``, ``replay_`` and
            ``_callable_`` names of each are renamed.
    """
    text = _TOKENS.sub(lambda match: match.group("string") or " ", source)
    if suffixes:
        text = re.sub("_(?:" + "|".join(map(re.escape, suffixes)) + ")(?![0-9A-Za-z])", "_HASH", text)
    if functions:
        names = "|".join(map(re.escape, sorted(functions, key=len, reverse=True)))
        pattern = r"(?<![0-9A-Za-z_])((?:adj_|replay_)?)(" + names + r")(?=_callable_|(?![0-9A-Za-z_]))"
        text = re.sub(pattern, lambda match: match.group(1) + functions[match.group(2)], text)
    lines = (line.strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line and not line.startswith("#line"))


def stable_function_names(functions):
    """Name built user functions by key and build order instead of by process-wide creation count."""
    names, counts = {}, {}
    for function in functions:
        if function.is_builtin():
            continue
        position = counts[function.key] = counts.get(function.key, -1) + 1
        names[function.native_func] = f"{function.key}__fn{position}"
    return names


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class Named:
    """A factory argument that labels show by name instead of by value."""

    def __init__(self, name, value):
        self.name, self.value = name, value

    def __repr__(self):
        return self.name


def label(name, args=(), kwargs=None):
    """Name a factory call as ``test_feather_pgs_point_streams`` names its pins."""
    kwargs = kwargs or {}
    parts = [repr(arg) for arg in args] + [f"{key}={kwargs[key]!r}" for key in sorted(kwargs)]
    return f"{name}({', '.join(parts)})"


def _value(argument):
    return argument.value if isinstance(argument, Named) else argument


def _unwrapped(factory):
    """Bypass a factory's cache so that every record builds its kernels anew."""
    return getattr(factory, "__wrapped__", factory)


def factory_calls():
    """Yield ``(name, factory, args, kwargs)`` for factory variants the point-stream pins do not reach.

    Each matrix-free Gauss-Seidel flag is toggled alone on two bases (the pins
    hold the suite's shapes); the fallback and linked parallel tiers of a
    second keyed-row coordinate layout (one articulation, no secondary block)
    follow. Calls a factory rejects are recorded as rejected.
    """
    from newton._src.solvers.feather_pgs import (  # noqa: PLC0415
        keyed_rows,
        sweep_routes,
        sweep_routes_tiers,
    )
    from newton._src.solvers.feather_pgs.keyed_row_layout import KeyedLayout  # noqa: PLC0415

    mf_gs = _unwrapped(sweep_routes.mf_gs_kernel)
    toggles = (
        {},
        {"shared_metadata": False},
        {"stream_mf_impulses": True},
        {"contact_torsion": True},
        {"skip_local_internal_worlds": True, "local_internal_max_constraints": 19},
        {"skip_tier_rows": 32},
        {"factor_coordinates": True},
        {"factor_coordinates": True, "independent_components": True},
    )
    for base in ({}, {"has_dense_velocity_limit_rows": False}):
        for change in toggles:
            yield "sweep_routes.mf_gs_kernel", mf_gs, (64, 32, 12, ARCH), {**base, **change}

    def parallel(layout, *args, **kwargs):
        factory = keyed_rows.get_parallel_factory.__wrapped__(sweep_routes_tiers.parallel_kernel, layout)
        return factory(*args, **kwargs)

    layout = Named("KeyedLayout(18, 0, 0, 0, 48, False, 0)", KeyedLayout(18, 0, 0, 0, 48, False, 0))
    for rows, min_rows in ((32, 0), (48, 32)):
        kwargs = {
            "rows": rows,
            "min_rows": min_rows,
            "sweeps": 24,
            "inkernel_response": (18, 0, 0, 0),
            "linked_rows": True,
        }
        yield "keyed_rows.get_parallel_factory", parallel, (layout, 72, 32, 18, ARCH), kwargs


def _schedules(layout, chain):
    """A held layout's scan schedules: every scan on the heavy chains (``chain``), the shared-memory jumps and
    depth levels of two warps, or the warp schedules (two warps: shared memory and levels)."""
    from newton._src.solvers.feather_pgs import tree_kinetic_state as native  # noqa: PLC0415
    from newton._src.solvers.feather_pgs import tree_scan as ts  # noqa: PLC0415

    schedule = ts.Schedule
    if layout.lanes > 32 and not chain:
        return native.HeldSchedules(schedule.JUMP_SMEM, schedule.JUMP_SMEM, schedule.LEVEL)
    if chain:
        return native.HeldSchedules(*(schedule.HEAVY_CHAIN,) * 3)
    return native.HeldSchedules(schedule.JUMP_SHFL, schedule.JUMP_SHFL, schedule.DFS_INTERVAL)


def held_tree_layouts():
    """Return ``{name: (layout, chain, response, repair_options)}`` at the suite's held-tree shapes; ``chain``
    runs every scan on the heavy chains (:func:`_schedules`).

    Chain plans with free and fixed roots (two at 64 lanes, one of them with a
    stored factor permutation), kinetic plans at 4, 8, 16, 32 and 64 lanes, and
    forests with a free and a prescribed root, with sparse, Cholesky and
    diagonal responses and an alternate repair motion. Every layout but
    ``chain_free44_l64_permuted``, ``kinetic_free1_l8_unlimited``,
    ``kinetic_free17``, ``kinetic_free44_l64`` and ``forest10_free`` has the
    sizes, entry count, depth and permutation of a tree the suite runs, so its
    finish and repair compile as that tree's do. ``kinetic_free1_l8_unlimited``
    is the suite's 8-lane free body without velocity limits: its repairs
    compare input banks, so the grid records the comparing repair and the
    comparison pass at every packed width. The depth of a chain plan reaches
    none of its kernels.
    """
    from newton._src.solvers.feather_pgs.tree_kinetic_layout import TreeLayout  # noqa: PLC0415

    def reverse(dofs):
        return tuple(range(dofs - 1, -1, -1))

    def tree(bodies, dofs, depth, root, *, lanes=0, entries=None, inverse=None, permutation=None, **forest):
        entries = dofs * (dofs + 1) // 2 if entries is None else entries
        inverse = dofs if inverse is None else inverse
        permutation = reverse(dofs) if permutation is None else permutation
        return TreeLayout(bodies, dofs, entries, inverse, depth, root, permutation, lanes, **forest)

    motion = {"motion_velocity": True}
    return {
        "chain_free9": (tree(9, 14, 2, 6, entries=81, inverse=8), True, "sparse", {}),
        "chain_free13": (tree(13, 18, 5, 6, entries=135, inverse=12), True, "sparse", {}),
        "chain_free16": (tree(16, 27, 5, 6, entries=243), True, "sparse", {}),
        "chain_free17": (tree(17, 18, 4, 6, entries=117, inverse=9), True, "sparse", {}),
        "chain_free19": (tree(19, 18, 4, 6, entries=117, inverse=9), True, "sparse", {}),
        "chain_free20": (tree(20, 25, 5, 6, entries=194, inverse=11), True, "sparse", {}),
        "chain_free44_l64": (tree(44, 43, 10, 6, entries=434), True, "sparse", {}),
        "chain_free44_l64_permuted": (
            tree(44, 43, 10, 6, entries=434, permutation=tuple(range(43))),
            True,
            "sparse",
            {},
        ),
        "chain_fixed8": (tree(8, 6, 6, 0), True, "sparse", motion),
        "chain_fixed11": (tree(11, 9, 8, 0, entries=44), True, "sparse", motion),
        "kinetic_fixed3_l4": (tree(3, 2, 2, 0, lanes=4), False, "cholesky", {}),
        "kinetic_free1_l8": (tree(1, 6, 0, 6, lanes=8), False, "cholesky", motion),
        "kinetic_free1_l8_unlimited": (tree(1, 6, 0, 6, lanes=8), False, "cholesky", {}),
        "kinetic_fixed11_l16": (tree(11, 9, 9, 0, lanes=16, entries=44), False, "cholesky", {}),
        "kinetic_fixed9_l16_diagonal": (tree(9, 4, 2, 0, lanes=16, entries=4), False, "diagonal", {}),
        "kinetic_free17": (tree(17, 18, 4, 6), False, "cholesky", {}),
        "kinetic_fixed21": (tree(21, 16, 6, 0, entries=40), False, "cholesky", motion),
        "kinetic_free44_l64": (tree(44, 43, 10, 6), False, "sparse", {}),
        "forest13_l16": (
            tree(
                13,
                21,
                9,
                0,
                lanes=16,
                entries=86,
                roots=(0, 11, 12),
                root_coordinates=(0, 6, 6),
                root_offsets=(0, 9, 15),
                root_families=(9, 6, 0),
            ),
            False,
            "cholesky",
            {},
        ),
        "forest32": (
            tree(
                32,
                35,
                14,
                0,
                entries=222,
                roots=(0, 30, 31),
                root_coordinates=(0, 6, 6),
                root_offsets=(0, 23, 29),
                root_families=(23, 6, 0),
            ),
            False,
            "cholesky",
            {},
        ),
        "forest10_free": (
            tree(
                10,
                15,
                8,
                0,
                lanes=16,
                entries=66,
                roots=(0, 9),
                root_coordinates=(0, 6),
                root_offsets=(0, 9),
                root_families=(9, 6),
            ),
            False,
            "cholesky",
            {},
        ),
    }


def walk_classes(layout):
    """Each root's walking class of a held layout (the predictor solves by the walk): a chain over the root's bodies,
    the root body holding its root's coordinates (one for a fixed root) and every next body one until the root's
    family is spent; ``None`` for a prescribed root."""
    from newton._src.solvers.feather_pgs.tree_scan import WalkClass  # noqa: PLC0415

    bounds = (*layout.roots, layout.bodies)
    families = layout.root_families if layout.forest else (layout.dofs,)
    classes = []
    for r, family in enumerate(families):
        if family == 0:
            classes.append(None)
            continue
        bodies = bounds[r + 1] - bounds[r]
        counts = [min(layout.root_coordinates[r] or 1, family)]
        for _ in range(1, bodies):
            counts.append(min(1, family - sum(counts)))
        counts[-1] += family - sum(counts)
        first = [0]
        for count in counts:
            first.append(first[-1] + count)
        classes.append(WalkClass(tuple(range(-1, bodies - 1)), tuple(first)))
    return tuple(classes)


def held_tree_calls():
    """Yield every held-tree layout's kernels as they run, plus the switch variants of four layouts.

    Every held-tree operator requires body velocity, so the state kernels are
    recorded with it. Each layout yields its repair, finish and predictor and,
    outside motion-velocity layouts, whose stamps a copied bank may name, the
    one-pass repair that compares input banks (the first repair of a captured
    step) and the kernels of a launch large enough to classify: the
    classification, the listed repair and the comparison pass. Owners classify
    only layouts of ordinary motion; a motion-velocity layout keeps its
    one-pass repair at any size. The switched layouts add the comparison
    forms, the stage clocks and the repair and finish without body velocity.
    A scalar-branch group of the held-tree owner has no layout: its repair,
    predict, finish and listed drive kernels are module-level kernels of
    ``scalar_branches``, which the pins family records.
    """
    from newton._src.solvers.feather_pgs import tree_kinetic_state as native  # noqa: PLC0415
    from newton._src.solvers.feather_pgs.tree_factor import RELEASE_SLIPS  # noqa: PLC0415

    state = _unwrapped(native.get_state_kernel)
    predictor = _unwrapped(native.get_predictor_kernel)
    classify = _unwrapped(native.get_classify_kernel)
    bank = _unwrapped(native.get_bank_kernel)
    switched = ("chain_free13", "kinetic_fixed11_l16", "forest13_l16", "forest32")
    for layout_name, (tree, chain, response, repair_options) in held_tree_layouts().items():
        layout = {"layout": Named(layout_name, tree)}
        # The alternate motion is a repair-only contract, body velocity a state-kernel one.
        finish_options = {key: value for key, value in repair_options.items() if key != "motion_velocity"}
        variants = [{}]
        if layout_name in switched:
            variants += [{"clocked": True}, {"body_velocity": False}]
        schedules = _schedules(tree, chain)
        for options in variants:
            state_options = {"body_velocity": True, **options}
            kwargs = {**layout, **repair_options, **state_options}
            yield "tree_kinetic_state.get_state_kernel", state, (False, schedules), kwargs
            kwargs = {**layout, **finish_options, **state_options}
            yield "tree_kinetic_state.get_state_kernel", state, (True, schedules), kwargs
            if options.keys() <= {"clocked"}:
                # The CUDA predictor: it holds the twist publication where the layout admits it.
                twists = native.publishes_twists(tree, schedules, cuda=True)
                kwargs = {
                    **layout,
                    "response": response,
                    "twists": twists,
                    "classes": walk_classes(tree),
                    "slips": RELEASE_SLIPS,
                    **options,
                }
                yield "tree_kinetic_state.get_predictor_kernel", predictor, (schedules,), kwargs
        # A group's coordinates: a free root's quaternion takes one more than its three rotation velocities.
        coordinates = tree.dofs + tree.free_root_count
        schedules = _schedules(tree, chain)
        if "motion_velocity" not in repair_options:
            kwargs = {**layout, "body_velocity": True, "bank_coordinates": coordinates}
            yield "tree_kinetic_state.get_state_kernel", state, (False, schedules), kwargs
            classify_options = {key: value for key, value in repair_options.items() if key != "motion_velocity"}
            yield "tree_kinetic_state.get_classify_kernel", classify, (), {**layout, **classify_options}
            kwargs = {**layout, **repair_options, "body_velocity": True, "listed": True}
            yield "tree_kinetic_state.get_state_kernel", state, (False, schedules), kwargs
            yield "tree_kinetic_state.get_bank_kernel", bank, (), {**layout, "coordinates": coordinates}


def _walk_class(parent, first, half):
    from newton._src.solvers.feather_pgs.tree_scan import WalkClass  # noqa: PLC0415

    return WalkClass(tuple(parent), tuple(first), tuple(half))


def _limbs(base, limbs, length, start):
    """Parents of ``limbs`` chains of ``length`` bodies each hung from body ``base``, numbered from ``start``."""
    parent = []
    for limb in range(limbs):
        first = start + limb * length
        parent += [base, *range(first, first + length - 1)]
    return parent


def walk_classes_of_the_suite():
    """Return ``{name: WalkClass}`` at the suite's walking shapes, each coordinate's half compiled in.

    ``g1``: a free root of six, two legs of six, a waist of three, two arms of seven with two two-link fingers each
    (38 bodies, 43 coordinates, joins at the pelvis, the torso and the hands); ``franka``: a fixed arm of seven and two
    fingers on the last link (9); ``allegro``: a palm fixed to the world and four fingers of four (16); ``cube``: a
    free body (6); ``kuka``: an arm of seven, a palm and the allegro fingers (23); ``chain106``: 106 one-coordinate
    bodies, past the emitter's straight-line limit (the loop form); ``d6``: a free root, a D6 joint whose turns follow
    its slides (each turn's column of both halves, the record's side) and a revolute child.
    """
    linear, angular, both = 0, 1, 2
    g1 = [-1, *_limbs(0, 2, 6, 1), 0, 13, 14]
    g1 += [15, *range(16, 22), 22, 23, 22, 25, 15, *range(27, 33), 33, 34, 33, 36]
    shapes = {
        "g1": (g1, [0, 6, *range(7, 44)], (linear,) * 3 + (angular,) * 40),
        "franka": ([-1, *range(7), 6], range(10), (angular,) * 7 + (linear,) * 2),
        "allegro": ([-1, *_limbs(0, 4, 4, 1)], [0, *range(17)], (angular,) * 16),
        "cube": ([-1], [0, 6], (linear,) * 3 + (angular,) * 3),
        "kuka": ([-1, *range(7), *_limbs(7, 4, 4, 8)], [*range(8), *range(7, 24)], (angular,) * 23),
        "chain106": ([-1, *range(105)], range(107), (angular,) * 106),
        "d6": ([-1, 0, 1], [0, 6, 12, 13], (linear,) * 3 + (angular,) * 3 + (linear,) * 3 + (both,) * 3 + (angular,)),
    }
    return {name: _walk_class(*shape) for name, shape in shapes.items()}


def walk_calls():
    """Yield ``(name, factory, args, kwargs)`` for the walk's readers over the suite's class shapes
    (:func:`walk_classes_of_the_suite`): every kernel whose text a walk operator writes.

    The step's leave of each class, with and without the slips' release (and the support placement's nodes on g1),
    and of allegro's hand with its cube; the audit's enter and the refresh; the walk's solve of each class (chain106
    in the loop form); the predictor's force and response halves on a tree of each held width: one word a lane
    (franka), two (g1, with joins) and four (chain106); the support writer on g1 and kuka; the
    general writer with and without sides; the free-body headers; and the sweeps that walk rows (the matrix-free
    Gauss-Seidel's free bodies, the local owners' block, the parallel tiers' rows).
    """
    from newton._src.solvers.feather_pgs import (  # noqa: PLC0415
        sweep_routes,
        sweep_routes_local,
        sweep_routes_tiers,
        tree_factor,
        walk_rows,
    )
    from newton._src.solvers.feather_pgs import tree_kinetic_state as native  # noqa: PLC0415
    from newton._src.solvers.feather_pgs.tree_kinetic_layout import TreeLayout  # noqa: PLC0415

    classes = walk_classes_of_the_suite()
    named = {name: Named(f"({name},)", (cls,)) for name, cls in classes.items()}
    leave = _unwrapped(tree_factor.walk_leave_kernel)
    for name in classes:
        for release in (False, True):
            yield "tree_factor.walk_leave_kernel", leave, (named[name], release), {}
    pair = Named("(allegro, cube)", (classes["allegro"], classes["cube"]))
    yield "tree_factor.walk_leave_kernel", leave, (pair, True), {}
    nodes = Named("g1 nodes", (tuple(range(classes["g1"].coordinates - 1, -1, -1)),))
    yield "tree_factor.walk_leave_kernel", leave, (named["g1"], True, nodes), {}
    yield "tree_factor.walk_enter_kernel", _unwrapped(tree_factor.walk_enter_kernel), (), {}
    yield "tree_factor.walk_step_kernel", lambda: tree_factor.walk_step_kernel, (), {}

    def layout(name, root):
        cls = classes[name]
        bodies, dofs = len(cls.parent), cls.coordinates
        depth = max(_depth(cls.parent, b) for b in range(bodies))
        tree = TreeLayout(bodies, dofs, dofs * (dofs + 1) // 2, dofs, depth, root, tuple(range(dofs - 1, -1, -1)))
        return Named(f"{name} layout", tree)

    solve = _unwrapped(native.get_walk_solve_kernel)
    force = _unwrapped(native.get_force_kernel)
    predictor = _unwrapped(native.get_predictor_kernel)
    roots = {"g1": 6, "franka": 1, "kuka": 1, "chain106": 1, "d6": 6}
    for name, root in roots.items():
        tree = layout(name, root)
        yield "tree_kinetic_state.get_walk_solve_kernel", solve, (), {"layout": tree, "classes": named[name]}
    for name in ("franka", "g1", "chain106"):
        tree = layout(name, roots[name])
        schedules = Named("schedules", _schedules(tree.value, False))
        kwargs = {"layout": tree, "response": "sparse", "classes": named[name]}
        yield "tree_kinetic_state.get_force_kernel", force, (schedules,), kwargs
        twists = native.publishes_twists(tree.value, schedules.value, cuda=True)
        kwargs = {**kwargs, "twists": twists, "slips": tree_factor.RELEASE_SLIPS}
        yield "tree_kinetic_state.get_predictor_kernel", predictor, (schedules,), kwargs
        if name == "g1":
            yield "tree_kinetic_state.get_predictor_kernel", predictor, (schedules,), {**kwargs, "side": False}
    support = _unwrapped(walk_rows.support_rows_kernel)
    for name in ("g1", "kuka"):
        cls = classes[name]
        node = tuple(range(cls.coordinates - 1, -1, -1))
        for records, torsion in ((False, False), (True, True)):
            args = (Named(f"{name} parent", cls.parent), Named(f"{name} first", cls.first), Named(f"{name} node", node))
            yield "walk_rows.support_rows_kernel", support, args, {"records": records, "torsion": torsion}
    for side in (False, True):
        yield "walk_rows.walked_rows_kernel", _unwrapped(walk_rows.walked_rows_kernel), (8, True, side), {}
    mf_gs = _unwrapped(sweep_routes.mf_gs_kernel)
    free = {"factor_coordinates": True, "independent_components": True, "has_dense_velocity_limit_rows": False}
    yield "sweep_routes.mf_gs_kernel", mf_gs, (64, 32, 12, ARCH), free
    local = _unwrapped(sweep_routes_local.local_owned_kernel)
    yield "sweep_routes_local.local_owned_kernel", local, (192, 20, 9, ARCH), {"warps_per_block": 2}
    tiers = {"rows": 32, "min_rows": 0, "sweeps": 24, "inkernel_response": (18, 0, 0, 0)}
    yield "sweep_routes_tiers.parallel_kernel", sweep_routes_tiers.parallel_kernel, (64, 32, 18, ARCH), tiers


def _depth(parent, body):
    depth = 0
    while parent[body] >= 0:
        body, depth = parent[body], depth + 1
    return depth


def _axis(name, values=(False, True)):
    """One census axis: a keyword over its values."""
    return tuple({name: value} for value in values)


def _combinations(axes):
    """Merge one option of every axis, for every combination of options."""
    for options in itertools.product(*axes):
        merged = {}
        for option in options:
            merged.update(option)
        yield merged


def census_calls():
    """Yield ``(name, factory, args, kwargs)`` for every flag combination of the sweep factories that vote an exit.

    A rewrite of these factories' friction-law splices and exit votes, or of
    the matrix-free Gauss-Seidel generator's text rewrites, can be exact for
    each flag alone and still change a combination of flags. The census takes
    the product of each factory's flags at fixed shapes: matrix-free
    Gauss-Seidel (one shape within a warp's width of coordinates, one beyond
    it), sparse (one shape small enough for subwarp sweeps), paired factor,
    local owners plain and over packet rows, sparse diagonal, parallel and
    keyed parallel tiers. The matrix-free Gauss-Seidel sweep is crossed with
    resident row metadata.
    """
    from newton._src.solvers.feather_pgs import (  # noqa: PLC0415
        keyed_rows,
        row_packets,
        sweep_routes,
        sweep_routes_diag,
        sweep_routes_local,
        sweep_routes_tiers,
    )
    from newton._src.solvers.feather_pgs.keyed_row_layout import KeyedLayout  # noqa: PLC0415
    from newton._src.solvers.feather_pgs.sweep_routes_sparse import sparse_kernel  # noqa: PLC0415

    mf_gs = (
        _axis("shared_metadata", (True, False)),
        _axis("stream_mf_impulses"),
        _axis("has_dense_velocity_limit_rows", (True, False)),
        ({}, {"skip_local_internal_worlds": True, "local_internal_max_constraints": 19}),
        _axis("contact_torsion"),
        _axis("factor_coordinates"),
        _axis("skip_tier_rows", (0, 32)),
        _axis("independent_components"),
        _axis("resident_row_metadata"),
    )
    local = (
        _axis("paired_dof_count", (0, 6)),
        _axis("persistent_queue"),
        _axis("warps_per_block", (1, 2)),
        _axis("lanes_per_world", (32, 16, 8)),
        _axis("contact_capable", (True, False)),
        ({}, {"mf_max_constraints": 64, "local_mf_max_constraints": 12}),
    )
    keyed = KeyedLayout(16, 6, 0, 16, 128, False, 0)
    specs = (
        (
            "sweep_routes.mf_gs_kernel",
            sweep_routes.mf_gs_kernel,
            ((64, 32, 12, ARCH), (192, 64, 40, ARCH)),
            mf_gs,
        ),
        (
            "sweep_routes_sparse.sparse_kernel",
            sparse_kernel,
            ((48, 18, 12), (32, 8, 8)),
            (
                _axis("active_rows"),
                _axis("use_support_table"),
                _axis("contact_triplets"),
                _axis("velocity_limits"),
                _axis("lanes_per_world", (32, 16, 8)),
                _axis("local_factors"),
                _axis("warm_start"),
                _axis("linked_rows"),
            ),
        ),
        (
            "sweep_routes_local.paired_factor_kernel",
            sweep_routes_local.paired_factor_kernel,
            ((192, 29, 23, 6, ARCH),),
            (_axis("contact_triples"), _axis("independent_components"), _axis("contact_units")),
        ),
        (
            "sweep_routes_local.local_owned_kernel",
            sweep_routes_local.local_owned_kernel,
            ((192, 20, 9, ARCH),),
            local,
        ),
        (
            "row_packets.get_local_factory(9, 6, 40, 11)",
            row_packets.get_local_factory.__wrapped__(9, 6, 40, 11),
            ((192, 20, 9, ARCH),),
            local,
        ),
        (
            "sweep_routes_diag.sparse_diagonal_kernel",
            sweep_routes_diag.sparse_diagonal_kernel,
            ((64, 10, 9, ARCH),),
            (_axis("contact_triples"), _axis("speculative_contact_batches"), _axis("contact_units")),
        ),
        (
            "sweep_routes_tiers.parallel_kernel",
            sweep_routes_tiers.parallel_kernel,
            ((192, 32, 18, ARCH),),
            (
                tuple(
                    {"rows": rows, "min_rows": low, "sweeps": 24, "inkernel_response": (18, 0, 0, 0)}
                    for rows, low in ((32, 0), (48, 32), (128, 96))
                ),
                _axis("linked_rows"),
                _axis("velocity_limit_pass"),
            ),
        ),
        (
            f"keyed_rows.get_parallel_factory({keyed!r})",
            keyed_rows.get_parallel_factory.__wrapped__(sweep_routes_tiers.parallel_kernel, keyed),
            ((128, 32, 22, ARCH),),
            (
                tuple(
                    {
                        "rows": rows,
                        "min_rows": low,
                        "sweeps": 24,
                        "inkernel_response": (16, 6, 0, 16),
                    }
                    for rows, low in ((32, 0), (64, 32), (96, 64), (128, 96))
                ),
                _axis("linked_rows"),
                _axis("velocity_limit_pass"),
            ),
        ),
    )
    for factory_name, cached, shapes, axes in specs:
        factory = _unwrapped(cached)
        for args, kwargs in itertools.product(shapes, _combinations(axes)):
            yield factory_name, factory, args, kwargs


def census(select=(), skip=()):
    """Return ``({label: row}, {factory: rejection summary})`` for :func:`census_calls`, by hash only.

    Code digests of its thousands of variants would take minutes, and an
    unchanged hash proves the code unchanged. Labels omit keywords given at a
    factory's default; calls labeled in ``skip`` are left to the families that
    recorded them. Rejected combinations are summarized per factory by count,
    messages and a digest of their labels, so that a change of admission shows
    in a diff as a changed rejection.
    """
    rows, rejected, defaults = {}, {}, {}
    for name, factory, args, options in census_calls():
        if id(factory) not in defaults:
            parameters = inspect.signature(factory).parameters.values()
            defaults[id(factory)] = {p.name: p.default for p in parameters if p.default is not p.empty}
        kwargs = {
            key: value
            for key, value in options.items()
            if key not in defaults[id(factory)] or _value(value) != defaults[id(factory)][key]
        }
        key = label(name, args, kwargs)
        if key in skip or key in rows or (select and not any(text in key for text in select)):
            continue
        try:
            built = factory(*map(_value, args), **{k: _value(v) for k, v in kwargs.items()})
        except (ValueError, NotImplementedError, RuntimeError, AssertionError) as error:
            # Generators also refuse a combination when a source seam guard fails.
            rejected.setdefault(name, []).append((key, f"{type(error).__name__}: {error}"))
            continue
        rows[key] = {"family": "census", **kernel_digests(getattr(built, "kernel", built), codegen=False)}
    summaries = {}
    for name, entries in rejected.items():
        messages = collections.Counter(message for _, message in entries)
        listed = "; ".join(f"{count} x {message}" for message, count in sorted(messages.items()))
        summaries[f"{name} [census]"] = (
            f"{len(entries)} rejected, labels {digest(json.dumps(sorted(entries)))[:16]}: {listed}"
        )
    return rows, summaries


def _switches():
    return sorted(name for name in os.environ if name.startswith(SWITCH_PREFIXES))


def _import_newton(root):
    """Import Newton from ``root`` (default: the checkout holding this script) and return its package path."""
    root = Path(root or Path(__file__).resolve().parents[2]).resolve()
    sys.path.insert(0, str(root))
    import newton  # noqa: PLC0415

    package = Path(newton.__file__).resolve().parent
    if package.parent != root:
        raise SystemExit(f"Imported Newton from {package}, not from {root}")
    return package


def inventory(families, select=()):
    """Return ``({label: (family, kernel)}, {label: rejection})`` for the chosen families but the census."""
    import warp as wp  # noqa: PLC0415

    kernels, rejected = {}, {}
    if "pins" in families:
        from newton.tests import test_feather_pgs_point_streams as pins  # noqa: PLC0415

        kernels.update((name, ("pins", kernel)) for name, kernel in pins._inventory().items())
    for family, calls in (("factories", factory_calls), ("held_tree", held_tree_calls), ("walk", walk_calls)):
        if family not in families:
            continue
        for name, factory, args, kwargs in calls():
            key = label(name, args, kwargs)
            if key in kernels or (select and not any(text in key for text in select)):
                continue
            try:
                built = factory(*map(_value, args), **{k: _value(v) for k, v in kwargs.items()})
            except (ValueError, NotImplementedError) as error:
                rejected[key] = f"{type(error).__name__}: {error}"
                continue
            # A tier factory returns its launch descriptor, which holds the kernel.
            built = getattr(built, "kernel", built)
            if isinstance(built, wp.Kernel):
                kernels[key] = (family, built)
            else:
                for position, kernel in enumerate(built):
                    kernels[f"{key}[{position}]"] = (family, kernel)
    if select:
        kernels = {key: value for key, value in kernels.items() if any(text in key for text in select)}
    return kernels, rejected


def kernel_digests(kernel, *, codegen=True):
    """Return the kernel key, its Warp content hash and its normalized CUDA and CPU code digests."""
    import warp as wp  # noqa: PLC0415
    from warp._src.context import ModuleHasher  # noqa: PLC0415

    module = kernel.module
    # Resolves deferred wp.static expressions; the content hash of one kernel
    # does not depend on the other kernels of its module.
    module.get_module_hash()
    options = module.resolve_options(wp.config)
    ModuleHasher([kernel], options)
    row = {"key": kernel.key, "hash": kernel.hash.hex()}
    if codegen:
        built = set()
        # Warp builds CPU modules with output_arch None, which tile builtins read.
        for target, arch in (("cuda", int(ARCH)), ("cpu", None)):
            target_options = dict(options, output_arch=arch)
            try:
                builder = _build(kernel, target_options, built)
                source = builder.codegen(target)
                functions = stable_function_names(builder.functions)
                row[target] = digest(normalize(source, (row["hash"][:8],), functions))
            except Exception as error:
                # A code generation failure is recorded, and compared, like a digest.
                row[target] = f"error: {type(error).__name__}: {error}"
    return row


def _build(kernel, options, built):
    """Build one kernel as a fresh process would, whatever this process built before.

    A function stays marked as used by a backward kernel once any build in the
    process marked it, and a marked function also emits its adjoint in later
    forward-only kernels. Clear the mark on every function the kernel builds,
    then build again whenever the kernel reached a function not cleared yet.
    """
    from warp._src.context import ModuleBuilder, ModuleHasher  # noqa: PLC0415

    while True:
        for function in built:
            function.adj.used_by_backward_kernel = False
        builder = ModuleBuilder(kernel.module, options, hasher=ModuleHasher([kernel], options))
        if built.issuperset(builder.functions):
            return builder
        built.update(builder.functions)


def record(path, *, root=None, families=FAMILIES, select=(), codegen=True, allow_switches=False):
    switches = _switches()
    if switches and not allow_switches:
        raise SystemExit(f"Unset the import-time switches, or pass --allow-switches: {', '.join(switches)}")
    start = time.perf_counter()
    package = _import_newton(root)
    import warp as wp  # noqa: PLC0415

    wp.config.log_level = wp.LOG_WARNING
    if codegen:
        # Code generation of some tile builtins (Cholesky, triangular solves) queries the runtime.
        wp.init()
    kernels, rejected = inventory(families, select)
    rows = {}
    for key in sorted(kernels):
        family, kernel = kernels[key]
        rows[key] = {"family": family, **kernel_digests(kernel, codegen=codegen)}
    if "census" in families:
        variants, summaries = census(select, skip=rows.keys())
        rows = dict(sorted({**rows, **variants}.items()))
        rejected.update(summaries)
    result = {
        "warp": wp.config.version,
        "newton": str(package),
        "hash_seed": os.environ.get("PYTHONHASHSEED"),
        "switches": {name: os.environ[name] for name in switches},
        "families": list(families),
        "select": list(select),
        "codegen": codegen,
        "seconds": round(time.perf_counter() - start, 1),
        "kernels": rows,
        "rejected": dict(sorted(rejected.items())),
    }
    Path(path).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"Recorded {len(rows)} kernels and {len(rejected)} rejected calls in {result['seconds']} s: {path}")
    return result


def compare(parent, child, columns=COLUMNS):
    """Split two records into changed and uncompared kernels, added and removed ones, and changed rejections.

    A shared kernel is compared on every requested column both rows carry. A
    requested column that either row lacks (a ``--hash-only`` row has no code
    digests) leaves the kernel uncompared, unless both rows carry the same
    content hash: the hash covers every input of code generation, so an
    unchanged hash also proves the generated code unchanged.
    """
    old, new = parent["kernels"], child["kernels"]
    changed, uncompared = {}, {}
    for key in sorted(old.keys() & new.keys()):
        before, after = old[key], new[key]
        differ = [c for c in columns if c in before and c in after and before[c] != after[c]]
        if differ:
            changed[key] = differ
        absent = [c for c in columns if c not in before or c not in after]
        if absent and not ("hash" in before and "hash" in after and before["hash"] == after["hash"]):
            uncompared[key] = absent
    rejections = sorted(
        key
        for key in parent.get("rejected", {}).keys() | child.get("rejected", {}).keys()
        if parent.get("rejected", {}).get(key) != child.get("rejected", {}).get(key)
    )
    return {
        "changed": changed,
        "uncompared": uncompared,
        "added": sorted(new.keys() - old.keys()),
        "removed": sorted(old.keys() - new.keys()),
        "rejections": rejections,
    }


def report(parent, child, columns=COLUMNS):
    """Print the comparison; return 1 when a shared kernel changed or went uncompared, or a kernel disappeared."""
    for field in ("warp", "hash_seed", "switches", "families", "select", "codegen"):
        if parent.get(field) != child.get(field):
            print(f"warning: the records differ in {field}: {parent.get(field)!r} against {child.get(field)!r}")
    result = compare(parent, child, columns)
    shared = len(parent["kernels"].keys() & child["kernels"].keys())
    uncompared = {key: absent for key, absent in result["uncompared"].items() if key not in result["changed"]}
    print(
        f"{shared} shared kernels compared on {', '.join(columns)}: "
        f"{len(result['changed'])} changed, {len(uncompared)} not compared"
    )
    for key, absent in uncompared.items():
        print(f"  not compared (no {', '.join(absent)} on both sides, and the hash changed or is missing): {key}")
    for key, differ in result["changed"].items():
        compared = [
            c for c in ("cuda", "cpu") if c in columns and c in parent["kernels"][key] and c in child["kernels"][key]
        ]
        if set(differ) != {"hash"}:
            kind = "generated code changed"
        elif compared:
            kind = "hash only, generated code identical"
        else:
            kind = "hash changed, generated code not compared"
        print(f"  changed ({kind}): {key}")
    for field, item in (("added", "added"), ("removed", "removed"), ("rejections", "rejection changed")):
        print(f"{len(result[field])} {field}")
        for key in result[field]:
            print(f"  {item}: {key}")
    return int(bool(result["changed"] or uncompared or result["removed"] or result["rejections"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    record_parser = commands.add_parser("record", help="record the kernels of one checkout")
    record_parser.add_argument("output", help="JSON file to write")
    record_parser.add_argument("--root", help="Newton checkout to import (default: the one holding this script)")
    record_parser.add_argument("--family", action="append", choices=FAMILIES, help="restrict to these families")
    record_parser.add_argument("--select", action="append", default=[], help="keep labels containing this text")
    record_parser.add_argument(
        "--hash-only", action="store_true", help="skip the generated code digests (the census never records them)"
    )
    record_parser.add_argument("--allow-switches", action="store_true", help="record under FEATHER_PGS_* switches")
    diff_parser = commands.add_parser("diff", help="compare two records")
    diff_parser.add_argument("parent")
    diff_parser.add_argument("child")
    diff_parser.add_argument(
        "--columns",
        default=",".join(COLUMNS),
        help="columns that must match: 'hash,cuda,cpu' (default) or 'cuda,cpu' for generated-code identity",
    )
    arguments = parser.parse_args(argv)
    if arguments.command == "record":
        if argv is None and os.environ.get("PYTHONHASHSEED") != "0":
            # Warp orders some loop-carried variables by string hash, which Python salts per process.
            os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, "PYTHONHASHSEED": "0"})
        record(
            arguments.output,
            root=arguments.root,
            families=tuple(arguments.family or FAMILIES),
            select=tuple(arguments.select),
            codegen=not arguments.hash_only,
            allow_switches=arguments.allow_switches,
        )
        return 0
    columns = tuple(column for column in arguments.columns.split(",") if column)
    if not columns or any(column not in COLUMNS for column in columns):
        parser.error(f"--columns takes a comma-separated subset of {', '.join(COLUMNS)}")
    parent, child = (json.loads(Path(path).read_text(encoding="utf-8")) for path in (arguments.parent, arguments.child))
    return report(parent, child, columns)


if __name__ == "__main__":
    sys.exit(main())
