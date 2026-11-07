# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pin the generated code of every FeatherPGS kernel.

Warp's per-kernel content hash covers the kernel source, every ``@wp.func`` it
calls (``func_native`` snippets included), struct layouts and ``wp.static``
values. A function passed by name to a builtin, such as the operator of
``wp.tile_map``, is compiled but not hashed, so kernels pass such functions
through ``wp.static``. The hash does not depend on line numbers or on other
kernels of the same module, and it is computed on the host without compiling
anything.

Every kernel keeps its pin until the pin is recorded on purpose: a change that
moves a kernel's generated code, or retires a kernel, fails here until its own
commit re-records the pins, so each move is a decision that commit states, never
a side effect.

Run ``python -m newton.tests.test_feather_pgs_kernel_pins --record`` to add pins
for new kernels; ``--record --rebaseline`` also refreshes the changed pins and
drops the retired ones, and lists each. Pins taken with another Warp version are
recorded anew.
"""

import argparse
import ast
import importlib
import itertools
import json
import os
import pkgutil
import sys
import unittest

import warp as wp

from newton._src.solvers.feather_pgs import environment_flags
from newton._src.solvers.feather_pgs.sweep import EXACT, LevelLaw
from newton.tests.feather_pgs_law_harness import built, keyed_classes

_PINS = os.path.join(os.path.dirname(__file__), "golden_data", "test_feather_pgs_kernel_pins", "pins.json")
_PACKAGE = "newton._src.solvers.feather_pgs"
# CUDA architecture of the factory sweeps; only measured opt-in layouts depend on it.
_ARCH = "120"


def _label(name, args, kwargs):
    parts = [repr(arg) for arg in args] + [f"{key}={kwargs[key]!r}" for key in sorted(kwargs)]
    return f"{name}({', '.join(parts)})"


def _linked_tier_calls():
    """Linked parallel tiers, whose friction pairs pool their patch ring's normal impulses.

    Each also with the last-word velocity-limit pass, which the solver builds for linked tiers with joint velocity
    limits.
    """
    for rows, last_word in itertools.product((32, 48), (False, True)):
        kwargs = {
            "rows": rows,
            "levels": LevelLaw.split(24),
            "inkernel_response": (18, 0, 0, 0),
            "linked_rows": True,
        }
        if last_word:
            kwargs["velocity_limit_pass"] = True
        yield "sweep_routes_tiers.TierSweep", (64, 32, 18, _ARCH), kwargs


def _keyed_layout(*, velocity_limits=False):
    """The 16+6 coordinate layout of the keyed-row sweeps.

    ``velocity_limits`` gives the 16 primary coordinates joint velocity limits, as a
    solver with ``enable_joint_velocity_limits`` plans them (the free block has none).
    """
    from newton._src.solvers.feather_pgs.keyed_row_layout import KeyedLayout  # noqa: PLC0415

    if velocity_limits:
        return KeyedLayout(16, 6, 0, 16, 128, True, (1 << 16) - 1)
    return KeyedLayout(16, 6, 0, 16, 128, False, 0)


def _keyed_tier_kwargs(*, linked):
    """Keyed parallel tiers; ``linked`` selects the patch row law and the ring-pooled projection."""
    for rows in (32, 64, 96, 128):
        kwargs = {"rows": rows, "levels": LevelLaw.split(24), "inkernel_response": (16, 6, 0, 16)}
        if linked:
            kwargs["linked_rows"] = True
        yield kwargs


def _suite_sparse_sweeps():
    """Native sparse sweeps at the shapes the benchmark suite builds: ``(size, kwargs)``."""
    active = {"active_rows": True}
    triplets = {"active_rows": True, "use_support_table": True, "contact_triplets": True}
    for size, kwargs in (
        ((100, 43, 18), active),
        ((112, 18, 12), triplets),
        ((128, 25, 16), triplets),
        ((128, 27, 21), {"use_support_table": True}),
        ((16, 6, 6), {"use_support_table": True, "velocity_limits": True, "lanes_per_world": 8, "local_factors": True}),
        ((256, 18, 12), triplets),
        ((48, 18, 18), triplets),
        ((48, 25, 16), triplets),
        ((56, 43, 18), active),
        ((64, 14, 10), {"use_support_table": True, "lanes_per_world": 16}),
        ((64, 9, 9), {"velocity_limits": True, "lanes_per_world": 16}),
        ((80, 18, 18), triplets),
    ):
        yield size, kwargs
    # The friction law links the patch rings of the active sweeps.
    for size, kwargs in (
        ((100, 43, 18), active),
        ((112, 18, 12), triplets),
        ((128, 25, 16), triplets),
        ((256, 18, 12), triplets),
        ((48, 18, 18), triplets),
        ((48, 25, 16), triplets),
        ((56, 43, 18), active),
        ((72, 18, 12), triplets),
        ((80, 18, 18), triplets),
    ):
        yield size, {**kwargs, "linked_rows": True}


def _suite_diagonal_sweeps():
    """Sparse-diagonal sweeps at the suite's shapes: ``(args, kwargs)``, over contact units."""
    for shape in ((1024, 114, 6, _ARCH), (256, 13, 9, _ARCH)):
        yield shape, {"speculative_contact_batches": True, "levels": LevelLaw(EXACT)}


def _keyed_last_word_calls():
    """Linked keyed tiers of a velocity-limited layout with the last-word pass.

    The solver builds the pass for linked tiers only with joint velocity limits, so these
    tiers read a layout that plans velocity-limit rows.
    """
    for kwargs in _keyed_tier_kwargs(linked=True):
        yield (_keyed_layout(velocity_limits=True), 128, 32, 22, _ARCH), {**kwargs, "velocity_limit_pass": True}


def _audit_calls():
    """The audit's one rows kernel at each support class the suite's routes read: ``(name, factory, args, kwargs)``."""
    from newton._src.solvers.feather_pgs import audit  # noqa: PLC0415

    def kernel(row_class):
        return lambda *args: audit.rows_kernel(row_class(*args))

    for support, wide in ((12, False), (18, False), (25, True)):
        yield "audit.packed_class", kernel(audit.packed_class), (support, wide), {}
    for support, table in ((12, False), (12, True), (16, True)):
        yield "audit.sparse_class", kernel(audit.sparse_class), (support, table), {}
    yield "audit.slot_class", kernel(audit.slot_class), (6, 1024), {}
    yield "audit.free_class", kernel(audit.free_class), (), {}


def _readout_calls():
    """Opt-in sweep readouts of the suite's counted sweeps: ``(name, factory, args, kwargs)``.

    ``Diagnostics(solve_readout="sweeps")`` builds these variants of the sparse, sparse-diagonal,
    matrix-free Gauss-Seidel and paired factor sweeps.
    """
    from newton._src.solvers.feather_pgs import (  # noqa: PLC0415
        sweep_routes,
        sweep_routes_diag,
        sweep_routes_local,
        sweep_routes_sparse,
    )
    from newton._src.solvers.feather_pgs.solve_policy import SolvePolicy  # noqa: PLC0415

    readout = {"solve_policy": SolvePolicy(readout="sweeps")}
    triplets = {"active_rows": True, "use_support_table": True, "contact_triplets": True}
    factory = built(sweep_routes_sparse.SparseSweep)
    for size in ((112, 18, 12), (128, 25, 16), (256, 18, 12)):
        yield "sweep_routes_sparse.SparseSweep", factory, size, {**triplets, **readout}
        yield "sweep_routes_sparse.SparseSweep", factory, size, {**triplets, "linked_rows": True, **readout}
    factory = built(sweep_routes_diag.DiagonalSweep)
    kwargs = {"levels": LevelLaw(EXACT), "speculative_contact_batches": True, **readout}
    yield "sweep_routes_diag.DiagonalSweep", factory, (1024, 114, 6, _ARCH), kwargs
    factory = built(sweep_routes.GeneralSweep)
    for args, kwargs in (
        ((192, 64, 22, _ARCH), {}),
        ((192, 64, 22, _ARCH), {"free_rows": False, "speed_rows": True}),
    ):
        yield "sweep_routes.GeneralSweep", factory, args, {**kwargs, **readout}
    factory = built(sweep_routes_local.PairedSweep)
    for units in (False, True):
        kwargs = {"contact_units": True, **readout} if units else readout
        yield (
            "sweep_routes_local.PairedSweep",
            factory,
            (192, 29, 23, 6, _ARCH),
            {**kwargs, "speed_rows": True},
        )


def _module_kernels():
    """Every non-generic kernel defined at module level in the FeatherPGS package and its subpackages."""
    package = importlib.import_module(_PACKAGE)
    kernels = {}
    for info in pkgutil.walk_packages(package.__path__, f"{_PACKAGE}."):
        module = importlib.import_module(info.name)
        for name, value in vars(module).items():
            # A re-exported kernel is pinned where it is defined. Generic kernels
            # are hashed per instantiation.
            if isinstance(value, wp.Kernel) and value.func.__module__ == module.__name__ and not value.is_generic:
                kernels[f"{info.name.removeprefix(_PACKAGE + '.')}.{name}"] = value
    return kernels


def _sparse_sweep_calls():
    """Native sparse sweeps over every flag combination without linked rows (sweep_routes_sparse.py); the law's routes
    build the linked ones (:func:`_linked_sweep_calls`)."""
    from newton._src.solvers.feather_pgs.sweep_routes_sparse import SparseSweep  # noqa: PLC0415

    factory = built(SparseSweep)
    for size, flags in itertools.product(((32, 8, 8), (48, 18, 12)), itertools.product((False, True), repeat=6)):
        active, table, triplets, velocity, local, warm = flags
        # The factory folds unsatisfiable requests into another variant; pin each variant once.
        if triplets and not (active and table):
            continue
        for lanes in (8, 16, 32):
            kwargs = {
                "active_rows": active,
                "use_support_table": table,
                "contact_triplets": triplets,
                "velocity_limits": velocity,
                "lanes_per_world": lanes,
                "local_factors": local,
                "warm_start": warm,
            }
            yield "sweep_routes_sparse.SparseSweep", factory, size, kwargs


def _linked_sweep_calls():
    """Native sparse sweeps over linked rows (sweep_routes_sparse.py)."""
    from newton._src.solvers.feather_pgs.sweep_routes_sparse import SparseSweep  # noqa: PLC0415

    factory = built(SparseSweep)
    for table, warm in itertools.product((False, True), repeat=2):
        kwargs = {"active_rows": True, "use_support_table": table, "warm_start": warm, "linked_rows": True}
        yield "sweep_routes_sparse.SparseSweep", factory, (48, 18, 12), kwargs
    for active, warm in itertools.product((False, True), repeat=2):
        kwargs = {
            "active_rows": active,
            "use_support_table": True,
            "contact_triplets": True,
            "warm_start": warm,
            "linked_rows": True,
        }
        yield "sweep_routes_sparse.SparseSweep", factory, (48, 18, 12), kwargs


def _support_writer_calls():
    """The support placement's writer (case K1, walk_rows.py) on a branched tree of a free root and three limbs, under
    each record and torsion law, and the step's one leave of that tree's class (factor/walk.py)."""
    from newton._src.solvers.feather_pgs import walk_rows  # noqa: PLC0415

    tree = ((-1, 0, 1, 0, 3, 0), (0, 6, 7, 8, 9, 10, 11), tuple(range(10, -1, -1)))
    for records, torsion in ((False, False), (True, False), (True, True)):
        kwargs = {"records": records, "torsion": torsion}
        yield "walk_rows.support_rows_kernel", walk_rows.support_rows_kernel.__wrapped__, tree, kwargs
    for unit_rows in (3, 4):
        yield "walk_rows.support_units_kernel", walk_rows.support_units_kernel.__wrapped__, (unit_rows, 16), {}
    from newton._src.solvers.feather_pgs import tree_scan  # noqa: PLC0415
    from newton._src.solvers.feather_pgs.factor import walk as factor_walk  # noqa: PLC0415

    walk = (tree_scan.WalkClass(*tree[:2]),)
    for release in (False, True):
        yield "factor.walk.walk_leave_kernel", factor_walk.walk_leave_kernel.__wrapped__, (walk, release), {}


def _dense_build_calls():
    """The dense rows' builds: the one writer's variants."""
    # The general placement's one writer: plain, and with blocks of up to eight rows (the closure), mimics among them.
    from newton._src.solvers.feather_pgs import walk_rows  # noqa: PLC0415

    # Its trees' classes: a branched tree of a free root and three limbs, and a free body.
    tables = (((-1, 0, 1, 0, 3, 0), (0, 6, 7, 8, 9, 10, 11)), ((-1,), (0, 6)))
    for args in ((0, False), (8, False), (8, True)):
        yield "walk_rows.walked_rows_kernel", walk_rows.walked_rows_kernel.__wrapped__, (*args, True, tables), {}
    # Its topology part, the rows walked in blocks: plain and with blocks.
    for args in ((0, False), (8, True)):
        block = (*args, True, tables, walk_rows.WRITER_BLOCK_ROWS)
        yield "walk_rows.walked_rows_kernel", walk_rows.walked_rows_kernel.__wrapped__, block, {}
    # The sparse diagonal's one writer, the star's walker on the same classes.
    yield "walk_rows.slot_rows_kernel", walk_rows.slot_rows_kernel.__wrapped__, (True, tables), {}
    # With velocity-limit rows: a key's joint row is one step to the world, into a slot.
    yield "walk_rows.slot_rows_kernel", walk_rows.slot_rows_kernel.__wrapped__, (True, tables, True), {}


def _solver_factory_calls():
    from newton._src.core.memory import Place  # noqa: PLC0415
    from newton._src.solvers.feather_pgs import (  # noqa: PLC0415
        sweep_levels,
        sweep_routes,
        sweep_routes_diag,
        sweep_routes_local,
        sweep_routes_tiers,
        sweep_units,
    )

    name = "sweep_levels."
    factory = built(sweep_routes_diag.DiagonalSweep)
    yield "sweep_routes_diag.DiagonalSweep", factory, (64, 10, 9, _ARCH), {}
    for serial in (False, True):
        factory = sweep_levels._get_sparse_contact_levels_kernel.__wrapped__
        kwargs = {"build_serial_contacts": serial}
        yield name + "_get_sparse_contact_levels_kernel", factory, (64, 10, _ARCH), kwargs
    for dofs in (1, 2):
        factory = sweep_levels._get_mark_independent_sparse_contact_candidates_kernel.__wrapped__
        yield name + "_get_mark_independent_sparse_contact_candidates_kernel", factory, (dofs, _ARCH), {}
    # The unit schedules of the sparse diagonal.
    for batches in (False, True):
        kwargs = {"levels": LevelLaw(EXACT), "speculative_contact_batches": batches}
        factory = built(sweep_routes_diag.DiagonalSweep)
        yield "sweep_routes_diag.DiagonalSweep", factory, (64, 10, 9, _ARCH), kwargs
        # The velocity-limit rows' last word.
        kwargs = {**kwargs, "velocity_limits": True}
        yield "sweep_routes_diag.DiagonalSweep", factory, (64, 10, 9, _ARCH), kwargs
    for limits, free, speed in itertools.product(*([(False, True)] * 3)):
        kwargs = {
            "has_dense_velocity_limit_rows": limits,
            "free_rows": free,
            "speed_rows": speed,
        }
        factory = built(sweep_routes.GeneralSweep)
        yield "sweep_routes.GeneralSweep", factory, (64, 32, 12, _ARCH), kwargs
    # Streamed row records, which large dense capacities select; limits add the limit pass's record reads.
    for limits in (False, True):
        kwargs = {
            "places": sweep_routes.RowPlaces(Place.GLOBAL),
            "has_dense_velocity_limit_rows": limits,
        }
        factory = built(sweep_routes.GeneralSweep)
        yield "sweep_routes.GeneralSweep", factory, (64, 32, 12, _ARCH), kwargs
    # A line past the stage: every impulse read and written in place.
    factory = built(sweep_routes.GeneralSweep)
    places = {"places": sweep_routes.RowPlaces(Place.GLOBAL, Place.GLOBAL, Place.GLOBAL)}
    yield "sweep_routes.GeneralSweep", factory, (64, 32, 12, _ARCH), places
    # The colored units' slot staging, under each law (the weights a constant of the plan).
    for weighted in (False, True):
        yield "sweep_units.stage_slots", sweep_units.stage_slots.__wrapped__, (weighted,), {}
    for speed in (False, True):
        factory = built(sweep_routes_local.PairedSweep)
        yield "sweep_routes_local.PairedSweep", factory, (192, 29, 23, 6, _ARCH), {"speed_rows": speed}
    # The unit walks of the same factory.
    for speed in (False, True):
        factory = built(sweep_routes_local.PairedSweep)
        args = (192, 29, 23, 6, _ARCH)
        yield "sweep_routes_local.PairedSweep", factory, args, {"contact_units": True, "speed_rows": speed}

    # Parallel tiers: one articulation per world, its response formed in-kernel.
    for rows in (32, 48):
        kwargs = {"rows": rows, "levels": LevelLaw.split(24), "inkernel_response": (18, 0, 0, 0)}
        yield "sweep_routes_tiers.TierSweep", built(sweep_routes_tiers.TierSweep), (64, 32, 18, _ARCH), kwargs
    for label, args, kwargs in _linked_tier_calls():
        yield label, built(sweep_routes_tiers.TierSweep), args, kwargs


def _suite_shape_calls():
    """Factory kernels at the shapes the benchmark suite's routes build, beyond those pinned above.

    Each call reproduces the kernel name its route records and, where a head
    profile ran the route, the profile's kernel suffix: the matrix-free
    Gauss-Seidel, sparse, sparse-diagonal and keyed parallel sweeps, the keyed
    limit rows, the sparse mass factors and the sparse contact responses. A byte-identical
    rewrite of their generators then keeps the binaries the suite runs.
    """
    from newton._src.solvers.feather_pgs import (  # noqa: PLC0415
        keyed_row_layout,
        keyed_rows,
        sweep_routes,
        sweep_routes_diag,
        sweep_routes_sparse,
        sweep_routes_tiers,
    )
    from newton._src.solvers.feather_pgs.keyed_row_layout import KeyedLayout  # noqa: PLC0415

    no_limits = {"has_dense_velocity_limit_rows": False}
    factory = built(sweep_routes.GeneralSweep)
    for args, kwargs in (
        ((192, 64, 22, _ARCH), {}),
        ((192, 64, 22, _ARCH), {"free_rows": False, "speed_rows": True}),
        ((72, 32, 18, _ARCH), no_limits),
    ):
        yield "sweep_routes.GeneralSweep", factory, args, kwargs
    factory = built(sweep_routes_sparse.SparseSweep)
    for size, kwargs in _suite_sparse_sweeps():
        yield "sweep_routes_sparse.SparseSweep", factory, size, kwargs
    factory = built(sweep_routes_diag.DiagonalSweep)
    for args, kwargs in _suite_diagonal_sweeps():
        yield "sweep_routes_diag.DiagonalSweep", factory, args, kwargs

    def parallel(layout, *args, **kwargs):
        return built(sweep_routes_tiers.TierSweep)(*args, keyed=layout, classes=keyed_classes(layout), **kwargs)

    def limit_prefix(layout):
        return keyed_row_layout.prefix_kernel.__wrapped__(layout, keyed_rows.PrefixInput, keyed_rows.KineticRows)

    # The tiers read only a layout's coordinate blocks; its limit masks shape the limit rows.
    limited = KeyedLayout(16, 6, 0, 16, 128, True, 0xFFFF, 0xFFFF)
    single = KeyedLayout(18, 0, 0, 0, 48, False, 0)
    for layout, shape, tiers in (
        (limited, (192, 64, 22, _ARCH), (32, 64, 96, 128)),
        (single, (72, 32, 18, _ARCH), (32, 48)),
    ):
        blocks = (layout.primary, layout.secondary, layout.primary_offset, layout.secondary_offset)
        for rows in tiers:
            kwargs = {"rows": rows, "levels": LevelLaw.split(24), "inkernel_response": blocks}
            yield "sweep_routes_tiers.TierSweep[keyed]", parallel, (layout, *shape), kwargs
    yield "keyed_row_layout.prefix_kernel", limit_prefix, (limited,), {}


def _keyed_row_calls():
    """Keyed rows of a 16+6 coordinate world (keyed_rows.py), including their parallel tiers."""
    from newton._src.solvers.feather_pgs import sweep_routes_tiers  # noqa: PLC0415

    def parallel(layout, *args, **kwargs):
        return built(sweep_routes_tiers.TierSweep)(*args, keyed=layout, classes=keyed_classes(layout), **kwargs)

    layout = _keyed_layout()
    for kwargs in _keyed_tier_kwargs(linked=False):
        yield "sweep_routes_tiers.TierSweep[keyed]", parallel, (layout, 128, 32, 22, _ARCH), kwargs
    # The same rows linked into their patch rings.
    for kwargs in _keyed_tier_kwargs(linked=True):
        yield "sweep_routes_tiers.TierSweep[keyed]", parallel, (layout, 128, 32, 22, _ARCH), kwargs
    for args, kwargs in _keyed_last_word_calls():
        yield "sweep_routes_tiers.TierSweep[keyed]", parallel, args, kwargs


def _factory_kernels():
    """Kernels of the factories, built at each call's arguments."""
    kernels = {}
    for name, factory, args, kwargs in itertools.chain(
        _sparse_sweep_calls(),
        _linked_sweep_calls(),
        _support_writer_calls(),
        _dense_build_calls(),
        _solver_factory_calls(),
        _suite_shape_calls(),
        _keyed_row_calls(),
        _readout_calls(),
        _audit_calls(),
    ):
        try:
            kernel = factory(*args, **kwargs)
        except ValueError:
            continue  # The factory rejects this combination.
        kernels[_label(name, args, kwargs)] = kernel
    return kernels


def _kernel_hash(kernel):
    """Return Warp's content hash of one kernel without compiling its module."""
    from warp._src.context import ModuleHasher  # noqa: PLC0415

    module = kernel.module
    # Resolves deferred wp.static expressions before hashing.
    module.get_module_hash()
    ModuleHasher(module._get_live_kernels(), module.resolve_options(wp.config))
    return kernel.hash.hex()


def _unhashed_functions(kernel):
    """Return the user functions that a kernel, or a function it hashes, passes to a call unhashed.

    The hash follows called functions and ``wp.static`` values, not a function
    passed by name as an argument (the operator of ``wp.tile_map``). Editing such
    a function would change neither a pin nor the kernel cache key. Run after
    :func:`_kernel_hash`, which resolves deferred ``wp.static`` values.
    """
    from warp._src.context import Function  # noqa: PLC0415

    missing, visited, pending = set(), set(), [kernel.adj]
    while pending:
        adj = pending.pop()
        if id(adj) in visited:
            continue
        visited.add(id(adj))
        _, _, functions = adj.get_references()
        statics = itertools.chain(
            adj.resolved_static_expressions.values(), (v for _, v in adj.deferred_static_expressions)
        )
        hashed = set(functions) | {value for value in statics if isinstance(value, Function)}
        for node in adj.reference_nodes():
            if not isinstance(node, ast.Call):
                continue
            for argument in (*node.args, *(keyword.value for keyword in node.keywords)):
                if isinstance(argument, (ast.Name, ast.Attribute)):
                    value, _ = adj.resolve_static_expression(argument, eval_types=False)
                    if isinstance(value, Function) and not value.is_builtin() and value not in hashed:
                        missing.add(value.key)
        for function in hashed:
            if not function.is_builtin():
                overloads = {**function.user_overloads, **function.user_templates}.values()
                pending.extend(overload.adj for overload in overloads if overload.generic_parent is None)
    return sorted(missing)


def _inventory():
    """Return ``{pin name: kernel}`` for the module-level and factory-built kernels."""
    return {**_module_kernels(), **_factory_kernels()}


def collect(kernels=None):
    """Return ``{pin name: content hash}`` for ``kernels``, by default the whole inventory."""
    kernels = _inventory() if kernels is None else kernels
    return {name: _kernel_hash(kernel) for name, kernel in sorted(kernels.items())}


def _stored():
    with open(_PINS, encoding="utf-8") as stream:
        return json.load(stream)


def _environment_switches():
    return sorted(name for name in os.environ if name.startswith(environment_flags.PREFIXES))


def compare(stored, current):
    """Split differences into changed and retired pins, and unpinned kernels."""
    changed = sorted(name for name, pin in stored.items() if name in current and current[name] != pin)
    missing = sorted(name for name in stored if name not in current)
    unpinned = sorted(name for name in current if name not in stored)
    return changed, missing, unpinned


class TestFeatherPGSKernelPins(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stored = _stored()
        cls.kernels = cls.current = None
        if cls.stored["warp"] == wp.config.version and not _environment_switches():
            cls.kernels = _inventory()
            cls.current = collect(cls.kernels)

    def setUp(self):
        if self.stored["warp"] != wp.config.version:
            self.skipTest(
                f"Pins use Warp's private ModuleHasher of {self.stored['warp']}; running Warp {wp.config.version}"
            )
        switches = _environment_switches()
        if switches:
            self.skipTest(f"Import-time switches select other generated kernels: {', '.join(switches)}")

    def test_every_kernel_keeps_its_pin(self):
        """Keep every pinned kernel's generated code until its pin is recorded on purpose."""
        changed, missing, _ = compare(self.stored["kernels"], self.current)
        rebaseline = (
            "Re-record on purpose with: python -m newton.tests.test_feather_pgs_kernel_pins --record --rebaseline"
        )
        self.assertEqual(changed, [], f"Kernels changed their generated code. {rebaseline}")
        self.assertEqual(missing, [], f"Pinned kernels disappeared from the inventory. {rebaseline}")

    def test_every_kernel_is_pinned(self):
        """Pin each new kernel before a route can launch it."""
        _, _, unpinned = compare(self.stored["kernels"], self.current)
        self.assertEqual(unpinned, [], "Record pins with: python -m newton.tests.test_feather_pgs_kernel_pins --record")

    def test_hashes_cover_function_arguments(self):
        """Hash every function a kernel passes to a builtin, so pins and the kernel cache follow its edits."""
        unhashed = {name: _unhashed_functions(kernel) for name, kernel in self.kernels.items()}
        self.assertEqual(
            {name: functions for name, functions in unhashed.items() if functions},
            {},
            "Pass these functions through wp.static() so that Warp hashes them",
        )

    def test_inventory_covers_the_varied_factories(self):
        """Reach every factory and variant family the routes build."""
        families = {name.split("(", 1)[0] for name in self.current}
        for family in (
            "sweep_routes_sparse.SparseSweep",
            "walk_rows.support_rows_kernel",
            "sweep_routes_diag.DiagonalSweep",
            "sweep_levels._get_sparse_contact_levels_kernel",
            "sweep_levels._get_mark_independent_sparse_contact_candidates_kernel",
            "sweep_routes_tiers.TierSweep",
            "sweep_routes_local.PairedSweep",
            "sweep_routes_tiers.TierSweep[keyed]",
            "row_slots.allocate_world_contact_slots",
            "walk_rows.slot_rows_kernel",
            "walk_rows.walked_rows_kernel",
            "resolved_worlds.finalize_resolved_worlds",
        ):
            with self.subTest(family=family):
                self.assertIn(family, families)
        sparse = [name for name in self.current if name.startswith("sweep_routes_sparse.")]
        for flag in (
            "active_rows=True",
            "contact_triplets=True",
            "velocity_limits=True",
            "warm_start=True",
            "linked_rows=True",
        ):
            with self.subTest(flag=flag):
                self.assertTrue(any(flag in name for name in sparse))
        # The MF-GS kernels that stream their row records are pinned.
        self.assertTrue([name for name in self.current if "RowPlaces(GLOBAL, SHARED)" in name])
        # The linked tiers are pinned.
        linked = [name for name in self.current if "linked_rows=True" in name]
        self.assertTrue(linked)
        # Their last-word velocity-limit pass, in the plain and the keyed-row tiers.
        last_word = {name.split("(", 1)[0] for name in linked if "velocity_limit_pass=True" in name}
        self.assertEqual(last_word, {"sweep_routes_tiers.TierSweep", "sweep_routes_tiers.TierSweep[keyed]"})

    def test_inventory_builds_the_suite_shapes(self):
        """Build every factory call at the suite's shapes; the suite's routes build them, so none may be rejected."""
        labels = {_label(name, args, kwargs) for name, _, args, kwargs in _suite_shape_calls()}
        self.assertEqual(sorted(labels - set(self.current)), [])


def record(rebaseline=False):
    """Write pins for new kernels; with ``rebaseline`` also refresh changed pins and drop retired ones, listing each.
    Without it, refuse to change or drop a pin."""
    try:
        stored = _stored()
    except FileNotFoundError:
        stored = {"warp": wp.config.version, "kernels": {}}
    switches = _environment_switches()
    if switches:
        raise SystemExit(f"Unset the import-time switches before recording: {', '.join(switches)}")
    current = collect()
    if stored["warp"] == wp.config.version:
        changed, missing, _ = compare(stored["kernels"], current)
        if (changed or missing) and not rebaseline:
            raise SystemExit(f"Pins changed or disappeared; not recording without --rebaseline: {changed + missing}")
        for name in changed:
            print(f"changed: {name}")
        for name in missing:
            print(f"retired: {name}")
    os.makedirs(os.path.dirname(_PINS), exist_ok=True)
    with open(_PINS, "w", encoding="utf-8") as stream:
        json.dump({"warp": wp.config.version, "kernels": current}, stream, indent=1, sort_keys=True)
        stream.write("\n")
    print(f"Recorded {len(current)} kernel pins in {_PINS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record", action="store_true", help="add pins for new kernels")
    parser.add_argument(
        "--rebaseline", action="store_true", help="with --record, also refresh changed and drop retired pins"
    )
    arguments, remaining = parser.parse_known_args()
    if arguments.record:
        record(rebaseline=arguments.rebaseline)
    else:
        unittest.main(argv=[sys.argv[0], *remaining])
