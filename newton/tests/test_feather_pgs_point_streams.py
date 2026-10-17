# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pin the generated code of every kernel point friction can launch.

Warp's per-kernel content hash covers the kernel source, every ``@wp.func`` it
calls (``func_native`` snippets included), struct layouts and ``wp.static``
values. A function passed by name to a builtin, such as the operator of
``wp.tile_map``, is compiled but not hashed, so kernels pass such functions
through ``wp.static``. The hash does not depend on line numbers or on other
kernels of the same module, and it is computed on the host without compiling
anything. Changes that admit persistent patch friction
(``friction_anchor_beta > 0``) on a route must leave these pins unchanged: point
friction keeps its kernels, and patch work lands as new variants.

Run ``python -m newton.tests.test_feather_pgs_point_streams --record`` to add
pins for new kernels. Recording refreshes patch-only pins but refuses to change
a point pin, unless the pins were taken with another Warp version.
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

_PINS = os.path.join(os.path.dirname(__file__), "golden_data", "test_feather_pgs_point_streams", "pins.json")
_PACKAGE = "newton._src.solvers.feather_pgs"
# CUDA architecture of the factory sweeps; only measured opt-in layouts depend on it.
_ARCH = "120"

# Kernels that only persistent patch friction launches. Their pins are recorded but
# may change; a change that adds a patch-only kernel lists it here.
_PATCH_ONLY = frozenset(
    (
        "compact_contact._produce_ordered_patch_contacts",
        "compact_contact.produce_patch_contacts",
        "keyed_rows.scatter_patch_contact_keys",
        "friction_patches._arange",
        "friction_patches._build",
        "friction_patches._finish_and_store",
        "friction_patches._invalidate_geometry_history",
        "friction_patches._merge_runs",
        "friction_patches._prepare",
        "friction_patches._reset_history",
        "friction_patches._sort_chunks",
        "friction_patches.finish_patch_impulses",
        "friction_patches.link_patch_routes",
        "friction_patches.link_patch_rows",
        "friction_patches.seed_patch_impulses",
        "friction_patches.seed_patch_routes",
        "row_packets.materialize_patch_fallback",
        "row_packets.produce_patch_contacts",
        # Sparse-diagonal schedules over patch contact units.
        "solver_feather_pgs._get_pgs_solve_sparse_diagonal_kernel(64, 10, 9, '120', contact_units=True, "
        "speculative_contact_batches=False)",
        "solver_feather_pgs._get_pgs_solve_sparse_diagonal_kernel(64, 10, 9, '120', contact_units=True, "
        "speculative_contact_batches=True)",
        "solver_feather_pgs._get_build_independent_sparse_contact_groups_kernel(64, 10, '120', "
        "build_serial_contacts=False, units=True)",
        "solver_feather_pgs._get_build_independent_sparse_contact_groups_kernel(64, 10, '120', "
        "build_serial_contacts=True, units=True)",
        "solver_feather_pgs._get_mark_independent_sparse_contact_candidates_kernel(1, '120', units=True)",
        "solver_feather_pgs._get_mark_independent_sparse_contact_candidates_kernel(2, '120', units=True)",
    )
)


def _label(name, args, kwargs):
    parts = [repr(arg) for arg in args] + [f"{key}={kwargs[key]!r}" for key in sorted(kwargs)]
    return f"{name}({', '.join(parts)})"


def _linked_tier_calls():
    """Patch-only parallel tiers, whose friction pairs pool their patch ring's normal impulses."""
    for tier, matrix_free, inkernel in itertools.product(((32, 0), (48, 32)), (False, True), ((18, 0, 0, 0), None)):
        kwargs = {
            "rows": tier[0],
            "min_rows": tier[1],
            "sweeps": 24,
            "matrix_free": matrix_free,
            "inkernel_response": inkernel,
            "exact_row_sums": matrix_free,
            "linked_rows": True,
        }
        yield "solver_feather_pgs._get_pgs_solve_parallel_kernel", (64, 32, 18, _ARCH), kwargs


def _keyed_layout():
    """The 16+6 coordinate layout of the keyed-row sweeps."""
    from newton._src.solvers.feather_pgs.keyed_row_layout import KeyedLayout  # noqa: PLC0415

    return KeyedLayout(16, 6, 0, 16, 128, False, 0)


def _keyed_tier_kwargs(*, linked):
    """Keyed parallel tiers; ``linked`` selects the patch row law and the ring-pooled projection."""
    for rows, min_rows in ((32, 0), (64, 32), (96, 64), (128, 96)):
        kwargs = {
            "rows": rows,
            "min_rows": min_rows,
            "sweeps": 24,
            "matrix_free": True,
            "inkernel_response": (16, 6, 0, 16),
            "exact_row_sums": True,
        }
        if linked:
            kwargs["linked_rows"] = True
        yield kwargs


def _patch_only_factory_labels():
    """Labels of the factory-built kernels only patch friction launches."""
    for name, args, kwargs in _linked_tier_calls():
        yield _label(name, args, kwargs)
    yield _label("keyed_rows.get_patch_fallback_kernel", (_keyed_layout(),), {})
    for kwargs in _keyed_tier_kwargs(linked=True):
        yield _label("keyed_rows.get_parallel_factory", (_keyed_layout(), 128, 32, 22, _ARCH), kwargs)


_PATCH_ONLY |= frozenset(_patch_only_factory_labels())


def _module_kernels():
    """Every non-generic kernel defined at module level in the FeatherPGS package."""
    package = importlib.import_module(_PACKAGE)
    kernels = {}
    for info in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f"{_PACKAGE}.{info.name}")
        for name, value in vars(module).items():
            # A re-exported kernel is pinned where it is defined. Generic kernels
            # are hashed per instantiation; none serves a point-friction route.
            if isinstance(value, wp.Kernel) and value.func.__module__ == module.__name__ and not value.is_generic:
                kernels[f"{info.name}.{name}"] = value
    return kernels


def _sparse_sweep_calls():
    """Native sparse sweeps over the flags point friction selects (sparse_pgs.py)."""
    from newton._src.solvers.feather_pgs.sparse_pgs import _get_pgs_solve_sparse_kernel  # noqa: PLC0415

    factory = _get_pgs_solve_sparse_kernel.__wrapped__
    for size, free, flags in itertools.product(
        ((32, 8, 8), (48, 18, 12)), ((0, 0), (32, 6)), itertools.product((False, True), repeat=6)
    ):
        active, table, triplets, velocity, local, warm = flags
        # The factory folds unsatisfiable requests into another variant; pin each variant once.
        if (active and free != (0, 0)) or (triplets and not (active and table)):
            continue
        for lanes in (8, 16, 32):
            kwargs = {
                "mf_max_constraints": free[0],
                "free_row_dofs": free[1],
                "active_rows": active,
                "use_support_table": table,
                "contact_triplets": triplets,
                "velocity_limits": velocity,
                "lanes_per_world": lanes,
                "local_factors": local,
                "warm_start": warm,
            }
            yield "sparse_pgs._get_pgs_solve_sparse_kernel", factory, size, kwargs


def _sparse_contact_calls():
    from newton._src.solvers.feather_pgs.sparse_contact import _get_sparse_contact_response_kernel  # noqa: PLC0415

    factory = _get_sparse_contact_response_kernel.__wrapped__
    for dofs, prepare, table in itertools.product((8, 18), (False, True), (False, True)):
        kwargs = {"prepare_rows": prepare, "use_support_table": table}
        yield "sparse_contact._get_sparse_contact_response_kernel", factory, (dofs,), kwargs


def _solver_factory_calls():
    from newton._src.solvers.feather_pgs import solver_feather_pgs as solver  # noqa: PLC0415

    name = "solver_feather_pgs."
    for triples, batches in itertools.product((False, True), repeat=2):
        kwargs = {"contact_triples": triples, "speculative_contact_batches": batches}
        factory = solver._get_pgs_solve_sparse_diagonal_kernel.__wrapped__
        yield name + "_get_pgs_solve_sparse_diagonal_kernel", factory, (64, 10, 9, _ARCH), kwargs
    for serial in (False, True):
        factory = solver._get_build_independent_sparse_contact_groups_kernel.__wrapped__
        kwargs = {"build_serial_contacts": serial}
        yield name + "_get_build_independent_sparse_contact_groups_kernel", factory, (64, 10, _ARCH), kwargs
    for dofs in (1, 2):
        factory = solver._get_mark_independent_sparse_contact_candidates_kernel.__wrapped__
        yield name + "_get_mark_independent_sparse_contact_candidates_kernel", factory, (dofs, _ARCH), {}
    # Patch-only unit schedules of the same factories.
    for batches in (False, True):
        kwargs = {"contact_units": True, "speculative_contact_batches": batches}
        factory = solver._get_pgs_solve_sparse_diagonal_kernel.__wrapped__
        yield name + "_get_pgs_solve_sparse_diagonal_kernel", factory, (64, 10, 9, _ARCH), kwargs
    for serial in (False, True):
        factory = solver._get_build_independent_sparse_contact_groups_kernel.__wrapped__
        kwargs = {"build_serial_contacts": serial, "units": True}
        yield name + "_get_build_independent_sparse_contact_groups_kernel", factory, (64, 10, _ARCH), kwargs
    for dofs in (1, 2):
        factory = solver._get_mark_independent_sparse_contact_candidates_kernel.__wrapped__
        yield name + "_get_mark_independent_sparse_contact_candidates_kernel", factory, (dofs, _ARCH), {"units": True}
    for tier, drives, velocity, local in itertools.product(((32, 0), (64, 32)), *([(False, True)] * 3)):
        kwargs = {
            "incremental_rows": tier[0],
            "min_rows": tier[1],
            "has_drive_rows": drives,
            "has_dense_velocity_limit_rows": velocity,
            "skip_local_internal_worlds": local,
        }
        factory = solver._get_pgs_solve_mf_gs_incremental_kernel.__wrapped__
        yield name + "_get_pgs_solve_mf_gs_incremental_kernel", factory, (64, 32, 12, _ARCH), kwargs
    for skip_rows, drives, coordinates, components in itertools.product((0, 32), *([(False, True)] * 3)):
        kwargs = {
            "skip_incremental_rows": skip_rows,
            "has_drive_rows": drives,
            "factor_coordinates": coordinates,
            "independent_components": components,
        }
        factory = solver._get_pgs_solve_mf_gs_kernel.__wrapped__
        yield name + "_get_pgs_solve_mf_gs_kernel", factory, (64, 32, 12, _ARCH), kwargs
    for triples, components in itertools.product((False, True), repeat=2):
        factory = solver._get_pgs_solve_paired_factor_kernel.__wrapped__
        yield name + "_get_pgs_solve_paired_factor_kernel", factory, (192, 29, 23, 6, _ARCH, triples, components), {}
    # Parallel tiers: one articulation per world with in-kernel response and world rows,
    # and the response-block variants without them.
    for tier, matrix_free, inkernel, world_rows in itertools.product(
        ((32, 0), (48, 32)), (False, True), ((18, 0, 0, 0), None), (False, True)
    ):
        if world_rows and inkernel is None:
            continue
        kwargs = {
            "rows": tier[0],
            "min_rows": tier[1],
            "sweeps": 24,
            "matrix_free": matrix_free,
            "inkernel_response": inkernel,
            "world_rows": world_rows,
            "exact_row_sums": matrix_free,
        }
        yield (
            name + "_get_pgs_solve_parallel_kernel",
            solver._get_pgs_solve_parallel_kernel,
            (64, 32, 18, _ARCH),
            kwargs,
        )
    for label, args, kwargs in _linked_tier_calls():
        yield label, solver._get_pgs_solve_parallel_kernel, args, kwargs


def _keyed_row_calls():
    """Keyed rows of a 16+6 coordinate world (keyed_rows.py), including their parallel tiers."""
    from newton._src.solvers.feather_pgs import keyed_rows  # noqa: PLC0415
    from newton._src.solvers.feather_pgs import solver_feather_pgs as solver  # noqa: PLC0415

    def parallel(layout, *args, **kwargs):
        return keyed_rows.get_parallel_factory.__wrapped__(solver._get_pgs_solve_parallel_kernel, layout)(
            *args, **kwargs
        )

    layout = _keyed_layout()
    yield "keyed_rows.get_fallback_kernel", keyed_rows.get_fallback_kernel.__wrapped__, (layout,), {}
    for kwargs in _keyed_tier_kwargs(linked=False):
        yield "keyed_rows.get_parallel_factory", parallel, (layout, 128, 32, 22, _ARCH), kwargs
    # The persistent patch row law of the same rows.
    yield "keyed_rows.get_patch_fallback_kernel", keyed_rows.get_patch_fallback_kernel.__wrapped__, (layout,), {}
    for kwargs in _keyed_tier_kwargs(linked=True):
        yield "keyed_rows.get_parallel_factory", parallel, (layout, 128, 32, 22, _ARCH), kwargs


def _factory_kernels():
    """Kernels of the factories patch integration varies, built with point arguments."""
    from newton._src.solvers.feather_pgs.row_build_native import get_row_build_native_kernel  # noqa: PLC0415

    kernels = {"row_build_native.get_row_build_native_kernel()": get_row_build_native_kernel()}
    for name, factory, args, kwargs in itertools.chain(
        _sparse_sweep_calls(), _sparse_contact_calls(), _solver_factory_calls(), _keyed_row_calls()
    ):
        try:
            kernel = factory(*args, **kwargs)
        except ValueError:
            continue  # The factory rejects this combination for every friction law.
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
    """Split differences into changed and missing point pins, and unpinned kernels."""
    changed = sorted(
        name for name, pin in stored.items() if name in current and current[name] != pin and name not in _PATCH_ONLY
    )
    missing = sorted(name for name in stored if name not in current and name not in _PATCH_ONLY)
    unpinned = sorted(name for name in current if name not in stored)
    return changed, missing, unpinned


class TestFeatherPGSPointStreams(unittest.TestCase):
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

    def test_point_kernels_keep_their_pins(self):
        """Keep the generated code of every pinned point-friction kernel unchanged."""
        changed, missing, _ = compare(self.stored["kernels"], self.current)
        self.assertEqual(changed, [], "Point-friction kernels changed their generated code")
        self.assertEqual(missing, [], "Point-friction kernels disappeared from the inventory")

    def test_every_kernel_is_pinned(self):
        """Pin each new kernel, or list it as patch-only, before point friction can launch it."""
        _, _, unpinned = compare(self.stored["kernels"], self.current)
        self.assertEqual(
            unpinned, [], "Record pins with: python -m newton.tests.test_feather_pgs_point_streams --record"
        )

    def test_patch_only_kernels_exist(self):
        """List only kernels that still exist as patch-only."""
        self.assertTrue(_PATCH_ONLY <= set(self.current), sorted(_PATCH_ONLY - set(self.current)))

    def test_hashes_cover_function_arguments(self):
        """Hash every function a kernel passes to a builtin, so pins and the kernel cache follow its edits."""
        unhashed = {name: _unhashed_functions(kernel) for name, kernel in self.kernels.items()}
        self.assertEqual(
            {name: functions for name, functions in unhashed.items() if functions},
            {},
            "Pass these functions through wp.static() so that Warp hashes them",
        )

    def test_inventory_covers_the_varied_factories(self):
        """Reach every factory and variant family that patch integration extends."""
        families = {name.split("(", 1)[0] for name in self.current}
        for family in (
            "sparse_pgs._get_pgs_solve_sparse_kernel",
            "sparse_contact._get_sparse_contact_response_kernel",
            "solver_feather_pgs._get_pgs_solve_sparse_diagonal_kernel",
            "solver_feather_pgs._get_build_independent_sparse_contact_groups_kernel",
            "solver_feather_pgs._get_mark_independent_sparse_contact_candidates_kernel",
            "solver_feather_pgs._get_pgs_solve_mf_gs_incremental_kernel",
            "solver_feather_pgs._get_pgs_solve_parallel_kernel",
            "solver_feather_pgs._get_pgs_solve_paired_factor_kernel",
            "keyed_rows.get_fallback_kernel",
            "keyed_rows.get_parallel_factory",
            "kernels.allocate_world_contact_slots",
            "row_packets.produce_contacts",
            "compact_contact.produce_contacts",
            "row_build.populate_world_J_masked",
            "resolved_worlds.finalize_resolved_worlds",
        ):
            with self.subTest(family=family):
                self.assertIn(family, families)
        sparse = [name for name in self.current if name.startswith("sparse_pgs.")]
        for flag in ("active_rows=True", "contact_triplets=True", "velocity_limits=True", "warm_start=True"):
            with self.subTest(flag=flag):
                self.assertTrue(any(flag in name for name in sparse))
        # The patch-only linked tiers are pinned beside the point tiers they extend.
        linked = [name for name in self.current if "linked_rows=True" in name]
        self.assertTrue(linked)
        self.assertTrue(set(linked) <= _PATCH_ONLY)


def record():
    """Write pins for new kernels and refresh patch-only ones; refuse to change a point pin."""
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
        if changed or missing:
            raise SystemExit(f"Point pins changed or disappeared; not recording: {changed + missing}")
    os.makedirs(os.path.dirname(_PINS), exist_ok=True)
    with open(_PINS, "w", encoding="utf-8") as stream:
        json.dump({"warp": wp.config.version, "kernels": current}, stream, indent=1, sort_keys=True)
        stream.write("\n")
    print(f"Recorded {len(current)} kernel pins in {_PINS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record", action="store_true", help="add pins for new kernels and refresh patch-only pins")
    arguments, remaining = parser.parse_known_args()
    if arguments.record:
        record()
    else:
        unittest.main(argv=[sys.argv[0], *remaining])
