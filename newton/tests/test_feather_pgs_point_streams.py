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
from newton._src.solvers.feather_pgs.kernel_template import SweepTier

_PINS = os.path.join(os.path.dirname(__file__), "golden_data", "test_feather_pgs_point_streams", "pins.json")
_PACKAGE = "newton._src.solvers.feather_pgs"
# CUDA architecture of the factory sweeps; only measured opt-in layouts depend on it.
_ARCH = "120"

# Kernels that only persistent patch friction launches. Their pins are recorded but
# may change; a change that adds a patch-only kernel lists it here.
_PATCH_ONLY = frozenset(
    (
        "friction_patches._build",
        "friction_patches._finish_and_store",
        "friction_patches._invalidate_geometry_history",
        "friction_patches._prepare",
        "friction_patches._prepare_frame",
        "friction_patches._reset_history",
        "friction_patches.link_patch_routes",
        "friction_patches.seed_patch_routes",
        # The resolved-world classifier under persistent patch friction.
        "resolved_worlds.check_raw_contact_normals_by_route",
        "resolved_worlds.finalize_resolved_worlds_bounded",
        # Sparse-diagonal schedules over patch contact units.
        "sweep_routes_diag.sparse_diagonal_kernel(64, 10, 9, '120', contact_units=True, "
        "speculative_contact_batches=False)",
        "sweep_routes_diag.sparse_diagonal_kernel(64, 10, 9, '120', contact_units=True, "
        "speculative_contact_batches=True)",
        "solver_feather_pgs._get_build_independent_sparse_contact_groups_kernel(64, 10, '120', "
        "build_serial_contacts=False, units=True)",
        "solver_feather_pgs._get_build_independent_sparse_contact_groups_kernel(64, 10, '120', "
        "build_serial_contacts=True, units=True)",
        "solver_feather_pgs._get_mark_independent_sparse_contact_candidates_kernel(1, '120', units=True)",
        "solver_feather_pgs._get_mark_independent_sparse_contact_candidates_kernel(2, '120', units=True)",
        # Native sparse sweeps that select patch rings at sweep entry.
        "sweep_routes_sparse.sparse_kernel(48, 18, 12, active_rows=True, linked_rows=True, "
        "use_support_table=False, warm_start=False)",
        "sweep_routes_sparse.sparse_kernel(48, 18, 12, active_rows=True, linked_rows=True, "
        "use_support_table=False, warm_start=True)",
        "sweep_routes_sparse.sparse_kernel(48, 18, 12, active_rows=True, linked_rows=True, "
        "use_support_table=True, warm_start=False)",
        "sweep_routes_sparse.sparse_kernel(48, 18, 12, active_rows=True, linked_rows=True, "
        "use_support_table=True, warm_start=True)",
        # Their ring-pooled contact triplets, in the active and the full sweep.
        "sweep_routes_sparse.sparse_kernel(48, 18, 12, active_rows=False, contact_triplets=True, "
        "linked_rows=True, use_support_table=True, warm_start=False)",
        "sweep_routes_sparse.sparse_kernel(48, 18, 12, active_rows=False, contact_triplets=True, "
        "linked_rows=True, use_support_table=True, warm_start=True)",
        "sweep_routes_sparse.sparse_kernel(48, 18, 12, active_rows=True, contact_triplets=True, "
        "linked_rows=True, use_support_table=True, warm_start=False)",
        "sweep_routes_sparse.sparse_kernel(48, 18, 12, active_rows=True, contact_triplets=True, "
        "linked_rows=True, use_support_table=True, warm_start=True)",
        # Paired factor sweeps over patch contact units.
        "sweep_routes_local.paired_factor_kernel(192, 29, 23, 6, '120', False, False, contact_units=True)",
        "sweep_routes_local.paired_factor_kernel(192, 29, 23, 6, '120', False, True, contact_units=True)",
    )
)


def _label(name, args, kwargs):
    parts = [repr(arg) for arg in args] + [f"{key}={kwargs[key]!r}" for key in sorted(kwargs)]
    return f"{name}({', '.join(parts)})"


def _linked_tier_calls():
    """Patch-only parallel tiers, whose friction pairs pool their patch ring's normal impulses.

    Each also with the last-word velocity-limit pass, which the solver builds for linked tiers with joint velocity
    limits.
    """
    for tier, last_word in itertools.product(((32, 0), (48, 32)), (False, True)):
        kwargs = {
            "rows": tier[0],
            "min_rows": tier[1],
            "sweeps": 24,
            "inkernel_response": (18, 0, 0, 0),
            "linked_rows": True,
        }
        if last_word:
            kwargs["velocity_limit_pass"] = True
        yield "sweep_routes_tiers.parallel_kernel", (64, 32, 18, _ARCH), kwargs


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
    for rows, min_rows in ((32, 0), (64, 32), (96, 64), (128, 96)):
        kwargs = {"rows": rows, "min_rows": min_rows, "sweeps": 24, "inkernel_response": (16, 6, 0, 16)}
        if linked:
            kwargs["linked_rows"] = True
        yield kwargs


def _suite_sparse_sweeps():
    """Native sparse sweeps at the shapes the benchmark suite builds: ``(size, kwargs, patch_only)``."""
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
        yield size, kwargs, False
    # Persistent patch friction links the patch rings of the active sweeps.
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
        yield size, {**kwargs, "linked_rows": True}, True


def _suite_diagonal_sweeps():
    """Sparse-diagonal sweeps at the suite's shapes: ``(args, kwargs, patch_only)``; contact units are patch-only."""
    for shape in ((1024, 114, 6, _ARCH), (256, 13, 9, _ARCH)):
        yield shape, {"contact_triples": True, "speculative_contact_batches": True}, False
        yield shape, {"speculative_contact_batches": True, "contact_units": True}, True


def _keyed_last_word_calls():
    """Linked keyed tiers of a velocity-limited layout with the last-word pass (patch-only).

    The solver builds the pass for linked tiers only with joint velocity limits, so these
    tiers read a layout that plans velocity-limit rows.
    """
    for kwargs in _keyed_tier_kwargs(linked=True):
        yield (_keyed_layout(velocity_limits=True), 128, 32, 22, _ARCH), {**kwargs, "velocity_limit_pass": True}


def _readout_calls():
    """Opt-in sweep readouts of the suite's counted sweeps: ``(name, factory, args, kwargs, patch_only)``.

    ``pgs_solve_readout="sweeps"`` builds these variants of the sparse, sparse-diagonal,
    matrix-free Gauss-Seidel, paired factor and local-packet sweeps.
    """
    from newton._src.solvers.feather_pgs import (  # noqa: PLC0415
        row_packets,
        sweep_routes,
        sweep_routes_diag,
        sweep_routes_local,
        sweep_routes_sparse,
    )
    from newton._src.solvers.feather_pgs.solve_policy import SolvePolicy  # noqa: PLC0415

    readout = {"solve_policy": SolvePolicy(readout="sweeps")}
    triplets = {"active_rows": True, "use_support_table": True, "contact_triplets": True}
    factory = sweep_routes_sparse.sparse_kernel.__wrapped__
    for size in ((112, 18, 12), (128, 25, 16), (256, 18, 12)):
        yield "sweep_routes_sparse.sparse_kernel", factory, size, {**triplets, **readout}, False
        yield (
            "sweep_routes_sparse.sparse_kernel",
            factory,
            size,
            {**triplets, "linked_rows": True, **readout},
            True,
        )
    factory = sweep_routes_diag.sparse_diagonal_kernel.__wrapped__
    for schedule, patch_only in (("contact_triples", False), ("contact_units", True)):
        kwargs = {schedule: True, "speculative_contact_batches": True, **readout}
        yield "sweep_routes_diag.sparse_diagonal_kernel", factory, (1024, 114, 6, _ARCH), kwargs, patch_only
    factory = sweep_routes.mf_gs_kernel.__wrapped__
    no_limits = {"has_dense_velocity_limit_rows": False}
    for args, kwargs in (
        ((192, 64, 22, _ARCH), {}),
        ((192, 64, 29, _ARCH), {**no_limits, "factor_coordinates": True, "independent_components": True}),
        ((192, 64, 15, _ARCH), {**no_limits, "skip_local_internal_worlds": True, "local_internal_max_constraints": 19}),
    ):
        yield "sweep_routes.mf_gs_kernel", factory, args, {**kwargs, **readout}, False
    factory = sweep_routes_local.paired_factor_kernel.__wrapped__
    for units in (False, True):
        kwargs = {"contact_units": True, **readout} if units else readout
        yield "sweep_routes_local.paired_factor_kernel", factory, (192, 29, 23, 6, _ARCH, False, True), kwargs, units
    packets = row_packets.get_local_factory.__wrapped__(9, 6, 40)
    pair = {"paired_dof_count": 6, "persistent_queue": True, "warps_per_block": 2}
    for args, kwargs in (
        (
            (192, 9, 9, _ARCH),
            {"lanes_per_world": 8, "contact_capable": False, "warps_per_block": 2},
        ),
        ((192, 20, 9, _ARCH), pair),
        ((192, 40, 9, _ARCH), {**pair, "mf_max_constraints": 64, "local_mf_max_constraints": 12}),
    ):
        yield "row_packets.get_local_factory(9, 6, 40)", packets, args, {**kwargs, **readout}, False


def _patch_only_factory_labels():
    """Labels of the factory-built kernels only patch friction launches."""
    for name, args, kwargs in _linked_tier_calls():
        yield _label(name, args, kwargs)
    for kwargs in _keyed_tier_kwargs(linked=True):
        yield _label("keyed_rows.get_parallel_factory", (_keyed_layout(), 128, 32, 22, _ARCH), kwargs)
    for size, kwargs, patch_only in _suite_sparse_sweeps():
        if patch_only:
            yield _label("sweep_routes_sparse.sparse_kernel", size, kwargs)
    for args, kwargs, patch_only in _suite_diagonal_sweeps():
        if patch_only:
            yield _label("sweep_routes_diag.sparse_diagonal_kernel", args, kwargs)
    for args, kwargs in _keyed_last_word_calls():
        yield _label("keyed_rows.get_parallel_factory", args, kwargs)
    for name, _, args, kwargs, patch_only in _readout_calls():
        if patch_only:
            yield _label(name, args, kwargs)


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
    """Native sparse sweeps over the flags point friction selects (sweep_routes_sparse.py)."""
    from newton._src.solvers.feather_pgs.sweep_routes_sparse import sparse_kernel  # noqa: PLC0415

    factory = sparse_kernel.__wrapped__
    for size, flags in itertools.product(((32, 8, 8), (48, 18, 12)), itertools.product((False, True), repeat=6)):
        active, table, triplets, velocity, local, warm = flags
        # The factory folds unsatisfiable requests into another variant; pin each variant once.
        if triplets and not (active and table):
            continue
        for lanes in (8, 16, 32):
            kwargs = {
                # The pins name the free-body capacities, which native sparse factors leave at zero.
                "mf_max_constraints": 0,
                "free_row_dofs": 0,
                "active_rows": active,
                "use_support_table": table,
                "contact_triplets": triplets,
                "velocity_limits": velocity,
                "lanes_per_world": lanes,
                "local_factors": local,
                "warm_start": warm,
            }
            yield "sweep_routes_sparse.sparse_kernel", factory, size, kwargs


def _linked_sweep_calls():
    """Patch-only native sparse sweeps over linked rows (sweep_routes_sparse.py)."""
    from newton._src.solvers.feather_pgs.sweep_routes_sparse import sparse_kernel  # noqa: PLC0415

    factory = sparse_kernel.__wrapped__
    for table, warm in itertools.product((False, True), repeat=2):
        kwargs = {"active_rows": True, "use_support_table": table, "warm_start": warm, "linked_rows": True}
        yield "sweep_routes_sparse.sparse_kernel", factory, (48, 18, 12), kwargs
    for active, warm in itertools.product((False, True), repeat=2):
        kwargs = {
            "active_rows": active,
            "use_support_table": True,
            "contact_triplets": True,
            "warm_start": warm,
            "linked_rows": True,
        }
        yield "sweep_routes_sparse.sparse_kernel", factory, (48, 18, 12), kwargs


def _sparse_contact_calls():
    from newton._src.solvers.feather_pgs.response_builds_rows import factor_contact_rows_kernel  # noqa: PLC0415

    factory = factor_contact_rows_kernel.__wrapped__
    for dofs, prepare, table in itertools.product((8, 18), (False, True), (False, True)):
        kwargs = {"prepare_rows": prepare, "use_support_table": table}
        yield "response_builds_rows.factor_contact_rows_kernel", factory, (dofs,), kwargs
    # The block records of the sparse sweep's contact units, under either law.
    for dofs in (8, 18, 25):
        kwargs = {"prepare_rows": True, "use_support_table": True, "contact_records": True}
        yield "response_builds_rows.factor_contact_rows_kernel", factory, (dofs,), kwargs


def _dense_build_calls():
    """The dense response builds without variants (response_builds_dense.py): every stage the dense routes run."""
    from newton._src.solvers.feather_pgs import response_builds_dense  # noqa: PLC0415

    for name in (
        "response_rows_kernel",
        "response_lanes_kernel",
        "response_diagonal_kernel",
        "diagonal_rows_kernel",
        "diagonal_groups_kernel",
        "cfm_worlds_kernel",
    ):
        yield f"response_builds_dense.{name}", getattr(response_builds_dense, name).__wrapped__, (), {}
    from newton._src.solvers.feather_pgs import packed_rows  # noqa: PLC0415

    # The dense rows packed by support: their build (by the side sizes it whitens), the velocity of their impulses
    # and the bilateral projection.
    for sizes in ((6,), (6, 9), (18,), (6, 106)):
        yield "packed_rows.pack_kernel", packed_rows.pack_kernel.__wrapped__, (sizes,), {}
    from newton._src.solvers.feather_pgs import response_builds_rows  # noqa: PLC0415

    # The bilateral elimination of blocks of up to eight rows: its setup, correction and projection weights.
    for part in ("setup", "correct", "weights"):
        yield (
            f"response_builds_rows.elimination.{part}",
            lambda rows, part=part: getattr(response_builds_rows.elimination.__wrapped__(rows), part),
            (8,),
            {},
        )

    # The sparse-diagonal rows whitened in place, with their diagonals.
    yield (
        "response_builds_rows.sparse_pair_whiten_kernel",
        response_builds_rows.sparse_pair_whiten_kernel.__wrapped__,
        (),
        {},
    )


def _solver_factory_calls():
    from newton._src.solvers.feather_pgs import solver_feather_pgs as solver  # noqa: PLC0415
    from newton._src.solvers.feather_pgs import (  # noqa: PLC0415
        sweep_routes,
        sweep_routes_diag,
        sweep_routes_local,
        sweep_routes_tiers,
    )

    name = "solver_feather_pgs."
    for triples, batches in itertools.product((False, True), repeat=2):
        kwargs = {"contact_triples": triples, "speculative_contact_batches": batches}
        factory = sweep_routes_diag.sparse_diagonal_kernel.__wrapped__
        yield "sweep_routes_diag.sparse_diagonal_kernel", factory, (64, 10, 9, _ARCH), kwargs
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
        factory = sweep_routes_diag.sparse_diagonal_kernel.__wrapped__
        yield "sweep_routes_diag.sparse_diagonal_kernel", factory, (64, 10, 9, _ARCH), kwargs
    for serial in (False, True):
        factory = solver._get_build_independent_sparse_contact_groups_kernel.__wrapped__
        kwargs = {"build_serial_contacts": serial, "units": True}
        yield name + "_get_build_independent_sparse_contact_groups_kernel", factory, (64, 10, _ARCH), kwargs
    for dofs in (1, 2):
        factory = solver._get_mark_independent_sparse_contact_candidates_kernel.__wrapped__
        yield name + "_get_mark_independent_sparse_contact_candidates_kernel", factory, (dofs, _ARCH), {"units": True}
    for skip_rows, limits, coordinates, components in itertools.product((0, 32), *([(False, True)] * 3)):
        kwargs = {
            "skip_tier_rows": skip_rows,
            "has_dense_velocity_limit_rows": limits,
            "factor_coordinates": coordinates,
            "independent_components": components,
        }
        factory = sweep_routes.mf_gs_kernel.__wrapped__
        yield "sweep_routes.mf_gs_kernel", factory, (64, 32, 12, _ARCH), kwargs
    # Streamed metadata with the packed row types and parents resident, which single-wave launches
    # of large dense capacities select under either law; limits add the limit scans' global reads.
    for limits in (False, True):
        kwargs = {
            "shared_metadata": False,
            "resident_row_metadata": True,
            "has_dense_velocity_limit_rows": limits,
            "skip_tier_rows": 0 if limits else 32,
        }
        factory = sweep_routes.mf_gs_kernel.__wrapped__
        yield "sweep_routes.mf_gs_kernel", factory, (64, 32, 12, _ARCH), kwargs
    for triples, components in itertools.product((False, True), repeat=2):
        factory = sweep_routes_local.paired_factor_kernel.__wrapped__
        yield "sweep_routes_local.paired_factor_kernel", factory, (192, 29, 23, 6, _ARCH, triples, components), {}
    # Patch-only unit walks of the same factory.
    for components in (False, True):
        factory = sweep_routes_local.paired_factor_kernel.__wrapped__
        args = (192, 29, 23, 6, _ARCH, False, components)
        yield "sweep_routes_local.paired_factor_kernel", factory, args, {"contact_units": True}

    # Parallel tiers: one articulation per world, its response formed in-kernel.
    for tier in ((32, 0), (48, 32)):
        kwargs = {"rows": tier[0], "min_rows": tier[1], "sweeps": 24, "inkernel_response": (18, 0, 0, 0)}
        yield "sweep_routes_tiers.parallel_kernel", sweep_routes_tiers.parallel_kernel, (64, 32, 18, _ARCH), kwargs
    for label, args, kwargs in _linked_tier_calls():
        yield label, sweep_routes_tiers.parallel_kernel, args, kwargs


def _spliced_sweep_calls():
    """Sweeps that splice the friction law or vote a sweep exit, beyond the factories above.

    Local owners, plain and over packet rows: a contactless subwarp articulation,
    an articulation pair and the pair's residual with matrix-free rows. Then the
    tiled-row sweep with and without row budgets and the legacy matrix-free sweep.
    """
    from newton._src.solvers.feather_pgs import row_packets, sweep_routes_local  # noqa: PLC0415

    local = sweep_routes_local.local_owned_kernel.__wrapped__
    packets = row_packets.get_local_factory.__wrapped__(9, 6, 40)
    pair = {"paired_dof_count": 6, "persistent_queue": True}
    for args, options in (
        ((192, 9, 9, _ARCH), {"lanes_per_world": 8, "contact_capable": False}),
        ((192, 20, 9, _ARCH), pair),
        ((192, 40, 9, _ARCH), {**pair, "mf_max_constraints": 64, "local_mf_max_constraints": 12}),
    ):
        kwargs = {**options, "warps_per_block": 2}
        yield "sweep_routes_local.local_owned_kernel", local, args, kwargs
        yield "row_packets.get_local_factory(9, 6, 40)", packets, args, kwargs


def _suite_shape_calls():
    """Factory kernels at the shapes the benchmark suite's routes build, beyond those pinned above.

    Each call reproduces the kernel name its route records and, where a head
    profile ran the route, the profile's kernel suffix: the matrix-free
    Gauss-Seidel, sparse, sparse-diagonal and keyed parallel sweeps, the keyed
    limit rows, the sparse mass factors, the sparse contact responses, the
    packet prefix and the paired-component preparation. A byte-identical
    rewrite of their generators then keeps the binaries the suite runs.
    """
    from newton._src.solvers.feather_pgs import (  # noqa: PLC0415
        keyed_row_layout,
        keyed_rows,
        response_builds_rows,
        row_packets,
        sweep_routes,
        sweep_routes_diag,
        sweep_routes_sparse,
        sweep_routes_tiers,
    )
    from newton._src.solvers.feather_pgs.keyed_row_layout import KeyedLayout  # noqa: PLC0415

    no_limits = {"has_dense_velocity_limit_rows": False}
    factory = sweep_routes.mf_gs_kernel.__wrapped__
    for args, kwargs in (
        ((192, 64, 22, _ARCH), {}),
        ((192, 64, 22, _ARCH), {"skip_tier_rows": 128}),
        ((192, 64, 15, _ARCH), {**no_limits, "skip_local_internal_worlds": True, "local_internal_max_constraints": 19}),
        ((192, 64, 29, _ARCH), {**no_limits, "factor_coordinates": True, "independent_components": True}),
        ((72, 32, 18, _ARCH), {**no_limits, "skip_tier_rows": 48}),
    ):
        yield "sweep_routes.mf_gs_kernel", factory, args, kwargs
    factory = sweep_routes_sparse.sparse_kernel.__wrapped__
    for size, kwargs, _ in _suite_sparse_sweeps():
        yield "sweep_routes_sparse.sparse_kernel", factory, size, kwargs
    factory = sweep_routes_diag.sparse_diagonal_kernel.__wrapped__
    for args, kwargs, _ in _suite_diagonal_sweeps():
        yield "sweep_routes_diag.sparse_diagonal_kernel", factory, args, kwargs

    def parallel(layout, *args, **kwargs):
        return keyed_rows.get_parallel_factory.__wrapped__(sweep_routes_tiers.parallel_kernel, layout)(*args, **kwargs)

    def limit_prefix(layout):
        return keyed_row_layout.prefix_kernel.__wrapped__(layout, keyed_rows.PrefixInput, keyed_rows.KineticRows)

    # The tiers read only a layout's coordinate blocks; its limit masks shape the limit rows.
    limited = KeyedLayout(16, 6, 0, 16, 128, True, 0xFFFF, 0xFFFF)
    single = KeyedLayout(18, 0, 0, 0, 48, False, 0)
    for layout, shape, tiers in (
        (limited, (192, 64, 22, _ARCH), ((32, 0), (64, 32), (96, 64), (128, 96))),
        (single, (72, 32, 18, _ARCH), ((32, 0), (48, 32))),
    ):
        blocks = (layout.primary, layout.secondary, layout.primary_offset, layout.secondary_offset)
        for rows, min_rows in tiers:
            kwargs = {"rows": rows, "min_rows": min_rows, "sweeps": 24, "inkernel_response": blocks}
            yield "keyed_rows.get_parallel_factory", parallel, (layout, *shape), kwargs
    yield "keyed_row_layout.prefix_kernel", limit_prefix, (limited,), {}
    factory = response_builds_rows.sparse_factor_kernel.__wrapped__
    for size in ((6, 21), (9, 44), (14, 81), (18, 117), (18, 135), (25, 194), (27, 243), (43, 434)):
        yield "response_builds_rows.sparse_factor_kernel", factory, size, {}
    # 18 coordinates with a support table are pinned with the patch-varied factories.
    factory = response_builds_rows.factor_contact_rows_kernel.__wrapped__
    for dofs, table in ((6, True), (9, False), (14, True), (25, True), (27, True), (43, False)):
        kwargs = {"prepare_rows": True, "use_support_table": table}
        yield "response_builds_rows.factor_contact_rows_kernel", factory, (dofs,), kwargs
    yield "row_packets.get_prefix_kernel", row_packets.get_prefix_kernel.__wrapped__, (9, 1, 40), {}
    components = response_builds_rows.paired_components_kernel.__wrapped__
    yield "response_builds_rows.paired_components_kernel", components, (23, 6), {}


def _keyed_row_calls():
    """Keyed rows of a 16+6 coordinate world (keyed_rows.py), including their parallel tiers."""
    from newton._src.solvers.feather_pgs import keyed_rows, sweep_routes_tiers  # noqa: PLC0415

    def parallel(layout, *args, **kwargs):
        return keyed_rows.get_parallel_factory.__wrapped__(sweep_routes_tiers.parallel_kernel, layout)(*args, **kwargs)

    layout = _keyed_layout()
    for kwargs in _keyed_tier_kwargs(linked=False):
        yield "keyed_rows.get_parallel_factory", parallel, (layout, 128, 32, 22, _ARCH), kwargs
    # The persistent patch row law of the same rows.
    for kwargs in _keyed_tier_kwargs(linked=True):
        yield "keyed_rows.get_parallel_factory", parallel, (layout, 128, 32, 22, _ARCH), kwargs
    for args, kwargs in _keyed_last_word_calls():
        yield "keyed_rows.get_parallel_factory", parallel, args, kwargs


def _factory_kernels():
    """Kernels of the factories patch integration varies, built with point arguments."""
    kernels = {}
    for name, factory, args, kwargs in itertools.chain(
        _sparse_sweep_calls(),
        _linked_sweep_calls(),
        _sparse_contact_calls(),
        _dense_build_calls(),
        _solver_factory_calls(),
        _spliced_sweep_calls(),
        _suite_shape_calls(),
        _keyed_row_calls(),
        ((name, factory, args, kwargs) for name, factory, args, kwargs, _ in _readout_calls()),
    ):
        try:
            kernel = factory(*args, **kwargs)
        except ValueError:
            continue  # The factory rejects this combination for every friction law.
        # A tier factory returns the tier's launch descriptor.
        kernels[_label(name, args, kwargs)] = kernel.kernel if isinstance(kernel, SweepTier) else kernel
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
            "sweep_routes_sparse.sparse_kernel",
            "response_builds_rows.factor_contact_rows_kernel",
            "sweep_routes_diag.sparse_diagonal_kernel",
            "solver_feather_pgs._get_build_independent_sparse_contact_groups_kernel",
            "solver_feather_pgs._get_mark_independent_sparse_contact_candidates_kernel",
            "sweep_routes_tiers.parallel_kernel",
            "sweep_routes_local.paired_factor_kernel",
            "keyed_rows.get_parallel_factory",
            "kernels.allocate_world_contact_slots",
            "row_packets.produce_contacts",
            "compact_contact.produce_contacts",
            "row_build.populate_world_J_masked",
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
        # The MF-GS kernels that keep row metadata resident are pinned.
        self.assertTrue([name for name in self.current if "resident_row_metadata=True" in name])
        # The patch-only linked tiers are pinned beside the point tiers they extend.
        linked = [name for name in self.current if "linked_rows=True" in name]
        self.assertTrue(linked)
        self.assertTrue(set(linked) <= _PATCH_ONLY)
        # Their last-word velocity-limit pass, in the plain and the keyed-row tiers.
        last_word = {name.split("(", 1)[0] for name in linked if "velocity_limit_pass=True" in name}
        self.assertEqual(last_word, {"sweep_routes_tiers.parallel_kernel", "keyed_rows.get_parallel_factory"})

    def test_inventory_covers_the_spliced_sweeps(self):
        """Reach the sweeps that splice the friction law or vote an exit, beyond the patch-varied factories."""
        families = {name.split("(", 1)[0] for name in self.current}
        for family in (
            "sweep_routes_local.local_owned_kernel",
            "row_packets.get_local_factory",
        ):
            with self.subTest(family=family):
                self.assertIn(family, families)
                # Every friction law launches these sweeps, so their pins are point pins.
                self.assertFalse({name for name in self.current if name.startswith(family + "(")} & _PATCH_ONLY)

    def test_inventory_builds_the_suite_shapes(self):
        """Build every factory call at the suite's shapes; the suite's routes build them, so none may be rejected."""
        labels = {_label(name, args, kwargs) for name, _, args, kwargs in _suite_shape_calls()}
        self.assertEqual(sorted(labels - set(self.current)), [])


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
