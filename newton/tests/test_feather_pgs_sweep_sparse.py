# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The native sparse PGS sweep as a configuration of the sweep emitter (:mod:`.sweep_routes_sparse`).

These CPU tests check how the factory resolves its options (validation, defaults, variants and launch bounds), that
every configuration declares a staging that fits a block and one copy of each buffer per world, and that the first
lane stores an impulse or a cached metric only after every lane has read it. The kernels compile offline when
``ptxas`` and ``cuobjdump`` are on ``PATH``. The sweep's physics is the sparse tests' and the conformance gate's.

The configurations are the conformance routes' (``sparse``, ``sparse_split``, ``sparse_triplets``,
``sparse_warm_start``, ``sparse_readout``, ``sparse_velocity_limits``, ``sparse_velocity_mimic``), the native
sparse tasks' under both friction laws, and the transactions, schedules and policies their options select.
"""

import itertools
import os
import re
import shutil
import sys
import unittest
from unittest import mock

from newton._src.solvers.feather_pgs import kernel_template
from newton._src.solvers.feather_pgs import sweep_routes_sparse as routes
from newton._src.solvers.feather_pgs.solve_policy import SolvePolicy

EMITTER = routes.sparse_kernel.__wrapped__

_TABLE = {"use_support_table": True}
_ACTIVE = {"active_rows": True, "use_support_table": True}
_TRIPLETS = {**_ACTIVE, "contact_triplets": True}
_PATCH = {**_TRIPLETS, "linked_rows": True}
_LIMITS = {"velocity_limits": True}
_SMALL = {"lanes_per_world": 8, "local_factors": True}

# (label, max_constraints, max_world_dofs, max_row_dofs, options, policy)
CONFIGURATIONS = (
    # The conformance routes.
    ("sparse_velocity_limits", 32, 2, 2, {**_TABLE, **_LIMITS, **_SMALL}, None),
    (
        "sparse_velocity_limits_accumulated",
        32,
        2,
        2,
        {**_TABLE, **_LIMITS, **_SMALL, "velocity_limit_law": "accumulated"},
        None,
    ),
    ("sparse_velocity_mimic", 32, 2, 2, {**_LIMITS, **_SMALL}, None),
    ("sparse_point", 64, 14, 10, _ACTIVE, None),
    ("sparse_triplets", 64, 14, 10, _TRIPLETS, None),
    ("sparse_readout", 64, 14, 10, _ACTIVE, {"readout": "sweeps"}),
    ("sparse_warm_start", 64, 14, 10, {**_ACTIVE, "warm_start": True}, None),
    ("sparse_split_and_patch", 64, 14, 10, {**_TABLE, "lanes_per_world": 16, "linked_rows": True}, None),
    # The native sparse tasks.
    ("patch_single12", 256, 18, 12, _PATCH, None),
    ("patch_single16", 48, 18, 18, _PATCH, None),
    ("point_staged", 112, 18, 12, _TRIPLETS, None),
    ("velocity_limits_16_lanes", 64, 9, 9, {**_LIMITS, "lanes_per_world": 16}, None),
    ("explicit_rows_patch", 128, 43, 18, {"active_rows": True, "linked_rows": True}, None),
    ("explicit_rows_point", 56, 43, 18, {"active_rows": True}, None),
    ("full_sweep_table", 128, 27, 21, _TABLE, None),
    ("small_table_limits", 16, 6, 6, {**_TABLE, **_LIMITS, **_SMALL}, None),
    # Options the recipes select.
    ("patch_staged", 112, 18, 12, {**_PATCH, "contact_transaction": "staged"}, None),
    ("point_single", 256, 18, 12, {**_TRIPLETS, "contact_transaction": "single"}, None),
    ("patch_without_triplets", 256, 18, 12, {**_ACTIVE, "linked_rows": True}, None),
    ("patch_full_sweep_single", 112, 18, 12, {**_TABLE, "contact_triplets": True, "linked_rows": True}, None),
    (
        "patch_full_sweep_staged",
        112,
        18,
        12,
        {**_TABLE, "contact_triplets": True, "linked_rows": True, "contact_transaction": "staged"},
        None,
    ),
    ("patch_warm_start", 48, 18, 12, {**_PATCH, "warm_start": True}, None),
    # Readouts.
    ("point_staged_sweeps", 112, 18, 12, _TRIPLETS, {"readout": "sweeps"}),
    ("explicit_rows_trace", 128, 43, 18, {"active_rows": True}, {"readout": "trace"}),
    ("patch_trace", 112, 18, 12, _PATCH, {"readout": "trace"}),
    (
        "limits_sweeps",
        64,
        9,
        9,
        {**_LIMITS, "lanes_per_world": 16},
        {"readout": "sweeps"},
    ),
    ("small_trace", 16, 6, 6, {**_TABLE, **_LIMITS, **_SMALL}, {"readout": "trace"}),
)


def _options(kwargs, policy):
    return {**kwargs, **({"solve_policy": SolvePolicy(**policy)} if policy else {})}


def _template(M, D, S, options):
    """The KernelTemplate of a configuration, without building a Warp kernel."""
    with mock.patch.object(kernel_template.KernelTemplate, "kernel", lambda self: self):
        return EMITTER(M, D, S, **options)


def _outcome(factory, M, D, S, options):
    """``("ok", template)`` or ``("raises", message)`` of a factory call."""
    with mock.patch.object(kernel_template.KernelTemplate, "kernel", lambda self: self):
        try:
            return "ok", factory(M, D, S, **options)
        except ValueError as error:
            return "raises", str(error)


def _variant(name: str, prefix: str) -> str:
    if not name.startswith(prefix):
        raise AssertionError(f"{name} does not start with {prefix}")
    return name[len(prefix) :]


def _shared_arrays(snippet: str) -> list:
    """The ``(type, elements)`` of a snippet's ``__shared__`` arrays, sorted."""
    arrays = re.findall(r"__shared__ ((?:unsigned |signed )?\w+) \w+\[(\d+)\];", snippet)
    return sorted((ctype, int(count)) for ctype, count in arrays)


class TestSparseSweepConfiguration(unittest.TestCase):
    """The factory's options, variants, dispatch and shared memory."""

    def test_variants_resolve(self):
        """Raise for an invalid combination; otherwise build a composed variant of the declared ABI and dispatch.

        Triplets need the support table and three rows; linked rows take part only in the active scan or a
        transaction; ``"auto"`` takes the single-stage transaction for linked rows, whose launch bounds follow the
        capacity.
        """
        grid = itertools.product(
            (False, True),  # active_rows
            (False, True),  # use_support_table
            (False, True),  # contact_triplets
            (False, True),  # linked_rows
            ("auto", "single", "staged"),  # contact_transaction
            (False, True),  # velocity_limits
            (8, 32),  # lanes_per_world
            (False, True),  # local_factors
        )
        checked = 0
        for active, table, triplets, linked, transaction, limits, lanes, local in grid:
            options = {
                "active_rows": active,
                "use_support_table": table,
                "contact_triplets": triplets,
                "linked_rows": linked,
                "contact_transaction": transaction,
                "velocity_limits": limits,
                "lanes_per_world": lanes,
                "local_factors": local,
                "velocity_limit_law": "accumulated" if limits and local else "stateless",
                "warm_start": active and local,
            }
            for M, D, S in ((64, 8, 6), (2, 8, 6), (192, 18, 12)):
                outcome, template = _outcome(EMITTER, M, D, S, options)
                if outcome == "raises":
                    continue
                checked += 1
                _variant(template.name, f"sweep_sparse_{M}_{D}_{S}")
                self.assertTrue(template.composed)
                if template.launch_bounds is not None:
                    self.assertEqual(template.launch_bounds[0], 64)
        self.assertGreater(checked, 100)

    def test_rejects_invalid_options(self):
        """Reject invalid capacities, lanes, layouts, transactions, laws and free-body rows."""
        cases = (
            ((16, 6, 4), {"mf_max_constraints": 1}),
            ((16, 6, 4), {"free_row_dofs": 6}),
            ((0, 6, 4), {}),
            ((16, 6, 7), {}),
            ((16, 72, 4), {"use_support_table": True}),
            ((16, 6, 4), {"velocity_limits": True, "active_rows": True}),
            ((16, 6, 4), {"lanes_per_world": 4}),
            ((16, 18, 4), {"lanes_per_world": 16}),
            ((16, 6, 4), {"lanes_per_world": 8, "active_rows": True}),
            ((96, 6, 4), {"local_factors": True}),
            ((16, 6, 4), {"contact_transaction": "fused"}),
            ((16, 6, 4), {"velocity_limit_law": "clamped"}),
        )
        for args, options in cases:
            with self.subTest(args=args, options=options), self.assertRaises(ValueError):
                EMITTER(*args, **options)

    def test_shared_memory_is_the_staging(self):
        """Declare the staging's shared arrays: every buffer one array of every world's copy, in world order."""
        for label, M, D, S, kwargs, policy in CONFIGURATIONS:
            with self.subTest(label):
                options = _options(kwargs, policy)
                template = _template(M, D, S, options)
                _, staging = routes.sparse_source(M, D, S, **options)
                self.assertTrue(_shared_arrays(template.snippet))
                self.assertTrue(staging.fits())
                self.assertEqual(staging.groups, 64 // options.get("lanes_per_world", 32))

    def test_first_lane_stores_after_every_lane_read(self):
        """Store an impulse or a cached metric from the first lane only after a barrier behind every lane's read.

        A row's (and its pair's) impulse, the staged triplet's normal impulse and the metric cache's map entry are
        stored after a barrier behind every lane's read of them: shuffles order execution, not memory.
        """
        spans = (
            ("float old_impulse = row_impulse;", "if (lane == 0) s_impulse[i] = new_impulse;"),
            ("const float old_normal = s_impulse[i];", "if (lane == 0) s_impulse[i] = new_normal;"),
            ("const int cached = id >= 0 ? s_metric_map[id] : -1;", "s_metric_map[id] = static_cast<signed char>"),
        )
        seen = set()
        for label, M, D, S, kwargs, policy in CONFIGURATIONS:
            snippet, _ = routes.sparse_source(M, D, S, **_options(kwargs, policy))
            for read, store in spans:
                for start in (match.start() for match in re.finditer(re.escape(read), snippet)):
                    end = snippet.index(store, start)
                    seen.add(read)
                    with self.subTest(label, read=read):
                        self.assertIn("SYNC();", snippet[start:end])
        self.assertEqual(seen, {read for read, _ in spans})

    @unittest.skipUnless(shutil.which("ptxas") and shutil.which("cuobjdump"), "needs ptxas and cuobjdump on PATH")
    def test_compiles_offline(self):
        """Compile the sweeps for sm_120 at 64 threads without local memory."""
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if not os.path.isdir(os.path.join(root, "tools", "fpgs_bench")):
            self.skipTest("needs the repository's tools/fpgs_bench")
        sys.path.insert(0, root)
        try:
            from tools.fpgs_bench import kernel_sass  # noqa: PLC0415
        finally:
            sys.path.remove(root)
        for label in ("sparse_point", "patch_single12", "sparse_split_and_patch", "sparse_velocity_limits_accumulated"):
            _, M, D, S, kwargs, policy = next(entry for entry in CONFIGURATIONS if entry[0] == label)
            with self.subTest(label):
                options = _options(kwargs, policy)
                emitter = kernel_sass.compile_kernel(EMITTER(M, D, S, **options), arch=120, block_dim=64)
                self.assertEqual(emitter["local"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
