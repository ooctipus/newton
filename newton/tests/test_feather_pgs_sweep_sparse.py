# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The native sparse PGS sweep as a configuration of the sweep emitter (:mod:`.sweep_routes_sparse`).

The emitter's kernel takes the place of ``sparse_pgs._get_pgs_solve_sparse_kernel`` at its launch site. These CPU
tests check that it takes the same arguments and resolves them as the legacy factory does (validation, defaults,
variants), that it declares the same ABI, dispatch, launch bounds and shared memory, that the first lane stores an
impulse or a cached metric only after every lane has read it, and, under a CPU emulation of the lane groups
(:mod:`newton.tests.feather_pgs_simt_fixture`, when ``g++`` is present), that both sweeps give bitwise equal
impulses, velocity deltas, ring records and readouts on random problems of every schedule. The kernels compile
offline when ``ptxas`` and ``cuobjdump`` are on ``PATH``.

The configurations are the conformance routes' (``sparse``, ``sparse_split``, ``sparse_triplets``,
``sparse_warm_start``, ``sparse_readout``, ``sparse_velocity_limits``, ``sparse_velocity_mimic``), the native
sparse tasks' under both friction laws, and the transactions, schedules and policies their options select.
"""

import concurrent.futures
import dataclasses
import inspect
import itertools
import os
import re
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from newton._src.solvers.feather_pgs import kernel_template, sparse_pgs
from newton._src.solvers.feather_pgs import sweep_routes_sparse as routes
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT as CONTACT,
)
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_FRICTION as FRICTION,
)
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT as LIMIT,
)
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT as VELOCITY_LIMIT,
)
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_MIMIC as MIMIC,
)
from newton._src.solvers.feather_pgs.solve_policy import SolvePolicy
from newton._src.solvers.feather_pgs.sparse_contact import CONTACT_RECORD_FLOATS, CONTACT_RING_FLOATS
from newton.tests import feather_pgs_simt_fixture as simt

LEGACY = sparse_pgs._get_pgs_solve_sparse_kernel.__wrapped__
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
    ("patch_single14", 112, 18, 12, {**_PATCH, "transaction_blocks": 14}, None),
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
    # Exit laws and readouts.
    ("point_staged_tolerance", 112, 18, 12, _TRIPLETS, {"velocity_change_tolerance": 5e-2}),
    ("explicit_rows_tolerance", 128, 43, 18, {"active_rows": True}, {"velocity_change_tolerance": 5e-2}),
    ("patch_trace", 112, 18, 12, _PATCH, {"velocity_change_tolerance": 5e-2, "readout": "trace"}),
    (
        "limits_tolerance_sweeps",
        64,
        9,
        9,
        {**_LIMITS, "lanes_per_world": 16},
        {"velocity_change_tolerance": 1e-3, "readout": "sweeps"},
    ),
    ("small_trace", 16, 6, 6, {**_TABLE, **_LIMITS, **_SMALL}, {"readout": "trace"}),
)


def _options(kwargs, policy):
    return {**kwargs, **({"solve_policy": SolvePolicy(**policy)} if policy else {})}


def _templates(M, D, S, options):
    """The legacy and emitter KernelTemplates of a configuration, without building Warp kernels."""
    with mock.patch.object(kernel_template.KernelTemplate, "kernel", lambda self: self):
        return LEGACY(M, D, S, **options), EMITTER(M, D, S, **options)


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
    """The emitter's factory against the legacy factory's arguments, variants and ABI."""

    def test_signatures_match_legacy(self):
        """Take the legacy factory's parameters, kinds and defaults: the source and the kernel factory alike."""

        def parameters(function):
            return [(p.name, p.kind, p.default) for p in inspect.signature(function).parameters.values()]

        self.assertEqual(parameters(EMITTER), parameters(LEGACY))
        self.assertEqual(parameters(routes.sparse_source), parameters(LEGACY))

    def test_abi_matches_legacy(self):
        """Declare the legacy ABI with its optional operands, its tiled dispatch and its launch bounds."""
        self.assertEqual(routes.SPARSE_ABI, sparse_pgs._SWEEP_ABI)
        for label, M, D, S, kwargs, policy in CONFIGURATIONS:
            with self.subTest(label):
                legacy, emitter = _templates(M, D, S, _options(kwargs, policy))
                self.assertEqual(emitter.params, legacy.params)
                self.assertEqual(emitter.native_params, legacy.native_params)
                self.assertEqual(emitter.signature, legacy.signature)
                self.assertEqual(emitter.dispatch, legacy.dispatch)
                self.assertEqual(emitter.launch_bounds, legacy.launch_bounds)
                self.assertTrue(emitter.composed)

    def test_variants_resolve_like_legacy(self):
        """Raise where the legacy factory raises, with its message; otherwise build the same variant.

        The variant is the kernel name's suffix: active rows, support table, contact triplets and their
        transaction and launch bounds, linked rows, velocity limits and their law, lanes, local factors and
        warm start, as the legacy factory resolves them (triplets need the support table and three rows; linked
        rows take part only in the active scan or a transaction; ``"auto"`` takes the single-stage transaction
        for linked rows).
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
                legacy, emitter = _outcome(LEGACY, M, D, S, options), _outcome(EMITTER, M, D, S, options)
                self.assertEqual(legacy[0], emitter[0], options)
                if legacy[0] == "raises":
                    self.assertEqual(legacy[1], emitter[1], options)
                    continue
                checked += 1
                self.assertEqual(
                    _variant(emitter[1].name, f"sweep_sparse_{M}_{D}_{S}"),
                    _variant(legacy[1].name, f"pgs_solve_sparse_{M}_{D}_{S}_0_0"),
                    options,
                )
                self.assertEqual(emitter[1].launch_bounds, legacy[1].launch_bounds, options)
        self.assertGreater(checked, 100)

    def test_rejects_like_legacy(self):
        """Reject the legacy factory's invalid capacities, lanes, layouts, transactions, laws and free-body rows."""
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
            ((16, 6, 4), {"transaction_blocks": 12}),
            ((16, 6, 4), {"velocity_limit_law": "clamped"}),
        )
        for args, options in cases:
            with self.subTest(args=args, options=options):
                with self.assertRaises(ValueError) as legacy:
                    LEGACY(*args, **options)
                with self.assertRaises(ValueError) as emitter:
                    EMITTER(*args, **options)
                self.assertEqual(str(emitter.exception), str(legacy.exception))

    def test_shared_memory_matches_legacy(self):
        """Declare the legacy's shared arrays: every buffer one array of every world's copy, in world order."""
        for label, M, D, S, kwargs, policy in CONFIGURATIONS:
            with self.subTest(label):
                options = _options(kwargs, policy)
                legacy, emitter = _templates(M, D, S, options)
                self.assertEqual(_shared_arrays(emitter.snippet), _shared_arrays(legacy.snippet))
                _, staging = routes.sparse_source(M, D, S, **options)
                self.assertTrue(staging.fits())
                self.assertEqual(staging.groups, 64 // options.get("lanes_per_world", 32))

    def test_first_lane_stores_after_every_lane_read(self):
        """Store an impulse or a cached metric from the first lane only after a barrier behind every lane's read.

        The legacy sweep stores a row's (and its pair's) impulse after the reduction that follows every lane's read
        of it, the staged triplet the normal's impulse right after every lane read it, and the metric cache its map
        entry right after every lane read it: shuffles order execution, not memory.
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
    def test_compiles_offline_with_the_legacy_resources(self):
        """Compile both sweeps for sm_120 at 64 threads: the same shared memory, no spills, and registers in the
        legacy's allocation granule (eight per thread), so the same blocks stay resident."""
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
                legacy = kernel_sass.compile_kernel(LEGACY(M, D, S, **options), arch=120, block_dim=64)
                emitter = kernel_sass.compile_kernel(EMITTER(M, D, S, **options), arch=120, block_dim=64)
                self.assertEqual(emitter["shared"], legacy["shared"])
                self.assertEqual(emitter["local"], 0)
                self.assertEqual(-(-emitter["reg"] // 8), -(-legacy["reg"] // 8))


# ---------------------------------------------------------------------------------------------------------------------
# Bitwise equality under the lane emulation
# ---------------------------------------------------------------------------------------------------------------------

# The legacy's lane races, which lockstep execution does not expose and the emulation's free lanes can: a barrier
# where every lane has read what the first lane then overwrites. They change no value.
_LEGACY_BARRIERS = (
    (
        "\n                if (lane == 0) impulse[row] = new_normal;\n",
        "\n                SYNC();\n                if (lane == 0) impulse[row] = new_normal;\n",
    ),
    (
        "\n            if (id >= 0 && lane == 0 && *metric_count <",
        "\n            SYNC();\n            if (id >= 0 && lane == 0 && *metric_count <",
    ),
)


def _race_free(template, staged: bool):
    source = template.source
    if staged:
        for old, new in _LEGACY_BARRIERS:
            if source.count(old) != 1:
                raise AssertionError(f"the legacy staged transaction changed: {old!r}")
            source = source.replace(old, new)
    return dataclasses.replace(template, source=source)


def _metric(a, c, d):
    """The tangent pair's metric (``friction_pair_metric``) in float32, for the triplets' records."""
    f = np.float32
    scale = max(f(a), f(d), f(1e-20))
    a, c, d = f(a) / scale, f(c) / scale, f(d) / scale
    largest = f(0.5) * (a + d + np.sqrt((a - d) * (a - d) + f(4) * c * c))
    smallest = max((a * d - c * c) / largest, f(0)) if largest > 0 else f(0)
    axis = np.array([c, largest - a] if a < d else [largest - d, c], dtype=np.float32)
    norm = np.sqrt(axis @ axis)
    return largest, smallest, axis / norm if norm > 0 else np.array([1, 0], dtype=np.float32)


def _problem(M, D, S, options, worlds, rng, *, linked_world):
    """Random worlds of the configuration's rows: contact triplets (patch rings when linked), limits, bilateral and
    velocity-limit rows over random supports, with their factors, records and seeded impulses."""
    table = options.get("use_support_table", False)
    linked = options.get("linked_rows", False)
    classes = []
    for _ in range(6):
        coordinates = rng.choice(D, size=S, replace=False).astype(np.int32)
        coordinates[rng.choice(S, size=int(rng.integers(0, S // 3 + 1)), replace=False)] = -1
        classes.append(coordinates)
    class_dof = np.stack(classes)
    types = np.full((worlds, M), MIMIC, np.int32)
    parents = np.full((worlds, M), -1, np.int32)
    row_class = np.zeros((worlds, M), np.int32)
    row_dof = np.full((worlds, M, S), -1, np.int32)
    counts = np.zeros(worlds, np.int32)
    for world in range(worlds):
        count = M if world % 3 == 0 else int(rng.integers(max(1, 2 * M // 3), M + 1))
        counts[world] = count
        row, normals = 0, []
        if options.get("velocity_limits"):
            while row < min(count, int(rng.integers(1, 4))):
                types[world, row] = VELOCITY_LIMIT
                row += 1
        while row < count:
            choice = rng.random()
            size = 3 if choice < 0.6 and row + 3 <= count else 1
            if size == 3:
                types[world, row : row + 3] = (CONTACT, FRICTION, FRICTION)
                parents[world, row + 1 : row + 3] = row
                normals.append(row)
            else:
                kinds = (LIMIT, MIMIC, VELOCITY_LIMIT if options.get("velocity_limits") else CONTACT)
                types[world, row] = kinds[0] if choice < 0.8 else kinds[1] if choice < 0.9 else kinds[2]
            row_class[world, row : row + size] = int(rng.integers(len(classes)))
            row_dof[world, row : row + size] = rng.permutation(class_dof[row_class[world, row]])
            row += size
        if linked or world == linked_world:
            # Patch rings of up to six normals (point contacts keep -1).
            first = 0
            while first < len(normals):
                ring = normals[first : first + int(rng.integers(1, 7))]
                for normal, following in zip(ring, ring[1:] + ring[:1], strict=True):
                    if len(ring) > 1:
                        parents[world, normal] = following
                first += len(ring)
    support = class_dof[row_class] if table else row_dof
    factor = rng.normal(size=(worlds, M, S)).astype(np.float32)
    live = np.where(support >= 0, factor, 0).astype(np.float32)
    diagonal = (np.einsum("wms,wms->wm", live, live) + 1e-3).astype(np.float32)
    diagonal[rng.random(diagonal.shape) < 0.03] = 0.0
    impulses = np.where(rng.random((worlds, M)) < 0.3, rng.uniform(0, 0.3, size=(worlds, M)), 0).astype(np.float32)
    impulses[types == FRICTION] *= np.sign(rng.normal(size=int((types == FRICTION).sum()))).astype(np.float32)
    impulses[types == VELOCITY_LIMIT] = 0.0
    arrays = {
        "world_constraint_count": counts,
        "rhs_bias": rng.uniform(-1.0, 0.6, size=(worlds, M)),
        "world_diag": diagonal,
        "world_impulses": impulses,
        "row_dof": row_dof if not table else np.full((1, 1, 1), -1),
        "row_factor": factor,
        "row_incident": rng.uniform(-0.3, 0.3, size=(worlds, M)),
        "world_row_type": types,
        "world_row_parent": parents,
        "world_row_mu": rng.uniform(0.2, 1.0, size=(worlds, M)),
        "factor_velocity_delta": np.full((worlds, D), np.nan),
        "support.row_class": row_class,
        "support.class_dof": class_dof,
    }
    if options.get("contact_triplets"):
        records = np.zeros((worlds, M // 3, CONTACT_RECORD_FLOATS + CONTACT_RING_FLOATS * linked), np.float32)
        for world, normal in zip(*np.nonzero(types == CONTACT), strict=True):
            if normal + 2 >= counts[world] or types[world, normal + 1] != FRICTION:
                continue
            dense = np.zeros((3, D), np.float32)
            for row in range(3):
                columns = support[world, normal + row]
                dense[row, columns[columns >= 0]] = live[world, normal + row][columns >= 0]
            block = dense @ dense.T
            largest, smallest, axis = _metric(block[1, 1] + 1e-3, block[1, 2], block[2, 2] + 1e-3)
            records[world, normal // 3, :8] = (
                block[0, 1],
                block[0, 2],
                block[1, 2],
                1.0 / (block[0, 0] + 1e-3),
                largest,
                smallest,
                *axis,
            )
        arrays["contact_records"] = records
    return arrays


# Run parameters: the last two stop worlds early under loose tolerances, so the votes and the exit decide results.
_RUNS = (
    {"iterations": 8, "omega": 1.0, "change_tolerance": 0.0, "relative_change_tolerance": 0.0},
    {"iterations": 8, "omega": 0.85, "change_tolerance": 1e-4, "friction_start_iteration": 2},
    {
        "iterations": 8,
        "omega": 1.0,
        "relative_change_tolerance": 1e-2,
        "friction_start_iteration": 1,
        "iteration_offset": 1,
    },
    {"iterations": 12, "omega": 1.1, "change_tolerance": 5e-2, "relative_change_tolerance": 0.3},
    {
        "iterations": 12,
        "omega": 1.0,
        "change_tolerance": 2e-2,
        "relative_change_tolerance": 0.1,
        "friction_start_iteration": 1,
    },
)

# One configuration of each schedule, layout and law.
_EMULATED = (
    "sparse_velocity_limits",
    "sparse_velocity_limits_accumulated",
    "sparse_velocity_mimic",
    "sparse_point",
    "sparse_triplets",
    "sparse_readout",
    "sparse_warm_start",
    "sparse_split_and_patch",
    "patch_single16",
    "point_single",
    "patch_full_sweep_staged",
    "explicit_rows_patch",
    "patch_trace",
    "explicit_rows_tolerance",
)


def _emulate(label: str, seed: int, directory: str, templates) -> list:
    """Run both bodies of configuration ``label`` on the same problems: ``(run, array, changed, equal)`` per output."""
    _, M, D, S, kwargs, policy = next(entry for entry in CONFIGURATIONS if entry[0] == label)
    options = _options(kwargs, policy)
    legacy, emitter = templates
    staged = routes._resolve(M, D, S, **options).staged
    lanes = options.get("lanes_per_world", 32)
    program = simt.SimtProgram(
        {"legacy": _race_free(legacy, staged), "emitter": emitter}, lanes, 64, os.path.join(directory, label)
    )
    # Every world slot of a block, and one world of a partial second block.
    worlds = 64 // lanes + 1
    rng = np.random.default_rng(seed)
    outcomes = []
    for run in _EMULATED_RUNS:
        scalars = _RUNS[run]
        arrays = _problem(M, D, S, options, worlds, rng, linked_world=1 if run % 2 else -1)
        if policy and policy.get("readout"):
            arrays["readout.sweeps"] = np.zeros((scalars["iterations"] + 1, worlds))
            arrays["readout.exhausted"] = np.zeros(worlds)
            if policy["readout"] == "trace":
                arrays["readout.stops"] = np.zeros((13, scalars["iterations"] + 2))
                arrays["readout.reasons"] = np.zeros(3)
                arrays["readout.limits"] = np.zeros((9, 26))
        expected = program.run("legacy", worlds, arrays, scalars)
        result = program.run("emitter", worlds, arrays, scalars)
        changed = not np.array_equal(expected["world_impulses"], arrays["world_impulses"].astype(np.float32))
        outcomes += [
            (run, name, changed, result[name].tobytes() == value.tobytes()) for name, value in expected.items()
        ]
    return outcomes


# Problems of every configuration: a plain solve, delayed friction with a relative tolerance and an offset, and an
# early exit under loose tolerances.
_EMULATED_RUNS = (0, 2, 3)


@unittest.skipUnless(simt.compiler(), "needs g++ for the lane emulation")
class TestSparseSweepEmulation(unittest.TestCase):
    """The emitter's and the legacy sweep's bodies run under the lane emulation, on the same random problems."""

    def test_matches_legacy_bit_for_bit(self):
        """Give the legacy's impulses, factor velocity delta, ring records and readout counters, bit for bit."""
        templates = {}
        for label in _EMULATED:
            _, M, D, S, kwargs, policy = next(entry for entry in CONFIGURATIONS if entry[0] == label)
            templates[label] = _templates(M, D, S, _options(kwargs, policy))
        with tempfile.TemporaryDirectory() as directory, concurrent.futures.ThreadPoolExecutor(4) as pool:
            futures = {
                label: pool.submit(_emulate, label, seed, directory, templates[label])
                for seed, label in enumerate(_EMULATED)
            }
            for label, future in futures.items():
                for run, name, changed, equal in future.result():
                    with self.subTest(label, run=run, array=name):
                        self.assertTrue(changed, "the sweep left every impulse unchanged")
                        self.assertTrue(equal)


if __name__ == "__main__":
    unittest.main(verbosity=2)
