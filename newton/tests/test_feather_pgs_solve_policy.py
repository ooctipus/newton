# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compile the Gauss-Seidel stopping rule and its readout as one frozen solve policy.

These host tests build kernels and records without a GPU. The default policy
must build exactly the kernels built before policies existed; the
point-stream pins check that on the whole inventory, and the tests here check
it on one call of every factory that takes a policy.
"""

import dataclasses
import inspect
import itertools
import json
import re
import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import (
    kernels,
    mf_dispatch,
    mf_gs_source,
    paired_factor_source,
    path_selection,
    row_packets,
)
from newton._src.solvers.feather_pgs import solver_feather_pgs as solver
from newton._src.solvers.feather_pgs.device_preludes import ARCHITECTURE_GUARD
from newton._src.solvers.feather_pgs.row_inputs import make_row_kernel
from newton._src.solvers.feather_pgs.solve_policy import (
    EXACT_POLICY,
    SWEEP_READOUT,
    SWEEP_TRACE,
    TRACE_BINS,
    TRACE_GRID,
    TRACE_KINDS,
    SolvePolicy,
    SolvePolicyUnsupported,
    SolveReadout,
    SolveReadoutRecord,
    SolveTrace,
    sweep_exit,
    vote,
)
from newton._src.solvers.feather_pgs.sparse_pgs import _get_pgs_solve_sparse_kernel
from newton._src.solvers.feather_pgs.velocity_limit_law import VELOCITY_LIMIT_LAWS
from newton.solvers import SolverFeatherPGS
from newton.tests.test_feather_pgs_native_prelude import _native_snippets
from newton.tests.test_feather_pgs_point_streams import _kernel_hash

SWEEPS = SolvePolicy(readout="sweeps")
VELOCITY = SolvePolicy(velocity_change_tolerance=1.0e-3)
TRACE = SolvePolicy(readout="trace")
TRACED_VELOCITY = SolvePolicy(readout="trace", velocity_change_tolerance=1.0e-3)

# Velocity-scaled votes each factory call must emit at a positive tolerance: every vote site with its own A_ii.
_SCALED_VOTES = {
    "sparse": (
        "fabsf(change) * (denominator)",
        "fabsf(sibling_change) * (diagonal[sibling])",
        "fabsf(normal_change) * (diagonal[row])",
        "fabsf(tangent_change) * (diagonal[tangent])",
        "sweep_violation <= 0.001f",
    ),
    "sparse_lanes": ("fabsf(change) * (denominator)",),
    "mf_gs": (
        "fabsf(delta_impulse) * (denom)",
        "fabsf(sib_delta) * (s_diag_dense[sib])",
        "fabsf(delta_impulse) * (mf_diag > 0.0f ? 1.0f / mf_diag : 0.0f)",
        "fabsf(delta_impulse_vlim) * (denom)",
        "fabsf(delta_impulse_mf_vlim) * (1.0f / mf_diag)",
    ),
    "paired": (
        "fabsf(delta_impulse) * (denom)",
        "fabsf(tangent1_delta) * (s_diag[tangent1])",
        "fabsf(tangent2_delta) * (s_diag[tangent2])",
    ),
    "local": ("fabsf(delta_impulse) * (1.0f / inverse_diagonal)",),
    "packets": (
        "fabsf(delta_impulse) * (1.0f / inverse_diagonal)",
        "fabsf(sibling_delta) * (sibling_inverse_diagonal > 0.0f ? 1.0f / sibling_inverse_diagonal : 0.0f)",
    ),
    "sparse_diagonal": (
        "fabsf(delta_impulse) * (denominator)",
        "fabsf(sibling_delta) * (s_diag[sibling])",
        "fabsf(delta_impulse) * (normal_denom)",
        "fabsf(tangent1_delta) * (s_diag[tangent1])",
    ),
    "incremental": (
        "fabsf(delta_impulse) * (s_diag[i])",
        "fabsf(sib_delta) * (s_diag[sib])",
        "fabsf(delta_impulse_vlim) * (s_diag[i])",
    ),
}


def _factory_calls():
    """One call of every sweep factory that takes a policy: ``(name, factory, args, kwargs)``."""
    packets = row_packets.get_local_factory.__wrapped__(9, 6, 40)
    return (
        (
            "sparse",
            _get_pgs_solve_sparse_kernel.__wrapped__,
            (48, 18, 12),
            {"active_rows": True, "use_support_table": True, "contact_triplets": True},
        ),
        ("sparse_lanes", _get_pgs_solve_sparse_kernel.__wrapped__, (16, 6, 6), {"lanes_per_world": 8}),
        (
            "mf_gs",
            solver._get_pgs_solve_mf_gs_kernel.__wrapped__,
            (64, 32, 12, "120"),
            {"has_drive_rows": False},
        ),
        (
            "paired",
            solver._get_pgs_solve_paired_factor_kernel.__wrapped__,
            (192, 29, 23, 6, "120", False, True),
            {"contact_units": True},
        ),
        (
            "local",
            solver._get_pgs_solve_local_owned_kernel.__wrapped__,
            (192, 9, 9, "120"),
            {"lanes_per_world": 8, "contact_capable": False, "warps_per_block": 2, "dense_response_matrix": True},
        ),
        (
            "packets",
            packets,
            (192, 20, 9, "120"),
            {"paired_dof_count": 6, "persistent_queue": True, "dense_response_matrix": True},
        ),
        (
            "sparse_diagonal",
            solver._get_pgs_solve_sparse_diagonal_kernel.__wrapped__,
            (64, 10, 9, "120"),
            {"speculative_contact_batches": True, "contact_units": True},
        ),
        (
            "incremental",
            solver._get_pgs_solve_mf_gs_incremental_kernel.__wrapped__,
            (64, 32, 12, "120"),
            {"incremental_rows": 32},
        ),
    )


# An accumulated velocity-limit impulse, max(old - r / d, 0) with the unrelaxed step, as velocity_limit_law emits it.
_LAW_SITE = re.compile(r"fmaxf\(old_\w+ - residual")


def _axis(name, values=(False, True)):
    """One census axis: a keyword over its values."""
    return tuple({name: value} for value in values)


def _vote_census():
    """Every combination of the flags that add or remove vote sites, per policy factory: ``(name, factory, args, kwargs)``.

    The axes follow the exit-voting factories' census in ``tools/fpgs_bench/kernel_identity.py`` at one
    shape each, crossed with the factories' other options that rewrite their sweeps: the velocity-limit
    law of the sparse, matrix-free and incremental sweeps, the sparse sweep's triplet transaction and
    the matrix-free sweep's resident row metadata. The matrix-free sweep keeps the current friction law
    without torsion, the variants that compile the velocity law; the census pins the others' text
    unchanged.
    """
    laws = _axis("velocity_limit_law", VELOCITY_LIMIT_LAWS)
    local = (
        _axis("paired_dof_count", (0, 6)),
        _axis("persistent_queue"),
        _axis("warps_per_block", (1, 2)),
        _axis("lanes_per_world", (32, 16, 8)),
        _axis("contact_capable"),
        ({}, {"mf_max_constraints": 64, "local_mf_max_constraints": 12}),
        _axis("dense_response_matrix"),
    )
    specs = (
        (
            "sparse",
            _get_pgs_solve_sparse_kernel.__wrapped__,
            (48, 18, 12),
            (
                ({}, {"mf_max_constraints": 32, "free_row_dofs": 6}),
                _axis("active_rows"),
                _axis("use_support_table"),
                _axis("contact_triplets"),
                _axis("velocity_limits"),
                _axis("local_factors"),
                _axis("warm_start"),
                _axis("linked_rows"),
                _axis("contact_transaction", ("auto", "single", "staged")),
                laws,
            ),
        ),
        (
            "sparse_lanes",
            _get_pgs_solve_sparse_kernel.__wrapped__,
            (32, 8, 8),
            (
                _axis("lanes_per_world", (16, 8)),
                _axis("velocity_limits"),
                _axis("local_factors"),
                _axis("warm_start"),
                laws,
            ),
        ),
        (
            "mf_gs",
            solver._get_pgs_solve_mf_gs_kernel.__wrapped__,
            (64, 32, 12, "120"),
            (
                _axis("shared_metadata"),
                _axis("stream_mf_impulses"),
                _axis("has_drive_rows"),
                _axis("has_dense_velocity_limit_rows"),
                _axis("fuse_vel_limits"),
                ({}, {"skip_local_internal_worlds": True, "local_internal_max_constraints": 19}),
                _axis("factor_coordinates"),
                _axis("skip_incremental_rows", (0, 32)),
                _axis("independent_components"),
                _axis("resident_row_metadata"),
                laws,
            ),
        ),
        (
            "paired",
            solver._get_pgs_solve_paired_factor_kernel.__wrapped__,
            (192, 29, 23, 6, "120"),
            (_axis("contact_triples"), _axis("independent_components"), _axis("contact_units")),
        ),
        ("local", solver._get_pgs_solve_local_owned_kernel.__wrapped__, (192, 20, 9, "120"), local),
        ("packets", row_packets.get_local_factory.__wrapped__(9, 6, 40), (192, 20, 9, "120"), local),
        (
            "sparse_diagonal",
            solver._get_pgs_solve_sparse_diagonal_kernel.__wrapped__,
            (64, 10, 9, "120"),
            (_axis("contact_triples"), _axis("speculative_contact_batches"), _axis("contact_units")),
        ),
        (
            "incremental",
            solver._get_pgs_solve_mf_gs_incremental_kernel.__wrapped__,
            (64, 32, 12, "120"),
            (
                ({"incremental_rows": 32}, {"incremental_rows": 64, "min_rows": 32}),
                _axis("has_drive_rows"),
                _axis("has_dense_velocity_limit_rows"),
                _axis("skip_local_internal_worlds"),
                laws,
            ),
        ),
    )
    for name, factory, args, axes in specs:
        for options in itertools.product(*axes):
            kwargs = {}
            for option in options:
                kwargs.update(option)
            yield name, factory, args, kwargs


class TestSolvePolicy(unittest.TestCase):
    """The frozen policy, its emitters and the factories that take it."""

    def test_policy_is_frozen_hashable_and_validated(self):
        """Freeze the policy, compare it by value and reject an unknown readout."""
        self.assertEqual(SolvePolicy(), EXACT_POLICY)
        self.assertEqual(SolvePolicy(readout="sweeps"), SWEEPS)
        self.assertEqual(hash(SolvePolicy(readout="sweeps")), hash(SWEEPS))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            SWEEPS.readout = "off"
        with self.assertRaisesRegex(ValueError, "readout"):
            SolvePolicy(readout="every_sweep")
        self.assertEqual(
            (EXACT_POLICY.counts, EXACT_POLICY.suffix, EXACT_POLICY.preludes, EXACT_POLICY.operands),
            (False, "", (), {}),
        )
        self.assertEqual(
            (SWEEPS.suffix, SWEEPS.preludes, SWEEPS.operands), ("_sweeps", (SWEEP_READOUT,), {"readout": SolveReadout})
        )

    def test_equal_policies_share_cached_kernels(self):
        """Return one cached kernel for equal policies, and another for a different policy."""
        first = _get_pgs_solve_sparse_kernel(16, 6, 4, solve_policy=SolvePolicy(readout="sweeps"))
        second = _get_pgs_solve_sparse_kernel(16, 6, 4, solve_policy=SolvePolicy(readout="sweeps"))
        self.assertIs(first, second)
        self.assertIsNot(first, _get_pgs_solve_sparse_kernel(16, 6, 4))

    def test_policy_and_readout_options_are_keyword_only(self):
        """Take the policy and the solver's readout and tolerance options by keyword only, defaulting to the exact law."""
        parameters = inspect.signature(SolverFeatherPGS).parameters
        for option, default in (("pgs_solve_readout", "off"), ("pgs_velocity_change_tolerance", 0.0)):
            with self.subTest(option=option):
                self.assertEqual(
                    (parameters[option].kind, parameters[option].default), (inspect.Parameter.KEYWORD_ONLY, default)
                )
        factories = [(name, factory) for name, factory, _, _ in _factory_calls()]
        factories += [("mf_gs_source", mf_gs_source.get_source), ("paired_source", paired_factor_source.get_source)]
        for name, factory in factories:
            with self.subTest(factory=name):
                parameter = inspect.signature(factory).parameters["solve_policy"]
                self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
                self.assertIs(parameter.default, EXACT_POLICY)

    def test_exact_policy_builds_the_kernels_of_a_policy_less_call(self):
        """Build the same Warp content hash, hence the same binary, with the exact policy passed or omitted."""
        for name, factory, args, kwargs in _factory_calls():
            with self.subTest(factory=name):
                kernel = factory(*args, **kwargs)
                self.assertEqual(
                    _kernel_hash(factory(*args, solve_policy=EXACT_POLICY, **kwargs)), _kernel_hash(kernel)
                )
                self.assertFalse(getattr(kernel, "_fpgs_solve_readout", False))
                self.assertNotIn("solve_readout_sweeps", "".join(_native_snippets(kernel)))

    def test_readout_appends_one_operand_prelude_and_exit_record(self):
        """Append the readout operand last, declare its prelude after the law and record every solve once."""
        for name, factory, args, kwargs in _factory_calls():
            with self.subTest(factory=name):
                exact = factory(*args, **kwargs)
                kernel = factory(*args, solve_policy=SWEEPS, **kwargs)
                self.assertEqual(kernel.key, exact.key + "_sweeps")
                self.assertTrue(kernel._fpgs_solve_readout)
                labels = [argument.label for argument in kernel.adj.args]
                self.assertEqual(labels, [argument.label for argument in exact.adj.args] + ["readout"])
                self.assertIs(kernel.adj.args[-1].type, SolveReadout)
                snippets = {snippet for snippet in _native_snippets(kernel) if "solve_readout_sweeps" in snippet}
                self.assertEqual(len(snippets), 1)
                (snippet,) = snippets
                self.assertEqual(snippet.count(ARCHITECTURE_GUARD), 1)
                # The readout helper follows the friction law, and each sweep records at its one exit.
                if "friction_pair" in snippet:
                    self.assertLess(snippet.index("friction_pair"), snippet.index(SWEEP_READOUT.source))
                body = snippet.split(SWEEP_READOUT.source, 1)[1]
                self.assertEqual(body.count("solve_readout_sweeps("), 1)
                self.assertEqual(body.count("if (sweep_stop) break;"), 1)

    def test_sweep_exit_keeps_the_exact_exit_text(self):
        """Return a sweep's own exit verbatim without a readout, and a recorded exit with one."""
        exact = "        if (__ballot_sync(MASK, changed != 0) == 0u) break;\n"
        self.assertIs(sweep_exit(EXACT_POLICY, exact, quiet="unused"), exact)
        counted = sweep_exit(
            SWEEPS, exact, quiet="done", gate="open", world="w", iteration="k", iterations="n", lane="l"
        )
        self.assertIn("const bool sweep_stop = open && done;", counted)
        self.assertIn("if (l == 0 && (sweep_stop || k + 1 == n))", counted)
        self.assertIn("solve_readout_sweeps(w, k + 1, sweep_stop);", counted)

    def test_world_dispatch_requires_the_world_first(self):
        """Reject the world dispatches of generated kernels unless the native's first operand is the world."""

        def signature(candidate: int, value: wp.array[float]): ...

        for dispatch in ("tiles", "worlds"):
            with self.subTest(dispatch=dispatch), self.assertRaisesRegex(ValueError, "world"):
                make_row_kernel("bad_dispatch", "", signature, dispatch=dispatch)

    def test_dispatch_operands_extend_the_native_abi(self):
        """Append private operands to the matrix-free dispatch exactly as to its native declaration."""
        native = mf_dispatch.make_native("", extra={"readout": SolveReadout})
        dispatch = mf_dispatch.make_dispatch(native, extra={"readout": SolveReadout})
        names = list(inspect.signature(dispatch).parameters)
        self.assertEqual(names[-2:], ["v_out", "readout"])
        plain = mf_dispatch.make_dispatch(mf_dispatch.make_native(""))
        self.assertEqual(list(inspect.signature(plain).parameters), names[:-1])


def _policy_snippet(test, kernel, marker):
    """The one native snippet of ``kernel`` that holds ``marker``."""
    snippets = {snippet for snippet in _native_snippets(kernel) if marker in snippet}
    test.assertEqual(len(snippets), 1, kernel.key)
    return snippets.pop()


class TestVelocityExitLaw(unittest.TestCase):
    """The velocity-scaled exit law and the trace, compiled by every factory that takes a policy."""

    def test_tolerance_is_validated_named_and_keyword_only(self):
        """Reject a negative or non-finite tolerance, name its variants and take fields by keyword only."""
        for value in (-1.0e-3, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "velocity_change_tolerance"):
                SolvePolicy(velocity_change_tolerance=value)
        with self.assertRaises(TypeError):
            SolvePolicy("sweeps")
        self.assertEqual(repr(VELOCITY), "SolvePolicy(velocity_change_tolerance=0.001)")
        self.assertEqual(VELOCITY.suffix, "_vtol0p001")
        self.assertEqual(SolvePolicy(readout="trace", velocity_change_tolerance=1.0e-7).suffix, "_vtol1em07_trace")
        self.assertEqual((TRACE.preludes, TRACE.operands), ((SWEEP_TRACE,), {"readout": SolveTrace}))
        self.assertEqual(SolvePolicy(velocity_change_tolerance=0), EXACT_POLICY)

    def test_vote_keeps_the_exact_text_and_scales_each_change(self):
        """Keep a vote verbatim under the exact law and bound every term's velocity-scaled change otherwise."""
        exact = "            if (delta != 0.0f) changed = 1;\n"
        terms = (("delta", "diag[i]", "row_type"), ("other", "diag[j]", "2"))
        self.assertIs(vote(EXACT_POLICY, exact, flag="changed", terms=terms), exact)
        self.assertIs(vote(SWEEPS, exact, flag="changed", terms=terms), exact)
        scaled = vote(VELOCITY, exact, flag="changed", terms=terms, slack=("tol", "tol"))
        self.assertEqual(
            scaled,
            "            if ((delta != 0.0f && !(fabsf(delta) * (diag[i]) <= 0.001f + (diag[i]) * (tol)))"
            " || (other != 0.0f && !(fabsf(other) * (diag[j]) <= 0.001f + (diag[j]) * (tol)))) changed = 1;\n",
        )
        traced = vote(TRACE, exact, flag="changed", terms=terms).splitlines()
        self.assertEqual(
            traced,
            [
                "            if (delta != 0.0f) solve_trace_change(fabsf(delta) * (diag[i]), row_type);",
                "            if (other != 0.0f) solve_trace_change(fabsf(other) * (diag[j]), 2);",
                exact.rstrip("\n"),
            ],
        )

    def test_every_vote_site_scales_by_its_diagonal(self):
        """Replace every change vote of every policy factory by its velocity-scaled form at a positive tolerance."""
        for name, factory, args, kwargs in _factory_calls():
            with self.subTest(factory=name):
                kernel = factory(*args, solve_policy=VELOCITY, **kwargs)
                self.assertTrue(kernel.key.endswith("_vtol0p001"), kernel.key)
                snippet = _policy_snippet(self, kernel, "0.001f")
                votes = [line for line in snippet.splitlines() if "changed = 1;" in line]
                self.assertTrue(votes)
                for line in votes:
                    self.assertIn("<= 0.001f", line)
                for expected in _SCALED_VOTES[name]:
                    self.assertIn(expected, snippet)

    def test_every_flag_combination_compiles_the_law_at_every_vote(self):
        """Compile the velocity law, traced or not, into every vote of every valid flag combination.

        A combination is valid when the factory builds it under the exact law; under a tolerance it
        must then build as well, with every change vote bounded by the tolerance and, traced, one
        exit record. Vote sites that only some flags emit (velocity-limit lanes, free-body rows, fused
        drive clamps, streamed impulses, local matrix-free rows) would otherwise keep the exact law
        unnoticed.

        Every policy, the plain readout included, must also compose with the factory's other options:
        its kernel is the policy-less kernel's name plus the policy's suffix, it appends one readout
        operand when it counts, and it keeps the policy-less sweep's velocity-limit law sites, the
        law's ``_vacc`` name, its row-metadata storage and its record of the law.
        """
        checked = dict.fromkeys(_SCALED_VOTES, 0)
        for name, factory, args, kwargs in _vote_census():
            try:
                plain = factory(*args, **kwargs)
            except (ValueError, NotImplementedError, RuntimeError, AssertionError):
                continue
            body = max(_native_snippets(plain), key=len)
            law_sites = len(_LAW_SITE.findall(body))
            accumulated = kwargs.get("velocity_limit_law") == "accumulated"
            # The law emits what its name says: accumulated sites under its _vacc name, none without it.
            if ("_vacc" in plain.key) != (law_sites > 0) or (law_sites and not accumulated):
                self.fail(f"{plain.key} {kwargs}: {law_sites} accumulated law sites")
            if kwargs.get("resident_row_metadata") and ("_smeta" not in plain.key or "s_meta_dense" not in body):
                self.fail(f"{plain.key} {kwargs}: resident row metadata is not resident")
            for policy in (SWEEPS, VELOCITY, TRACED_VELOCITY):
                kernel = factory(*args, solve_policy=policy, **kwargs)
                self.assertEqual(kernel.key, plain.key + policy.suffix, kwargs)
                self.assertEqual(getattr(kernel, "_fpgs_solve_readout", False), policy.counts and policy.readout)
                labels = [argument.label for argument in kernel.adj.args]
                self.assertEqual(labels, [argument.label for argument in plain.adj.args] + ["readout"] * policy.counts)
                if hasattr(plain, "_fpgs_velocity_limit_law"):
                    self.assertEqual(getattr(kernel, "_fpgs_velocity_limit_law", None), plain._fpgs_velocity_limit_law)
                snippet = max(_native_snippets(kernel), key=len)
                if len(_LAW_SITE.findall(snippet)) != law_sites or ("s_meta_dense" in snippet) != (
                    "s_meta_dense" in body
                ):
                    self.fail(f"{kernel.key} {kwargs}: the policy rewrote the law or the row metadata")
                if not policy.exact:
                    snippets = {snippet for snippet in _native_snippets(kernel) if "0.001f" in snippet}
                    if len(snippets) != 1:
                        self.fail(f"{kernel.key}: {len(snippets)} snippets hold the tolerance")
                    (snippet,) = snippets
                votes = [line.strip() for line in snippet.splitlines() if "changed = 1;" in line]
                unbounded = [line for line in votes if not policy.exact and "<= 0.001f" not in line]
                if not votes or unbounded:
                    self.fail(f"{kernel.key} {kwargs}: unbounded votes {unbounded[:2] or 'none found'}")
                if policy.traces and (
                    snippet.count("solve_trace_sweep(") != 1 or snippet.count("solve_trace_change(") < len(votes)
                ):
                    self.fail(f"{kernel.key} {kwargs}: the trace misses a vote or the exit")
                if policy.counts and not policy.traces and snippet.count("solve_readout_sweeps(") != 1:
                    self.fail(f"{kernel.key} {kwargs}: the readout records no exit or more than one")
            checked[name] += 1
        # The admitted combinations per factory: a factory gate that starts refusing some must not shrink the census.
        self.assertEqual(
            checked,
            {
                "sparse": 288,
                "sparse_lanes": 32,
                "mf_gs": 336,
                "paired": 6,
                "local": 48,
                "packets": 48,
                "sparse_diagonal": 5,
                "incremental": 32,
            },
        )

    def test_trace_records_every_vote_and_each_sweep(self):
        """Record each change before its vote and every sweep at its exit, under the exact law."""
        for name, factory, args, kwargs in _factory_calls():
            with self.subTest(factory=name):
                kernel = factory(*args, solve_policy=TRACE, **kwargs)
                self.assertEqual(kernel._fpgs_solve_readout, "trace")
                self.assertIs(kernel.adj.args[-1].type, SolveTrace)
                snippet = _policy_snippet(self, kernel, "solve_trace_sweep(")
                body = snippet.split(SWEEP_TRACE.source, 1)[1]
                votes = sum("changed = 1;" in line for line in body.splitlines())
                self.assertGreaterEqual(body.count("solve_trace_change("), votes)
                self.assertEqual(body.count("solve_trace_sweep("), 1)
                self.assertNotIn("0.001f", body)

    def test_trace_names_the_fused_drive_clamp_a_velocity_limit(self):
        """Trace the matrix-free sweep's fused drive velocity clamp under the velocity-limit kind, not the drive's."""
        kernel = solver._get_pgs_solve_mf_gs_kernel.__wrapped__(
            64, 32, 12, "120", has_drive_rows=True, fuse_vel_limits=True, solve_policy=TRACE
        )
        snippet = _policy_snippet(self, kernel, "solve_trace_sweep(")
        velocity_limit = int(kernels.PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)
        self.assertEqual(TRACE_KINDS[velocity_limit], "velocity_limit")
        records = [line.strip() for line in snippet.splitlines() if "solve_trace_change(fabsf(delta_fvl)" in line]
        self.assertEqual(
            records, [f"if (delta_fvl != 0.0f) solve_trace_change(fabsf(delta_fvl) * (denom_fvl), {velocity_limit});"]
        )

    def test_variants_without_the_law_refuse_it(self):
        """Refuse the velocity law and trace where the matrix-free sweep keeps uncovered votes."""
        for options in ({"friction_mode": "bisection"}, {"contact_torsion": True}):
            for policy in (VELOCITY, TRACE):
                with self.subTest(options=options, policy=policy), self.assertRaises(SolvePolicyUnsupported):
                    solver._get_pgs_solve_mf_gs_kernel.__wrapped__(64, 32, 12, "120", solve_policy=policy, **options)
            kernel = solver._get_pgs_solve_mf_gs_kernel.__wrapped__(64, 32, 12, "120", solve_policy=SWEEPS, **options)
            self.assertEqual(kernel._fpgs_solve_readout, "sweeps")

    def test_active_rows_wait_for_violated_rows_under_a_tolerance(self):
        """Stop an active-row sparse sweep under a tolerance only after a scan bounds every unvisited violation.

        The bound is the velocity change a visit would make, the violation times the relaxation factor, in
        the law and in the trace alike.
        """
        kernel = _get_pgs_solve_sparse_kernel.__wrapped__(48, 18, 12, active_rows=True, solve_policy=VELOCITY)
        snippet = _policy_snippet(self, kernel, "0.001f")
        self.assertIn("if (sweep_stop) {", snippet)
        self.assertIn("sweep_stop = sweep_violation <= 0.001f;", snippet)
        traced = _get_pgs_solve_sparse_kernel.__wrapped__(48, 18, 12, active_rows=True, solve_policy=TRACE)
        for text in (snippet, _policy_snippet(self, traced, "solve_trace_sweep(")):
            self.assertTrue("? omega * fmaxf(-residual, 0.0f) :" in text, "the unvisited-row bound ignores omega")
        full = _get_pgs_solve_sparse_kernel.__wrapped__(48, 18, 12, solve_policy=VELOCITY)
        self.assertNotIn("sweep_violation", _policy_snippet(self, full, "0.001f"))
        exact = _get_pgs_solve_sparse_kernel.__wrapped__(48, 18, 12, active_rows=True, solve_policy=SWEEPS)
        self.assertNotIn("sweep_violation", _policy_snippet(self, exact, "solve_readout_sweeps("))


class TestSolveReadoutRecord(unittest.TestCase):
    """Host summaries of readout counters."""

    def test_record_allocates_slots_and_summarizes_counts(self):
        """Allocate counters for readout kernels only and summarize their cumulative histogram."""
        record = SolveReadoutRecord(SWEEPS)
        counted = SimpleNamespace(key="counted_kernel", _fpgs_solve_readout=True)
        plain = SimpleNamespace(key="plain_kernel")
        record.register(counted, "sparse_pgs", world_count=3, budget=4, device="cpu")
        record.register(plain, "mf_gs", world_count=3, budget=4, device="cpu")
        record.register(None, "paired_factor", world_count=3, budget=4, device="cpu")
        self.assertEqual(record.kernels, {"counted_kernel": "sparse_pgs"})
        self.assertEqual(record.operands(plain), [])
        (operand,) = record.operands(counted)
        self.assertEqual(operand.sweeps.shape, (5, 3))
        # World 0: two solves of four sweeps, one of them without stopping; world 1: one solve of two; world 2: none.
        sweeps = np.zeros((5, 3), dtype=np.int32)
        sweeps[4, 0], sweeps[2, 1] = 2, 1
        operand.sweeps.assign(sweeps)
        operand.exhausted.assign(np.array([1, 0, 0], dtype=np.int32))
        (summary,) = record.summary()["slots"]
        self.assertEqual(summary["sweeps"], [0, 0, 1, 0, 2])
        self.assertEqual((summary["solves"], summary["exhausted"], summary["budget"], summary["worlds"]), (3, 1, 4, 2))
        self.assertAlmostEqual(summary["full_budget_fraction"], 2.0 / 3.0)
        self.assertAlmostEqual(summary["mean_sweeps"], 10.0 / 3.0)
        self.assertEqual(summary["world_mean_sweeps"]["p50"], 3.0)
        self.assertEqual(summary["pass"], "position")
        self.assertNotIn("trace", summary)
        json.dumps(record.summary(), allow_nan=False)

    def test_summary_separates_solves_past_their_first_sweep(self):
        """Summarize the solves that ran past their first sweep apart from the one-sweep solves.

        The sparse sweeps record a world without rows as a solve that stopped after one sweep, which
        would dilute the full-budget fraction of the worlds a tolerance could shorten.
        """
        record = SolveReadoutRecord(SWEEPS)
        counted = SimpleNamespace(key="counted_kernel", _fpgs_solve_readout="sweeps")
        record.register(counted, "sparse_pgs", world_count=3, budget=4, device="cpu")
        (operand,) = record.operands(counted)
        # World 0: two solves of four sweeps; world 1: one of two; world 2 has no rows: three one-sweep solves.
        sweeps = np.zeros((5, 3), dtype=np.int32)
        sweeps[4, 0], sweeps[2, 1], sweeps[1, 2] = 2, 1, 3
        operand.sweeps.assign(sweeps)
        (summary,) = record.summary()["slots"]
        self.assertEqual((summary["solves"], summary["sweeps"]), (6, [0, 3, 1, 0, 2]))
        self.assertAlmostEqual(summary["full_budget_fraction"], 2.0 / 6.0)
        past = summary["past_first_sweep"]
        self.assertEqual(past["solves"], 3)
        self.assertAlmostEqual(past["full_budget_fraction"], 2.0 / 3.0)
        self.assertAlmostEqual(past["mean_sweeps"], 10.0 / 3.0)
        # A one-sweep budget leaves nothing past the first sweep.
        short = SimpleNamespace(key="short_kernel", _fpgs_solve_readout="sweeps")
        record.register(short, "mf_gs", world_count=1, budget=1, device="cpu")
        self.assertEqual(
            record.summary()["slots"][1]["past_first_sweep"],
            {"solves": 0, "full_budget_fraction": None, "mean_sweeps": None},
        )
        json.dumps(record.summary(), allow_nan=False)

    def test_velocity_pass_counts_in_its_own_slots(self):
        """Count launches inside the velocity pass in slots of their own budget, apart from the position solve.

        A velocity pass of one or two sweeps exhausts nearly every solve; counted in the position
        solve's slot, those solves would read as full-budget position solves.
        """
        record = SolveReadoutRecord(SWEEPS)
        counted = SimpleNamespace(key="counted_kernel", _fpgs_solve_readout="sweeps")
        position_only = SimpleNamespace(key="position_only", _fpgs_solve_readout="sweeps")
        record.register(counted, "mf_gs", world_count=2, budget=8, device="cpu", velocity_budget=2)
        record.register(position_only, "sparse_pgs", world_count=2, budget=8, device="cpu")
        (position,) = record.operands(counted)
        with record.velocity_pass():
            (velocity,) = record.operands(counted)
            self.assertEqual(record.operands(position_only), [])
        self.assertIs(record.operands(counted)[0], position)
        self.assertEqual((position.sweeps.shape, velocity.sweeps.shape), ((9, 2), (3, 2)))
        self.assertEqual(record.kernels, {"counted_kernel": "mf_gs", "position_only": "sparse_pgs"})
        # Two velocity-pass solves of world 0, both of two sweeps without stopping.
        sweeps = np.zeros((3, 2), dtype=np.int32)
        sweeps[2, 0] = 2
        velocity.sweeps.assign(sweeps)
        velocity.exhausted.assign(np.array([2, 0], dtype=np.int32))
        slots = {(slot["kernel"], slot["pass"]): slot for slot in record.summary()["slots"]}
        self.assertEqual(
            sorted(slots),
            [("counted_kernel", "position"), ("counted_kernel", "velocity"), ("position_only", "position")],
        )
        self.assertEqual(slots[("counted_kernel", "position")]["solves"], 0)
        velocity_slot = slots[("counted_kernel", "velocity")]
        self.assertEqual((velocity_slot["budget"], velocity_slot["solves"], velocity_slot["exhausted"]), (2, 2, 2))

    def test_worlds_without_rows_count_as_skipped_per_pass(self):
        """Count, per pass, one skipped solve per launch of every world without dense, matrix-free or propagation rows.

        The matrix-free sweeps return before their first sweep for such a world, so no slot records
        it; a world's solves are its slots' solves plus its skipped ones. On the propagation contact
        routes the propagation sweeps solve a world whose rows are all propagation rows, so it is not
        skipped; a route without propagation rows passes no counts of them.
        """
        record = SolveReadoutRecord(SWEEPS)
        counted = SimpleNamespace(key="counted_kernel", _fpgs_solve_readout="sweeps")
        record.register(counted, "mf_gs_incremental", world_count=4, budget=4, device="cpu", velocity_budget=2)
        record.register_skipped(world_count=4, device="cpu", velocity_budget=2)
        # World 0 holds dense rows, world 2 matrix-free rows and world 3 propagation rows; world 1 has none.
        dense = wp.array([4, 0, 0, 0], dtype=wp.int32, device="cpu")
        free = wp.array([0, 0, 2, 0], dtype=wp.int32, device="cpu")
        propagation = wp.array([0, 0, 0, 3], dtype=wp.int32, device="cpu")
        for _ in range(2):
            record.count_skipped(dense, free, propagation_constraint_count=propagation)
        with record.velocity_pass():
            record.count_skipped(dense, free, propagation_constraint_count=propagation)
        # A second registration keeps the counters.
        record.register_skipped(world_count=4, device="cpu")
        summary = record.summary()
        self.assertEqual(
            summary["skipped"],
            [{"pass": "position", "solves": 2, "worlds": 1}, {"pass": "velocity", "solves": 1, "worlds": 1}],
        )
        self.assertEqual([slot["solves"] for slot in summary["slots"]], [0, 0])
        json.dumps(summary, allow_nan=False)
        # Without propagation rows world 3 has no rows either.
        record.count_skipped(dense, free)
        self.assertEqual(record.summary()["skipped"][0], {"pass": "position", "solves": 4, "worlds": 2})

    def test_skip_counters_need_a_readout_and_a_registering_route(self):
        """Count no skipped solve without a readout, or on a route that registered no skip counters."""
        empty = wp.zeros(2, dtype=wp.int32, device="cpu")
        for policy, register in ((EXACT_POLICY, True), (SWEEPS, False)):
            with self.subTest(policy=policy, register=register):
                record = SolveReadoutRecord(policy)
                if register:
                    record.register_skipped(world_count=2, device="cpu")
                record.count_skipped(empty, empty)
                self.assertEqual(record.summary()["skipped"], [])

    def test_trace_record_summarizes_stops_reasons_and_final_changes(self):
        """Summarize the trace's per-tolerance stopping sweeps, exit reasons and final-sweep changes."""
        record = SolveReadoutRecord(TRACE)
        traced = SimpleNamespace(key="traced_kernel", _fpgs_solve_readout="trace")
        record.register(traced, "mf_gs", world_count=2, budget=4, device="cpu")
        (operand,) = record.operands(traced)
        self.assertEqual(operand.stops.shape, (len(TRACE_GRID), 6))
        self.assertEqual(operand.limits.shape, (len(TRACE_KINDS), TRACE_BINS))
        stops = np.zeros((len(TRACE_GRID), 6), dtype=np.int32)
        stops[:, 5] = 2  # Below 1e-1 neither solve's changes ever fell to the tolerance...
        stops[-1, 5], stops[-1, 3] = 1, 1  # ...at 1e-1 one of them did after its third sweep.
        operand.stops.assign(stops)
        operand.reasons.assign(np.array([0, 0, 2], dtype=np.int32))
        limits = np.zeros((len(TRACE_KINDS), TRACE_BINS), dtype=np.int32)
        limits[2, 17] = 2  # Two friction changes in [10^-1, 10^-0.5).
        operand.limits.assign(limits)
        (summary,) = record.summary()["slots"]
        trace = summary["trace"]
        self.assertEqual(trace["stops"][-1], [0, 0, 0, 1, 0, 1])
        self.assertEqual(trace["reasons"], {"stationary": 0, "tolerance": 0, "budget": 2})
        self.assertEqual(list(trace["final_change"]), ["friction"])
        self.assertAlmostEqual(trace["final_change"]["friction"]["at_most"]["p50"], 10.0**-0.5)
        self.assertAlmostEqual(trace["final_change_all"]["p99"], 10.0**-0.5)
        json.dumps(record.summary(), allow_nan=False)

    def test_trace_totals_hold_counts_beyond_32_bits(self):
        """Sum the trace's stopping, reason and final-change counts over every world without wrapping at 2**31.

        16384 worlds solving a few times per step pass 2**31 solves within about 3e4 steps, a
        training-length run; the per-world counts stay 32-bit.
        """
        record = SolveReadoutRecord(TRACE)
        traced = SimpleNamespace(key="traced_kernel", _fpgs_solve_readout="trace")
        record.register(traced, "sparse_pgs", world_count=16384, budget=8, device="cpu")
        (operand,) = record.operands(traced)
        self.assertEqual(
            {name: getattr(operand, name).dtype for name in ("sweeps", "exhausted")},
            dict.fromkeys(("sweeps", "exhausted"), wp.int32),
        )
        for name in ("stops", "reasons", "limits"):
            self.assertEqual(getattr(operand, name).dtype, wp.int64, name)
        total = 3 * 2**31
        stops = np.zeros((len(TRACE_GRID), 10), dtype=np.int64)
        stops[:, 9] = total
        operand.stops.assign(stops)
        operand.reasons.assign(np.array([0, 0, total], dtype=np.int64))
        limits = np.zeros((len(TRACE_KINDS), TRACE_BINS), dtype=np.int64)
        limits[0, 20] = total
        operand.limits.assign(limits)
        trace = record.summary()["slots"][0]["trace"]
        self.assertEqual([row[-1] for row in trace["stops"]], [total] * len(TRACE_GRID))
        self.assertEqual(trace["reasons"]["budget"], total)
        self.assertEqual(trace["final_change"]["contact"]["count"], total)

    def test_trace_kinds_name_every_constraint_type(self):
        """Name every constraint type at its own index of the trace's kinds, followed by the kind of no change."""
        names = {
            kernels.PGS_CONSTRAINT_TYPE_CONTACT: "contact",
            kernels.PGS_CONSTRAINT_TYPE_JOINT_TARGET: "drive",
            kernels.PGS_CONSTRAINT_TYPE_FRICTION: "friction",
            kernels.PGS_CONSTRAINT_TYPE_JOINT_LIMIT: "joint_limit",
            kernels.PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT: "velocity_limit",
            kernels.PGS_CONSTRAINT_TYPE_MIMIC: "mimic",
            kernels.PGS_CONSTRAINT_TYPE_CONNECT: "connect",
            kernels.PGS_CONSTRAINT_TYPE_TORSION: "torsion",
        }
        self.assertEqual(len(names), kernels.PGS_CONSTRAINT_TYPE_COUNT)
        self.assertEqual(TRACE_KINDS, (*(names[kind] for kind in range(kernels.PGS_CONSTRAINT_TYPE_COUNT)), "none"))


class TestSolverReadoutOption(unittest.TestCase):
    """The solver option and its records on a CPU model, where no native sweep is built."""

    @staticmethod
    def _model():
        builder = newton.ModelBuilder()
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_articulation([builder.add_joint_revolute(-1, body)])
        return builder.finalize(device="cpu")

    def test_invalid_readout_raises(self):
        """Reject an unknown readout or an invalid velocity tolerance at construction."""
        with self.assertRaisesRegex(ValueError, "pgs_solve_readout"):
            SolverFeatherPGS(self._model(), pgs_solve_readout="every_sweep")
        for value in (-1.0, float("nan")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "pgs_velocity_change_tolerance"):
                SolverFeatherPGS(self._model(), pgs_velocity_change_tolerance=value)

    def test_unhonored_tolerance_warns_and_records_the_exact_law(self):
        """Warn when no sweep compiles the velocity law, and record the exact law it keeps."""
        with self.assertWarnsRegex(UserWarning, "pgs_velocity_change_tolerance"):
            selection = SolverFeatherPGS(self._model(), pgs_velocity_change_tolerance=1.0e-3).get_path_selection()
        json.dumps(selection, allow_nan=False)
        option = selection["options"]["pgs_velocity_change_tolerance"]
        self.assertEqual((option["requested"], option["applied"], option["source"]), (1.0e-3, 0.0, "argument"))
        law = selection["routes"]["sweep"]["exit_law"]
        self.assertEqual((law["law"], law["velocity_change_tolerance"], law["honored"]), ("exact", 1.0e-3, {}))
        self.assertIn("No sweep", law["reason"])

    def test_records_name_the_readout_and_its_kernels(self):
        """Record the requested readout, an off applied value without counted sweeps, and no measurement."""
        model = self._model()
        for requested in ("off", "sweeps"):
            with self.subTest(requested=requested):
                selection = SolverFeatherPGS(model, pgs_solve_readout=requested).get_path_selection()
                json.dumps(selection, allow_nan=False)
                option = selection["options"]["pgs_solve_readout"]
                self.assertEqual((option["requested"], option["applied"], option["default"]), (requested, "off", "off"))
                readout = selection["routes"]["sweep"]["readout"]
                self.assertEqual((readout["mode"], readout["counted"], readout["uncounted"]), ("off", {}, []))
                self.assertEqual(readout["skipped_passes"], [])
                self.assertIn("off" if requested == "off" else "No sweep", readout["reason"])
                self.assertNotIn("solve_readout", selection["measurements"])
                law = selection["routes"]["sweep"]["exit_law"]
                self.assertEqual((law["law"], law["honored"], law["declined"], law["ignored"]), ("exact", {}, {}, []))
                self.assertEqual(readout["declined"], {})


def _sweep_record(**fields):
    """A matrix-free ``routes.sweep`` record naming only its matrix-free Gauss-Seidel kernel, with ``fields`` set."""
    record = dict.fromkeys(
        (
            "sparse",
            "sparse_diagonal",
            "paired_factor",
            "free_body_matrix_free",
            "tiers",
            "local",
            "dense",
            "propagation",
        )
    )
    record.update(route="matrix_free_gs", driver="direct", mf_gs="mf_gs_kernel")
    record.update(fields)
    return record


def _policy_solver(*, readout="sweeps", tolerance=0.0, declined=(), honored=(), skipped=("position",), **attributes):
    """The solver attributes the sweep records read, counting the matrix-free kernel and its skipped solves."""
    return SimpleNamespace(
        pgs_solve_readout=readout,
        pgs_velocity_change_tolerance=tolerance,
        _solve_readout=SimpleNamespace(kernels={"mf_gs_kernel": "mf_gs"}, skipped_passes=tuple(skipped)),
        _solve_policy_declined=dict.fromkeys(declined, "mf_gs: the variant keeps the exact law"),
        _solve_policy_honored=dict.fromkeys(honored, "mf_gs"),
        **attributes,
    )


class TestSweepPolicyRecords(unittest.TestCase):
    """The readout and exit-law records of a route's sweeps and the warnings they raise."""

    def test_declined_trace_is_recorded_and_warned_apart_from_the_law(self):
        """Record and warn a declined trace under the readout, and a declined tolerance under the exit law."""
        for readout, tolerance in (("trace", 0.0), ("sweeps", 1.0e-3), ("trace", 1.0e-3)):
            with self.subTest(readout=readout, tolerance=tolerance):
                solver = _policy_solver(readout=readout, tolerance=tolerance, declined=("mf_gs_kernel",))
                record = _sweep_record()
                sweep = {
                    "readout": path_selection._readout(solver, record),
                    "exit_law": path_selection._exit_law(solver, record),
                }
                self.assertEqual(sweep["readout"]["mode"], readout)
                self.assertEqual(list(sweep["readout"]["declined"]), ["mf_gs_kernel"] if readout == "trace" else [])
                self.assertEqual(list(sweep["exit_law"]["declined"]), ["mf_gs_kernel"] if tolerance else [])
                messages = path_selection.policy_warnings(sweep)
                self.assertEqual(
                    [message.split(" ", 1)[0] for message in messages],
                    ["pgs_velocity_change_tolerance"] * bool(tolerance)
                    + ["pgs_solve_readout='trace'"] * (readout == "trace"),
                )
                json.dumps(sweep, allow_nan=False)

    def test_phase_launches_count_once_each(self):
        """Say that a contact-then-internal solve counts once per row-phase launch.

        A driver that launches one row phase at a time counts per launch too, and its sweeps
        return before their first sweep for a world without rows of the launch's phase (the
        matrix-free sweep at the velocity-limit phase, the incremental tier at the drive-and-limit,
        contact and velocity-limit phases), so the record says that such a launch counts the world
        nowhere.
        """
        record = _sweep_record(driver="contact_then_internal")
        self.assertIn("once per launch", path_selection._readout(_policy_solver(), record)["reason"])
        self.assertIn("solves its worlds", path_selection._readout(_policy_solver(), _sweep_record())["reason"])
        for driver in ("physx_grasp", "propagation_phases"):
            with self.subTest(driver=driver):
                reason = path_selection._readout(_policy_solver(), _sweep_record(driver=driver))["reason"]
                self.assertIn("counts are per launch", reason)
                self.assertIn("in no slot and not as skipped", reason)

    def test_records_name_the_worlds_no_counter_counts(self):
        """Name each owner that solves whole worlds without counters, and none where counted sweeps own every world.

        The matrix-free sweep returns before its first sweep for the worlds the parallel tiers own,
        and the propagation sweeps solve the worlds whose rows are all propagation rows. Those
        worlds hold rows, so the skip counter leaves them out too: without a record their solves
        would vanish from the summary. The incremental tiers count their own worlds.
        """
        parallel = {"family": "parallel", "tiers": [{"rows": 32, "kernel": "tier32"}, {"rows": 48, "kernel": "tier48"}]}
        incremental = {"family": "incremental", "tiers": [{"kernel": "incremental_kernel"}]}
        propagation = {"mode": "propagation", "row_solve": "propagation_kernel", "cached_response": False}
        for fields, owners in (
            ({}, []),
            ({"tiers": incremental}, []),
            ({"tiers": parallel}, ["parallel_tiers"]),
            ({"tiers": {**parallel, "family": "keyed_parallel"}}, ["parallel_tiers"]),
            ({"propagation": propagation, "driver": "propagation_phases"}, ["propagation"]),
        ):
            with self.subTest(fields=sorted(fields)):
                readout = path_selection._readout(_policy_solver(), _sweep_record(**fields))
                self.assertEqual(sorted(readout["uncounted_worlds"]), owners)
                if owners:
                    self.assertIn("uncounted_worlds", readout["reason"])
                json.dumps(readout, allow_nan=False)
        # The tiers' worlds are bounded by the largest tier's rows.
        readout = path_selection._readout(_policy_solver(), _sweep_record(tiers=parallel))
        self.assertIn("1 to 48 dense rows", readout["uncounted_worlds"]["parallel_tiers"])
        # Without counted sweeps nothing is counted, so no owner is named.
        solver = _policy_solver(readout="off")
        solver._solve_readout = SimpleNamespace(kernels={}, skipped_passes=())
        self.assertEqual(path_selection._readout(solver, _sweep_record(tiers=parallel))["uncounted_worlds"], {})

    def test_fused_propagation_records_why_nothing_counts(self):
        """Say that the fused propagation kernel runs every solve where no sweep counts or honors a tolerance.

        Under ``articulated_contact_response="propagation-fused"`` none of the counted sweeps
        launches, so the solver builds them under the exact policy, without counters: the records
        then name the fused kernel as the reason, not fixed-iteration or Jacobi-tier sweeps.
        """
        fused = {
            "mode": "propagation-fused",
            "row_solve": "propagation_kernel",
            "full_iteration": "fused_kernel",
            "cached_response": False,
        }
        solver = _policy_solver(tolerance=1.0e-3)
        solver._solve_readout = SimpleNamespace(kernels={}, skipped_passes=())
        record = _sweep_record(driver="propagation_fused", propagation=fused)
        sweep = {
            "readout": path_selection._readout(solver, record),
            "exit_law": path_selection._exit_law(solver, record),
        }
        readout, law = sweep["readout"], sweep["exit_law"]
        self.assertEqual((readout["mode"], readout["counted"], readout["uncounted_worlds"]), ("off", {}, {}))
        self.assertIn("fused propagation kernel", readout["reason"])
        self.assertEqual((law["law"], law["honored"], law["driver_honors"]), ("exact", {}, False))
        self.assertIn("fused_kernel", law["ignored"])
        self.assertIn("fused propagation kernel", law["reason"])
        (message,) = path_selection.policy_warnings(sweep)
        self.assertIn("fused propagation kernel", message)
        json.dumps(sweep, allow_nan=False)
        # The propagation route that launches its row phases one by one keeps the general reasons.
        record = _sweep_record(driver="propagation_phases", propagation={**fused, "mode": "propagation"})
        self.assertIn("No sweep", path_selection._readout(solver, record)["reason"])
        self.assertIn("No sweep", path_selection._exit_law(solver, record)["reason"])

    def test_records_name_the_passes_that_count_skipped_solves(self):
        """Name the passes whose solves of worlds without rows count as skipped, and none without counted sweeps."""
        record = _sweep_record()
        for skipped in ((), ("position",), ("position", "velocity")):
            with self.subTest(skipped=skipped):
                readout = path_selection._readout(_policy_solver(skipped=skipped), record)
                self.assertEqual(readout["skipped_passes"], list(skipped))
        solver = _policy_solver()
        solver._solve_readout = SimpleNamespace(kernels={}, skipped_passes=("position",))
        self.assertEqual(path_selection._readout(solver, record)["skipped_passes"], [])


if __name__ == "__main__":
    unittest.main()
