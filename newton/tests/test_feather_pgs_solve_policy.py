# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compile the sweeps' one exit law into every vote, and its readout as one frozen solve policy.

These host tests build kernels and records without a GPU. Every vote of every
factory reads the law's rounding floor, under every policy; the default policy
builds exactly the kernels of a policy-less call, and the kernel pins
check the whole inventory.
"""

import dataclasses
import inspect
import itertools
import json
import re
import unittest
import warnings
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import (
    kernel_template,
    path_selection,
    row_kernels,
    sweep_routes,
    sweep_routes_diag,
    sweep_routes_local,
)
from newton._src.solvers.feather_pgs.device_preludes import ARCHITECTURE_GUARD
from newton._src.solvers.feather_pgs.kernel_template import KernelTemplate
from newton._src.solvers.feather_pgs.solve_policy import (
    DEFAULT_POLICY,
    QUIET_LITERAL,
    SWEEP_READOUT,
    SWEEP_TRACE,
    TRACE_BINS,
    TRACE_GRID,
    TRACE_KINDS,
    Change,
    SolvePolicy,
    SolveReadout,
    SolveReadoutRecord,
    SolveTrace,
    bound,
    sweep_exit,
    vote,
)
from newton._src.solvers.feather_pgs.sweep_routes_sparse import sparse_kernel
from newton.solvers import SolverFeatherPGS
from newton.tests.test_feather_pgs_kernel_pins import _kernel_hash
from newton.tests.test_feather_pgs_native_prelude import _native_snippets

SWEEPS = SolvePolicy(readout="sweeps")
TRACE = SolvePolicy(readout="trace")

# A vote's loud test: a change against its bound.
_LOUD = re.compile(r"\(!\(fabsf\(.+\) <= .+\)\)")

# Velocity-scaled changes each factory call's law must read: every vote site with its own A_ii.
_SCALED_VOTES = {
    "sparse": ("fabsf(delta_impulse) * (denom)", "fabsf(sib_delta) * (s_diag[sib])", "!sweep_violation_any"),
    "sparse_lanes": ("fabsf(delta_impulse) * (denom)", "fabsf(sib_delta) * (s_diag[sib])"),
    "mf_gs": (
        "fabsf(delta_impulse) * (denom)",
        "fabsf(sib_delta) * (s_diag_dense[sib])",
        "fabsf(delta_impulse) * (mf_diag > 0.0f ? 1.0f / mf_diag : 0.0f)",
        "fabsf(delta_impulse_vlim) * (denom)",
        "fabsf(speed_row[1]) * (1.0f / mf_diag)",
    ),
    "paired": (
        "fabsf(delta_impulse) * (s_diag[i])",
        "fabsf(delta_impulse) * ((r == 0 ? unit_diag0 : unit_diag1))",
        "fabsf(sib_delta) * (unit_diag2)",
        "fabsf(sib_delta) * (s_diag[sib])",
    ),
    "sparse_diagonal": (
        "fabsf(delta_impulse) * (denom)",
        "fabsf(sib_delta) * (s_diag_dense[sib])",
        "fabsf(delta_lambda) * (limit_denom)",
    ),
}


def _factory_calls():
    """One call of every sweep factory that takes a policy: ``(name, factory, args, kwargs)``."""
    return (
        (
            "sparse",
            sparse_kernel.__wrapped__,
            (48, 18, 12),
            {"active_rows": True, "use_support_table": True, "contact_triplets": True},
        ),
        ("sparse_lanes", sparse_kernel.__wrapped__, (16, 6, 6), {"lanes_per_world": 8}),
        (
            "mf_gs",
            sweep_routes.mf_gs_kernel.__wrapped__,
            (64, 32, 12, "120"),
            {},
        ),
        (
            "paired",
            sweep_routes_local.paired_factor_kernel.__wrapped__,
            (192, 29, 23, 6, "120", True),
            {"contact_units": True},
        ),
        (
            "sparse_diagonal",
            sweep_routes_diag.sparse_diagonal_kernel.__wrapped__,
            (64, 10, 9, "120"),
            {"speculative_contact_batches": True, "contact_units": True},
        ),
    )


# A velocity-limit visit: its law call (projection_laws.velocity_limit_update or speed_row).
_LAW_SITE = re.compile(r"ProjectionLaw::(?:velocity_limit_update|speed_row)\([^;]*\);")


def _axis(name, values=(False, True)):
    """One census axis: a keyword over its values."""
    return tuple({name: value} for value in values)


def _vote_census():
    """Every combination of the flags that add or remove vote sites, per policy factory: ``(name, factory, args, kwargs)``.

    The axes follow the exit-voting factories' census in ``tools/fpgs_bench/kernel_identity.py`` at one
    shape each, crossed with the factories' other options that rewrite their sweeps: the matrix-free
    sweep's resident row metadata. The matrix-free sweep keeps the current friction law without torsion; the
    census pins the others' text unchanged.
    """
    specs = (
        (
            "sparse",
            sparse_kernel.__wrapped__,
            (48, 18, 12),
            (
                _axis("active_rows"),
                _axis("use_support_table"),
                _axis("contact_triplets"),
                _axis("velocity_limits"),
                _axis("local_factors"),
                _axis("warm_start"),
                _axis("linked_rows"),
            ),
        ),
        (
            "sparse_lanes",
            sparse_kernel.__wrapped__,
            (32, 8, 8),
            (
                _axis("lanes_per_world", (16, 8)),
                _axis("velocity_limits"),
                _axis("local_factors"),
                _axis("warm_start"),
            ),
        ),
        (
            "mf_gs",
            sweep_routes.mf_gs_kernel.__wrapped__,
            (64, 32, 12, "120"),
            (
                _axis("places", sweep_routes.ROW_PLACES),
                _axis("has_dense_velocity_limit_rows"),
                _axis("factor_coordinates"),
                _axis("skip_tier_rows", (0, 32)),
                _axis("independent_components"),
            ),
        ),
        (
            "paired",
            sweep_routes_local.paired_factor_kernel.__wrapped__,
            (192, 29, 23, 6, "120"),
            (_axis("independent_components"), _axis("contact_units")),
        ),
        (
            "sparse_diagonal",
            sweep_routes_diag.sparse_diagonal_kernel.__wrapped__,
            (64, 10, 9, "120"),
            (_axis("speculative_contact_batches"), _axis("contact_units")),
        ),
    )
    for name, factory, args, axes in specs:
        for options in itertools.product(*axes):
            kwargs = {}
            for option in options:
                kwargs.update(option)
            yield name, factory, args, kwargs


def _with_readout(name: str, labels: list[str], count: int) -> list[str]:
    """Factory ``name``'s labels with ``count`` readout operands appended: last, but before the paired sweep's walk,
    which it takes after them."""
    if name == "paired":
        return labels[:-1] + ["readout"] * count + labels[-1:]
    return labels + ["readout"] * count


def _readout_arg(kernel):
    """The kernel's readout operand."""
    return next(argument for argument in kernel.adj.args if argument.label == "readout")


class TestSolvePolicy(unittest.TestCase):
    """The frozen policy, its emitters and the factories that take it."""

    def test_policy_is_frozen_hashable_and_validated(self):
        """Freeze the policy, compare it by value and reject an unknown readout."""
        self.assertEqual(SolvePolicy(), DEFAULT_POLICY)
        self.assertEqual(SolvePolicy(readout="sweeps"), SWEEPS)
        self.assertEqual(hash(SolvePolicy(readout="sweeps")), hash(SWEEPS))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            SWEEPS.readout = "off"
        with self.assertRaisesRegex(ValueError, "readout"):
            SolvePolicy(readout="every_sweep")
        self.assertEqual(
            (DEFAULT_POLICY.counts, DEFAULT_POLICY.suffix, DEFAULT_POLICY.preludes, DEFAULT_POLICY.operands),
            (False, "", (), {}),
        )
        self.assertEqual(
            (SWEEPS.suffix, SWEEPS.preludes, SWEEPS.operands),
            ("_sweeps", (SWEEP_READOUT,), {"readout": SolveReadout}),
        )

    def test_equal_policies_share_cached_kernels(self):
        """Return one cached kernel for equal policies, and another for a different policy."""
        first = sparse_kernel(16, 6, 4, solve_policy=SolvePolicy(readout="sweeps"))
        second = sparse_kernel(16, 6, 4, solve_policy=SolvePolicy(readout="sweeps"))
        self.assertIs(first, second)
        self.assertIsNot(first, sparse_kernel(16, 6, 4))

    def test_policy_and_readout_options_are_keyword_only(self):
        """Take the policy and the solver's diagnostics (the readout's option) by keyword only; the exit law takes no
        option."""
        parameters = inspect.signature(SolverFeatherPGS).parameters
        self.assertEqual(
            (parameters["diagnostics"].kind, parameters["diagnostics"].default),
            (inspect.Parameter.KEYWORD_ONLY, None),
        )
        for option in ("pgs_change_tolerance", "pgs_relative_change_tolerance", "pgs_velocity_change_tolerance"):
            self.assertNotIn(option, parameters)
        factories = [(name, factory) for name, factory, _, _ in _factory_calls()]
        factories += [
            ("mf_gs_source", sweep_routes.mf_gs_source),
            ("paired_source", sweep_routes_local.paired_factor_source),
        ]
        for name, factory in factories:
            with self.subTest(factory=name):
                parameter = inspect.signature(factory).parameters["solve_policy"]
                self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
                self.assertIs(parameter.default, DEFAULT_POLICY)

    def test_default_policy_builds_the_kernels_of_a_policy_less_call(self):
        """Build the same Warp content hash, hence the same binary, with the default policy passed or omitted."""
        for name, factory, args, kwargs in _factory_calls():
            with self.subTest(factory=name):
                kernel = factory(*args, **kwargs)
                self.assertEqual(
                    _kernel_hash(factory(*args, solve_policy=DEFAULT_POLICY, **kwargs)), _kernel_hash(kernel)
                )
                self.assertNotIn("readout", [argument.label for argument in kernel.adj.args])
                self.assertNotIn("solve_readout_sweeps", "".join(_native_snippets(kernel)))

    def test_readout_appends_one_operand_prelude_and_exit_record(self):
        """Append the readout operand last (the paired sweep's walk follows it), declare its prelude after the law
        and record every solve once."""
        for name, factory, args, kwargs in _factory_calls():
            with self.subTest(factory=name):
                exact = factory(*args, **kwargs)
                kernel = factory(*args, solve_policy=SWEEPS, **kwargs)
                self.assertEqual(kernel.key, exact.key + "_sweeps")
                labels = [argument.label for argument in kernel.adj.args]
                self.assertEqual(labels, _with_readout(name, [argument.label for argument in exact.adj.args], 1))
                self.assertIs(_readout_arg(kernel).type, SolveReadout)
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

    def test_sweep_exit_stops_a_quiet_sweep_and_records_it(self):
        """Stop once the sweep was quiet; a readout records the solve at its one exit."""
        plain = sweep_exit(DEFAULT_POLICY, quiet="done")
        self.assertEqual(
            plain,
            "        {\n            const bool sweep_stop = done;\n            if (sweep_stop) break;\n        }\n",
        )
        counted = sweep_exit(SWEEPS, quiet="done", world="w", iteration="k", iterations="n", lane="l")
        self.assertIn("const bool sweep_stop = done;", counted)
        self.assertIn("if (l == 0 && (sweep_stop || k + 1 == n))", counted)
        self.assertIn("solve_readout_sweeps(w, k + 1, sweep_stop);", counted)
        bounded = sweep_exit(DEFAULT_POLICY, quiet="done", violation=("scan();", "loud", "value", "kind"))
        self.assertIn("bool sweep_stop = done;", bounded)
        self.assertLess(bounded.index("scan();"), bounded.index("sweep_stop = !loud;"))

    def test_world_dispatch_requires_the_world_first(self):
        """Reject the world dispatches of generated kernels unless the native's first operand is the world."""

        def signature(candidate: int, value: wp.array[float]): ...

        for dispatch in (
            kernel_template.items("world", groups=1, lanes=32, guard="world_count"),
            sweep_routes.worlds(False),
            sweep_routes.worlds(True),
        ):
            with self.subTest(dispatch=dispatch.body), self.assertRaisesRegex(ValueError, "world"):
                KernelTemplate.from_abi("bad_dispatch", signature, "", dispatch)

    def test_dispatch_operands_extend_the_native_abi(self):
        """Append the readout operand to the matrix-free sweep's kernel signature exactly as to its native ABI."""

        def template(policy):
            return KernelTemplate.from_abi(
                "mf_gs", sweep_routes.MF_GS_ABI, "", sweep_routes.worlds(True), extra=policy.arguments, composed=True
            )

        counted, plain = template(SWEEPS), template(DEFAULT_POLICY)
        for names in ([name for name, _ in counted.signature], [name for name, _ in counted.native_params]):
            self.assertEqual(names[-2:], ["v_out", "readout"])
        self.assertIs(counted.signature[-1][1], SolveReadout)
        self.assertEqual(plain.signature, counted.signature[:-1])
        self.assertEqual(plain.native_params, counted.native_params[:-1])
        # The world queue's operands lead the kernel's signature; the native function takes the world instead.
        queue = sweep_routes.worlds(True).extra
        self.assertEqual(plain.signature[: len(queue)], queue)
        self.assertEqual(plain.native_params[0], ("world", int))


def _policy_snippet(test, kernel, marker):
    """The one native snippet of ``kernel`` that holds ``marker``."""
    snippets = {snippet for snippet in _native_snippets(kernel) if marker in snippet}
    test.assertEqual(len(snippets), 1, kernel.key)
    return snippets.pop()


class TestExitLaw(unittest.TestCase):
    """The exit law at the rounding floor and the trace, compiled by every factory that takes a policy."""

    def test_policy_takes_only_a_readout_by_keyword(self):
        """Name a policy by its readout alone, by keyword: the law has no option."""
        with self.assertRaises(TypeError):
            SolvePolicy("sweeps")
        with self.assertRaises(TypeError):
            SolvePolicy(velocity_change_tolerance=1.0e-3)
        self.assertEqual(repr(TRACE), "SolvePolicy(readout='trace')")
        self.assertEqual((TRACE.suffix, SWEEPS.suffix, DEFAULT_POLICY.suffix), ("_trace", "_sweeps", ""))
        self.assertEqual((TRACE.preludes, TRACE.operands), ((SWEEP_TRACE,), {"readout": SolveTrace}))

    def test_vote_bounds_each_change_by_its_floor(self):
        """Vote a change loud beyond its row's bound in its storage's form: phi against 2^-20 (|b| + S + A |lambda|)
        from a stored diagonal, |dlambda| against 2^-20 ((|b| + S) / A + |lambda|) from a stored inverse; a NaN
        change is loud; trace each change's velocity."""
        self.assertEqual(QUIET_LITERAL, "9.5367431640625e-07f")
        self.assertEqual(bound("b", "S", "l", step="s"), "9.5367431640625e-07f * ((fabsf(b) + (S)) * (s) + fabsf(l))")
        self.assertEqual(bound("b", "S", "l", diagonal="A"), "9.5367431640625e-07f * (fabsf(b) + (S) + (A) * fabsf(l))")
        self.assertEqual(bound("b", "S", "0.0f", diagonal="A"), "9.5367431640625e-07f * (fabsf(b) + (S))")
        for forms in ({}, {"step": "s", "diagonal": "A"}):
            with self.assertRaises(ValueError):
                bound("b", "S", "l", **forms)
        scaled = Change("delta", "tau", "row_type", "diag[i]", scaled=True)
        self.assertEqual(scaled.loud, "!(fabsf(delta) * (diag[i]) <= tau)")
        changes = (Change("delta", "tau", "row_type", "diag[i]"), Change("other", "rho", "2", "diag[j]"))
        expected = "    if ((!(fabsf(delta) <= tau)) || (!(fabsf(other) <= rho))) changed = 1;"
        self.assertEqual(vote(DEFAULT_POLICY, changes, flag="changed", pad="    "), expected)
        self.assertEqual(vote(SWEEPS, changes, flag="changed", pad="    "), expected)
        traced = vote(TRACE, changes, flag="changed", pad="    ").splitlines()
        self.assertEqual(
            traced,
            [
                "    if (delta != 0.0f) solve_trace_change(fabsf(delta) * (diag[i]), row_type);",
                "    if (other != 0.0f) solve_trace_change(fabsf(other) * (diag[j]), 2);",
                expected,
            ],
        )

    def test_every_vote_site_reads_its_floor(self):
        """Bound every change vote of every policy factory by its row's bound; the trace scales every change by its
        own diagonal."""
        for name, factory, args, kwargs in _factory_calls():
            with self.subTest(factory=name):
                kernel = factory(*args, **kwargs)
                snippet = _policy_snippet(self, kernel, QUIET_LITERAL)
                votes = [line for line in snippet.splitlines() if "changed = 1;" in line]
                self.assertTrue(votes)
                for line in votes:
                    self.assertRegex(line, _LOUD)
                traced = _policy_snippet(self, factory(*args, solve_policy=TRACE, **kwargs), "solve_trace_sweep(")
                for expected in _SCALED_VOTES[name]:
                    self.assertIn(expected, traced)

    def test_every_flag_combination_compiles_the_law_at_every_vote(self):
        """Compile the exit law, traced or not, into every vote of every valid flag combination.

        A combination is valid when the factory builds it without a readout; under a readout it must
        then build as well, with every change vote bounded by its floor and, traced, one exit record.
        Vote sites that only some flags emit (velocity-limit lanes, free-body rows, fused drive clamps,
        streamed impulses) would otherwise escape the law unnoticed.

        Every policy, the plain readout included, must also compose with the factory's other options:
        its kernel is the policy-less kernel's name plus the policy's suffix, it appends one readout
        operand when it counts, and it keeps the policy-less sweep's velocity-limit law sites and its
        row-metadata storage.
        """
        checked = dict.fromkeys(_SCALED_VOTES, 0)
        for name, factory, args, kwargs in _vote_census():
            try:
                plain = factory(*args, **kwargs)
            except (ValueError, NotImplementedError, RuntimeError, AssertionError):
                continue
            body = max(_native_snippets(plain), key=len)
            law_sites = len(_LAW_SITE.findall(body))
            places = kwargs.get("places")
            if (
                places is not None
                and places.staging == "resident"
                and ("_smeta" not in plain.key or "s_meta_dense" not in body)
            ):
                self.fail(f"{plain.key} {kwargs}: resident row metadata is not resident")
            for policy in (DEFAULT_POLICY, SWEEPS, TRACE):
                kernel = factory(*args, solve_policy=policy, **kwargs)
                self.assertEqual(kernel.key, plain.key + policy.suffix, kwargs)
                labels = [argument.label for argument in kernel.adj.args]
                self.assertEqual(
                    labels, _with_readout(name, [argument.label for argument in plain.adj.args], policy.counts)
                )
                snippet = max(_native_snippets(kernel), key=len)
                if len(_LAW_SITE.findall(snippet)) != law_sites or ("s_meta_dense" in snippet) != (
                    "s_meta_dense" in body
                ):
                    self.fail(f"{kernel.key} {kwargs}: the policy rewrote the law or the row metadata")
                snippets = {snippet for snippet in _native_snippets(kernel) if QUIET_LITERAL in snippet}
                if len(snippets) != 1:
                    self.fail(f"{kernel.key}: {len(snippets)} snippets hold the floor")
                (snippet,) = snippets
                votes = [line.strip() for line in snippet.splitlines() if "changed = 1;" in line]
                unbounded = [line for line in votes if not _LOUD.search(line)]
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
                "sparse": 40,
                "sparse_lanes": 16,
                "mf_gs": 48,
                "paired": 4,
                "sparse_diagonal": 3,
            },
        )

    def test_trace_records_every_vote_and_each_sweep(self):
        """Record each change before its vote and every sweep at its exit, under the exact law."""
        for name, factory, args, kwargs in _factory_calls():
            with self.subTest(factory=name):
                kernel = factory(*args, solve_policy=TRACE, **kwargs)
                self.assertIs(_readout_arg(kernel).type, SolveTrace)
                snippet = _policy_snippet(self, kernel, "solve_trace_sweep(")
                body = snippet.split(SWEEP_TRACE.source, 1)[1]
                votes = sum("changed = 1;" in line for line in body.splitlines())
                self.assertGreaterEqual(body.count("solve_trace_change("), votes)
                self.assertEqual(body.count("solve_trace_sweep("), 1)

    def test_torsion_votes_under_every_policy(self):
        """Vote a spin row's change (contact torsion) under the exit law and the trace, scaled by its row's diagonal,
        with its friction pair's."""
        torsion = int(row_kernels.PGS_CONSTRAINT_TYPE_TORSION)
        options = {"contact_torsion": True}
        traced = sweep_routes.mf_gs_kernel.__wrapped__(64, 32, 12, "120", solve_policy=TRACE, **options)
        snippet = _policy_snippet(self, traced, "solve_trace_sweep(")
        self.assertIn(
            f"if (spin_delta != 0.0f) solve_trace_change(fabsf(spin_delta) * (s_diag_dense[spin]), {torsion});",
            snippet,
        )
        plain = sweep_routes.mf_gs_kernel.__wrapped__(64, 32, 12, "120", **options)
        snippet = _policy_snippet(self, plain, "fabsf(spin_delta) * (s_diag_dense[spin]) <=")
        self.assertIn("(!(fabsf(spin_delta) * (s_diag_dense[spin]) <= spin_bound))", snippet)
        self.assertIn("ProjectionLaw::friction_ball(", snippet)
        self.assertIn("ConeLaw::ball(", snippet)
        kernel = sweep_routes.mf_gs_kernel.__wrapped__(64, 32, 12, "120", solve_policy=SWEEPS, **options)
        self.assertIs(_readout_arg(kernel).type, SolveReadout)

    def test_active_rows_stop_only_once_every_unvisited_row_is_quiet(self):
        """Stop an active-row sparse sweep only after a scan finds no unvisited zero-impulse row a visit would move
        beyond its floor: the violation times the relaxation factor, in the law and in the trace alike."""
        kernel = sparse_kernel.__wrapped__(48, 18, 12, active_rows=True)
        snippet = _policy_snippet(self, kernel, QUIET_LITERAL)
        self.assertIn("if (sweep_stop) {", snippet)
        self.assertIn("sweep_stop = !sweep_violation_any;", snippet)
        self.assertIn(f"(!(fabsf(excess) <= {bound('s_rhs[row]', 'fabsf(dot)', '0.0f', diagonal='1.0f')}))", snippet)
        traced = sparse_kernel.__wrapped__(48, 18, 12, active_rows=True, solve_policy=TRACE)
        for text in (snippet, _policy_snippet(self, traced, "solve_trace_sweep(")):
            self.assertTrue("? omega * fmaxf(-residual, 0.0f) :" in text, "the unvisited-row bound ignores omega")
        full = sparse_kernel.__wrapped__(48, 18, 12)
        self.assertNotIn("sweep_violation", _policy_snippet(self, full, QUIET_LITERAL))


class TestSolveReadoutRecord(unittest.TestCase):
    """Host summaries of readout counters."""

    def test_record_allocates_slots_and_summarizes_counts(self):
        """Allocate counters for readout kernels only and summarize their cumulative histogram."""
        record = SolveReadoutRecord(SWEEPS)
        counted = SimpleNamespace(key="counted_kernel")
        plain = SimpleNamespace(key="plain_kernel")
        record.register(counted, "sparse_pgs", readout="sweeps", world_count=3, budget=4, device="cpu")
        record.register(plain, "mf_gs", readout=None, world_count=3, budget=4, device="cpu")
        record.register(None, "paired_factor", readout="sweeps", world_count=3, budget=4, device="cpu")
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
        self.assertNotIn("trace", summary)
        json.dumps(record.summary(), allow_nan=False)

    def test_summary_separates_solves_past_their_first_sweep(self):
        """Summarize the solves that ran past their first sweep apart from the one-sweep solves.

        The sparse sweeps record a world without rows as a solve that stopped after one sweep, which
        would dilute the full-budget fraction of the worlds a tolerance could shorten.
        """
        record = SolveReadoutRecord(SWEEPS)
        counted = SimpleNamespace(key="counted_kernel")
        record.register(counted, "sparse_pgs", readout="sweeps", world_count=3, budget=4, device="cpu")
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
        short = SimpleNamespace(key="short_kernel")
        record.register(short, "mf_gs", readout="sweeps", world_count=1, budget=1, device="cpu")
        self.assertEqual(
            record.summary()["slots"][1]["past_first_sweep"],
            {"solves": 0, "full_budget_fraction": None, "mean_sweeps": None},
        )
        json.dumps(record.summary(), allow_nan=False)

    def test_worlds_without_rows_count_as_skipped(self):
        """Count one skipped solve per launch of every world without dense or matrix-free rows.

        The matrix-free sweeps return before their first sweep for such a world, so no slot records
        it; a world's solves are its slots' solves plus its skipped ones.
        """
        record = SolveReadoutRecord(SWEEPS)
        counted = SimpleNamespace(key="counted_kernel")
        record.register(counted, "parallel_tier", readout="sweeps", world_count=4, budget=4, device="cpu")
        record.register_skipped(world_count=4, device="cpu")
        # World 0 holds dense rows and world 2 matrix-free rows; worlds 1 and 3 have none.
        dense = wp.array([4, 0, 0, 0], dtype=wp.int32, device="cpu")
        free = wp.array([0, 0, 2, 0], dtype=wp.int32, device="cpu")
        for _ in range(2):
            record.count_skipped(dense, free)
        # A second registration keeps the counters.
        record.register_skipped(world_count=4, device="cpu")
        summary = record.summary()
        self.assertEqual(summary["skipped"], {"solves": 4, "worlds": 2})
        self.assertEqual([slot["solves"] for slot in summary["slots"]], [0])
        json.dumps(summary, allow_nan=False)

    def test_skip_counters_need_a_readout_and_a_registering_route(self):
        """Count no skipped solve without a readout, or on a route that registered no skip counters."""
        empty = wp.zeros(2, dtype=wp.int32, device="cpu")
        for policy, register in ((DEFAULT_POLICY, True), (SWEEPS, False)):
            with self.subTest(policy=policy, register=register):
                record = SolveReadoutRecord(policy)
                if register:
                    record.register_skipped(world_count=2, device="cpu")
                record.count_skipped(empty, empty)
                self.assertIsNone(record.summary()["skipped"])

    def test_trace_record_summarizes_stops_reasons_and_final_changes(self):
        """Summarize the trace's per-tolerance stopping sweeps, exit reasons and final-sweep changes."""
        record = SolveReadoutRecord(TRACE)
        traced = SimpleNamespace(key="traced_kernel")
        record.register(traced, "mf_gs", readout="trace", world_count=2, budget=4, device="cpu")
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
        traced = SimpleNamespace(key="traced_kernel")
        record.register(traced, "sparse_pgs", readout="trace", world_count=16384, budget=8, device="cpu")
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
            row_kernels.PGS_CONSTRAINT_TYPE_CONTACT: "contact",
            row_kernels.PGS_CONSTRAINT_TYPE_SPEED: "speed",
            row_kernels.PGS_CONSTRAINT_TYPE_FRICTION: "friction",
            row_kernels.PGS_CONSTRAINT_TYPE_JOINT_LIMIT: "joint_limit",
            row_kernels.PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT: "velocity_limit",
            row_kernels.PGS_CONSTRAINT_TYPE_MIMIC: "mimic",
            row_kernels.PGS_CONSTRAINT_TYPE_LINK: "link",
            row_kernels.PGS_CONSTRAINT_TYPE_TORSION: "torsion",
        }
        self.assertEqual(len(names), row_kernels.PGS_CONSTRAINT_TYPE_COUNT)
        self.assertEqual(TRACE_KINDS, (*(names[kind] for kind in range(row_kernels.PGS_CONSTRAINT_TYPE_COUNT)), "none"))


class TestSolverReadoutOption(unittest.TestCase):
    """The solver option and its records on a CPU model, whose split-mode loop sweep compiles every policy."""

    @staticmethod
    def _model():
        builder = newton.ModelBuilder()
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_articulation([builder.add_joint_revolute(-1, body, limit_lower=-0.1, limit_upper=0.1)])
        builder.joint_q[-1] = 0.2
        return builder.finalize(device="cpu")

    def test_invalid_readout_raises(self):
        """Reject an unknown readout at construction, and the retired tolerance options."""
        with self.assertRaisesRegex(ValueError, "solve_readout"):
            SolverFeatherPGS(self._model(), diagnostics=SolverFeatherPGS.Diagnostics(solve_readout="every_sweep"))
        for option in ("pgs_change_tolerance", "pgs_relative_change_tolerance", "pgs_velocity_change_tolerance"):
            with self.subTest(option=option), self.assertRaises(TypeError):
                SolverFeatherPGS(self._model(), **{option: 1.0e-3})

    def test_records_name_the_exit_law(self):
        """Record the one exit law, its floor in rounding units of float32, without a warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            selection = SolverFeatherPGS(self._model()).get_path_selection()
        json.dumps(selection, allow_nan=False)
        law = selection["routes"]["sweep"]["exit_law"]
        self.assertEqual((law["law"], law["quiet_units"], law["unit_roundoff"]), ("rounding_floor", 16, 2.0**-24))

    def test_records_name_the_readout_and_its_kernels(self):
        """Record the requested readout, the matrix-free sweep's counters and, after a step, its counted solve."""
        model = self._model()
        for requested in ("off", "sweeps", "trace"):
            with self.subTest(requested=requested):
                diagnostics = SolverFeatherPGS.Diagnostics(solve_readout=requested)
                solver = SolverFeatherPGS(model, diagnostics=diagnostics, enable_joint_limits=True)
                selection = solver.get_path_selection()
                json.dumps(selection, allow_nan=False)
                option = selection["options"]["diagnostics"]
                self.assertEqual(
                    (option["requested"]["solve_readout"], option["applied"]["solve_readout"]), (requested, requested)
                )
                readout = selection["routes"]["sweep"]["readout"]
                self.assertEqual(readout["mode"], requested)
                self.assertEqual(list(readout["counted"].values()), [] if requested == "off" else ["mf_gs"])
                self.assertEqual(selection["routes"]["sweep"]["exit_law"]["law"], "rounding_floor")
                state_0, state_1, control = model.state(), model.state(), model.control()
                solver.step(state_0, state_1, control, None, 1.0 / 60.0)
                measurements = solver.get_path_selection()["measurements"]
                if requested == "off":
                    self.assertNotIn("solve_readout", measurements)
                    continue
                # The violated limit is the world's one row: one solve, stopped once a sweep is quiet.
                (slot,) = measurements["solve_readout"]["slots"]
                self.assertEqual((slot["solves"], slot["exhausted"]), (1, 0))
                self.assertGreaterEqual(slot["mean_sweeps"], 2.0)


def _sweep_record(**fields):
    """A matrix-free ``routes.sweep`` record naming only its matrix-free Gauss-Seidel kernel, with ``fields`` set."""
    record = dict.fromkeys(
        (
            "sparse",
            "sparse_diagonal",
            "paired_factor",
            "mf_colored",
            "tiers",
        )
    )
    record.update(route="matrix_free_gs", mf_gs="mf_gs_kernel")
    record.update(fields)
    return record


def _policy_solver(*, readout="sweeps", skipped=True, **attributes):
    """The solver attributes the sweep records read, counting the matrix-free kernel and its skipped solves."""
    return SimpleNamespace(
        sweep=SimpleNamespace(
            policy=SolvePolicy(readout=readout),
            readout=SimpleNamespace(kernels={"mf_gs_kernel": "mf_gs"}, counts_skipped=skipped),
        ),
        **attributes,
        options=SimpleNamespace(diagnostics=SolverFeatherPGS.Diagnostics(solve_readout=readout)),
    )


class TestSweepPolicyRecords(unittest.TestCase):
    """The readout and exit-law records of a route's sweeps."""

    def test_a_sweep_records_the_one_law(self):
        """Record the exit law, and that a launch solves its worlds, so a solve counts once."""
        solver, record = _policy_solver(), _sweep_record()
        sweep = {
            "readout": path_selection._readout(solver, record),
            "exit_law": path_selection._exit_law(solver, record),
        }
        self.assertEqual(sweep["exit_law"]["law"], "rounding_floor")
        self.assertIn("solves its worlds", sweep["readout"]["reason"])
        json.dumps(sweep, allow_nan=False)

    def test_records_say_whether_skipped_solves_count(self):
        """Say whether the solves of worlds without rows count as skipped, and not without counted sweeps."""
        record = _sweep_record()
        for skipped in (False, True):
            with self.subTest(skipped=skipped):
                readout = path_selection._readout(_policy_solver(skipped=skipped), record)
                self.assertEqual(readout["counts_skipped"], skipped)
        solver = _policy_solver()
        solver.sweep.readout = SimpleNamespace(kernels={}, counts_skipped=True)
        self.assertFalse(path_selection._readout(solver, record)["counts_skipped"])


if __name__ == "__main__":
    unittest.main()
