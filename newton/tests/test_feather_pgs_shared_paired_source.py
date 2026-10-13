# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Preserve qualified native laws while sharing representation source construction."""

import hashlib
import inspect
import unittest
from unittest import mock

from newton._src.solvers.feather_pgs import (
    component_qualification,
    friction,
    kinetic_compact_coupled,
    kinetic_hybrid,
    kinetic_solve,
    paired_factor_source,
)
from newton._src.solvers.feather_pgs.solver_feather_pgs import (
    _get_pgs_solve_mf_gs_kernel,
    _get_pgs_solve_paired_factor_kernel,
)


class TestSharedPairedSource(unittest.TestCase):
    def test_compact_owner_does_not_rewrite_sibling_source(self):
        """Preserve compact and fallback bodies through direct policy binding."""
        kernel = _get_pgs_solve_mf_gs_kernel(
            192,
            64,
            29,
            "cpu",
            has_drive_rows=False,
            has_dense_velocity_limit_rows=False,
            factor_coordinates=True,
            independent_components=True,
        )
        with (
            mock.patch.object(kinetic_solve, "cuda_recurrence", side_effect=AssertionError("sibling source")),
            mock.patch.object(
                kinetic_solve, "replace_once", side_effect=AssertionError("source rewriting"), create=True
            ),
        ):
            source, baseline, mf_phase = kinetic_compact_coupled.compact_source(kernel)
            fallback, _ = kinetic_compact_coupled.fallback_source(baseline)
        self.assertEqual(
            hashlib.sha256(source.encode()).hexdigest(),
            "37441d9cfb9a88eff531fdae61cb0b36f05f7a88d04bbf38d657919081beda4a",
        )
        self.assertEqual(
            hashlib.sha256(fallback.encode()).hexdigest(),
            "8963b592c49eabea3681347df4f400b3722c08846bcc222258c2ce7b3ce1bcad",
        )
        self.assertIn(mf_phase, source)

    def test_scalar_owners_share_secondary_response_policy(self):
        """Preserve both existing CPU bodies, including converted free columns."""
        from newton._src.solvers.feather_pgs import kinetic_coupled_cpu  # noqa: PLC0415

        for converted, expected in (
            (False, "2700160add075b99e651912cb6ffb04057248af06d6f4fb38d6a2c6985d279c3"),
            (True, "191c1da9cfdbd4dbaa82c25dcdf85532caabe67a13bc57990b759965eb5cff8c"),
        ):
            with self.subTest(converted=converted):
                source = kinetic_coupled_cpu.get_source(converted_secondary=converted)
                self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), expected)

    def test_pair_intrinsics_preserve_cpu_law(self):
        """Share the bounded friction helper with explicit backend intrinsics."""
        self.assertEqual(
            hashlib.sha256(friction.get_pair_source(cpu=True).encode()).hexdigest(),
            "238560e12935fbc0a3a6fa16ade522b170194422b93a59b10a716423dcce2c57",
        )
        self.assertEqual(friction.get_pair_source(), friction.FRICTION_PAIR_CUDA)

    def test_physical_bodies_preserve_qualified_source(self):
        """Preserve complete native bodies from qualified Newton 9f17ebed."""
        cases = (
            (192, 29, 23, 6, True, False, "4d555fb5cbd4f06cd2c29c4621f79ad48225b24c0d7f271ec5de79573afe5c89"),
            (192, 29, 23, 6, True, True, "2af1540362ce01c8e3e26c34f0c3f56e649592e7aac7eac3115074d62ee3c913"),
            (24, 11, 7, 4, False, False, "9fe52763b31563b9c35bd36d9fc8cd8452334f49470e9100ce6de23e14d9e779"),
            (24, 11, 7, 4, True, False, "114b45226c380eb44cae09528948f28744e030d4fed54d7f146b4600b79fc355"),
        )
        for M, D, P, S, triples, independent, expected in cases:
            with self.subTest(shape=(M, D, P, S), triples=triples, independent=independent):
                source = paired_factor_source.get_source(
                    M, D, P, S, contact_triples=triples, independent_components=independent
                )
                self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), expected)
                kernel = _get_pgs_solve_paired_factor_kernel(
                    M, D, P, S, "cpu", contact_triples=triples, independent_components=independent
                )
                native = inspect.getclosurevars(kernel.func).nonlocals["pgs_solve_paired_factor_native"]
                self.assertEqual(native.native_snippet, source)

    def test_kinetic_body_preserves_publication_fences(self):
        """Preserve held coordinates and all six current shared-impulse publication fences."""
        source = paired_factor_source.get_source(192, 29, 23, 6, contact_triples=True, kinetic=True)
        self.assertEqual(
            hashlib.sha256(source.encode()).hexdigest(),
            "15e4693a894065519665d82bb91f607cf8387c1aed7df7db1d7d189dbcce2e34",
        )
        self.assertEqual(source, kinetic_solve.cuda_recurrence())
        for index in range(6):
            self.assertEqual(source.count(f"IMPULSE_READ_COMPLETE_{index}"), 1)

    def test_kinetic_shape_guard(self):
        """Reject unsupported kinetic representations before generating native code."""
        with self.assertRaises(ValueError):
            paired_factor_source.get_source(64, 11, 7, 4, kinetic=True)

    def test_component_predicates_preserve_qualified_source(self):
        """Preserve physical materialization and both held-coordinate admission predicates."""
        hashes = {
            "physical": "e545381d57064dfa1cce0abd6d58ffe51c1de14d987cf05d5c868e971ed5fc03",
            "kinetic": "e7d7f4f78ae8aef5a0288f6144b6dad755a4cb3c4421fa5cbe98b88b7b25888e",
            "hybrid": "bb51171cd981c27c1d1a441ac982659d8bacb8c47b980f47b7c10e404d1ebc21",
        }
        for layout, expected in hashes.items():
            with self.subTest(layout=layout):
                source = component_qualification.get_source(layout)
                self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), expected)
        self.assertEqual(kinetic_solve.qualifier_source(), component_qualification.get_source("kinetic"))
        self.assertEqual(kinetic_hybrid.qualify_source(), component_qualification.get_source("hybrid"))
        with self.assertRaises(ValueError):
            component_qualification.get_source("unsupported")


if __name__ == "__main__":
    unittest.main()
