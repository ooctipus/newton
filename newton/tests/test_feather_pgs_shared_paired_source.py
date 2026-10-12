# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Preserve qualified native laws while sharing representation source construction."""

import hashlib
import inspect
import unittest

from newton._src.solvers.feather_pgs import component_qualification, kinetic_hybrid, kinetic_solve, paired_factor_source
from newton._src.solvers.feather_pgs.solver_feather_pgs import _get_pgs_solve_paired_factor_kernel


class TestSharedPairedSource(unittest.TestCase):
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
