"""Check the conditional register-residual owner and its exact fallback seam."""

import unittest

from newton._src.solvers.feather_pgs import sparse_register_gram


class TestSparseRegister(unittest.TestCase):
    """Exercise the scoped native factory before integration."""

    def test_api_exists(self):
        """Require both intrinsic row-count factories and the installer."""
        self.assertTrue(callable(sparse_register_gram.get_solve_kernel))
        self.assertTrue(callable(sparse_register_gram.install))

    def test_register_columns_and_cross_bank_siblings(self):
        """Keep128 named64-row Gram scalars and statically addressed siblings."""
        source = sparse_register_gram.cuda_source(100, 64)
        self.assertEqual(source.count("float g"), 128)
        recurrence = source.split("for(int iteration=", 1)[1].split("impulses.data[", 1)[0]
        self.assertNotIn("__syncwarp", recurrence)
        self.assertNotIn("__shfl_down", recurrence)
        self.assertNotIn("for(int row=", recurrence)
        self.assertIn("z[43*65]", source)
        projection = sparse_register_gram._projection(31, 64)
        self.assertIn("lam1,0", projection)
        self.assertIn("g0_32*sibling_delta", projection)
        self.assertIn("g1_32*sibling_delta", projection)
        self.assertIn("applied1+=sibling_delta", projection)

    def test_only_intrinsic_classes(self):
        """Reject unsupported class construction instead of silently truncating."""
        for rows in (31, 33, 65):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                sparse_register_gram.get_solve_kernel(100, rows)


if __name__ == "__main__":
    unittest.main()
