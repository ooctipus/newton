# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The lane-parallel held factor solve of the tree predictor matches the serial law."""

import unittest

from newton._src.solvers.feather_pgs import response_operator


class TestLaneCholeskyEmitter(unittest.TestCase):
    def test_emitter_is_sized_by_the_family(self):
        """Every loop bound and row stride is the family size; the lane, mask and width are the caller's."""
        text = response_operator.LaneCholeskySolve(23, "L").solve("$SIX_BODIES+0", "$WHITE+0")
        self.assertIn("for(int k=0;k<23;++k)", text)
        self.assertIn("for(int k=23-1;k>=0;--k)", text)
        self.assertIn("row[k]=Lr[ic*23+k];col[k]=Lr[k*23+ic];", text)
        self.assertIn("float row[23],col[23];", text)
        self.assertIn("__shfl_sync(MASK,x,k,WIDTH)", text)
        self.assertIn("diagonal!=0.0f?value/diagonal:0.0f", text)
        custom = response_operator.LaneCholeskySolve(6, "L").solve("a", "b", lane="l", mask="m", width="16")
        self.assertIn("__shfl_sync(m,z,k,16)", custom)
        self.assertIn("const int i=l;", custom)


if __name__ == "__main__":
    unittest.main()
