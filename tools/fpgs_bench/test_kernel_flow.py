# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the PTX dataflow classes: identical, schedule-only and rounding."""

import unittest

from tools.fpgs_bench import kernel_flow as flow

_BASE = """
    ld.global.f32 %f1, [%rd1];
    ld.global.f32 %f2, [%rd1+4];
    mul.rn.f32 %f3, %f1, %f2;
    add.f32 %f4, %f3, %f1;
    st.global.f32 [%rd2], %f4;
    st.global.f32 [%rd2+4], %f3;
"""
# The same operations, the operands commuted and the stores and registers reordered.
_SCHEDULED = """
    ld.global.f32 %f7, [%rd1+4];
    ld.global.f32 %f8, [%rd1];
    mul.rn.f32 %f9, %f7, %f8;
    st.global.f32 [%rd2+4], %f9;
    add.f32 %f10, %f8, %f9;
    st.global.f32 [%rd2], %f10;
"""
# The product contracted into the sum.
_CONTRACTED = """
    ld.global.f32 %f1, [%rd1];
    ld.global.f32 %f2, [%rd1+4];
    mul.f32 %f3, %f1, %f2;
    fma.rn.f32 %f4, %f1, %f2, %f1;
    st.global.f32 [%rd2], %f4;
    st.global.f32 [%rd2+4], %f3;
"""


class TestKernelFlow(unittest.TestCase):
    def test_classes(self):
        """Equal text is identical, the same operations in another order schedule-only, a contraction rounding."""
        self.assertEqual(flow.classify(_BASE, _BASE)[0], "identical")
        self.assertEqual(flow.classify(_BASE, _SCHEDULED)[0], "schedule-only")
        kind, detail = flow.classify(_BASE, _CONTRACTED)
        self.assertEqual(kind, "rounding")
        self.assertIn("fma.rn.f32", detail)

    def test_entry_takes_one_function_without_hash_suffixes(self):
        """Cut one function's entry from a module's PTX and drop the kernel's hash suffix."""
        ptx = (
            ".entry a_0123abcd_cuda_kernel_forward(\n)\n{\n ret;\n}\n.entry b_89abcdef_cuda_kernel_forward(\n)\n{\n}\n"
        )
        text = flow.entry(ptx, "a_0123abcd_cuda_kernel_forward")
        self.assertTrue(text.startswith("a_H_cuda_kernel_forward"))
        self.assertNotIn("b_", text)


if __name__ == "__main__":
    unittest.main()
