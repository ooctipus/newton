# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the SASS record's branch resolution, listing normalization and comparison."""

import contextlib
import io
import unittest

from tools.fpgs_bench import kernel_sass as sass


def _record(**kernels):
    return {"kernels": kernels}


def _row(reg=80, digest="s", cpu="c"):
    return {"reg": reg, "stack": 72, "shared": 3600, "local": 0, "instructions": 100, "sass": digest, "cpu_twin": cpu}


class TestKernelSass(unittest.TestCase):
    def test_resolve_cuda_branches(self):
        """Keep the selected side of CUDA tests, nested or negated, and every other conditional as written."""
        source = "\n".join(
            (
                "a;",
                "#if defined(__CUDA_ARCH__)",
                "cuda;",
                "#if WP_ENABLE_MATHDX",
                "dx;",
                "#endif",
                "#else",
                "cpu;",
                "#endif",
                "#ifndef __CUDA_ARCH__",
                "host;",
                "#endif",
                "#if !defined(__CUDA_ARCH__) // comment",
                "free(p);",
                "#endif",
                "b;",
            )
        )
        self.assertEqual(
            sass.resolve_cuda_branches(source, cuda=True), "a;\ncuda;\n#if WP_ENABLE_MATHDX\ndx;\n#endif\nb;"
        )
        self.assertEqual(sass.resolve_cuda_branches(source, cuda=False), "a;\ncpu;\nhost;\nfree(p);\nb;")
        with self.assertRaises(ValueError):
            sass.resolve_cuda_branches("#if defined(__CUDA_ARCH__)\nx;\n", cuda=True)

    def test_sass_instructions_drop_addresses_encodings_and_hash_suffix(self):
        """Keep instruction text only, with the kernel's hash suffix replaced."""
        listing = (
            "\tFunction : k_0123abcd_cuda_kernel_forward\n"
            "        /*0000*/                   LDC R1, c[0x0][0x37c] ;                /* 0x0000df00ff017b82 */\n"
            "                                                                          /* 0x000e220000000800 */\n"
            "        /*0010*/                   CALL.REL k_89abcdef_cuda_kernel_forward ;  /* 0x0 */\n"
        )
        self.assertEqual(sass.sass_instructions(listing), ["LDC R1, c[0x0][0x37c]", "CALL.REL KERNEL_forward"])

    def test_parse_usage(self):
        """Read registers, stack, shared and local memory of the kernel function."""
        usage = " Function k_cuda_kernel_forward:\n  REG:80 STACK:72 SHARED:3600 LOCAL:0 CONSTANT[2]:36"
        self.assertEqual(sass.parse_usage(usage), {"reg": 80, "stack": 72, "shared": 3600, "local": 0})

    def test_compare_flags_register_granules(self):
        """Report changed SASS, resources and CPU twins, and flag a register count crossing a granule."""
        parent = _record(same=_row(), within=_row(78), across=_row(80), twin=_row(), gone=_row())
        child = _record(same=_row(), within=_row(74, "t"), across=_row(82, "t"), twin=_row(cpu="d"), new=_row())
        result = sass.compare(parent, child)
        self.assertEqual(sorted(result["changed"]), ["across", "twin", "within"])
        self.assertFalse(result["changed"]["within"]["granule"])
        self.assertTrue(result["changed"]["across"]["granule"])
        self.assertEqual(result["changed"]["twin"], {"cpu_twin": ("c", "d")})
        self.assertEqual((result["added"], result["removed"]), (["new"], ["gone"]))
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(sass.report(parent, child), 1)
            self.assertEqual(sass.report(parent, _record(**parent["kernels"])), 0)
        self.assertIn("REGISTER GRANULE CROSSED", output.getvalue())


if __name__ == "__main__":
    unittest.main()
