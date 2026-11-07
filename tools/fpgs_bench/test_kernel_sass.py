# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the SASS record's branch resolution, listing parsing, occupancy model and comparison."""

import contextlib
import io
import unittest

from tools.fpgs_bench import kernel_sass as sass


def _record(arch=120, **kernels):
    return {"arch": arch, "kernels": kernels}


def _row(reg=80, digest="s", cpu="c", shared=3600, block_dim=32, local=0):
    return {
        "reg": reg,
        "stack": 72,
        "shared": shared,
        "local": local,
        "instructions": 100,
        "sass": digest,
        "cpu_twin": cpu,
        "block_dim": block_dim,
    }


# Two kernels of one module, as cuobjdump lists them.
_USAGE = """Resource usage:
 Common:
  GLOBAL:147 CONSTANT[3]:5720
 Function first_0123abcd_cuda_kernel_forward:
  REG:40 STACK:0 SHARED:1152 LOCAL:0 CONSTANT[0]:976 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function second_89abcdef_cuda_kernel_forward:
  REG:86 STACK:72 SHARED:2880 LOCAL:8 CONSTANT[0]:976 TEXTURE:0 SURFACE:0 SAMPLER:0
"""
_SASS = """
\tcode for sm_120
\t\tFunction : second_89abcdef_cuda_kernel_forward
\t.headerflags\t@"EF_CUDA_SM120 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM120)"
        /*0000*/                   LDC R1, c[0x0][0x37c] ;                /* 0x0000df00ff017b82 */
        /*0010*/                   EXIT ;                                 /* 0x000000000000794d */
\t\t..........

\t\tFunction : first_0123abcd_cuda_kernel_forward
        /*0000*/                   MOV R2, 0x1 ;                          /* 0x0000000100027802 */
"""


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
        """Read registers, stack, shared and local memory of one function, and refuse a listing of several."""
        usage = " Function k_cuda_kernel_forward:\n  REG:80 STACK:72 SHARED:3600 LOCAL:0 CONSTANT[2]:36"
        self.assertEqual(sass.parse_usage(usage), {"reg": 80, "stack": 72, "shared": 3600, "local": 0})
        with self.assertRaises(ValueError):
            sass.parse_usage(_USAGE)

    def test_listings_split_by_function(self):
        """Credit each kernel of a module with its own resources and instructions, whatever the listing order."""
        usage, listing = sass.function_sections(_USAGE), sass.function_sections(_SASS)
        self.assertEqual(sorted(usage), ["first_0123abcd_cuda_kernel_forward", "second_89abcdef_cuda_kernel_forward"])
        self.assertEqual(sorted(listing), sorted(usage))
        second = "second_89abcdef_cuda_kernel_forward"
        self.assertEqual(sass.parse_usage(usage[second]), {"reg": 86, "stack": 72, "shared": 2880, "local": 8})
        self.assertEqual(sass.sass_instructions(listing[second]), ["LDC R1, c[0x0][0x37c]", "EXIT"])
        self.assertEqual(sass.sass_instructions(listing["first_0123abcd_cuda_kernel_forward"]), ["MOV R2, 0x1"])

    def test_resident_blocks(self):
        """Count resident blocks per SM from registers, shared memory, block size and the architecture's limits."""
        # A 72-register one-warp kernel: the block limit binds on sm_120, registers (7 warps per
        # sub-partition) on sm_103; at 80 registers both keep 24 (6 warps per sub-partition).
        self.assertEqual(sass.resident_blocks(72, 2704, 32, 120), 24)
        self.assertEqual(sass.resident_blocks(72, 2704, 32, 103), 28)
        self.assertEqual(sass.resident_blocks(80, 2704, 32, 103), 24)
        # 86 registers allocate 88: 5 warps per sub-partition.
        self.assertEqual(sass.resident_blocks(86, 2880, 32, 120), 20)
        # Shared memory binds: 6928 bytes allocate 7040, 14 of them in 100 KB.
        self.assertEqual(sass.resident_blocks(80, 6928, 32, 120), 14)
        # A two-warp block: 12 by registers, 11 by shared memory.
        self.assertEqual(sass.resident_blocks(79, 9024, 64, 120), 11)

    def test_compare_reports_granule_direction_and_resident_blocks(self):
        """Report changed SASS, resources and CPU twins with each granule crossing's direction and blocks per SM."""
        parent = _record(same=_row(), within=_row(78), up=_row(72), down=_row(91), twin=_row(), gone=_row())
        child = _record(
            same=_row(), within=_row(74, "t"), up=_row(78, "t"), down=_row(79, "t"), twin=_row(cpu="d"), new=_row()
        )
        result = sass.compare(parent, child)
        self.assertEqual(sorted(result["changed"]), ["down", "twin", "up", "within"])
        self.assertIsNone(result["changed"]["within"]["granule"])
        self.assertEqual(result["changed"]["up"]["granule"], "up")
        self.assertEqual(result["changed"]["down"]["granule"], "down")
        self.assertEqual(result["changed"]["up"]["blocks"], (24, 24))
        self.assertEqual(result["changed"]["down"]["blocks"], (20, 24))
        self.assertEqual(result["changed"]["twin"], {"cpu_twin": ("c", "d")})
        self.assertEqual((result["added"], result["removed"]), (["new"], ["gone"]))
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(sass.report(parent, child), 1)
            self.assertEqual(sass.report(parent, _record(**parent["kernels"])), 0)
        self.assertIn("granule up, blocks/SM 24->24", output.getvalue())
        self.assertIn("granule down, blocks/SM 20->24", output.getvalue())
        self.assertNotIn("REGRESSION", output.getvalue())

    def test_launch_blocks_follow_the_owner_launches(self):
        """Compile every held-tree kernel at its layout's block, as the owner launches it."""
        calls = list(sass.identity.held_tree_calls())
        blocks = sass.launch_blocks(("held_tree",))
        self.assertEqual(len(blocks), len(calls))
        for name, _, args, kwargs in calls:
            self.assertEqual(blocks[sass.identity.label(name, args, kwargs)], kwargs["layout"].value.block)

    def test_walk_kernels_compile_at_their_launch_blocks(self):
        """Compile the walk family's predictor halves at their layout's block, its solve and refresh at the owner's."""
        blocks = sass.launch_blocks(("walk",))
        for name, _, args, kwargs in sass.identity.walk_calls():
            block = blocks.get(sass.identity.label(name, args, kwargs))
            if name.endswith(("get_force_kernel", "get_predictor_kernel")):
                self.assertEqual(block, kwargs["layout"].value.block)
            elif name.endswith("get_walk_solve_kernel"):
                self.assertEqual(block, 32)
            elif name.endswith("walk_step_kernel"):
                self.assertEqual(block, 64)
            else:
                self.assertIsNone(block)

    def test_coordinate_tree_kernels_group_at_their_launch_blocks(self):
        """Group the one-coordinate class's kernels at the blocks kernel_sass compiles them at: each repair at its
        launch block.

        The held-tree owner's one-coordinate class has no layout, so its kernels are in the pins family: the
        repairs launch one warp per articulation in blocks of four, the predicts and finishes at Warp's default
        block. Nothing is compiled here; this checks the grouping a record uses.
        """
        from newton._src.solvers.feather_pgs import coordinate_trees  # noqa: PLC0415

        kernels, rejected = sass.identity.inventory(("pins",), ("coordinate_trees.",))
        self.assertEqual(rejected, {})
        blocks = {
            key: block for keys, block in sass._module_groups(kernels, sass.launch_blocks(("pins",))) for key in keys
        }
        self.assertEqual(len(blocks), 6)
        for kind in ("slides", "turns"):
            self.assertEqual(blocks.pop(f"coordinate_trees.repair_{kind}"), 128)
            self.assertNotIn(f"coordinate_trees.repair_{kind}", sass.launch_blocks(("held_tree",)))
        self.assertEqual(coordinate_trees.REPAIR_BLOCK_DIM, 128)
        self.assertEqual(set(blocks.values()), {256})

    def test_report_fails_kernels_that_lose_resident_blocks_or_spill(self):
        """Exit 2 when a kernel keeps fewer blocks resident or spills more to local memory."""
        parent = _record(103, kernel=_row(72, shared=4752))
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(sass.report(parent, _record(103, kernel=_row(80, "t", shared=4752))), 2)
            self.assertEqual(sass.report(parent, _record(103, kernel=_row(72, "t", shared=4752, local=16))), 2)
        self.assertIn("blocks/SM 28->24, REGRESSION", output.getvalue())


if __name__ == "__main__":
    unittest.main()
