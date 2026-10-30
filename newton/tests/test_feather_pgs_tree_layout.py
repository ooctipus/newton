# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""A tree layout fills its kernels' sizes; thread primitives come from one prelude sized by the lane count."""

import re
import unittest

from newton._src.solvers.feather_pgs import tree_kinetic_state
from newton._src.solvers.feather_pgs import tree_scan as ts
from newton._src.solvers.feather_pgs.tree_kinetic_layout import TreeLayout

# A packed fixed-root tree, a one-warp free-root tree and a two-warp tree.
_LAYOUTS = (
    TreeLayout(4, 3, 6, 3, 2, 0, (2, 1, 0), lanes=8),
    TreeLayout(21, 22, 60, 22, 4, 6, tuple(range(22))),
    TreeLayout(44, 43, 434, 18, 10, 6, tuple(range(42, -1, -1))),
)
# Warp-level intrinsics whose first argument is a lane mask.
_MASKED = re.compile(r"__(?:shfl_sync|shfl_down_sync|syncwarp|all_sync|any_sync)\(([^,)]*)")


class TestTreeLayoutRender(unittest.TestCase):
    def test_sizes_are_numbers(self):
        """Give every layout value as a number: a layout supplies sizes, never code."""
        for layout in _LAYOUTS:
            with self.subTest(lanes=layout.lanes):
                self.assertTrue(all(isinstance(value, int) for value in layout.sizes().values()))

    def test_render_rejects_code_tokens(self):
        """Fail on a thread or scratch token: the prelude declares those, the layout does not."""
        for token in ("SYNC", "MASK", "LANE", "STRIDE", "SUB", "SHFL", "ALL", "ANY", "STATE_SCRATCH_DECL"):
            with self.subTest(token=token), self.assertRaisesRegex(ValueError, "without values"):
                _LAYOUTS[0].render(f"${token};")

    def test_render_places_fragments_in_one_pass(self):
        """Fill sizes inside a fragment and never rescan a placed fragment for slots."""
        layout = _LAYOUTS[0]
        text = layout.render("a[$DOFS];$PART;$OTHER", PART="b[$BODIES]", OTHER="$$PART")
        self.assertEqual(text, "a[3];b[4];$PART")
        with self.assertRaises(ValueError):
            layout.render("$DOFS", DOFS="7")

    def test_prelude_differs_only_in_the_lane_count(self):
        """Open every tree native with one lane-group prelude per layout that differs only in its lane count."""
        rendered = {layout.lanes: layout.native("") for layout in _LAYOUTS}
        for lanes, text in rendered.items():
            self.assertIn(f"const int LANES = {lanes},", text)
        self.assertEqual(len({re.sub(r"LANES = \d+,", "LANES = N,", text) for text in rendered.values()}), 1)
        for name in ("WIDTH", "MASK", "LANE", "STRIDE", "SUB", "SYNC", "ALL", "ANY"):
            self.assertRegex(rendered[_LAYOUTS[0].lanes], rf"\b{name} =")

    def test_natives_take_the_prelude_mask(self):
        """Emit tree natives whose warp intrinsics take the prelude's mask, never a mask written per layout: on
        every schedule a layout's lanes run (heavy chains on a warp or two per tree)."""
        schedule = ts.Schedule
        for layout in _LAYOUTS:
            warp = layout.lanes <= 32
            choices = [
                (schedule.JUMP_SHFL, schedule.JUMP_SHFL, schedule.DFS_INTERVAL, schedule.LEVEL)
                if warp
                else (schedule.JUMP_SMEM, schedule.JUMP_SMEM, schedule.LEVEL, schedule.LEVEL)
            ]
            if layout.lanes >= 32:
                choices.append((schedule.HEAVY_CHAIN,) * 4)
            for choice in choices:
                schedules = tree_kinetic_state.HeldSchedules(*choice)
                with self.subTest(lanes=layout.lanes, schedules=schedules.tag):
                    helpers = tree_kinetic_state.get_helpers.__wrapped__(layout, schedules)
                    natives = [helpers._collect, helpers._storage, helpers._scan_poses, helpers._scan_motion]
                    # The static-scratch form a CPU owner builds shares the CUDA side of the default form.
                    natives.append(tree_kinetic_state.get_helpers.__wrapped__(layout, schedules, host=True)._storage)
                    natives += tree_kinetic_state._thread_helpers.__wrapped__(layout)
                    # The bank comparison votes over the group's lanes (a free root's quaternion adds a coordinate).
                    natives.append(
                        tree_kinetic_state._bank_binding.__wrapped__(layout.dofs + layout.free_root_count, layout)
                    )
                    # The predictor's native: the held solve, the subtree reduction and the twist publication.
                    predictor = tree_kinetic_state.get_predictor_kernel.__wrapped__(
                        schedules,
                        layout=layout,
                        response="cholesky",
                        twists=tree_kinetic_state.publishes_twists(layout, schedules, cuda=True),
                    )
                    predictor_natives = [
                        f for f in predictor.adj.get_references()[2] if getattr(f, "native_snippet", None)
                    ]
                    self.assertTrue(any("LANES" in native.native_snippet for native in predictor_natives))
                    natives += predictor_natives
                    for native in natives:
                        self.assertNotIn("$", native.native_snippet)
                        self.assertEqual(set(_MASKED.findall(native.native_snippet)) - {"MASK"}, set())

    def test_block_vote_spans_the_launch_block(self):
        """Vote over a launch block from the prelude's lane count: a packed block's whole warp, else the block.

        A tree's vote (the prelude's ``ANY``) would leave the other trees of a
        packed block out of the vote that lists the block for a repair.
        """
        for layout in _LAYOUTS:
            with self.subTest(lanes=layout.lanes):
                snippet = tree_kinetic_state._block_vote.__wrapped__(layout).native_snippet
                self.assertNotIn("$", snippet)
                self.assertIn(f"const int LANES = {layout.lanes},", snippet)
                self.assertIn("if constexpr(LANES<32)return __any_sync(0xffffffffu,value);", snippet)
                self.assertIn("else return __syncthreads_or(value);", snippet)


if __name__ == "__main__":
    unittest.main()
