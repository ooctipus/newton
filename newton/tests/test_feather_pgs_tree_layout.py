# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""A tree layout fills its kernels' sizes; thread primitives come from one prelude sized by the lane count."""

import contextlib
import dataclasses
import re
import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import tree_kinetic_plan, tree_kinetic_state
from newton._src.solvers.feather_pgs import tree_scan as ts
from newton._src.solvers.feather_pgs.tree_kinetic_layout import TreeLayout
from newton._src.solvers.feather_pgs.tree_plan import TreePlan
from newton.solvers import SolverFeatherPGS
from newton.tests import feather_pgs_scenes as scenes

# A packed fixed-root tree, a one-warp free-root tree and a two-warp tree.
_LAYOUTS = (
    TreeLayout(4, 3, 6, 3, 2, 0, (2, 1, 0), lanes=8),
    TreeLayout(21, 22, 60, 22, 4, 6, tuple(range(22))),
    TreeLayout(44, 43, 434, 18, 10, 6, tuple(range(42, -1, -1))),
)
# A chain of more bodies and coordinates than a block's lanes.
_LONG_CHAIN = TreePlan.build(np.arange(-1, 105), np.r_[6, np.ones(105, np.int64)])
# Warp-level intrinsics whose first argument is a lane mask.
_MASKED = re.compile(r"__(?:shfl_sync|shfl_down_sync|syncwarp|all_sync|any_sync)\(([^,)]*)")


class TestLargeTrees(unittest.TestCase):
    def test_factor_pattern_holds_any_size(self):
        """Keep the fill-free pattern past the masks' 64 coordinates: a chain's every entry, a star's diagonal."""
        chain = _LONG_CHAIN.factor
        self.assertEqual(chain.nonzero_count, 111 * 112 // 2)
        self.assertEqual(chain.nonzero_count, _LONG_CHAIN.nonzero_count)
        self.assertEqual(np.count_nonzero(chain.lookup >= 0), chain.nonzero_count)
        self.assertFalse(np.any(chain.ancestor_mask) or np.any(chain.joint_ancestor_mask))
        star = TreePlan.build(np.r_[-1, np.zeros(70, np.int64)]).factor
        # The root's coordinate (the last in leaf-first order) couples to every leaf.
        self.assertEqual(star.nonzero_count, 71 + 70)
        self.assertEqual(TreePlan.build(np.arange(-1, 63)).factor.joint_ancestor_mask[-1], (1 << 64) - 1)

    def test_a_tree_past_the_lanes_takes_the_depth_levels(self):
        """Hold a tree of more bodies than a block's lanes: 64 lanes, and the depth-level scans on CUDA."""
        lanes = tree_kinetic_state.held_lanes(_LONG_CHAIN, 3)
        self.assertEqual(lanes, 64)
        schedules = tree_kinetic_state.held_schedules(_LONG_CHAIN, lanes, chains=True)
        self.assertEqual(set(dataclasses.asdict(schedules).values()), {ts.Schedule.LEVEL})
        layout = TreeLayout(106, 111, _LONG_CHAIN.nonzero_count, 111, 105, 6, tuple(range(110, -1, -1)), lanes=lanes)
        self.assertEqual((layout.block, layout.pack), (64, 1))
        with self.assertRaises(ValueError):
            TreeLayout(106, 111, 111, 111, 105, 6, tuple(range(111)), lanes=32)


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


class TestStateBlockPlacement(unittest.TestCase):
    def test_a_block_past_the_budget_reads_the_global_scratch(self):
        """Place a tree's block by its size alone: the rule reads the budget, the native reads the rule."""
        layout = _LAYOUTS[0]
        fits = 4 * layout.total * layout.pack
        self.assertFalse(dataclasses.replace(layout, shared_budget=fits).global_block)
        tight = dataclasses.replace(layout, shared_budget=fits - 1)
        self.assertTrue(tight.global_block)
        self.assertIn("__shared__", tree_kinetic_state._placed_block(layout, "values", "$TOTAL"))
        self.assertNotIn("__shared__", tree_kinetic_state._placed_block(tight, "values", "$TOTAL"))
        self.assertIn("plan.scratch.data+group*$TOTAL", tree_kinetic_state._placed_block(tight, "values", "$TOTAL"))

    @unittest.skipUnless(wp.is_cuda_available(), "Placement is a CUDA choice")
    def test_both_placements_step_alike(self):
        """Step two different trees with their state blocks in shared and in global memory: the same state to rounding."""
        scene, steps, finals = scenes.sprawl(worlds=2), 8, []
        for budget in (None, 64):
            with (
                mock.patch.object(tree_kinetic_plan, "static_budget", return_value=budget)
                if budget
                else contextlib.nullcontext()
            ):
                model = scene.build("cuda:0")
                solver = SolverFeatherPGS(model, **scene.solver_options)
            self.assertEqual(solver.factor.held.groups[0].layout.global_block, budget is not None)
            # The row order must not follow thread scheduling: sorted contacts order the rows, and the scene's row limit
            # keeps them all, since a thread race picks the rows that survive a limit.
            pipeline = newton.CollisionPipeline(model, rigid_contact_max=scene.rigid_contact_max, deterministic=True)
            contacts = pipeline.contacts()
            state, out, control = model.state(), model.state(), model.control()
            # Different trees: a block read at another tree's offset then holds other values.
            joint_q = state.joint_q.numpy()
            joint_q[model.joint_coord_count // 2 + 7] += 0.05
            state.joint_q.assign(joint_q)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            for _ in range(steps):
                pipeline.collide(state, contacts)
                solver.step(state, out, control, contacts, scene.dt)
                state, out = out, state
            finals.append({name: getattr(state, name).numpy() for name in ("joint_q", "joint_qd", "body_q", "body_qd")})
        # Only the address space differs, so ptxas may fuse multiply-adds apart (BACKBONE rule 4), though both devices
        # now read the same bits. One rounding of a position, u at the scene's size, reads as u / dt in a contact's
        # target; over the steps that sums to steps * u / dt in velocity and steps * (steps + 1) / 2 * u in position.
        # Every joint coordinate moved by one ulp stays inside these, and a lost store of one body fails them 14x over.
        u = np.finfo(np.float32).eps * np.abs(finals[0]["body_q"]).max()
        position, velocity = 0.5 * steps * (steps + 1) * u, steps * u / scene.dt
        for name, tolerance in (
            ("joint_q", position),
            ("body_q", position),
            ("joint_qd", velocity),
            ("body_qd", velocity),
        ):
            np.testing.assert_allclose(finals[1][name], finals[0][name], rtol=0.0, atol=tolerance, err_msg=name)


if __name__ == "__main__":
    unittest.main()
