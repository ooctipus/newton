# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The held-tree finish and repair are authored stage functions over one tree-group operand."""

import unittest
from pathlib import Path

from newton._src.solvers.feather_pgs import tree_kinetic_state
from newton._src.solvers.feather_pgs.tree_kinetic_layout import TreeLayout

# A packed fixed-root tree, a free root body, a two-warp free-root tree and a forest with a free root.
_LAYOUTS = (
    TreeLayout(4, 3, 6, 3, 2, 0, (2, 1, 0), lanes=8),
    TreeLayout(1, 6, 21, 6, 0, 6, tuple(range(5, -1, -1)), lanes=8),
    TreeLayout(44, 43, 434, 18, 10, 6, tuple(range(42, -1, -1))),
    TreeLayout(
        5,
        9,
        18,
        9,
        3,
        0,
        tuple(range(8, -1, -1)),
        roots=(0, 4),
        root_coordinates=(0, 6),
        root_offsets=(0, 3),
        root_families=(3, 6),
    ),
)
_STAGES = ("accelerations", "transport", "integrate", "relative_poses", "poses", "motion", "publish", "close")


def _kernels(layout):
    """The finish and every repair form: ordinary, alternate-motion, listed (with each motion) and bank-comparing."""
    state = tree_kinetic_state.get_state_kernel
    yield "finish", state(True, layout=layout)
    yield "repair", state(False, layout=layout)
    yield "repair_motion", state(False, layout=layout, motion_velocity=True)
    yield "repair_listed", state(False, layout=layout, listed=True)
    yield "repair_listed_motion", state(False, layout=layout, motion_velocity=True, listed=True)
    # A group's coordinate count: a free root's quaternion takes one more than its three rotation velocities.
    yield "repair_banked", state(False, layout=layout, bank_coordinates=layout.dofs + layout.free_roots)


class TestTreeKineticStages(unittest.TestCase):
    def test_kernels_are_authored_code(self):
        """Build every finish and repair from functions written in the module, not from executed text."""
        source = Path(tree_kinetic_state.__file__).resolve()
        for layout in _LAYOUTS:
            for name, kernel in _kernels(layout):
                with self.subTest(layout=layout.bodies, kernel=name):
                    self.assertEqual(Path(kernel.func.__code__.co_filename).resolve(), source)

    def test_stages_read_one_tree_group(self):
        """Call each stage on the TreeGroup admission returns, one call below the kernel."""
        for layout in _LAYOUTS:
            for name, kernel in _kernels(layout):
                with self.subTest(layout=layout.bodies, kernel=name):
                    functions = {f.func.__name__: f for f in kernel.adj.get_references()[2] if not f.is_builtin()}
                    self.assertIs(functions["admit"].func.__annotations__["return"], tree_kinetic_state.TreeGroup)
                    for stage in _STAGES:
                        arguments = list(functions[stage].input_types.values())
                        self.assertIs(arguments[0], tree_kinetic_state.TreeGroup, stage)

    def test_finish_and_repair_share_one_launch_abi(self):
        """Take every repair form's inputs, after the six common arguments, in every finish and repair."""
        for layout in _LAYOUTS:
            labels = {name: [arg.label for arg in kernel.adj.args] for name, kernel in _kernels(layout)}
            with self.subTest(layout=layout.bodies):
                self.assertEqual(len({tuple(args) for args in labels.values()}), 1)
                self.assertEqual(labels["finish"][6:], ["motion_qd", "repair_blocks", "repair_count", "bank"])


if __name__ == "__main__":
    unittest.main()
