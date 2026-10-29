# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The held-tree finish and repair are authored stage functions over one tree-group operand."""

import re
import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import tree_kinetic_state
from newton._src.solvers.feather_pgs import tree_scan as ts
from newton._src.solvers.feather_pgs.held_tree import HeldTree
from newton._src.solvers.feather_pgs.publication_data import PublicationData
from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS
from newton._src.solvers.feather_pgs.tree_kinetic_layout import TreeLayout
from newton.tests.test_feather_pgs_mixed_tree_free_body import DT, _build

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


def _schedules(layout):
    """Schedules a layout's lanes run: shuffles and intervals within a warp, shared memory and levels over two."""
    schedule = ts.Schedule
    if layout.lanes <= 32:
        return tree_kinetic_state.HeldSchedules(
            schedule.JUMP_SHFL, schedule.JUMP_SHFL, schedule.DFS_INTERVAL, schedule.LEVEL
        )
    return tree_kinetic_state.HeldSchedules(schedule.JUMP_SMEM, schedule.JUMP_SMEM, schedule.LEVEL, schedule.LEVEL)


_ADMISSION = re.compile(r"admit_\d+\(")


def _kernels(layout):
    """The finish and every repair form: ordinary, alternate-motion, listed and bank-comparing.

    The listed and bank-comparing repairs are contracts of ordinary motion: owners keep the one-pass
    repair of a motion-velocity layout.
    """
    state, schedules = tree_kinetic_state.get_state_kernel, _schedules(layout)
    yield "finish", state(True, schedules, layout=layout)
    yield "repair", state(False, schedules, layout=layout)
    yield "repair_motion", state(False, schedules, layout=layout, motion_velocity=True)
    yield "repair_listed", state(False, schedules, layout=layout, listed=True)
    # A group's coordinate count: a free root's quaternion takes one more than its three rotation velocities.
    yield "repair_banked", state(False, schedules, layout=layout, bank_coordinates=layout.dofs + layout.free_roots)


def _input_uses(kernel, name="motion_qd"):
    """Statements of a kernel's generated CPU code that use launch input ``name`` other than to declare or admit it."""
    from warp._src.context import ModuleBuilder, ModuleHasher  # noqa: PLC0415

    module = kernel.module
    module.get_module_hash()
    options = dict(module.resolve_options(wp.config), output_arch=None)
    source = ModuleBuilder(module, options, hasher=ModuleHasher([kernel], options)).codegen("cpu")
    # The launch argument and the parameters and fields that declare the input.
    declaration = re.compile(rf"((wp::array_t<wp::\w+>|StateBank_\w+)\s*&?\s*)?(var_|adj_)?{name}\s*[,);]?")
    word = re.compile(rf"\b{name}\b|\bvar_{name}\b")
    uses = []
    for line in source.splitlines():
        code = line.split("//", 1)[0].strip()
        if not word.search(code) or f"_wp_args->{name}" in code or _ADMISSION.search(code):
            continue
        if not declaration.fullmatch(code):
            uses.append(code)
    return uses


def _state_arrays(group, data=None):
    """Every array a finish or repair of ``group`` can write: its cache's, and its publication's when given."""
    structs = [("cache", group.data, tree_kinetic_state.KineticData)]
    if data is not None:
        structs.append(("data", data, PublicationData))
    arrays = {}
    for prefix, instance, struct in structs:
        for name in struct.vars:
            value = getattr(instance, name)
            if isinstance(value, wp.array) and value.size:
                arrays[f"{prefix}.{name}"] = value
    return arrays


def _launch(group, kernel, data, velocity, start):
    """Launch a state kernel from the arrays in ``start`` with ``velocity`` as its motion input; return what it left."""
    arrays = _state_arrays(group, data)
    for name, array in arrays.items():
        array.assign(start[name])
    wp.launch_tiled(
        kernel,
        dim=[group.launch_count],
        inputs=[
            group.sparse.plan,
            group.plan,
            data,
            group.data,
            group.solver._mass_update_requested,
            1,
            velocity,
            group._repair_blocks,
            group._list_counts,
            group._no_bank,
        ],
        block_dim=group.block_dim,
        device=group.solver.model.device,
    )
    # A CPU array's numpy() is a view; copy what the launch left.
    return {name: array.numpy().copy() for name, array in arrays.items()}


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

    def test_only_the_alternate_motion_repair_reads_the_motion_input(self):
        """Generate no use of the motion input but its hand-off to admission, except in the alternate-motion repair."""
        wp.init()
        for layout in _LAYOUTS:
            for name, kernel in _kernels(layout):
                with self.subTest(layout=layout.bodies, kernel=name):
                    uses = _input_uses(kernel)
                    if name == "repair_motion":
                        self.assertTrue(any("different_motion" in use for use in uses), uses)
                    else:
                        self.assertEqual(uses, [])

    def test_listed_and_comparing_repairs_refuse_an_alternate_motion(self):
        """Build no listed or bank-comparing repair of an alternate motion, which no owner launches."""
        state = tree_kinetic_state.get_state_kernel
        for layout in _LAYOUTS:
            schedules = _schedules(layout)
            with self.subTest(layout=layout.bodies):
                with self.assertRaisesRegex(ValueError, "ordinary motion"):
                    state(False, schedules, layout=layout, motion_velocity=True, listed=True)
                with self.assertRaisesRegex(ValueError, "ordinary motion"):
                    state(
                        False,
                        schedules,
                        layout=layout,
                        motion_velocity=True,
                        bank_coordinates=layout.dofs + layout.free_roots,
                    )

    def test_only_listed_and_comparing_repairs_read_their_inputs(self):
        """Read the block list only in the listed repair and the bank only in the repair that compares banks."""
        wp.init()
        readers = {
            "repair_blocks": ("repair_listed",),
            "repair_count": ("repair_listed",),
            "bank": ("repair_banked",),
        }
        for layout in _LAYOUTS:
            for name, kernel in _kernels(layout):
                for argument, forms in readers.items():
                    with self.subTest(layout=layout.bodies, kernel=name, input=argument):
                        uses = _input_uses(kernel, argument)
                        if name in forms:
                            self.assertTrue(uses)
                        else:
                            self.assertEqual(uses, [])

    def test_owners_without_alternate_motion_reject_another_velocity(self):
        """Refuse a velocity other than the state's in begin and predict, before any cache changes."""
        model = _build(wp.get_device("cpu"), worlds=2)
        solver = SolverFeatherPGS(model, friction_anchor_beta=0.0)
        owner = HeldTree(solver)
        state, output, control = model.state(), model.state(), model.control()
        augmented = solver._prepare_augmented_state(state, output, control)
        other = wp.clone(state.joint_qd)
        self.assertTrue(owner.groups)
        for group in owner.groups:
            with self.subTest(layout=group.layout.root_families):
                self.assertFalse(group.motion_velocity)
                before = {name: array.numpy().copy() for name, array in _state_arrays(group).items()}
                with self.assertRaisesRegex(RuntimeError, "excludes alternate prescaled predictor velocity"):
                    group.begin(state, augmented, other, DT, True)
                with self.assertRaisesRegex(RuntimeError, "excludes alternate prescaled predictor velocity"):
                    group.predict(state, augmented, control, other, DT)
                for name, array in _state_arrays(group).items():
                    np.testing.assert_array_equal(array.numpy(), before[name], err_msg=name)

    def check_motion_input_ignored(self, device):
        """Launch each finish and ordinary repair with the state's velocity and with NaN as the motion input.

        Every launch must rebuild its groups' caches (a new generation), so
        that equal outputs show an unread input rather than an early exit. The
        repairs are admitted because the caches are stamped with the bank the
        preceding finish published into, not with the input bank.
        """
        device = wp.get_device(device)
        for alternate in (False, True):
            model = _build(device, worlds=2, speed_limit=0.8 if alternate else None)
            solver = SolverFeatherPGS(model, friction_anchor_beta=0.0)
            solver.enable_joint_velocity_limits = alternate
            owner = HeldTree(solver)
            state, output, control = model.state(), model.state(), model.control()
            rng = np.random.default_rng(41)
            state.joint_qd.assign(rng.uniform(-0.4, 0.4, model.joint_dof_count).astype(np.float32))
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            augmented = solver._prepare_augmented_state(state, output, control)
            nan = wp.full(state.joint_qd.shape, float("nan"), dtype=float, device=device)
            for group in owner.groups:
                for name, kernel, published in (
                    ("finish", group.finish_kernel, output),
                    ("repair", group.repair_kernel, state),
                ):
                    with self.subTest(alternate=alternate, layout=group.layout.root_families, kernel=name):
                        self.assertEqual(group.motion_velocity, alternate)
                        data = group._publication(state, augmented, published, DT)
                        start = {key: array.numpy().copy() for key, array in _state_arrays(group, data).items()}
                        plain = _launch(group, kernel, data, state.joint_qd, start)
                        poisoned = _launch(group, kernel, data, nan, start)
                        for launched in (plain, poisoned):
                            self.assertTrue((launched["cache.generation"] != start["cache.generation"]).all())
                        differ = [key for key in plain if plain[key].tobytes() != poisoned[key].tobytes()]
                        if alternate and name == "repair":
                            self.assertTrue(differ)
                        else:
                            self.assertEqual(differ, [])

    def test_finish_and_ordinary_repair_ignore_the_motion_input_cpu(self):
        """Leave bitwise-equal state after a CPU finish or ordinary repair whose motion input holds NaN.

        An alternate-motion repair reads the input, so its NaN shows: the
        comparison sees a read. CPU owners launch their kernels with static
        scratch.
        """
        self.check_motion_input_ignored("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA unavailable")
    def test_finish_and_ordinary_repair_ignore_the_motion_input_cuda(self):
        """Leave bitwise-equal state after a CUDA finish or ordinary repair whose motion input holds NaN."""
        self.check_motion_input_ignored("cuda:0")


if __name__ == "__main__":
    unittest.main()
