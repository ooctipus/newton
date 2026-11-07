# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The held-tree finish and repair are authored stage functions over one tree-group operand."""

import dataclasses
import re
import unittest
from unittest.mock import patch

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


def _schedules(layout):
    """Schedules a layout's lanes run: shuffles and intervals within a warp, shared memory and levels over two."""
    schedule = ts.Schedule
    if layout.lanes <= 32:
        return tree_kinetic_state.HeldSchedules(schedule.JUMP_SHFL, schedule.JUMP_SHFL, schedule.DFS_INTERVAL)
    return tree_kinetic_state.HeldSchedules(schedule.JUMP_SMEM, schedule.JUMP_SMEM, schedule.LEVEL)


_ADMISSION = re.compile(r"admit_\d+\(")


def _kernels(layout):
    """The finish and every repair form: ordinary, alternate-motion and bank-comparing.

    The bank-comparing repair is a contract of ordinary motion.
    """
    state, schedules = tree_kinetic_state.get_state_kernel, _schedules(layout)
    yield "finish", state(True, schedules, layout=layout)
    yield "repair", state(False, schedules, layout=layout)
    yield "repair_motion", state(False, schedules, layout=layout, motion_velocity=True)
    # A group's coordinate count: a free root's quaternion takes one more than its three rotation velocities.
    yield "repair_banked", state(False, schedules, layout=layout, bank_coordinates=layout.dofs + layout.free_root_count)


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
            group.solver.factor.mass_update_requested,
            group.solver.factor.clock,
            velocity,
            group._no_bank,
        ],
        block_dim=group.block_dim,
        device=group.solver.model.device,
    )
    # A CPU array's numpy() is a view; copy what the launch left.
    return {name: array.numpy().copy() for name, array in arrays.items()}


class TestTreeKineticStages(unittest.TestCase):
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

    def test_comparing_repair_refuses_an_alternate_motion(self):
        """Build no bank-comparing repair of an alternate motion, which no owner launches."""
        state = tree_kinetic_state.get_state_kernel
        for layout in _LAYOUTS:
            schedules = _schedules(layout)
            with self.subTest(layout=layout.bodies):
                with self.assertRaisesRegex(ValueError, "ordinary motion"):
                    state(
                        False,
                        schedules,
                        layout=layout,
                        motion_velocity=True,
                        bank_coordinates=layout.dofs + layout.free_root_count,
                    )

    def test_only_the_comparing_repair_reads_the_bank(self):
        """Read the bank only in the repair that compares banks."""
        wp.init()
        for layout in _LAYOUTS:
            for name, kernel in _kernels(layout):
                with self.subTest(layout=layout.bodies, kernel=name):
                    uses = _input_uses(kernel, "bank")
                    if name == "repair_banked":
                        self.assertTrue(uses)
                    else:
                        self.assertEqual(uses, [])

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
            solver = SolverFeatherPGS(model)
            solver.options = solver.laws.options = solver.factor.options = solver.placement.options = (
                solver.sweep.options
            ) = dataclasses.replace(solver.options, enable_joint_velocity_limits=alternate)
            owner = HeldTree(solver, solver.spec.classes.held)
            state, output = model.state(), model.state()
            rng = np.random.default_rng(41)
            state.joint_qd.assign(rng.uniform(-0.4, 0.4, model.joint_dof_count).astype(np.float32))
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            nan = wp.full(state.joint_qd.shape, float("nan"), dtype=float, device=device)
            for group in owner.groups:
                for name, kernel, published in (
                    ("finish", group.finish_kernel, output),
                    ("repair", group.repair_kernel, state),
                ):
                    with self.subTest(alternate=alternate, layout=group.layout.root_families, kernel=name):
                        self.assertEqual(group.motion_velocity, alternate)
                        data = group._publication(state, solver.factor, published, DT)
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


def _keys(device, *, based, worlds=4, keys=8):
    """``worlds`` worlds of ``keys`` one-coordinate trees each, a prismatic key a tree with a drive, limits and armature:
    one body on the world, or (``based``) a fixed base under it, two bodies; random coordinates in and past the limits."""
    template = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    for k in range(keys):
        frame = wp.transform(wp.vec3(0.02 * k, 0.0, 0.1), wp.quat_rpy(0.1 * k, -0.2, 0.3))
        key = wp.transform(wp.vec3(0.0, 0.01, 0.01), wp.quat_rpy(0.2, 0.1 * k, 0.0))
        body = template.add_link(
            mass=0.01 + 1.0e-3 * k,
            com=wp.vec3(0.0005, -0.0003, 0.002),
            inertia=wp.mat33(np.diag([2.0e-6, 2.0e-6, 3.0e-6])),
        )
        joints, parent, anchor = [], -1, wp.transform_multiply(frame, key)
        if based:
            parent = template.add_link(mass=0.2, com=wp.vec3(0.0, 0.0, 0.01), inertia=wp.mat33(np.eye(3) * 1.0e-4))
            joints.append(template.add_joint_fixed(parent=-1, child=parent, parent_xform=frame))
            anchor = key
        joints.append(
            template.add_joint_prismatic(
                parent=parent,
                child=body,
                parent_xform=anchor,
                axis=wp.vec3(0.2 + 0.1 * k, 0.5, 0.9),
                limit_lower=-0.004,
                limit_upper=0.0,
                target_ke=300.0 + 10.0 * k,
                target_kd=1.5,
                armature=1.0e-4,
            )
        )
        template.add_articulation(joints)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    builder.replicate(template, worlds, spacing=(1.0, 0.0, 0.0))
    model = builder.finalize(device=device)
    rng = np.random.default_rng(5)
    model.joint_q.assign(rng.uniform(-0.005, 0.001, model.joint_coord_count).astype(np.float32))
    model.joint_qd.assign(rng.uniform(-0.05, 0.05, model.joint_dof_count).astype(np.float32))
    return model


class TestNarrowTrees(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "CUDA unavailable")
    def test_cuda_one_and_two_lane_trees_match_eight_lanes(self):
        """Step one-body trees at one lane and two-body trees at two lanes bitwise as at eight.

        A narrow tree packs 32 or 16 trees a warp, a thread or two a tree: every thread of the block is its tree's
        lane, so the repair and the finish admit every tree (the CPU's one-thread group exits its block's other
        logical lanes; a one-lane tree's thread is no such lane).
        """
        for based, narrow in ((False, 1), (True, 2)):
            states = {}
            for lanes in (narrow, 8):
                model = _keys("cuda:0", based=based)
                with patch.object(tree_kinetic_state, "held_lanes", lambda tree, groups, device, lanes=lanes: lanes):
                    solver = SolverFeatherPGS(
                        model, pgs_iterations=8, enable_joint_limits=True, update_mass_matrix_interval=2
                    )
                groups = solver.factor.held.groups
                self.assertEqual(
                    {(group.layout.lanes, group.group_count % group.layout.pack) for group in groups}, {(lanes, 0)}
                )
                pair = [model.state(), model.state()]
                control = model.control()
                for _ in range(12):
                    solver.step(pair[0], pair[1], control, None, DT)
                    pair.reverse()
                states[lanes] = {
                    name: getattr(pair[0], name).numpy() for name in ("joint_q", "joint_qd", "body_q", "body_qd")
                }
            with self.subTest(based=based, lanes=narrow):
                for name, array in states[8].items():
                    np.testing.assert_array_equal(states[narrow][name], array, err_msg=name)


if __name__ == "__main__":
    unittest.main()
