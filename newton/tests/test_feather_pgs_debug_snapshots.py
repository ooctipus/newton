# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The per-substep stage-3 snapshots are opt-in diagnostics that never change physics.

Without ``FEATHER_PGS_DEBUG_SNAPSHOTS=1`` or ``pgs_debug=True`` the solver
allocates no snapshot and launches no snapshot copy. With either, the external
NaN dumpers read the same attributes, shapes and values as before the switch
existed, and the snapshots hold no gradient, so a tape records the same step.
"""

import ctypes
import os
import sys
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_test_devices

FLAG = "FEATHER_PGS_DEBUG_SNAPSHOTS"
# In copy order: the joint velocity the predictor starts from, the unconstrained
# joint acceleration and the predicted velocity before contacts.
SNAPSHOTS = ("_debug_stage3_qd_work", "_debug_stage3_joint_qdd", "_debug_stage3_v_hat")
STATE_FIELDS = ("joint_q", "joint_qd", "body_q", "body_qd")
DT = 1.0 / 240.0
# Driver node types (CUgraphNodeType) of a captured CUDA graph.
KERNEL_NODE, MEMCPY_NODE, MEMSET_NODE = 0, 1, 2
# Separate CUDA solver instances may order float atomic accumulations differently,
# so they compare within the repository's cross-instance bound; CPU runs serially.
CUDA_INSTANCE_ATOL = 2.0e-6


@contextmanager
def _snapshot_flag(enabled):
    """Set ``FEATHER_PGS_DEBUG_SNAPSHOTS=1``, or leave it unset, for the duration.

    The held-tree lane width is pinned to one tree per block: by default each
    held-tree group times its candidate widths at construction, so two CUDA
    instances could launch different kernels.
    """
    with patch.dict(os.environ, {"FEATHER_PGS_TREE_LANE_PACK": "0"}):
        os.environ.pop(FLAG, None)
        if enabled:
            os.environ[FLAG] = "1"
        yield


def _assert_same(test, device, actual, expected, label):
    """Require bitwise equality on CPU and the cross-instance bound between CUDA instances."""
    test.assertTrue(np.all(np.isfinite(expected)), label)
    if wp.get_device(device).is_cuda:
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=CUDA_INSTANCE_ATOL, err_msg=label)
    else:
        np.testing.assert_array_equal(actual, expected, err_msg=label)


def _scene(device, worlds=2):
    """A fixed-base arm whose tip touches the ground, a floating two-link body and a free box, per world."""
    builder = newton.ModelBuilder()
    for _world in range(worlds):
        builder.begin_world()
        parent, joints = -1, []
        for link in range(2):
            body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.05), lock_inertia=True)
            joints.append(
                builder.add_joint_revolute(
                    parent=parent,
                    child=body,
                    axis=newton.Axis.Y,
                    parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.42 if link == 0 else -0.2), wp.quat_identity()),
                    target_ke=5.0,
                    target_kd=1.0,
                    velocity_limit=1.5,
                )
            )
            parent = body
        builder.add_shape_sphere(parent, radius=0.03, xform=wp.transform(wp.vec3(0.0, 0.0, -0.2), wp.quat_identity()))
        builder.add_articulation(joints)
        root = builder.add_link(xform=wp.transform(wp.vec3(0.6, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_box(root, hx=0.1, hy=0.1, hz=0.1)
        tail = builder.add_link()
        builder.add_shape_box(tail, hx=0.1, hy=0.03, hz=0.03)
        builder.add_articulation(
            [
                builder.add_joint_free(child=root),
                builder.add_joint_revolute(
                    parent=root,
                    child=tail,
                    axis=newton.Axis.Z,
                    parent_xform=wp.transform(wp.vec3(0.15, 0.0, 0.0), wp.quat_identity()),
                    child_xform=wp.transform(wp.vec3(-0.1, 0.0, 0.0), wp.quat_identity()),
                ),
            ]
        )
        box = builder.add_body(xform=wp.transform(wp.vec3(-0.6, 0.0, 0.06), wp.quat_identity()))
        builder.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05)
        builder.end_world()
    builder.add_ground_plane()
    # Free joints with the legacy DOF-shaped target layout warn at finalize.
    with patch.object(newton, "use_coord_layout_targets", True):
        return builder.finalize(device=device)


@wp.kernel
def _sum_of_squares(values: wp.array[float], total: wp.array[float]):
    i = wp.tid()
    wp.atomic_add(total, 0, values[i] * values[i])


def _gradient_arm(device):
    """A fixed-base two-link arm whose model records gradients."""
    builder = newton.ModelBuilder()
    parent, joints = -1, []
    for link in range(2):
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.05), lock_inertia=True)
        joints.append(
            builder.add_joint_revolute(
                parent=parent,
                child=body,
                axis=newton.Axis.Y,
                parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0 if link == 0 else -0.2), wp.quat_identity()),
                target_ke=5.0,
                target_kd=1.0,
            )
        )
        parent = body
    builder.add_articulation(joints)
    return builder.finalize(device=device, requires_grad=True)


def _taped_step(model, *, snapshots):
    """Step once under a tape, then back-propagate a loss on the input and output joint velocities.

    The loss reads the input velocity itself, so its gradient reaches the input
    however much of it flows back through the step.
    """
    with _snapshot_flag(snapshots):
        solver = SolverFeatherPGS(model)
    state_in, state_out = model.state(requires_grad=True), model.state(requires_grad=True)
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    state_in.joint_qd.assign(np.array([0.3, -0.5], dtype=np.float32))
    loss = wp.zeros(1, dtype=float, requires_grad=True, device=model.device)
    tape = wp.Tape()
    with tape:
        solver.step(state_in, state_out, model.control(), None, DT)
        for velocity in (state_in.joint_qd, state_out.joint_qd):
            wp.launch(_sum_of_squares, dim=velocity.shape[0], inputs=[velocity], outputs=[loss], device=model.device)
    tape.backward(loss)
    return {
        "solver": solver,
        "tape_entries": len(tape.launches),
        "joint_qd_out": state_out.joint_qd.numpy().copy(),
        "loss": loss.numpy().copy(),
        "joint_qd_grad": state_in.joint_qd.grad.numpy().copy(),
        "joint_q_grad": state_in.joint_q.grad.numpy().copy(),
    }


def _is_snapshot(solver, array):
    return any(array is not None and array is getattr(solver, name, None) for name in SNAPSHOTS)


def _signature(array):
    return (tuple(array.shape), array.dtype, str(array.device))


def _rollout(model, *, snapshots, steps=6, **solver_kwargs):
    """Step eagerly through the collision pipeline, recording every copy and kernel launch per step.

    Returns the solver, the states after each step, and per step the copies
    (destination, source, the source's values when the destination is a
    snapshot) and the launched kernel keys.
    """
    with _snapshot_flag(snapshots):
        solver = SolverFeatherPGS(model, **solver_kwargs)
    pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = pipeline.contacts()
    states = [model.state() for _ in range(2)]
    newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
    states[0].joint_qd.assign(np.linspace(-0.6, 0.6, model.joint_dof_count, dtype=np.float32))
    control = model.control()
    copies, launches, inputs, trajectory = [], [], [], []
    copy, launch, launch_tiled = wp.copy, wp.launch, wp.launch_tiled
    reading = False

    def record_copy(dest, src, *args, **kwargs):
        nonlocal reading
        result = copy(dest, src, *args, **kwargs)
        if reading:
            return result  # a CUDA array's numpy() below copies to the host through wp.copy
        values = None
        if _is_snapshot(solver, dest):
            reading = True
            try:
                values = src.numpy().copy()
            finally:
                reading = False
        copies[-1].append((dest, src, values))
        return result

    def recording(original):
        def record(kernel, *args, **kwargs):
            launches[-1].append(getattr(kernel, "key", None))
            return original(kernel, *args, **kwargs)

        return record

    spies = {"copy": record_copy, "launch": recording(launch), "launch_tiled": recording(launch_tiled)}
    for _ in range(steps):
        copies.append([])
        launches.append([])
        pipeline.collide(states[0], contacts)
        states[0].clear_forces()
        inputs.append(states[0])
        with patch.multiple(wp, **spies):
            solver.step(states[0], states[1], control, contacts, DT)
        trajectory.append({name: getattr(states[1], name).numpy().copy() for name in STATE_FIELDS})
        trajectory[-1]["impulses"] = solver.impulses.numpy().copy()
        states.reverse()
    return {
        "solver": solver,
        "copies": copies,
        "launches": launches,
        "inputs": inputs,
        "trajectory": trajectory,
        "contacts": int(contacts.rigid_contact_count.numpy()[0]),
    }


def _configurations(device):
    configurations = {"patch": {}, "point": {"friction_anchor_beta": 0.0}}
    if wp.get_device(device).is_cuda:
        configurations["matrix_free"] = {"pgs_mode": "matrix_free"}
        # Velocity limits need matrix-free mode; the predictor then starts from the prescaled qd_work.
        configurations["velocity_limits"] = {"pgs_mode": "matrix_free", "enable_joint_velocity_limits": True}
    return configurations


def test_default_skips_snapshot_copies(test, device):
    """Allocate and copy no stage-3 snapshot unless the flag is set; launch everything else unchanged."""
    model = _scene(device)
    for label, options in _configurations(device).items():
        with test.subTest(configuration=label):
            off = _rollout(model, snapshots=False, **options)
            on = _rollout(model, snapshots=True, **options)
            solver = off["solver"]
            for name in SNAPSHOTS:
                test.assertIsNone(getattr(solver, name), name)
            for step, (off_copies, on_copies) in enumerate(zip(off["copies"], on["copies"], strict=True)):
                # The flag adds exactly the three snapshot copies and nothing else.
                kept = [
                    (_signature(dest), _signature(src))
                    for dest, src, _ in on_copies
                    if not _is_snapshot(on["solver"], dest)
                ]
                test.assertEqual([(_signature(dest), _signature(src)) for dest, src, _ in off_copies], kept, step)
                test.assertEqual(len(on_copies) - len(kept), len(SNAPSHOTS), step)
            test.assertEqual(off["launches"], on["launches"])
            selected = solver.get_path_selection()
            record = selected["routes"]["diagnostics"]["stage3_snapshots"]
            test.assertFalse(record["enabled"])
            test.assertIn(FLAG, record["reason"])
            test.assertNotIn(FLAG, selected["environment"]["applied"])


def test_opt_in_keeps_snapshot_attributes(test, device):
    """Copy the stage-3 velocity, acceleration and prediction into the dumper's attributes every substep."""
    model = _scene(device)
    for label, options in _configurations(device).items():
        with test.subTest(configuration=label):
            run = _rollout(model, snapshots=True, **options)
            solver = run["solver"]
            for name in SNAPSHOTS:
                snapshot = getattr(solver, name)
                test.assertIsInstance(snapshot, wp.array, name)
                test.assertEqual(snapshot.shape, model.joint_qd.shape, name)
                test.assertEqual(snapshot.dtype, wp.float32, name)
                test.assertEqual(snapshot.device, model.device, name)
            for step, step_copies in enumerate(run["copies"]):
                taken = [(dest, src) for dest, src, _ in step_copies if _is_snapshot(solver, dest)]
                # Velocity limits start the predictor from the prescaled qd_work, else from the input state.
                velocity = solver.qd_work if solver.enable_joint_velocity_limits else run["inputs"][step].joint_qd
                expected = [velocity, solver.joint_qdd, solver.v_hat]
                test.assertEqual(len(taken), len(SNAPSHOTS), step)
                for (dest, src), name, source in zip(taken, SNAPSHOTS, expected, strict=True):
                    test.assertIs(dest, getattr(solver, name), (step, name))
                    test.assertIs(src, source, (step, name))
            # Nothing later in the step overwrites them: they hold the last step's stage-3 values.
            last = [values for dest, _, values in run["copies"][-1] if _is_snapshot(solver, dest)]
            for name, values in zip(SNAPSHOTS, last, strict=True):
                np.testing.assert_array_equal(getattr(solver, name).numpy(), values, err_msg=name)
            test.assertTrue(np.all(np.isfinite(last[-1])))
            test.assertGreater(float(np.abs(last[-1]).max()), 0.0)
            selected = solver.get_path_selection()
            record = selected["routes"]["diagnostics"]["stage3_snapshots"]
            test.assertTrue(record["enabled"])
            test.assertIn(FLAG, record["reason"])
            test.assertEqual(selected["environment"]["applied"].get(FLAG), "1")


def test_snapshots_leave_physics_unchanged(test, device):
    """Step every configuration to the same states and impulses with and without the snapshots, bitwise on CPU."""
    model = _scene(device)
    for label, options in _configurations(device).items():
        with test.subTest(configuration=label):
            off = _rollout(model, snapshots=False, **options)
            on = _rollout(model, snapshots=True, **options)
            test.assertGreater(off["contacts"], 0)
            test.assertEqual(off["contacts"], on["contacts"])
            for step, (expected, actual) in enumerate(zip(off["trajectory"], on["trajectory"], strict=True)):
                for name, values in expected.items():
                    _assert_same(test, device, actual[name], values, f"step {step} {name}")
            moved = np.abs(off["trajectory"][-1]["joint_q"] - model.joint_q.numpy()).max()
            test.assertGreater(float(moved), 1.0e-4, "the scene must move")
            test.assertGreater(float(np.abs(off["trajectory"][-1]["impulses"]).max()), 0.0)


class TestFeatherPGSDebugSnapshots(unittest.TestCase):
    def test_flag_is_declared_as_a_diagnostic(self):
        """Declare the switch in the FeatherPGS registry as a diagnostic the solver reads."""
        from newton._src.solvers.feather_pgs import environment_flags  # noqa: PLC0415

        flag = environment_flags.FLAGS[FLAG]
        self.assertEqual(flag.kind, "diagnostic")
        self.assertEqual(flag.readers, ("solvers.feather_pgs.solver_feather_pgs",))

    def test_flag_other_than_one_keeps_snapshots_off(self):
        """Keep the snapshots off for every value of the flag except 1."""
        model = _scene("cpu", worlds=1)
        for value in ("0", "", "true", "yes"):
            with self.subTest(value=value), patch.dict(os.environ, {FLAG: value}):
                solver = SolverFeatherPGS(model)
                for name in SNAPSHOTS:
                    self.assertIsNone(getattr(solver, name), name)
                self.assertFalse(solver.get_path_selection()["routes"]["diagnostics"]["stage3_snapshots"]["enabled"])

    def test_model_without_joints_allocates_no_snapshots(self):
        """Allocate no snapshot and step particles when the flag is set but the model has no joints."""
        from newton._src.solvers.feather_pgs import path_selection  # noqa: PLC0415

        builder = newton.ModelBuilder()
        builder.add_particle(pos=(0.0, 0.0, 1.0), vel=(0.0, 0.0, 0.0), mass=1.0)
        model = builder.finalize(device="cpu")
        with _snapshot_flag(True):
            solver = SolverFeatherPGS(model)
        for name in SNAPSHOTS:
            self.assertIsNone(getattr(solver, name), name)
        # get_path_selection() cannot describe a model without joints yet: its friction
        # record reads the patch pipeline, which such a model never builds.
        record = path_selection._diagnostics(solver)["stage3_snapshots"]
        self.assertFalse(record["enabled"])
        self.assertEqual(record["reason"], f"{FLAG}=1, but the model has no joints to snapshot.")
        state_in, state_out = model.state(), model.state()
        solver.step(state_in, state_out, None, None, DT)
        self.assertTrue(np.all(np.isfinite(state_out.particle_q.numpy())))

    def test_pgs_debug_keeps_snapshots(self):
        """Copy the snapshots under pgs_debug without the flag, and record that pgs_debug enabled them."""
        model = _scene("cpu", worlds=1)
        run = _rollout(model, snapshots=False, steps=2, pgs_debug=True)
        solver = run["solver"]
        for name in SNAPSHOTS:
            self.assertIsInstance(getattr(solver, name), wp.array, name)
        for step, step_copies in enumerate(run["copies"]):
            taken = [dest for dest, _, _ in step_copies if _is_snapshot(solver, dest)]
            self.assertEqual(len(taken), len(SNAPSHOTS), step)
            for dest, name in zip(taken, SNAPSHOTS, strict=True):
                self.assertIs(dest, getattr(solver, name), (step, name))
        selected = solver.get_path_selection()
        record = selected["routes"]["diagnostics"]["stage3_snapshots"]
        self.assertTrue(record["enabled"])
        self.assertTrue(record["reason"].startswith("pgs_debug=True: "), record["reason"])
        self.assertNotIn(FLAG, selected["environment"]["applied"])

    def test_snapshots_record_no_gradients(self):
        """Leave a tape's record and its input gradients unchanged when the flag keeps the snapshots.

        The snapshots hold no gradient, so wp.copy neither copies the sources'
        gradients nor records an adjoint copy, which in the backward pass would
        overwrite the sources' gradients with the snapshots' zeros. CPU covers
        it: wp.copy's gradient handling is the same on every device.
        """
        model = _gradient_arm("cpu")
        runs = {snapshots: _taped_step(model, snapshots=snapshots) for snapshots in (False, True)}
        solver = runs[True]["solver"]
        for name in SNAPSHOTS:
            snapshot = getattr(solver, name)
            self.assertIsInstance(snapshot, wp.array, name)
            self.assertFalse(snapshot.requires_grad, name)
            self.assertIsNone(snapshot.grad, name)
        self.assertEqual(runs[True]["tape_entries"], runs[False]["tape_entries"])
        for name in ("joint_qd_out", "loss", "joint_qd_grad", "joint_q_grad"):
            np.testing.assert_array_equal(runs[True][name], runs[False][name], err_msg=name)
        # The loss's own term, 2 * qd_in, reaches the input: an overwrite would show.
        self.assertTrue(np.all(np.isfinite(runs[False]["joint_qd_grad"])))
        self.assertGreater(float(np.abs(runs[False]["joint_qd_grad"]).max()), 0.1)


def _graph_node_types(graph):
    """Count a captured CUDA graph's top-level nodes by driver node type, or None without a driver query."""
    try:
        library = ctypes.CDLL("nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1")
        get_nodes, get_type = library.cuGraphGetNodes, library.cuGraphNodeGetType
    except (OSError, AttributeError):
        return None
    get_nodes.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t)]
    get_type.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    get_nodes.restype = get_type.restype = ctypes.c_int
    count = ctypes.c_size_t()
    if get_nodes(graph.graph, None, ctypes.byref(count)) != 0:
        return None
    nodes = (ctypes.c_void_p * count.value)()
    if count.value and get_nodes(graph.graph, nodes, ctypes.byref(count)) != 0:
        return None
    types = {}
    for node in nodes[: count.value]:
        kind = ctypes.c_int()
        if get_type(node, ctypes.byref(kind)) != 0:
            return None
        types[kind.value] = types.get(kind.value, 0) + 1
    return types


def _graph_rollout(model, *, snapshots, steps=12, **solver_kwargs):
    """Capture one substep per state order, replay them, and record each graph's nodes and captured copies."""
    device = model.device
    with _snapshot_flag(snapshots):
        solver = SolverFeatherPGS(model, **solver_kwargs)
    pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = pipeline.contacts()
    states = [model.state() for _ in range(2)]
    newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
    states[0].joint_qd.assign(np.linspace(-0.6, 0.6, model.joint_dof_count, dtype=np.float32))
    control = model.control()

    def substep(state_in, state_out):
        state_in.clear_forces()
        solver.step(state_in, state_out, control, contacts, DT)

    # Compile and initialize the layouts before capture.
    pipeline.collide(states[0], contacts)
    substep(states[0], states[1])
    states.reverse()
    copy, copies, graphs = wp.copy, [], []

    def record_copy(dest, src, *args, **kwargs):
        copies[-1].append((dest, src))
        return copy(dest, src, *args, **kwargs)

    with patch.object(wp, "copy", record_copy):
        for order in ((states[0], states[1]), (states[1], states[0])):
            copies.append([])
            with wp.ScopedCapture(device=device) as capture:
                solver.seed_double_buffer_events()
                substep(*order)
            graphs.append(capture.graph)
    for step in range(steps):
        pipeline.collide(states[0], contacts)
        wp.capture_launch(graphs[step % 2])
        states.reverse()
    final = {name: getattr(states[0], name).numpy().copy() for name in STATE_FIELDS}
    final["impulses"] = solver.impulses.numpy().copy()
    return {
        "solver": solver,
        "final": final,
        "node_types": [_graph_node_types(graph) for graph in graphs],
        "copies": copies,
    }


@unittest.skipUnless(wp.is_cuda_available(), "Graph capture requires CUDA")
class TestFeatherPGSDebugSnapshotsGraph(unittest.TestCase):
    def test_captured_substeps_have_no_snapshot_memcpy_nodes(self):
        """Capture no snapshot memcpy node by default and three per substep under the flag; replay the same states."""
        device = wp.get_device("cuda:0")
        model = _scene(device, worlds=8)
        for mode in ("split", "matrix_free"):
            with self.subTest(pgs_mode=mode):
                off = _graph_rollout(model, snapshots=False, pgs_mode=mode)
                on = _graph_rollout(model, snapshots=True, pgs_mode=mode)
                for name in SNAPSHOTS:
                    self.assertIsNone(getattr(off["solver"], name), name)
                for off_copies, on_copies in zip(off["copies"], on["copies"], strict=True):
                    taken = [dest for dest, _ in on_copies if _is_snapshot(on["solver"], dest)]
                    self.assertEqual(len(taken), len(SNAPSHOTS))
                    for dest, name in zip(taken, SNAPSHOTS, strict=True):
                        self.assertIs(dest, getattr(on["solver"], name), name)
                    kept = [
                        (_signature(dest), _signature(src))
                        for dest, src in on_copies
                        if not _is_snapshot(on["solver"], dest)
                    ]
                    self.assertEqual([(_signature(dest), _signature(src)) for dest, src in off_copies], kept)
                for off_types, on_types in zip(off["node_types"], on["node_types"], strict=True):
                    if off_types is None or on_types is None:
                        continue  # no driver query: the captured-copy spy above still applies
                    self.assertEqual(on_types.get(MEMCPY_NODE, 0) - off_types.get(MEMCPY_NODE, 0), len(SNAPSHOTS))
                    for kind in (KERNEL_NODE, MEMSET_NODE):
                        self.assertEqual(on_types.get(kind, 0), off_types.get(kind, 0), kind)
                for name, values in off["final"].items():
                    _assert_same(self, device, on["final"][name], values, name)


for _device in get_test_devices():
    for _name, _func in (
        ("test_default_skips_snapshot_copies", test_default_skips_snapshot_copies),
        ("test_opt_in_keeps_snapshot_attributes", test_opt_in_keeps_snapshot_attributes),
        ("test_snapshots_leave_physics_unchanged", test_snapshots_leave_physics_unchanged),
    ):
        add_function_test(TestFeatherPGSDebugSnapshots, _name, _func, devices=[_device], check_output=False)


if __name__ == "__main__":
    unittest.main()
