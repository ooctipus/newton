# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check complete current/next G1 kinetics with retained sparse W consumers."""

import gc
import importlib
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS
from tools.fpgs_bench.test_sparse_contact_block import saved_records
from tools.fpgs_bench.test_sparse_factor import ASSET, fixture, physical_rows, unpack
from tools.fpgs_bench.test_sparse_metric_tangents import check_native, physical_metrics
from tools.fpgs_bench.test_world_scan_publication import compose, inverse, rotate

DT = 1.0 / 240.0
ENV = {
    "FEATHER_PGS_SPARSE_FACTOR": "1",
    "FEATHER_PGS_SINGLE_FACTOR": "0",
    "FEATHER_PGS_SPARSE_PACKETS": "0",
    "FEATHER_PGS_SPARSE_CONTACT_BLOCK": "0",
    "FEATHER_PGS_SPARSE_PARALLEL_LIMITS": "1",
    "FEATHER_PGS_SPARSE_LEVEL_UPDATE": "1",
    "FEATHER_PGS_SPARSE_METRIC_TANGENTS": "1",
}


def make_solver(model, enabled, **overrides):
    """Construct the actual complete owner; CPU is explicitly the original path."""
    options = {
        "pgs_mode": "matrix_free" if model.device.is_cuda else "split",
        "pgs_iterations": 8,
        "dense_max_constraints": 100,
        "enable_joint_limits": True,
        "joint_limit_activation_gap": 0.01,
        "update_mass_matrix_interval": 2,
        "mf_gs_incremental_rows": 0,
        "fuse_joint_velocity_limits": False,
        "grouped_dynamics": True,
        "use_parallel_streams": bool(model.device.is_cuda),
        "double_buffer": False,
    }
    options.update(overrides)
    with patch.dict(os.environ, {**ENV, "FEATHER_PGS_G1_KINETIC_STATE": str(int(enabled))}):
        return SolverFeatherPGS(model, **options)


def physical_mass(model, state):
    """Assemble full physical H from public COM twists, including every weld."""
    count = model.joint_dof_count
    probe, twists = model.state(), []
    for dof in range(count):
        unit = np.zeros(count, np.float32)
        unit[dof] = 1.0
        newton.eval_fk(model, state.joint_q, wp.array(unit, dtype=float, device=model.device), probe)
        twists.append(probe.body_qd.numpy().astype(float))
    twists = np.asarray(twists).transpose(1, 2, 0)
    mass, inertia, poses = model.body_mass.numpy(), model.body_inertia.numpy(), probe.body_q.numpy()
    result = np.zeros((count, count))
    for body in range(model.body_count):
        rotation = np.asarray(wp.quat_to_matrix(wp.quat(*poses[body, 3:])), float).reshape(3, 3)
        linear, angular = twists[body, :3], twists[body, 3:]
        # Use all authored inertia entries: do not symmetrize the only oracle.
        result += mass[body] * linear.T @ linear + angular.T @ (rotation @ inertia[body] @ rotation.T) @ angular
    return result


def augmentation(solver):
    """Retain original armature and current positive implicit-drive augmentation."""
    result = solver.R_by_size[43].numpy().astype(float).copy()
    rows, stiffness = solver._augmented_drive_row_by_dof.numpy(), solver.aug_row_K.numpy()
    result += np.asarray([max(0.0, float(stiffness[row])) if row >= 0 else 0.0 for row in rows]).reshape(-1, 43)
    return result


def backward_error(matrix, value, rhs):
    """Scale a physical linear action without hiding cancellation in coefficients."""
    return float(
        np.linalg.norm(matrix @ value - rhs, np.inf)
        / (1.0 + np.linalg.norm(matrix, np.inf) * np.linalg.norm(value, np.inf) + np.linalg.norm(rhs, np.inf))
    )


def public_reference(model, state):
    """Evaluate current public poses/COM twists in FP64, including free-root anchors.

    Reuse the independent publication reference's exact transform primitives,
    not the candidate's scan or its current S. Generic FP32 FK subtracts world
    positions at each joint: saved translations of 35--76 m make that a less
    accurate velocity oracle. No quaternion/input renormalization is introduced.
    """
    names = (
        "joint_type",
        "joint_parent",
        "joint_child",
        "joint_q_start",
        "joint_qd_start",
        "joint_X_p",
        "joint_X_c",
        "joint_axis",
        "body_com",
    )
    values = {name: getattr(model, name).numpy() for name in names}
    q, qd = (getattr(state, name).numpy().astype(np.float64) for name in ("joint_q", "joint_qd"))
    poses = np.zeros((model.body_count, 7), np.float64)
    anchors = np.zeros((model.joint_count, 7), np.float64)
    roots = np.zeros(model.body_count, int)
    for joint in range(model.joint_count):
        kind, parent, body, qs, ds = (int(values[name][joint]) for name in names[:5])
        anchor = values["joint_X_p"][joint].astype(np.float64)
        if parent >= 0:
            anchor = compose(poses[parent], anchor)
        transform = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        if kind == int(newton.JointType.REVOLUTE):
            transform[3:6] = values["joint_axis"][ds].astype(np.float64) * np.sin(q[qs] / 2)
            transform[6] = np.cos(q[qs] / 2)
        elif kind == int(newton.JointType.PRISMATIC):
            transform[:3] = values["joint_axis"][ds].astype(np.float64) * q[qs]
        elif kind == int(newton.JointType.FREE):
            transform = q[qs : qs + 7]
            if parent >= 0:
                raise ValueError("Public oracle admits free articulation roots only")
        elif kind != int(newton.JointType.FIXED):
            raise ValueError(f"Unsupported public oracle joint type: {kind}")
        anchors[joint] = anchor
        poses[body] = compose(compose(anchor, transform), inverse(values["joint_X_c"][joint].astype(np.float64)))
        roots[body] = roots[parent] if parent >= 0 else body
    com = poses[:, :3] + rotate(poses[:, 3:], values["body_com"].astype(np.float64))
    motion = np.zeros((model.body_count, 6), np.float64)
    for joint in range(model.joint_count):
        kind, parent, body, _, ds = (int(values[name][joint]) for name in names[:5])
        anchor, local = anchors[joint], np.zeros(6)
        if kind in (int(newton.JointType.PRISMATIC), int(newton.JointType.REVOLUTE)):
            axis = rotate(anchor[3:], values["joint_axis"][ds].astype(np.float64))
            if kind == int(newton.JointType.PRISMATIC):
                local[:3] = axis * qd[ds]
            else:
                local[3:] = axis * qd[ds]
                local[:3] = np.cross(anchor[:3] - com[roots[body]], local[3:])
        elif kind == int(newton.JointType.FREE):
            # Free-root linear DOFs describe its COM, in the parent-anchor basis.
            local[:3] = rotate(anchor[3:], qd[ds : ds + 3])
            local[3:] = rotate(anchor[3:], qd[ds + 3 : ds + 6])
        motion[body] = local + (motion[parent] if parent >= 0 else 0)
    public = motion.copy()
    public[:, :3] += np.cross(motion[:, 3:], com - com[roots])
    return {"body_q": poses, "body_qd": public}


def public_state(test, model, state):
    """Check FP64 poses and coordinate-invariant linear/angular velocity errors."""
    expected = public_reference(model, state)
    for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
        test.assertTrue(np.isfinite(getattr(state, name).numpy()).all(), name)
    np.testing.assert_allclose(state.body_q.numpy(), expected["body_q"], rtol=2e-5, atol=3e-6, err_msg="body_q")
    # A near-zero Cartesian component can cancel large ancestor contributions;
    # an elementwise relative gate depends on the arbitrary world orientation.
    # Keep linear and angular units separate, and use the stricter 3e-6
    # absolute-plus-relative physical vector bound against independent FP64.
    actual = state.body_qd.numpy().astype(np.float64).reshape(-1, 2, 3)
    reference = expected["body_qd"].reshape(-1, 2, 3)
    error = np.linalg.norm(actual - reference, axis=2)
    scale = 1.0 + np.linalg.norm(reference, axis=2)
    test.assertLess(float(np.max(error / scale)), 3e-6, "public linear/angular velocity")


def stage_predict(solver, state, output, control):
    """Run original production preparation through the force/predictor boundary."""
    solver._last_step_dt = DT
    augmented = solver._prepare_augmented_state(state, output, control)
    drives = solver._stage1_prepare_augmented_drives(state, augmented, control, DT)
    inertia, velocity = solver._stage1_fk_id(state, augmented, output)
    ready = solver._stage1_complete_joint_tau(state, augmented, control, DT, drives)
    solver._stage1_crba(augmented, inertia, drives)
    if ready is not None:
        wp.get_stream(solver.model.device).wait_event(ready)
    if solver._g1_kinetic_state is None:
        solver._stage3_zero_qdd(augmented)
        solver._sparse_factor.predict(augmented)
        solver._stage3_compute_v_hat(state, augmented, DT, velocity)
        solver._clamp_rigid_velocity_limits(solver.v_hat)
    else:
        solver._g1_kinetic_state.predict(state, augmented, control, velocity, DT)
        solver._clamp_rigid_velocity_limits(solver.v_hat)
    return augmented, velocity


def rotated_input(model, state, *, epoch=0):
    """Exercise arbitrary root orientation and nonzero current motion/COM forces."""
    worlds = model.world_count
    q = model.joint_q.numpy().reshape(worlds, 44).copy()
    q[:, 2] = 0.9
    for world in range(worlds):
        q[world, :3] += np.asarray([0.13 * world, -0.09 * world, 0.0], np.float32)
        q[world, 3:7] = np.asarray(
            wp.quat_from_axis_angle(wp.normalize(wp.vec3(1, -2, 3)), 0.31 + 0.09 * epoch + 0.04 * world)
        )
        q[world, 7:] += np.linspace(-0.06, 0.07 + 0.01 * epoch, 37, dtype=np.float32)
    state.joint_q.assign(q.reshape(-1))
    state.joint_qd.assign(np.linspace(-0.4, 0.6 + 0.03 * epoch, worlds * 43, dtype=np.float32))
    state.body_f.assign(np.random.default_rng(81 + epoch).normal(0, 0.8, (worlds * 44, 6)).astype(np.float32))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)


def set_anchor(model):
    """Use nontrivial authored free-root parent and child anchors before admission."""
    for name, angle, position in (("joint_X_p", 0.23, (0.11, -0.07, 0.04)), ("joint_X_c", -0.17, (-0.03, 0.02, 0.01))):
        values = getattr(model, name).numpy()
        for world in range(model.world_count):
            values[world * 44] = np.r_[position, np.asarray(wp.quat_from_axis_angle(wp.vec3(0, 1, 0), angle))]
        getattr(model, name).assign(values)


def solve_quality(test, case, held, *, jacobian=None):
    """Use current physical J, held physical H and the existing metric-law gate."""
    solver, owner = case["solver"], case["owner"]
    count = int(solver.constraint_count.numpy()[0])
    if jacobian is None:
        jacobian = physical_rows(case)
    test.assertEqual(jacobian.shape, (count, 43))
    vhat = solver.v_hat.numpy().copy()
    solver.impulses.zero_()
    actual, impulse, _ = check_native(test, case)
    defect = backward_error(held, actual - vhat, jacobian.T @ impulse)
    test.assertLess(defect, 2e-6)
    diagonal, rhs, types, parents, friction = (
        getattr(solver, name).numpy()[0, :count] for name in ("diag", "rhs", "row_type", "row_parent", "row_mu")
    )
    metrics = physical_metrics(jacobian, diagonal, rhs, types, parents, friction, vhat, actual, impulse)
    test.assertTrue(all(np.isfinite(value) for value in metrics.values()))
    test.assertLess(metrics["cone"], 3e-5)
    owner.check()
    return {"momentum": defect, **metrics}


def direct_geometry(test, device):
    """Compare the new representation against independent physical kinetic energy."""
    module = importlib.import_module("newton._src.solvers.feather_pgs.g1_kinetic_state")
    with patch.dict(os.environ, {**ENV, "FEATHER_PGS_G1_KINETIC_STATE": "0"}):
        case = fixture(device)
    solver, model, state, sparse = (case[name] for name in ("solver", "model", "state", "owner"))
    set_anchor(model)
    rotated_input(model, state)
    if model.device.is_cuda:
        solver = make_solver(model, True)
        sparse, owner = solver._sparse_factor, solver._g1_kinetic_state
        test.assertIsNotNone(owner)
    else:
        owner = module.G1KineticState(sparse)
    augmented = solver._prepare_augmented_state(state, model.state(), model.control())
    owner.begin(state, augmented, state.joint_qd, DT, True)
    current = physical_mass(model, state)
    expected = current[42 - sparse.host["row"], 42 - sparse.host["col"]]
    actual = owner.geometric.numpy()[0].astype(float)
    error = float(np.linalg.norm(actual - expected, np.inf) / (1.0 + np.linalg.norm(expected, np.inf)))
    test.assertLess(error, 3e-6)
    np.testing.assert_array_equal(owner.status.numpy(), [0])
    np.testing.assert_array_equal(owner.geometry_valid.numpy(), [1])
    public_state(test, model, state)
    owner.invalidate()
    np.testing.assert_array_equal(owner.geometry_valid.numpy(), [0])
    print(f"g1_kinetic_geometry physical_H={error:.8g}", flush=True)


def baseline_bias(solver):
    """Project original body bias to all generalized coordinates, including welds."""
    parent = solver.model.joint_parent.numpy()
    result = np.empty(43)
    screws, wrench = solver.joint_S_s.numpy().astype(float), solver.body_f_s.numpy().astype(float)
    for natural, joint in enumerate(solver._sparse_factor.host["dof_joint"]):
        descendants = []
        for body in range(44):
            cursor = body
            while cursor >= 0 and cursor != joint:
                cursor = parent[cursor]
            if cursor == joint:
                descendants.append(body)
        result[natural] = screws[natural] @ wrench[descendants].sum(axis=0)
    return result


def multiworld_fixture(device):
    """Use five copies of the same authored G1, with live contacts and a partial block."""
    part, builder = newton.ModelBuilder(), newton.ModelBuilder()
    part.add_usd(str(ASSET), load_visual_shapes=False)
    builder.replicate(part, 5)
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    model.rigid_contact_max = 16
    set_anchor(model)
    seed = model.state()
    rotated_input(model, seed)
    contacts = newton.Contacts(rigid_contact_max=16, soft_contact_max=0, device=device)
    shape_body = model.shape_body.numpy()
    floor = int(np.flatnonzero(shape_body < 0)[0])
    first, second = np.full(16, floor, np.int32), np.full(16, floor, np.int32)
    point0, point1, normals = (np.zeros((16, 3), np.float32) for _ in range(3))
    poses = seed.body_q.numpy()
    for world in range(5):
        body = world * 44 + 6
        first[world] = int(np.flatnonzero(shape_body == body)[0])
        point0[world] = [0.02, -0.01, -0.05]
        point1[world] = np.asarray(wp.transform_point(wp.transform(*poses[body]), wp.vec3(*point0[world])))
        point1[world, 2] += 0.003
        normals[world] = [0, 0, -1]
    for name, values in (
        ("shape0", first),
        ("shape1", second),
        ("point0", point0),
        ("point1", point1),
        ("normal", normals),
    ):
        getattr(contacts, "rigid_contact_" + name).assign(values)
    contacts.rigid_contact_margin0.zero_()
    contacts.rigid_contact_margin1.zero_()
    contacts.rigid_contact_count.fill_(5)
    return model, contacts


class TestG1KineticStateCPU(unittest.TestCase):
    def test_owner_api(self):
        """Require the bounded complete-owner API before any cache retirement."""
        module = importlib.import_module("newton._src.solvers.feather_pgs.g1_kinetic_state")
        for name in ("begin", "predict", "finish", "invalidate", "validate_model"):
            self.assertTrue(callable(getattr(module.G1KineticState, name)))

    def test_physical_geometry_and_invalidation(self):
        """Run the representation oracle on CPU without claiming CPU production admission."""
        direct_geometry(self, "cpu")

    def test_public_oracle_anchored_translation_and_rest(self):
        """World translation must not alter the anchored-root physical COM twist."""
        case = fixture("cpu")
        model, state = case["model"], case["state"]
        set_anchor(model)
        rotated_input(model, state)
        reference = public_reference(model, state)
        generic = model.state()
        newton.eval_fk(model, state.joint_q, state.joint_qd, generic)
        np.testing.assert_allclose(generic.body_qd.numpy(), reference["body_qd"], rtol=2e-5, atol=3e-6)
        before = state.joint_q.numpy().copy()
        shifted = before.copy()
        shifted[:3] += np.array([-36.0, -76.0, 0.0], np.float32)
        state.joint_q.assign(shifted)
        translated = public_reference(model, state)
        displacement = rotate(
            model.joint_X_p.numpy()[0, 3:].astype(np.float64), shifted[:3].astype(np.float64) - before[:3]
        )
        np.testing.assert_allclose(
            translated["body_q"][:, :3], reference["body_q"][:, :3] + displacement, atol=3e-12, rtol=0
        )
        np.testing.assert_allclose(translated["body_q"][:, 3:], reference["body_q"][:, 3:], atol=3e-12, rtol=0)
        np.testing.assert_allclose(translated["body_qd"], reference["body_qd"], atol=3e-12, rtol=0)
        module = importlib.import_module("newton._src.solvers.feather_pgs.g1_kinetic_state")
        owner = module.G1KineticState(case["owner"])
        augmented = case["solver"]._prepare_augmented_state(state, model.state(), model.control())
        owner.begin(state, augmented, state.joint_qd, DT, True)
        public_state(self, model, state)
        state.joint_qd.zero_()
        np.testing.assert_array_equal(public_reference(model, state)["body_qd"], np.zeros((44, 6)))

    def test_constructor_and_model_guards(self):
        """Do not retire original storage on unsupported CPU/kinematic configurations."""
        module = importlib.import_module("newton._src.solvers.feather_pgs.g1_kinetic_state")
        with patch.dict(os.environ, {"FEATHER_PGS_G1_KINETIC_STATE": "0"}):
            case = fixture()
        solver = make_solver(case["model"], True)
        self.assertIsNone(solver._g1_kinetic_state)
        self.assertFalse(module.supported(solver))
        owner = module.G1KineticState(case["owner"])
        owner.validate_model()
        flags = case["model"].body_flags.numpy()
        flags[7] |= int(newton.BodyFlags.KINEMATIC)
        case["model"].body_flags.assign(flags)
        with self.assertRaises((RuntimeError, ValueError)):
            owner.validate_model()

    def test_mass_requests_consumed_on_both_routes(self):
        """The new early return must retain the original forced/device-mask common tail."""
        for force, step, requested, expected in (
            (True, 1, [0], [1, 1, 1]),
            (False, 0, [0], [1, 1, 1]),
            (False, 1, [1], [1, 1, 1]),
            (False, 1, [0], [0, 0, 0]),
        ):
            with self.subTest(force=force, step=step):
                calls = []
                owner = SimpleNamespace(
                    model=SimpleNamespace(articulation_count=3, device=wp.get_device("cpu")),
                    _step=step,
                    update_mass_matrix_interval=2,
                    _force_mass_update=force,
                    _g1_kinetic_state=object(),
                    _mass_update_requested=wp.array(requested, dtype=int, device="cpu"),
                    mass_update_mask=wp.zeros(3, dtype=int, device="cpu"),
                )
                owner._sparse_factor = SimpleNamespace(refresh=calls.append)
                SolverFeatherPGS._stage1_crba(owner, owner, None, None)
                self.assertEqual(calls, [owner])
                np.testing.assert_array_equal(owner.mass_update_mask.numpy(), expected)
                np.testing.assert_array_equal(owner._mass_update_requested.numpy(), [0])
                self.assertFalse(owner._force_mass_update)


class TestG1KineticStateCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Use only the device made visible by the root-owned native runner."""
        wp.init()
        if not wp.get_cuda_devices():
            raise unittest.SkipTest("Root owns the native GPU lease")
        cls.device = wp.get_cuda_devices()[0]

    @classmethod
    def tearDownClass(cls):
        """Release cyclic solver/owner references while CUDA is still alive."""
        wp.synchronize_device(cls.device)
        gc.collect()

    def test_current_force_predictor_and_publication(self):
        """Check physical H, current force buckets and original anchored-root integration."""
        direct_geometry(self, self.device)
        with patch.dict(os.environ, {"FEATHER_PGS_G1_KINETIC_STATE": "0"}):
            case = fixture(self.device)
        model = case["model"]
        set_anchor(model)
        serial = make_solver(model, True, use_parallel_streams=False)
        self.assertIsNone(serial._g1_kinetic_state)
        solvers = [make_solver(model, enabled) for enabled in (False, True)]
        self.assertIsNone(solvers[0]._g1_kinetic_state)
        self.assertIsNotNone(solvers[1]._g1_kinetic_state)
        states, outputs = [model.state() for _ in solvers], [model.state() for _ in solvers]
        controls = [model.control() for _ in solvers]
        held = None
        for epoch in range(3):
            if epoch == 2:
                model.body_mass.assign(model.body_mass.numpy() * np.float32(1.015))
                com = model.body_com.numpy()
                com[:, 1] += np.float32(0.001)
                model.body_com.assign(com)
                for solver in solvers:
                    solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
            for index, (solver, state, output, control) in enumerate(
                zip(solvers, states, outputs, controls, strict=True)
            ):
                rotated_input(model, state, epoch=epoch)
                solver.reset(state)
                solver._step = epoch
                solver._force_mass_update = epoch == 2
                control.joint_f.assign(np.linspace(-0.2, 0.3 + 0.1 * epoch, 43, dtype=np.float32))
                target = control.joint_target_q.numpy()
                target += np.float32(0.02 * (epoch + 1))
                control.joint_target_q.assign(target)
                control.joint_target_qd.assign(np.linspace(-0.1, 0.15, control.joint_target_qd.size, dtype=np.float32))
                solver._passive_spring_stiffness.assign(np.linspace(0, 0.2, 43, dtype=np.float32))
                solver._passive_spring_ref.assign(np.linspace(-0.1, 0.1, 43, dtype=np.float32))
                solver._passive_joint_damping.assign(np.linspace(0, 0.3, 43, dtype=np.float32))
                augmented, velocity = stage_predict(solver, state, output, control)
                if index == 0:
                    expected_tau = solver.joint_tau.numpy().astype(float)
                    expected_bias = baseline_bias(solver)
                    expected_vhat = solver.v_hat.numpy().copy()
                else:
                    native = solver._g1_kinetic_state
                    self.assertTrue(np.any(augmentation(solver)[0] > solver.R_by_size[43].numpy()[0]))
                    np.testing.assert_allclose(native.bias.numpy()[0], expected_bias, rtol=2e-5, atol=3e-5)
                    np.testing.assert_allclose(solver.v_hat.numpy(), expected_vhat, rtol=2e-5, atol=3e-5)
                    if epoch != 1:
                        held = physical_mass(model, state) + np.diag(augmentation(solver)[0])
                        held_bytes = solver._sparse_factor.data.W.numpy().copy()
                    else:
                        np.testing.assert_array_equal(solver._sparse_factor.data.W.numpy(), held_bytes)
                    W = unpack(solver._sparse_factor)
                    inverse = (W.T @ W)[::-1, ::-1]
                    self.assertLess(backward_error(held, inverse, np.eye(43)), 3e-6)
                    qd = velocity.numpy().astype(float)
                    accel = (solver.v_hat.numpy().astype(float) - qd) / DT
                    accel[:3] -= np.cross(qd[3:6], qd[:3])
                    defect = backward_error(held, accel, expected_tau)
                    self.assertLess(defect, 3e-6)
                    self.assertFalse(solver._force_mass_update)
                    np.testing.assert_array_equal(solver._mass_update_requested.numpy(), [0])
                    print(f"g1_kinetic_predict epoch={epoch} momentum={defect:.8g}", flush=True)
                solver.v_out.assign(state.joint_qd.numpy() + np.linspace(0.1, -0.1, 43, dtype=np.float32))
                solver._stage6_update_qdd(state, augmented, DT)
                solver._stage6_integrate(state, augmented, output, DT)
                public_state(self, model, output)
            for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                np.testing.assert_allclose(
                    getattr(outputs[1], name).numpy(),
                    getattr(outputs[0], name).numpy(),
                    rtol=2e-5,
                    atol=3e-6,
                    err_msg=name,
                )

    def test_saved_current_geometry_held_operator_and_contacts(self):
        """Keep each captured physical held H while rebuilding current S and loaded contact geometry."""
        replay, records = saved_records()
        for gpu, record, data in records:
            for index, world in enumerate(data["worlds"]):
                with self.subTest(gpu=gpu, step=record["step"], world=int(world)):
                    with patch.dict(os.environ, {**ENV, "FEATHER_PGS_G1_KINETIC_STATE": "0"}):
                        case = replay.bind_world(record, data, index, self.device)
                    previous, model, state = (case[name] for name in ("solver", "model", "state"))
                    solver = make_solver(model, True)
                    self.assertIsNotNone(solver._g1_kinetic_state)
                    for name, value in record["settings"].items():
                        setattr(solver, name, value)
                    for name in ("shape_material_mu", "shape_material_restitution"):
                        getattr(solver, name).assign(getattr(previous, name).numpy())
                    output, control = model.state(), model.control()
                    augmented = solver._prepare_augmented_state(state, output, control)
                    solver._g1_kinetic_state.begin(state, augmented, state.joint_qd, 0.0025, False)
                    public_state(self, model, state)
                    sparse = solver._sparse_factor
                    case.update(solver=solver, owner=sparse, host=sparse.host)
                    held = case["H"]
                    W = np.linalg.solve(np.linalg.cholesky(held[::-1, ::-1]), np.eye(43))
                    packed = W[sparse.host["row"], sparse.host["col"]][None].astype(np.float32)
                    sparse.data.W.assign(packed)
                    sparse.data.valid.fill_(1)
                    solver.v_hat.assign(previous.v_hat.numpy())
                    sparse.build_rows(state, augmented, case["contacts"], 0.0025)
                    solver.check_constraint_capacity()
                    solver._stage4_compute_rhs_world(0.0025)
                    sparse.restitution(0.0025)
                    # Keep the captured ORIGINAL current screw basis in the
                    # physical oracle. Reusing candidate S for both sides could
                    # hide a held/current-basis error behind exact W algebra.
                    physical_solver = SimpleNamespace(**vars(solver))
                    physical_solver.joint_S_s = previous.joint_S_s
                    physical_solver.articulation_origin = previous.articulation_origin
                    jacobian = physical_rows({**case, "solver": physical_solver})
                    metrics = solve_quality(self, case, held, jacobian=jacobian)
                    np.testing.assert_array_equal(sparse.data.W.numpy(), packed)
                    print(
                        "g1_kinetic_saved "
                        + json.dumps(
                            {
                                "gpu_fixture": gpu,
                                "step": record["step"],
                                "world": int(world),
                                **metrics,
                            }
                        ),
                        flush=True,
                    )

    def test_actual_steps_masked_refresh_reset_and_graph(self):
        """Check actual complete dispatch, mixed cache repair and both graph mass epochs."""
        model, contacts = multiworld_fixture(self.device)
        solvers = [make_solver(model, enabled, double_buffer=True) for enabled in (False, True)]
        # Keep asynchronous owners alive until all device streams have drained,
        # including an assertion exit before the later graph lifecycle checks.
        self.addCleanup(lambda owners=solvers: wp.synchronize_device(owners[0].model.device))
        original, candidate = solvers
        self.assertIsNone(original._g1_kinetic_state)
        self.assertIsNotNone(candidate._g1_kinetic_state)
        native = candidate._g1_kinetic_state
        states = [[model.state(), model.state()] for _ in solvers]
        controls = [model.control() for _ in solvers]
        for pair, control in zip(states, controls, strict=True):
            rotated_input(model, pair[0])
            control.joint_f.assign(np.linspace(-0.2, 0.3, 215, dtype=np.float32))
        seen = []
        original_launch, original_tiled = wp.launch, wp.launch_tiled

        def watch(function):
            def launch(*args, **kwargs):
                kernel = kwargs.get("kernel", args[0] if args else None)
                seen.append(kernel.key)
                return function(*args, **kwargs)

            return launch

        def complete_step(source, destination):
            for index, (solver, pair, control) in enumerate(zip(solvers, states, controls, strict=True)):
                if index:
                    with (
                        patch.object(wp, "launch", watch(original_launch)),
                        patch.object(wp, "launch_tiled", watch(original_tiled)),
                    ):
                        solver.step(pair[source], pair[destination], control, contacts, DT)
                else:
                    solver.step(pair[source], pair[destination], control, contacts, DT)
                solver.check_constraint_capacity()
                self.assertFalse(solver._force_mass_update)
                np.testing.assert_array_equal(solver._mass_update_requested.numpy(), [0])
                public_state(self, model, pair[destination])
            # Compare physical outputs, not packed W coefficients or retired qdd buffers.
            for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                expected = getattr(states[0][destination], name).numpy()
                actual = getattr(states[1][destination], name).numpy()
                self.assertLess(np.linalg.norm(actual - expected) / (1 + np.linalg.norm(expected)), 7e-4, name)
            self.assertIs(candidate._fk_id_cache_source_state, states[1][destination])
            np.testing.assert_array_equal(candidate._fk_id_cache_valid.numpy(), np.ones(5, np.int32))

        complete_step(0, 1)
        held = candidate._sparse_factor.data.W.numpy().copy()
        complete_step(1, 0)
        np.testing.assert_array_equal(candidate.mass_update_mask.numpy(), np.zeros(5, np.int32))
        np.testing.assert_array_equal(candidate._sparse_factor.data.W.numpy(), held)

        # Current-world cache repair is selective. The original device mass
        # request, in contrast, is a scalar broadcast to every articulation.
        reset_mask = wp.array([False, True, False, True, False], dtype=wp.bool, device=self.device)
        for solver, pair in zip(solvers, states, strict=True):
            q = pair[0].joint_q.numpy().reshape(5, 44)
            q[[1, 3], 7:] += np.float32(0.02)
            pair[0].joint_q.assign(q.reshape(-1))
            solver.reset(pair[0], reset_mask)
            solver._step = 3
            solver._force_mass_update = False
            solver._mass_update_requested.fill_(1)
        np.testing.assert_array_equal(native.geometry_valid.numpy(), [1, 0, 1, 0, 1])
        np.testing.assert_array_equal(candidate._fk_id_cache_valid.numpy(), [1, 0, 1, 0, 1])
        complete_step(0, 1)
        np.testing.assert_array_equal(candidate.mass_update_mask.numpy(), np.ones(5, np.int32))

        # Notification changes current COM, mass, full angular inertia and gravity;
        # the late requested refresh must be consumed despite a non-global step.
        model.body_mass.assign(model.body_mass.numpy() * np.float32(1.01))
        com = model.body_com.numpy()
        com[:, 0] += np.float32(0.002)
        model.body_com.assign(com)
        model.body_inertia.assign(model.body_inertia.numpy() * np.float32(1.02))
        gravity = model.gravity.numpy()
        gravity[0] += np.array([0.1, -0.05, 0.03], np.float32)
        model.set_gravity(gravity)
        for solver, pair, control in zip(solvers, states, controls, strict=True):
            solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES | newton.ModelFlags.MODEL_PROPERTIES)
            solver._step = 5
            solver._force_mass_update = False
            pair[1].body_f.assign(np.random.default_rng(714).normal(0, 1.3, (220, 6)).astype(np.float32))
            control.joint_f.assign(-control.joint_f.numpy())
        np.testing.assert_array_equal(native.geometry_valid.numpy(), np.zeros(5, np.int32))
        complete_step(1, 0)
        np.testing.assert_array_equal(candidate.mass_update_mask.numpy(), np.ones(5, np.int32))

        # A different input bank must repair current S/pose even if the previous
        # complete step left every cache-valid bit set.
        replacement = model.state()
        rotated_input(model, replacement, epoch=2)
        reference_input, reference_output = model.state(), model.state()
        for name in ("joint_q", "joint_qd", "body_q", "body_qd", "body_f"):
            getattr(reference_input, name).assign(getattr(replacement, name).numpy())
        original._step = 7
        candidate._step = 7
        original.step(reference_input, reference_output, controls[0], contacts, DT)
        candidate.step(replacement, states[1][1], controls[1], contacts, DT)
        candidate.check_constraint_capacity()
        public_state(self, model, states[1][1])
        self.assertIs(candidate._fk_id_cache_source_state, states[1][1])
        self.assertLess(
            np.linalg.norm(candidate.v_hat.numpy() - original.v_hat.numpy())
            / (1.0 + np.linalg.norm(original.v_hat.numpy())),
            7e-4,
        )
        self.assertLess(
            np.linalg.norm(states[1][1].joint_qd.numpy() - reference_output.joint_qd.numpy())
            / (1.0 + np.linalg.norm(reference_output.joint_qd.numpy())),
            7e-4,
        )

        for retired in (
            "grouped_tau",
            "compute_composite_inertia",
            "template_fk_kinematics",
            "finalize_body_dynamics",
            "eval_rigid_fk_id",
            "integrate_generalized_joints",
            "refresh_masked_body_inertia",
            "compute_velocity_predictor",
            "update_qdd_from_velocity",
            "remove_free_root_transport_from_qdd",
            "apply_free_root_transport_to_predictor",
        ):
            self.assertFalse(any(retired in key for key in seen), retired)
        self.assertTrue(any("g1_kinetic" in key for key in seen))

        # Drain eager maintenance and seed any buffer events within capture,
        # exactly as the real manager does; retain all production streams.
        contacts.rigid_contact_count.zero_()
        candidate._step = 8
        wp.synchronize_device(self.device)
        with wp.ScopedCapture(device=self.device) as captured:
            candidate.seed_double_buffer_events()
            candidate.step(states[1][1], states[1][0], controls[1], contacts, DT)
            candidate.step(states[1][0], states[1][1], controls[1], contacts, DT)
        for contact_count in (0, 5, 0, 5):
            candidate.reset(states[1][1], reset_mask)
            candidate._mass_update_requested.fill_(1)
            contacts.rigid_contact_count.fill_(contact_count)
            wp.capture_launch(captured.graph)
            candidate.check_constraint_capacity()
            np.testing.assert_array_equal(candidate._mass_update_requested.numpy(), [0])
            public_state(self, model, states[1][1])
        print(
            "g1_kinetic_complete "
            + json.dumps({"worlds": 5, "graph_contacts": [0, 5, 0, 5], "kernels": sorted(set(seen))}),
            flush=True,
        )


if __name__ == "__main__":
    unittest.main()
