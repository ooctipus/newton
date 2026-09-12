# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent current-input controls for complete late Kuka publication.

The four pinned historical 512-world inputs provide solved velocities and
current numeric model data. This is matched-equation publication, not a replay
of recorded trajectories or a contact/convergence certificate.
"""

import ast
import copy
import inspect
import textwrap
import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import kernels
from newton._src.solvers.feather_pgs import kuka_joint_world as light
from newton._src.solvers.feather_pgs import world_scan_publication as scan
from tools.fpgs_bench import test_kuka_joint_world as saved

CHAIN = (
    kernels.update_qdd_from_velocity,
    kernels.remove_free_root_transport_from_qdd,
    kernels.integrate_generalized_joints,
    kernels.eval_rigid_fk_kinematics,
    kernels.finalize_body_dynamics,
)
OUTPUTS = {
    "v_new": "post_solve_v_out",
    "joint_qdd": "post3_aug_joint_qdd",
    "joint_q_new": "pre_state_joint_q",
    "joint_qd_new": "pre_state_joint_qd",
    "body_q": "pre_state_body_q",
    "body_qd": "pre_state_body_qd",
    "body_q_com": "post3_aug_body_q_com",
    "articulation_origin": "post3_solver_articulation_origin",
    "joint_S_s": "post3_aug_joint_S_s",
    "body_v_s": "post3_aug_body_v_s",
    "body_a_s": "post3_aug_body_a_s",
    "body_I_s": "post3_aug_body_I_s",
    "body_inertia_terms": "post3_solver__body_inertia_terms",
    "body_f_s": "post3_aug_body_f_s",
    "fk_id_cache_valid": "post3_solver__fk_id_cache_valid",
}


def bind(snapshot, device, *, refresh=(0, 0), changed=False):
    """Bind original kernel argument names to disjoint current input/output owners."""
    values = {
        "dt": 1 / 240,
        "inv_dt": 240.0,
        "angular_damping": 0.0,
        "materialize_all_body_inertia": refresh[0],
        "materialize_body_inertia_terms": refresh[1],
    }
    aliases = {
        "joint_q": "pre_state_joint_q",
        "joint_qd": "pre_state_joint_qd",
        "kinematic_dof_mask": "post3_solver__kinematic_dof_mask",
        "kinematic_joint_mask": "post3_solver__kinematic_joint_mask",
        "free_root_joint_indices": "post3_solver__free_root_joint_indices",
    }
    for kernel in CHAIN:
        for argument in kernel.adj.args:
            name = argument.label
            if name in values:
                continue
            key = OUTPUTS.get(name, aliases.get(name))
            if key is None:
                key = next(key for key in ("full_model_" + name, "post3_solver_" + name) if key in snapshot)
            values[name] = wp.array(snapshot[key], dtype=argument.type.dtype, device=device)
    plan = light.build_plan(**saved.plan_inputs(snapshot))
    scan.validate_scan_plan(plan)
    dimensions = (
        len(snapshot["pre_state_joint_qd"]),
        len(snapshot["post3_solver__free_root_joint_indices"]),
        len(snapshot["full_model_joint_type"]),
        len(snapshot["post3_solver_art_to_world"]),
        len(snapshot["full_model_body_mass"]),
    )
    bundle = SimpleNamespace(
        values=values,
        plan=plan,
        device_plan=plan.device_data(device),
        dimensions=dimensions,
        device=device,
    )
    if changed:
        change_numeric_inputs(bundle)
    # Every always-written output starts invalid. Held inertia retains its
    # original finite bytes, so missing or excess epoch writes are visible.
    for name in OUTPUTS.keys() - {"v_new", "body_I_s", "body_inertia_terms"}:
        values[name].fill_(0 if name == "fk_id_cache_valid" else np.nan)
    data = scan.PublicationData()
    for name in scan.PublicationData.vars:
        setattr(data, name, values["v_new" if name == "v_out" else name])
    bundle.data = data
    return bundle


def change_numeric_inputs(bundle):
    """Change live frames, COM, inertia, gravity and both free-root motion families."""
    values, plan = bundle.values, bundle.plan
    lanes = plan.joint_ids.reshape(-1)
    for name, scale in (("joint_X_p", 0.013), ("joint_X_c", -0.009), ("body_X_com", 0.006)):
        array = values[name].numpy().copy()
        indices = plan.body_ids.reshape(-1) if name == "body_X_com" else lanes
        angle = 0.07 * np.sin(np.arange(len(indices), dtype=np.float64) * 0.37)
        delta = np.zeros((len(indices), 4))
        delta[:, 1], delta[:, 3] = np.sin(angle / 2), np.cos(angle / 2)
        array[indices, :3] += scale * np.array([0.5, -0.25, 0.75], np.float32)
        array[indices, 3:] = quat_multiply(array[indices, 3:], delta).astype(np.float32)
        values[name].assign(array)
    com = values["body_com"].numpy().copy()
    com += np.array([0.003, -0.002, 0.004], np.float32)
    values["body_com"].assign(com)
    # Keep body_X_com's translation consistent with the model COM notification.
    transform = values["body_X_com"].numpy().copy()
    transform[:, :3] = com
    values["body_X_com"].assign(transform)
    values["body_mass"].assign(values["body_mass"].numpy() * np.float32(1.17))
    values["body_inertia"].assign(values["body_inertia"].numpy() * np.float32(1.09))
    values["gravity"].assign(np.array([[0.8, -1.1, -8.7]], np.float32))
    qd = values["joint_qd"].numpy().copy()
    solved = values["v_new"].numpy().copy()
    starts = values["joint_qd_start"].numpy()
    for offset, lane in enumerate((30, 31)):
        dofs = starts[plan.joint_ids[:, lane], None] + np.arange(6)
        motion = np.array([0.31, -0.17, 0.12, 0.27, 0.19, -0.23], np.float32) * (offset + 1)
        qd[dofs], solved[dofs] = motion, motion + np.float32(0.025)
    values["joint_qd"].assign(qd)
    values["v_new"].assign(solved)
    values["angular_damping"] = 0.03


def original(bundle):
    """Execute the original complete five-kernel boundary with original argument order."""
    for index, kernel in enumerate(CHAIN):
        values = dict(bundle.values)
        if index == 3:
            values["joint_q"], values["joint_qd"] = values["joint_q_new"], values["joint_qd_new"]
        wp.launch(
            kernel,
            dim=bundle.dimensions[index],
            inputs=[values[argument.label] for argument in kernel.adj.args],
            device=bundle.device,
        )


def launch(bundle):
    """Execute the actual new all-world kernel without predictor/row/GS changes."""
    wp.launch_tiled(
        scan.get_kernel(str(wp.get_device(bundle.device).arch)),
        dim=[len(bundle.plan.body_ids)],
        inputs=[bundle.device_plan, bundle.data],
        block_dim=32,
        device=bundle.device,
    )


def quat_multiply(a, b):
    """Multiply xyzw quaternions independently in FP64."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.concatenate(
        (
            a[..., 3:] * b[..., :3] + b[..., 3:] * a[..., :3] + np.cross(a[..., :3], b[..., :3]),
            a[..., 3:] * b[..., 3:] - np.sum(a[..., :3] * b[..., :3], axis=-1, keepdims=True),
        ),
        axis=-1,
    )


def rotate_inverse(q, vector):
    """Express a world vector in its current normalized root orientation in FP64."""
    q = np.asarray(q, np.float64).copy()
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    q[..., :3] *= -1
    vector = np.asarray(vector, np.float64)
    twice = 2 * np.cross(q[..., :3], vector)
    return vector + q[..., 3:] * twice + np.cross(q[..., :3], twice)


def rotate(q, vector):
    """Evaluate Warp's actual quaternion-rotation expression in FP64 without renormalizing inputs."""
    return (
        vector * (2 * q[..., 3:] ** 2 - 1)
        + q[..., :3] * (2 * np.sum(q[..., :3] * vector, axis=-1, keepdims=True))
        + 2 * q[..., 3:] * np.cross(q[..., :3], vector)
    )


def compose(a, b):
    """Compose the actual transform law independently in FP64."""
    return np.concatenate((a[..., :3] + rotate(a[..., 3:], b[..., :3]), quat_multiply(a[..., 3:], b[..., 3:])), axis=-1)


def inverse(transform):
    """Use the original conjugate-based transform inverse in FP64."""
    q = transform[..., 3:].copy()
    q[..., :3] *= -1
    return np.concatenate((rotate(q, -transform[..., :3]), q), axis=-1)


def fp64_reference(bundle):
    """Evaluate current fixed/scalar/free geometry and the original serial V/A law."""
    values = {k: x.numpy().astype(np.float64) for k, x in bundle.values.items() if hasattr(x, "numpy")}
    ids, joints, parents = bundle.plan.body_ids, bundle.plan.joint_ids, bundle.plan.body_parent
    worlds = len(ids)
    pose = np.zeros((worlds, 32, 7))
    anchors = np.zeros_like(pose)
    for lane in range(32):
        joint = joints[:, lane]
        qstart = values["joint_q_start"][joint].astype(int)
        dstart = values["joint_qd_start"][joint].astype(int)
        kind = int(values["joint_type"][joint[0]])
        if not np.all(values["joint_type"][joint] == kind):
            raise ValueError("Fixture has heterogeneous joint types")
        transform = np.zeros((worlds, 7))
        transform[:, 6] = 1
        if kind == 1:
            angle = values["joint_q_new"][qstart]
            transform[:, 3:6] = values["joint_axis"][dstart] * np.sin(angle[:, None] / 2)
            transform[:, 6] = np.cos(angle / 2)
        elif kind == 0:
            transform[:, :3] = values["joint_axis"][dstart] * values["joint_q_new"][qstart, None]
        elif kind == 4:
            transform = values["joint_q_new"][qstart[:, None] + np.arange(7)]
        elif kind != 3:
            raise ValueError("Fixture supports only actual fixed/scalar/free types")
        anchor = values["joint_X_p"][joint]
        parent = parents[:, lane]
        selected = parent >= 0
        anchor[selected] = compose(pose[np.arange(worlds)[selected], parent[selected]], anchor[selected])
        anchors[:, lane] = anchor
        pose[:, lane] = compose(compose(anchor, transform), inverse(values["joint_X_c"][joint]))
    origins = np.zeros((worlds, 3, 3))
    for group, lane in enumerate((0, 30, 31)):
        origins[:, group] = pose[:, lane, :3] + rotate(pose[:, lane, 3:], values["body_com"][ids[:, lane]])
    velocity, acceleration = np.zeros((worlds, 32, 6)), np.zeros((worlds, 32, 6))
    motion = np.zeros_like(values["joint_S_s"])
    for lane in range(32):
        joint = joints[:, lane]
        dstart = values["joint_qd_start"][joint].astype(int)
        kind = int(values["joint_type"][joint[0]])
        group = 0 if lane < 30 else lane - 29
        anchor = anchors[:, lane]
        radius = anchor[:, :3] - origins[:, group]
        u = np.zeros((worlds, 6))
        if kind in (0, 1):
            axis = rotate(anchor[:, 3:], values["joint_axis"][dstart])
            if kind == 0:
                motion[dstart, :3] = axis
            else:
                motion[dstart, 3:] = axis
                motion[dstart, :3] = np.cross(radius, axis)
            u = motion[dstart] * values["joint_qd_new"][dstart, None]
        elif kind == 4:
            for component in range(3):
                axis = rotate(anchor[:, 3:], np.broadcast_to(np.eye(3)[component], (worlds, 3)))
                motion[dstart + component, :3] = axis
                motion[dstart + component + 3, 3:] = axis
            u[:, :3] = rotate(anchor[:, 3:], values["joint_qd_new"][dstart[:, None] + np.arange(3)])
            u[:, 3:] = rotate(anchor[:, 3:], values["joint_qd_new"][dstart[:, None] + np.arange(3, 6)])
        parent = parents[:, lane]
        selected = parent >= 0
        pv, pa = np.zeros_like(u), np.zeros_like(u)
        pv[selected] = velocity[np.arange(worlds)[selected], parent[selected]]
        pa[selected] = acceleration[np.arange(worlds)[selected], parent[selected]]
        velocity[:, lane] = pv + u
        acceleration[:, lane, :3] = pa[:, :3] + np.cross(pv[:, 3:], u[:, :3]) + np.cross(pv[:, :3], u[:, 3:])
        acceleration[:, lane, 3:] = pa[:, 3:] + np.cross(pv[:, 3:], u[:, 3:])
    return {
        "body_q": pose,
        "body_q_com": compose(pose, values["body_X_com"][ids]),
        "body_v_s": velocity,
        "body_a_s": acceleration,
        "joint_S_s": motion,
    }


def physical_metrics(bundle):
    """Remove world/root translation and compare physical COM V/A and bias wrench."""
    values = {
        name: value.numpy().astype(np.float64) for name, value in bundle.values.items() if hasattr(value, "numpy")
    }
    arts = values["body_to_articulation"].astype(int)
    roots = values["joint_child"].astype(int)[values["articulation_start"].astype(int)[:-1]]
    root_pose = values["body_q"][roots[arts]]
    relative_position = rotate_inverse(root_pose[:, 3:], values["body_q"][:, :3] - root_pose[:, :3])
    inverse_root = root_pose[:, 3:].copy()
    inverse_root[:, :3] *= -1
    relative_rotation = quat_multiply(inverse_root, values["body_q"][:, 3:])
    relative_rotation /= np.linalg.norm(relative_rotation, axis=1, keepdims=True)
    relative_rotation *= np.where(relative_rotation[:, 3:] < 0, -1, 1)
    radius = values["body_q_com"][:, :3] - values["articulation_origin"][arts]
    velocity, acceleration = values["body_v_s"], values["body_a_s"]
    com_velocity = velocity[:, :3] + np.cross(velocity[:, 3:], radius)
    # The cache acceleration is bias acceleration (qdd=0), not full solved
    # body acceleration. Translate its spatial value and centrifugal term.
    com_acceleration = (
        acceleration[:, :3] + np.cross(acceleration[:, 3:], radius) + np.cross(velocity[:, 3:], com_velocity)
    )
    force = values["body_f_s"]
    com_wrench = np.concatenate((force[:, :3], force[:, 3:] - np.cross(radius, force[:, :3])), axis=1)
    return {
        "root_relative_position_m": relative_position,
        "root_relative_quaternion": relative_rotation,
        "com_velocity_m_s": com_velocity,
        "com_bias_acceleration_m_s2": com_acceleration,
        "com_bias_wrench_N_Nm": com_wrench,
    }


def check_outputs(test, candidate, reference):
    """Check all15 fields without making the less accurate original FP32 FK the oracle."""
    for name in OUTPUTS:
        test.assertTrue(np.isfinite(candidate.values[name].numpy()).all(), name)
    for name in ("v_new", "joint_qdd", "joint_q_new", "joint_qd_new", "articulation_origin", "fk_id_cache_valid"):
        actual, expected = candidate.values[name].numpy(), reference.values[name].numpy()
        np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-6, err_msg=name)
    oracle = fp64_reference(reference)
    for name, expected in oracle.items():
        errors = []
        for bundle in (reference, candidate):
            actual = bundle.values[name].numpy().astype(np.float64)
            if name != "joint_S_s":
                actual = actual[bundle.plan.body_ids]
            errors.append(float(np.max(np.abs(actual - expected))))
            # Absolute bounds are not scaled by global world translation.
            np.testing.assert_allclose(actual, expected, atol=5e-5, rtol=0, err_msg=name)
            if name in ("body_q", "body_q_com"):
                np.testing.assert_allclose(actual[..., 3:], expected[..., 3:], atol=3e-6, rtol=0)
        test.assertLessEqual(errors[1], errors[0] + 3e-6, (name, errors))
    # Original finalization on the actual candidate geometry/V/A independently
    # checks all inertia, compact terms, bias force and public COM velocity.
    final_values = dict(candidate.values)
    final_outputs = ("body_I_s", "body_inertia_terms", "body_f_s", "body_qd")
    for name in final_outputs:
        final_values[name] = wp.clone(candidate.values[name])
    wp.launch(
        CHAIN[4],
        dim=candidate.dimensions[4],
        inputs=[final_values[arg.label] for arg in CHAIN[4].adj.args],
        device=candidate.device,
    )
    for name in final_outputs:
        np.testing.assert_allclose(candidate.values[name].numpy(), final_values[name].numpy(), atol=3e-6, rtol=3e-6)
    actual, expected = physical_metrics(candidate), physical_metrics(reference)
    bounds = {
        "root_relative_position_m": 5e-5,
        "root_relative_quaternion": 3e-6,
        "com_velocity_m_s": 5e-5,
        "com_bias_acceleration_m_s2": 1e-4,
        "com_bias_wrench_N_Nm": 1e-4,
    }
    for name, bound in bounds.items():
        np.testing.assert_allclose(actual[name], expected[name], atol=bound, rtol=0, err_msg=name)


class TestWorldScanPublication(unittest.TestCase):
    def test_original_finalizer_statement_recovery(self):
        """Recover every original finalizer statement after restoring only its private input reads."""
        original_ast = ast.parse(textwrap.dedent(inspect.getsource(kernels.finalize_body_dynamics_body.func))).body[0]
        adapted = ast.parse(scan.finalizer_source()).body[0]
        replacements = {
            "pose": "body_q[body]",
            "pose_com": "body_q_com[body]",
            "origin_value": "articulation_origin[articulation]",
            "velocity": "body_v_s[body]",
            "acceleration": "body_a_s[body]",
        }

        class Restore(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id in replacements:
                    return ast.parse(replacements[node.id], mode="eval").body
                return node

        restored = Restore().visit(ast.Module(body=adapted.body, type_ignores=[]))
        self.assertEqual(ast.dump(restored), ast.dump(ast.Module(body=original_ast.body, type_ignores=[])))

    def test_actual_plan_global35_and_depth_guard(self):
        """Cover both free roots and reject unsupported depth before native dispatch."""
        self.assertTrue(callable(scan.get_kernel))
        for snapshot in saved.snapshots():
            plan = light.build_plan(**saved.plan_inputs(snapshot))
            scan.validate_scan_plan(plan)
            np.testing.assert_array_equal(np.sort(plan.dof_ids.reshape(-1)), np.arange(512 * 35))
            np.testing.assert_array_equal(np.sort(plan.body_ids.reshape(-1)), np.arange(512 * 32))
            invalid = copy.deepcopy(plan)
            invalid.body_parent[:, :18] = np.arange(-1, 17)
            with self.assertRaisesRegex(ValueError, "fifteen"):
                scan.validate_scan_plan(invalid)
            invalid.body_parent[:, 0] = 0
            with self.assertRaisesRegex(ValueError, "parent-first"):
                scan.validate_scan_plan(invalid)

    def check_case(self, snapshot, device, *, refresh, changed=False):
        """Compare independent original and new owners and enforce held/input immutability."""
        candidate = bind(snapshot, device, refresh=refresh, changed=changed)
        reference = bind(snapshot, device, refresh=refresh, changed=changed)
        before = {
            name: value.numpy().copy()
            for name, value in candidate.values.items()
            if name not in OUTPUTS and hasattr(value, "numpy")
        }
        held = {name: candidate.values[name].numpy().copy() for name in ("body_I_s", "body_inertia_terms")}
        original(reference)
        launch(candidate)
        check_outputs(self, candidate, reference)
        dof_mask = candidate.values["kinematic_dof_mask"].numpy() != 0
        joint_mask = candidate.values["kinematic_joint_mask"].numpy() != 0
        coordinate_mask = np.repeat(joint_mask, np.diff(candidate.values["joint_q_start"].numpy()))
        np.testing.assert_array_equal(candidate.values["v_new"].numpy()[dof_mask], before["joint_qd"][dof_mask])
        np.testing.assert_array_equal(candidate.values["joint_qd_new"].numpy()[dof_mask], before["joint_qd"][dof_mask])
        np.testing.assert_array_equal(
            candidate.values["joint_q_new"].numpy()[coordinate_mask], before["joint_q"][coordinate_mask]
        )
        np.testing.assert_array_equal(candidate.values["joint_qdd"].numpy()[dof_mask], 0)
        np.testing.assert_array_equal(candidate.values["fk_id_cache_valid"].numpy(), 1)
        for name, previous in before.items():
            np.testing.assert_array_equal(candidate.values[name].numpy(), previous, err_msg=name)
        if not refresh[0]:
            arts = candidate.values["body_to_articulation"].numpy()
            free = candidate.values["is_free_rigid"].numpy()
            mask = free[arts] == 0
            np.testing.assert_array_equal(candidate.values["body_I_s"].numpy()[mask], held["body_I_s"][mask])
        if not refresh[1]:
            np.testing.assert_array_equal(candidate.values["body_inertia_terms"].numpy(), held["body_inertia_terms"])
        return candidate, reference

    def test_loaded_all_four_refresh_and_reuse(self):
        """Match all512 worlds from both GPUs at reuse, compact refresh and full refresh."""
        for index, snapshot in enumerate(saved.snapshots()):
            for refresh in ((0, 0), (0, 1), (1, 0)):
                with self.subTest(snapshot=index, refresh=refresh):
                    self.check_case(snapshot, "cpu", refresh=refresh)

    def test_changed_frames_com_gravity_free_and_prescribed_roots(self):
        """Read changed numeric properties and moving prescribed/free roots on both epochs."""
        for index, snapshot in enumerate(saved.snapshots()):
            with self.subTest(snapshot=index):
                self.check_case(snapshot, "cpu", refresh=(index % 2, 1 - index % 2), changed=True)

    def test_physical_metric_rejects_global_translation_masking(self):
        """Reject a local millimetre defect even at large global world translation."""
        snapshots = saved.snapshots()
        snapshot = next(snapshots)
        reference = bind(snapshot, "cpu")
        original(reference)
        candidate = bind(snapshot, "cpu")
        original(candidate)
        pose = candidate.values["body_q"].numpy()
        pose[candidate.plan.body_ids[-1, 20], 0] += 0.001
        candidate.values["body_q"].assign(pose)
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(
                physical_metrics(candidate)["root_relative_position_m"],
                physical_metrics(reference)["root_relative_position_m"],
                atol=3e-6,
                rtol=3e-6,
            )

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA requires the root GPU lease")
    def test_cuda_all_four_and_graph_numeric_reset(self):
        """Compare actual native publication and stable graph replays after numeric resets."""
        device = wp.get_device("cuda:0")
        for index, snapshot in enumerate(saved.snapshots()):
            candidate, reference = self.check_case(snapshot, device, refresh=(0, index % 2))
            with wp.ScopedCapture(device=device) as captured:
                launch(candidate)
            for changed in (True, False):
                fresh = bind(snapshot, device, refresh=(0, index % 2), changed=changed)
                reference = bind(snapshot, device, refresh=(0, index % 2), changed=changed)
                # Keep captured pointers; reset all arrays, including immutable
                # topology-compatible numeric properties and writable seeds.
                for name, value in candidate.values.items():
                    if hasattr(value, "numpy"):
                        wp.copy(value, fresh.values[name])
                # Captured scalar parameters cannot change; match that contract.
                reference.values["angular_damping"] = candidate.values["angular_damping"]
                original(reference)
                wp.capture_launch(captured.graph)
                check_outputs(self, candidate, reference)


if __name__ == "__main__":
    unittest.main()
