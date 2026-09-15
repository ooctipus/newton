# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check complete Franka state ownership with unchanged constraint consumers."""

import hashlib
import importlib
import importlib.util
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import world_scan_owner
from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS
from tools.fpgs_bench.test_franka_packet_local import captures

REFERENCE = Path(
    "/home/octi/Projects/newton-fpgs-franka-scan-publication-20260914/tools/fpgs_bench/test_world_scan_publication.py"
)
REFERENCE_SHA = "dbd042225e9f5bc8ce9f8ef3688f97e950a6255c4ea03fa7b352cac3d4637084"
DT = 1.0 / 240.0


def reference_module():
    """Reuse the pinned generalized FP64/original publication test, not its runtime owner."""
    assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest() == REFERENCE_SHA
    spec = importlib.util.spec_from_file_location("franka_kinetic_public_reference", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bind_saved(snapshot, capture, device, *, changed=False):
    """Bind only current model/state/force and held-factor inputs from the four original payloads."""
    native = importlib.import_module("newton._src.solvers.feather_pgs.franka_kinetic_state")
    model = SimpleNamespace(**capture["settings"]["model"])
    model.device = wp.get_device(device)
    solver = SimpleNamespace(model=model, world_count=model.world_count, angular_damping=0.05, requires_grad=False)

    def array(prefix, name, dtype):
        return wp.array(snapshot[prefix + name], dtype=dtype, device=device)

    for name in set(world_scan_owner.MODEL_FIELDS) | set(world_scan_owner.PLAN_FIELDS):
        variable = native.PublicationData.vars.get(name)
        dtype = variable.type.dtype if variable is not None else int
        setattr(model, name, array("model__", name, dtype))
    for name, source in world_scan_owner.SOLVER_FIELDS.items():
        setattr(solver, source, array("solver__", source, native.PublicationData.vars[name].type.dtype))
    for name in world_scan_owner.CACHE_FIELDS:
        field = "_body_inertia_terms" if name == "body_inertia_terms" else name
        setattr(solver, field, array("solver__", field, native.PublicationData.vars[name].type.dtype))
    for name in (
        "art_to_world",
        "articulation_joint_end",
        "articulation_dof_start",
        "articulation_response_dof_count",
        "_prescribed_articulation",
        "body_response_dof_mask",
        "_mass_update_requested",
    ):
        setattr(solver, name, array("solver__", name, int))
    for name in ("group_to_art", "_crba_source_dof_by_size", "L_by_size"):
        setattr(
            solver,
            name,
            {size: array("solver__", f"{name}__{size}", float if name == "L_by_size" else int) for size in (9, 6)},
        )
    solver.n_arts_by_size = {size: solver.group_to_art[size].size for size in (9, 6)}
    for name in ("_passive_spring_stiffness", "_passive_spring_ref", "_passive_joint_damping", "joint_qdd", "v_hat"):
        setattr(solver, name, array("solver__", name, float))
    # Captured joint_tau is already total force, not the augmented u0 input.
    solver.joint_tau = wp.zeros(model.joint_dof_count, dtype=float, device=device)
    state = SimpleNamespace(requires_grad=False)
    for name, dtype in (
        ("joint_q", float),
        ("joint_qd", float),
        ("body_q", wp.transform),
        ("body_qd", wp.spatial_vector),
        ("body_f", wp.spatial_vector),
    ):
        setattr(state, name, array("state_in__", name, dtype))
    output = SimpleNamespace(
        requires_grad=False,
        **{name: wp.clone(getattr(state, name)) for name in ("joint_q", "joint_qd", "body_q", "body_qd", "body_f")},
    )
    control = SimpleNamespace(joint_f=array("control__", "joint_f", float))
    solver.v_out = array("solved__", "v_out", float)
    if changed:
        for name, angle in (("joint_X_p", 0.17), ("joint_X_c", -0.13)):
            values = getattr(model, name).numpy().copy()
            roots = np.flatnonzero(model.joint_parent.numpy() < 0)
            values[roots, :3] += np.array([0.03, -0.02, 0.01], np.float32)
            values[roots, 3:] = np.asarray(wp.quat_from_axis_angle(wp.normalize(wp.vec3(1, 2, -1)), angle))
            getattr(model, name).assign(values)
        model.body_mass.assign(model.body_mass.numpy() * np.float32(1.02))
        model.body_inertia.assign(model.body_inertia.numpy() * np.float32(0.99))
        com = model.body_com.numpy().copy()
        com[:, 1] += np.float32(0.002)
        model.body_com.assign(com)
        values = solver.body_X_com.numpy().copy()
        values[:, :3] = com
        solver.body_X_com.assign(values)
        gravity = model.gravity.numpy().copy()
        gravity[0] += np.array([0.12, -0.08, 0.04], np.float32)
        model.gravity.assign(gravity)
    owner = native.FrankaKineticState(solver)
    return SimpleNamespace(
        model=model,
        solver=solver,
        state=state,
        output=output,
        control=control,
        owner=owner,
        device=model.device,
        dt=capture["dt"],
    )


def launch_state(case, *, finish=False, refresh=True):
    """Exercise the actual native ABI without pretending a captured namespace is a production constructor."""
    owner = case.owner
    output = case.output if finish else case.state
    wp.launch_tiled(
        owner.finish_kernel if finish else owner.repair_kernel,
        dim=[case.model.world_count],
        inputs=[
            owner.plan,
            owner._publication(case.state, case.solver, output, case.dt),
            owner.data,
            case.solver._mass_update_requested,
            int(refresh),
        ],
        block_dim=32,
        device=case.device,
    )


def publication_values(case, state):
    """Provide existing reference arguments from independently bound current inputs."""
    data = case.owner._publication(case.state, case.solver, state, case.dt)
    values = {name: getattr(data, name) for name in world_scan_owner.world_scan_publication.PublicationData.vars}
    values.update(
        articulation_start=case.model.articulation_start,
        articulation_joint_end=case.solver.articulation_joint_end,
        v_new=data.v_out,
        inv_dt=1 / case.dt,
    )
    plan = SimpleNamespace(**case.owner.host_plan)
    return SimpleNamespace(
        values=values,
        plan=plan,
        physical_bodies=13,
        primary_bodies=11,
        primary_dofs=9,
        device=case.device,
        dimensions=(
            case.model.joint_dof_count,
            case.solver._free_root_joint_indices.size,
            case.model.joint_count,
            case.model.articulation_count,
            case.model.body_count,
        ),
    )


def physical_reference(case, state):
    """Assemble H, positive bias and COM velocity from current FP64 poses/motion and authored inertias."""
    module = reference_module()
    result = module.fp64_reference(publication_values(case, state))
    plan, model = case.owner.host_plan, case.model
    ids = plan["body_ids"][:, :13]
    mass = model.body_mass.numpy()[ids].astype(float)
    inertia = model.body_inertia.numpy()[ids].astype(float)
    pose, velocity, acceleration = (result[name] for name in ("body_q", "body_v_s", "body_a_s"))
    rotation = np.stack(
        [module.rotate(pose[..., 3:], np.broadcast_to(axis, pose[..., :3].shape)) for axis in np.eye(3)], axis=-1
    )
    inertia = rotation @ inertia @ rotation.swapaxes(-1, -2)
    com = pose[..., :3] + module.rotate(pose[..., 3:], model.body_com.numpy()[ids].astype(float))
    origins = com[:, [0, 11, 12]]
    radius = com - origins[:, np.r_[np.zeros(11, int), 1, 2]]
    public = velocity.copy()
    public[..., :3] += np.cross(velocity[..., 3:], radius)
    omega, alpha = velocity[..., 3:], acceleration[..., 3:]
    force = mass[..., None] * (
        acceleration[..., :3] + np.cross(alpha, radius) + np.cross(omega, public[..., :3]) - model.gravity.numpy()[0]
    )
    torque = np.einsum("...ij,...j->...i", inertia, alpha)
    torque += np.cross(omega, np.einsum("...ij,...j->...i", inertia, omega)) + np.cross(radius, force)
    wrench = np.concatenate((force, torque), axis=-1)
    axes = result["joint_S_s"][plan["dof_ids"]]
    h = np.zeros((model.world_count, 9, 9))
    bias = np.zeros((model.world_count, 9))
    for body in range(11):
        chain, cursor = [], body
        while cursor >= 0:
            dof = int(plan["body_local_dof"][0, cursor])
            if dof >= 0:
                chain.append(dof)
            cursor = int(plan["body_parent"][0, cursor])
        jac = np.zeros((model.world_count, 6, 9))
        for dof in chain:
            jac[:, 3:, dof] = axes[:, dof, 3:]
            jac[:, :3, dof] = axes[:, dof, :3] + np.cross(axes[:, dof, 3:], radius[:, body])
            bias[:, dof] += np.sum(axes[:, dof] * wrench[:, body], axis=-1)
        h += mass[:, body, None, None] * np.einsum("wki,wkj->wij", jac[:, :3], jac[:, :3])
        h += jac[:, 3:].swapaxes(-1, -2) @ inertia[:, body] @ jac[:, 3:]
    return dict(
        result, public=public, H=h, bias=bias, wrench=wrench, radius=radius, axes=axes, inertia=inertia, mass=mass
    )


def record_error(test, name, value):
    """Record observed normalized errors without changing any acceptance threshold."""
    if not hasattr(test, "_physical_errors"):
        test._physical_errors = {}
    test._physical_errors[name] = max(test._physical_errors.get(name, 0.0), float(value))


def report_errors(test):
    """Print the measured error maxima alongside the unchanged native assertions."""
    print(
        "franka_kinetic_errors "
        + json.dumps(
            {"selector": test._testMethodName, "max_normalized_errors": getattr(test, "_physical_errors", {})},
            sort_keys=True,
        ),
        flush=True,
    )


def scaled(test, actual, expected, *, tolerance=3e-6, name="physical"):
    """Use dimensionally separate physical vectors/matrices, not near-zero component identity."""
    actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    test.assertTrue(np.isfinite(actual).all(), name)
    error = np.linalg.norm((actual - expected).reshape(len(actual), -1), axis=1)
    scale = 1 + np.linalg.norm(expected.reshape(len(expected), -1), axis=1)
    measured = float(np.max(error / scale))
    record_error(test, name, measured)
    test.assertLessEqual(measured, tolerance, name)


def check_current(test, case, *, geometry=True):
    """Check all physical/public outputs and the retained free/prescribed consumer services."""
    ref = physical_reference(case, case.state)
    plan, owner, solver = case.owner.host_plan, case.owner, case.solver
    ids = plan["body_ids"][:, :13]
    np.testing.assert_allclose(case.state.body_q.numpy()[ids], ref["body_q"], atol=3e-6, rtol=2e-5)
    for half in (slice(0, 3), slice(3, 6)):
        scaled(
            test,
            case.state.body_qd.numpy()[ids][..., half].reshape(-1, 3),
            ref["public"][..., half].reshape(-1, 3),
            name="public COM/angular velocity",
        )
    if geometry:
        scaled(test, owner.geometric.numpy()[plan["primary_group"]].reshape(-1, 9, 9), ref["H"], name="current H9")
    scaled(test, owner.bias.numpy(), ref["bias"], name="positive current bias")
    free = np.zeros((case.model.world_count, 6, 6))
    free[:, :3, :3] = ref["mass"][:, 11, None, None] * np.eye(3)
    free[:, 3:, 3:] = ref["inertia"][:, 11]
    scaled(test, solver.body_I_s.numpy()[ids[:, 11]], free, name="free full inertia for MF")
    scaled(test, solver.body_v_s.numpy()[ids[:, 12]], ref["body_v_s"][:, 12], name="prescribed current motion")
    owner.check()
    return ref


def check_predictor(test, case, ref):
    """Apply current force buckets against the independently retained captured L9/L6."""
    native = importlib.import_module("newton._src.solvers.feather_pgs.franka_kinetic_state")
    solver, state, model, plan = case.solver, case.state, case.model, case.owner.host_plan
    solver.joint_tau.assign(np.linspace(-0.13, 0.17, model.joint_dof_count, dtype=np.float32))
    # Nonzero external torque/force and passive controls deliberately differ from capture.
    state.body_f.assign(np.random.default_rng(902).normal(0, 0.4, (model.body_count, 6)).astype(np.float32))
    force = native.ForceInput()
    aliases = {
        "joint_q": state.joint_q,
        "joint_qd": state.joint_qd,
        "predictor_qd": state.joint_qd,
        "joint_f": case.control.joint_f,
        "body_f": state.body_f,
        "body_flags": model.body_flags,
        "stiffness": solver._passive_spring_stiffness,
        "reference": solver._passive_spring_ref,
        "damping": solver._passive_joint_damping,
        "u0": solver.joint_tau,
        "lower9": solver.L_by_size[9],
        "lower6": solver.L_by_size[6],
        "kinematic_dof": solver._kinematic_dof_mask,
        "kinematic_joint": solver._kinematic_joint_mask,
        "qdd": solver.joint_qdd,
        "vhat": solver.v_hat,
        "dt": case.dt,
        "S": solver.joint_S_s,
    }
    for name in native.ForceInput.vars:
        setattr(force, name, aliases[name])
    ids, dofs = plan["body_ids"][:, :13], plan["dof_ids"]
    ext = state.body_f.numpy()[ids].astype(float)
    ext[..., 3:] += np.cross(ref["radius"], ext[..., :3])
    ext[(model.body_flags.numpy()[ids] & int(newton.BodyFlags.KINEMATIC)) != 0] = 0
    tau = case.control.joint_f.numpy()[dofs].astype(float) + solver.joint_tau.numpy()[dofs]
    for body in range(10, 0, -1):
        ext[:, int(plan["body_parent"][0, body])] += ext[:, body]
    q, qd = state.joint_q.numpy(), state.joint_qd.numpy()[dofs].astype(float)
    for dof, body in enumerate(plan["dof_body"]):
        tau[:, dof] += np.sum(ref["axes"][:, dof] * ext[:, body], axis=-1) - ref["bias"][:, dof]
        index = dofs[:, dof]
        tau[:, dof] += solver._passive_spring_stiffness.numpy()[index] * (
            solver._passive_spring_ref.numpy()[index] - q[plan["q_index"][:, dof]]
        )
        tau[:, dof] -= solver._passive_joint_damping.numpy()[index] * qd[:, dof]
    tau[:, 9:15] += np.einsum("wki,wi->wk", ref["axes"][:, 9:15], ext[:, 11] - ref["wrench"][:, 11])
    expected_qdd = np.zeros_like(qd)
    held = {}
    for size, start, group in ((9, 0, plan["primary_group"]), (6, 9, plan["secondary_group"])):
        held[size] = solver.L_by_size[size].numpy().copy()
        lower = np.tril(held[size][group].astype(float))
        H = lower @ lower.swapaxes(-1, -2)
        expected_qdd[:, start : start + size] = np.linalg.solve(H, tau[:, start : start + size, None])[..., 0]
    expected_vhat = qd + case.dt * expected_qdd
    expected_vhat[:, 9:12] += case.dt * np.cross(qd[:, 12:15], qd[:, 9:12])
    wp.launch_tiled(
        case.owner.predictor_kernel,
        dim=[model.world_count],
        inputs=[case.owner.plan, case.owner.data, force],
        block_dim=32,
        device=case.device,
    )
    scaled(test, solver.v_hat.numpy()[dofs], expected_vhat, tolerance=1e-5, name="held/current predictor velocity")
    for size, start, group in ((9, 0, plan["primary_group"]), (6, 9, plan["secondary_group"])):
        lower = np.tril(held[size][group].astype(float))
        H = lower @ lower.swapaxes(-1, -2)
        actual = solver.joint_qdd.numpy()[dofs[:, start : start + size]].astype(float)
        residual = np.einsum("wij,wj->wi", H, actual) - tau[:, start : start + size]
        denominator = (
            1
            + np.linalg.norm(H, axis=(1, 2)) * np.linalg.norm(actual, axis=1)
            + np.linalg.norm(tau[:, start : start + size], axis=1)
        )
        measured = float(np.max(np.linalg.norm(residual, axis=1) / denominator))
        record_error(test, f"held momentum{size}", measured)
        test.assertLess(measured, 3e-6)
        np.testing.assert_array_equal(solver.L_by_size[size].numpy(), held[size])
    np.testing.assert_array_equal(solver.joint_qdd.numpy()[dofs[:, 15:]], np.zeros((model.world_count, 6)))
    case.owner.check()


def check_finish(test, snapshot, capture, device, *, changed=False, refresh=True):
    """Match original integration and independently check post-damping physical publication."""
    case, original = (bind_saved(snapshot, capture, device, changed=changed) for _ in range(2))
    read_only = [
        case.state.joint_q,
        case.state.joint_qd,
        case.control.joint_f,
        case.model.body_mass,
        case.model.body_com,
        case.model.body_inertia,
        case.model.joint_X_p,
        case.model.joint_X_c,
        case.solver.L_by_size[9],
        case.solver.L_by_size[6],
    ]
    inputs = [value.numpy().copy() for value in read_only]
    launch_state(case)
    geometry = case.owner.geometric.numpy().copy()
    original_values = publication_values(original, original.output)
    reference_module().original(original_values)
    launch_state(case, finish=True, refresh=refresh)
    for name in ("joint_q", "joint_qd"):
        scaled(
            test,
            getattr(case.output, name).numpy().reshape(512, -1),
            getattr(original.output, name).numpy().reshape(512, -1),
            name="original integration " + name,
        )
    before = case.state
    case.state = case.output
    check_current(test, case, geometry=refresh)
    case.state = before
    if not refresh:
        np.testing.assert_array_equal(case.owner.geometric.numpy(), geometry)
    for value, expected in zip(read_only, inputs, strict=True):
        np.testing.assert_array_equal(value.numpy(), expected)
    return case


def model_fixture(device="cpu", worlds=2):
    """Reconstruct the captured physical13-body model; simple witness boxes are test-only geometry."""
    capture = next(captures())
    with np.load(capture["path"], allow_pickle=False) as snapshot:
        part = newton.ModelBuilder()
        for body in range(13):
            part.add_link(
                xform=wp.transform(*snapshot["model__body_q"][body]),
                com=wp.vec3(*snapshot["model__body_com"][body]),
                inertia=wp.mat33(*snapshot["model__body_inertia"][body].reshape(-1)),
                mass=float(snapshot["model__body_mass"][body]),
                lock_inertia=True,
                is_kinematic=bool(snapshot["model__body_flags"][body] & int(newton.BodyFlags.KINEMATIC)),
            )
        fields = (
            "limit_lower",
            "limit_upper",
            "limit_ke",
            "limit_kd",
            "target_ke",
            "target_kd",
            "damping",
            "spring_stiffness",
            "spring_ref",
            "armature",
            "effort_limit",
            "velocity_limit",
            "friction",
        )
        for joint in range(13):
            start = int(snapshot["model__joint_qd_start"][joint])
            linear, angular = map(int, snapshot["model__joint_dof_dim"][joint])
            axes = []
            for dof in range(start, start + linear + angular):
                config = {name: float(snapshot["model__joint_" + name][dof]) for name in fields}
                config.update(
                    axis=wp.vec3(*snapshot["model__joint_axis"][dof]),
                    actuator_mode=newton.JointTargetMode(int(snapshot["model__joint_target_mode"][dof])),
                )
                axes.append(newton.ModelBuilder.JointDofConfig(**config))
            part.add_joint(
                newton.JointType(int(snapshot["model__joint_type"][joint])),
                int(snapshot["model__joint_parent"][joint]),
                int(snapshot["model__joint_child"][joint]),
                linear_axes=axes[:linear],
                angular_axes=axes[linear:],
                parent_xform=wp.transform(*snapshot["model__joint_X_p"][joint]),
                child_xform=wp.transform(*snapshot["model__joint_X_c"][joint]),
            )
        for joints in (list(range(11)), [11], [12]):
            part.add_articulation(joints)
        for index in range(2):
            part.add_constraint_mimic(
                int(snapshot["model__constraint_mimic_joint0"][index]),
                int(snapshot["model__constraint_mimic_joint1"][index]),
                float(snapshot["model__constraint_mimic_coef0"][index]),
                float(snapshot["model__constraint_mimic_coef1"][index]),
            )
        for body in (9, 11, 12):
            part.add_shape_box(body, hx=0.02, hy=0.02, hz=0.02)
        builder = newton.ModelBuilder()
        builder.replicate(part, worlds)
        model = builder.finalize(device=device)
        model.rigid_contact_max = 32
        for name, width in (("joint_q", 23), ("joint_qd", 21), ("joint_target_q", 21), ("joint_target_qd", 21)):
            getattr(model, name).assign(np.tile(snapshot["model__" + name][:width], worlds))
        model.set_gravity(np.broadcast_to(snapshot["model__gravity"][0], (worlds, 3)))
    # Nontrivial authored anchors are admitted before constructing either solver.
    for name, angle in (("joint_X_p", 0.09), ("joint_X_c", -0.07)):
        values = getattr(model, name).numpy().copy()
        roots = np.flatnonzero(model.joint_parent.numpy() < 0)
        values[roots, 3:] = np.asarray(wp.quat_from_axis_angle(wp.vec3(0, 1, 0), angle))
        values[roots, :3] += np.array([0.02, -0.01, 0.03], np.float32)
        getattr(model, name).assign(values)
    return model


def make_solver(model, enabled, **overrides):
    """Use the actual unchanged Franka matrix-free/local packet eight-sweep recipe."""
    options = {
        "pgs_mode": "matrix_free" if model.device.is_cuda else "split",
        "pgs_iterations": 8,
        "pgs_beta": 0.05,
        "pgs_cfm": 1e-6,
        "dense_max_constraints": 192,
        "mf_max_constraints": 64,
        "enable_joint_limits": True,
        "joint_limit_activation_gap": 0.1,
        "enable_joint_velocity_limits": False,
        "update_mass_matrix_interval": 2,
        "grouped_dynamics": False,
        "double_buffer": True,
        "use_parallel_streams": model.device.is_cuda,
        "angular_damping": 0.05,
    }
    options.update(overrides)
    with patch.dict(
        os.environ,
        {
            "FEATHER_PGS_FRANKA_KINETIC_STATE": str(int(enabled)),
            "FEATHER_PGS_LOCAL_ROW_PACKETS": "1" if model.device.is_cuda else "0",
            "FEATHER_PGS_SPARSE_FACTOR": "0",
            "FEATHER_PGS_WORLD_SCAN_PUBLICATION": "0",
        },
    ):
        return SolverFeatherPGS(model, **options)


def loaded_contacts(model, state):
    """Create bounded finger/free and finger/free-to-prescribed witnesses for unchanged row owners."""
    contacts = newton.Contacts(rigid_contact_max=32, soft_contact_max=0, device=model.device)
    first, second = np.full(32, -1, np.int32), np.full(32, -1, np.int32)
    point0, point1, normal = (np.zeros((32, 3), np.float32) for _ in range(3))
    shape_body, pose = model.shape_body.numpy(), state.body_q.numpy()
    for world in range(model.world_count):
        for offset, (a, b) in enumerate(((9, 11), (9, 12), (11, 12))):
            row, body0, body1 = 3 * world + offset, 13 * world + a, 13 * world + b
            first[row], second[row] = (int(np.flatnonzero(shape_body == body)[0]) for body in (body0, body1))
            normal[row] = [0, 0, 1]
            point0[row] = [0.01, -0.01, 0.0]
            world_point = wp.transform_point(wp.transform(*pose[body0]), wp.vec3(*point0[row]))
            point1[row] = np.asarray(
                wp.transform_point(wp.transform_inverse(wp.transform(*pose[body1])), world_point - wp.vec3(0, 0, 0.001))
            )
    for name, value in (
        ("shape0", first),
        ("shape1", second),
        ("point0", point0),
        ("point1", point1),
        ("normal", normal),
    ):
        getattr(contacts, "rigid_contact_" + name).assign(value)
    contacts.rigid_contact_count.fill_(3 * model.world_count)
    return contacts


def actual_case(solver, state, output):
    """Bind the same independent reference to an actual admitted production owner."""
    return SimpleNamespace(
        model=solver.model,
        solver=solver,
        owner=solver._franka_kinetic_state,
        state=state,
        output=output,
        device=solver.model.device,
        dt=DT,
    )


def check_factor(test, case):
    """Check actual refreshed L9/L6 against current physical H plus the original R/K augmentation."""
    ref, solver, plan = physical_reference(case, case.state), case.solver, case.owner.host_plan
    free = np.zeros((case.model.world_count, 6, 6))
    free[:, :3, :3] = ref["mass"][:, 11, None, None] * np.eye(3)
    free[:, 3:, 3:] = ref["inertia"][:, 11]
    axes = ref["axes"][:, 9:15]
    free = axes @ free @ axes.swapaxes(-1, -2)
    for size, H, start, groups in ((9, ref["H"], 0, plan["primary_group"]), (6, free, 9, plan["secondary_group"])):
        diagonal = solver.R_by_size[size].numpy()[groups].astype(float).copy()
        rows = solver._augmented_drive_row_by_dof.numpy()[plan["dof_ids"][:, start : start + size]]
        stiffness = solver.aug_row_K.numpy()
        valid = rows >= 0
        diagonal[valid] += np.maximum(stiffness[rows[valid]], 0)
        H[:, np.arange(size), np.arange(size)] += diagonal
        lower = np.tril(solver.L_by_size[size].numpy()[groups].astype(float))
        scaled(test, lower @ lower.swapaxes(-1, -2), H, name=f"actual refreshed factor{size}")


class TestFrankaKineticStateCPU(unittest.TestCase):
    def test_owner_api(self):
        """Require complete current/next ownership before retiring original producers."""
        module = importlib.import_module("newton._src.solvers.feather_pgs.franka_kinetic_state")
        for name in ("begin", "predict", "finish", "invalidate", "validate_model", "check"):
            self.assertTrue(callable(getattr(module.FrankaKineticState, name)))

    def test_structural_notification_reuses_complete_proof(self):
        """Unchanged fixed-root notifications must not repeat the per-world topology proof."""
        native = importlib.import_module("newton._src.solvers.feather_pgs.franka_kinetic_state")
        capture = next(captures())
        with np.load(capture["path"], allow_pickle=False) as snapshot:
            case = bind_saved(snapshot, capture, "cpu")
        with patch.object(native, "build_plan", side_effect=AssertionError("repeated topology proof")):
            for flags in (None, 0, newton.ModelFlags.JOINT_PROPERTIES):
                case.owner.validate_model(flags)
        # Preserve the existing numeric-notification contract, not a wider one.
        with patch.object(native, "_fingerprint", side_effect=AssertionError("numeric proof rescan")):
            for flags in (
                newton.ModelFlags.JOINT_DOF_PROPERTIES,
                newton.ModelFlags.BODY_INERTIAL_PROPERTIES,
                newton.ModelFlags.SHAPE_PROPERTIES,
                newton.ModelFlags.MODEL_PROPERTIES,
            ):
                case.owner.validate_model(flags)

    def test_structural_notification_rejects_changed_proof_inputs(self):
        """Every plan input and allocation dimension still requires reconstruction when changed."""
        capture = next(captures())
        with np.load(capture["path"], allow_pickle=False) as snapshot:
            case = bind_saved(snapshot, capture, "cpu")
        model, solver = case.model, case.solver
        arrays = [
            (name, getattr(model, name))
            for name in (*world_scan_owner.PLAN_FIELDS, "joint_axis", "joint_X_p", "joint_X_c")
        ]
        arrays.extend(
            (name, getattr(solver, name))
            for name in (
                "art_to_world",
                "articulation_joint_end",
                "articulation_dof_start",
                "articulation_response_dof_count",
                "body_to_articulation",
                "_prescribed_articulation",
                "body_response_dof_mask",
                "_free_root_joint_indices",
            )
        )
        arrays.extend((f"group_to_art[{size}]", solver.group_to_art[size]) for size in (9, 6))
        arrays.append(("source9", solver._crba_source_dof_by_size[9]))
        for name, array in arrays:
            original = array.numpy().copy()
            changed = original.copy()
            changed.flat[0] += 1
            with self.subTest(array=name):
                try:
                    array.assign(changed)
                    with self.assertRaisesRegex(RuntimeError, "reconstruct"):
                        case.owner.validate_model(newton.ModelFlags.JOINT_PROPERTIES)
                finally:
                    array.assign(original)
        for target, fields in (
            (
                model,
                (
                    "world_count",
                    "body_count",
                    "joint_count",
                    "articulation_count",
                    "joint_coord_count",
                    "joint_dof_count",
                ),
            ),
            (solver, ("world_count",)),
        ):
            for name in fields:
                original = getattr(target, name)
                with self.subTest(scalar=name):
                    try:
                        setattr(target, name, original + 1)
                        with self.assertRaisesRegex(RuntimeError, "reconstruct"):
                            case.owner.validate_model(newton.ModelFlags.JOINT_PROPERTIES)
                    finally:
                        setattr(target, name, original)
        for size in (9, 6):
            original = solver.n_arts_by_size[size]
            with self.subTest(group_count=size):
                try:
                    solver.n_arts_by_size[size] += 1
                    with self.assertRaisesRegex(RuntimeError, "reconstruct"):
                        case.owner.validate_model(newton.ModelFlags.JOINT_PROPERTIES)
                finally:
                    solver.n_arts_by_size[size] = original
        case.owner.validate_model(newton.ModelFlags.JOINT_PROPERTIES)

    def test_saved_current_physical_geometry_and_held_force(self):
        """Check four real current epochs, held factors and every retained free/prescribed service."""
        for capture in captures():
            with self.subTest(path=capture["path"]), np.load(capture["path"], allow_pickle=False) as snapshot:
                case = bind_saved(snapshot, capture, "cpu")
                launch_state(case)
                ref = check_current(self, case)
                check_predictor(self, case, ref)

    def test_saved_original_integration_and_current_publication(self):
        """Check both held/current publication epochs and changed free-root anchors against original integration."""
        for index, capture in enumerate(captures()):
            with self.subTest(path=capture["path"]), np.load(capture["path"], allow_pickle=False) as snapshot:
                check_finish(self, snapshot, capture, "cpu", changed=bool(index & 1), refresh=bool(index & 1))

    def test_actual_constructor_and_unsupported_guards(self):
        """Construct the actual physical model and reject changed structural ownership before cache reads."""
        native = importlib.import_module("newton._src.solvers.feather_pgs.franka_kinetic_state")
        model = model_fixture("cpu")
        solver = make_solver(model, True)
        self.assertIsNone(solver._franka_kinetic_state)
        self.assertFalse(native.supported(solver))
        # CPU split intentionally retains prescribed response coordinates;
        # validate the new complete plan against an actual captured MF owner.
        capture = next(captures())
        with np.load(capture["path"], allow_pickle=False) as snapshot:
            case = bind_saved(snapshot, capture, "cpu")
        owner, model = case.owner, case.model
        owner.validate_model()
        flags = model.body_flags.numpy().copy()
        flags[9] |= int(newton.BodyFlags.KINEMATIC)
        model.body_flags.assign(flags)
        with self.assertRaisesRegex(RuntimeError, "reconstruct"):
            owner.validate_model()


class TestFrankaKineticStateCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Use only the single root-leased visible GPU."""
        if not wp.get_cuda_devices():
            raise unittest.SkipTest("Root owns CUDA lease")
        cls.device = wp.get_cuda_devices()[0]

    def test_saved_current_geometry_held_force_and_services(self):
        """Check all four loaded current/held epochs with nonzero live forces and prescribed response."""
        for index, capture in enumerate(captures()):
            with self.subTest(path=capture["path"]), np.load(capture["path"], allow_pickle=False) as snapshot:
                case = bind_saved(snapshot, capture, self.device, changed=bool(index & 1))
                launch_state(case)
                ref = check_current(self, case)
                check_predictor(self, case, ref)
        report_errors(self)

    def test_saved_original_publication_and_graph_repair(self):
        """Match original publication and replay current-source repairs after selective invalidation."""
        for index, capture in enumerate(captures()):
            with self.subTest(path=capture["path"]), np.load(capture["path"], allow_pickle=False) as snapshot:
                case = check_finish(
                    self, snapshot, capture, self.device, changed=bool(index & 1), refresh=bool(index & 1)
                )
                case.state = case.output
                mask = wp.array(np.arange(512) % 2 == 0, dtype=wp.bool, device=self.device)
                wp.synchronize_device(self.device)
                with wp.ScopedCapture(device=self.device) as captured:
                    launch_state(case, refresh=True)
                for requested in (0, 1, 0):
                    case.owner.invalidate(mask)
                    case.solver._mass_update_requested.fill_(requested)
                    wp.capture_launch(captured.graph)
                    check_current(self, case)
        report_errors(self)

    def test_actual_steps_loaded_reset_masked_refresh_and_graph(self):
        """Exercise actual eight-sweep dispatch, current/held factors, public outputs and reset/graph lifetimes."""
        model = model_fixture(self.device, worlds=2)
        solvers = [make_solver(model, enabled) for enabled in (False, True)]
        self.addCleanup(lambda owners=solvers: wp.synchronize_device(owners[0].model.device))
        original, candidate = solvers
        self.assertIsNone(original._franka_kinetic_state)
        self.assertIsNotNone(candidate._franka_kinetic_state)
        self.assertIsNotNone(candidate._row_packets)
        states = [[model.state(), model.state()] for _ in solvers]
        controls = [model.control() for _ in solvers]
        for pair, control in zip(states, controls, strict=True):
            pair[0].joint_qd.assign(np.linspace(-0.12, 0.17, model.joint_dof_count, dtype=np.float32))
            pair[0].body_f.assign(np.random.default_rng(817).normal(0, 0.3, (model.body_count, 6)).astype(np.float32))
            control.joint_f.assign(np.linspace(-0.1, 0.15, model.joint_dof_count, dtype=np.float32))
            newton.eval_fk(model, pair[0].joint_q, pair[0].joint_qd, pair[0])
        contacts = loaded_contacts(model, states[0][0])
        seen = []
        old_launch, old_tiled = wp.launch, wp.launch_tiled

        def watch(original_launch):
            def launch(*args, **kwargs):
                kernel = kwargs.get("kernel", args[0] if args else None)
                seen.append(kernel.key)
                return original_launch(*args, **kwargs)

            return launch

        def complete(source, destination, refresh):
            for index, (solver, pair, control) in enumerate(zip(solvers, states, controls, strict=True)):
                inputs = {
                    name: getattr(pair[source], name).numpy().copy() for name in ("joint_q", "joint_qd", "body_f")
                }
                control_before = control.joint_f.numpy().copy()
                if index:
                    with (
                        patch.object(wp, "launch", watch(old_launch)),
                        patch.object(wp, "launch_tiled", watch(old_tiled)),
                    ):
                        solver.step(pair[source], pair[destination], control, contacts, DT)
                else:
                    solver.step(pair[source], pair[destination], control, contacts, DT)
                solver.check_constraint_capacity()
                self.assertFalse(solver._force_mass_update)
                np.testing.assert_array_equal(solver._mass_update_requested.numpy(), [0])
                np.testing.assert_array_equal(solver.mass_update_mask.numpy(), np.full(6, int(refresh), np.int32))
                for name, value in inputs.items():
                    np.testing.assert_array_equal(getattr(pair[source], name).numpy(), value)
                np.testing.assert_array_equal(control.joint_f.numpy(), control_before)
            if refresh:
                check_factor(self, actual_case(candidate, states[1][source], states[1][destination]))
            current = actual_case(candidate, states[1][destination], states[1][source])
            ref = physical_reference(current, current.state)
            ids = current.owner.host_plan["body_ids"][:, :13]
            np.testing.assert_allclose(current.state.body_q.numpy()[ids], ref["body_q"], atol=3e-6, rtol=2e-5)
            for half in (slice(0, 3), slice(3, 6)):
                scaled(
                    self,
                    current.state.body_qd.numpy()[ids][..., half].reshape(-1, 3),
                    ref["public"][..., half].reshape(-1, 3),
                    name="complete public velocity",
                )
            for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                # Secondary matched-eight-step ceiling inherited from the G1
                # complete-owner test before Franka native execution; this is
                # not a new Franka-calibrated tolerance. Independent physical
                # H/momentum/public-velocity gates retain their 3e-6 bounds.
                scaled(
                    self,
                    getattr(states[1][destination], name).numpy().reshape(2, -1),
                    getattr(states[0][destination], name).numpy().reshape(2, -1),
                    tolerance=7e-4,
                    name="matched8 forward " + name,
                )
            self.assertIs(candidate._fk_id_cache_source_state, states[1][destination])

        complete(0, 1, True)
        held = {size: candidate.L_by_size[size].numpy().copy() for size in (9, 6)}
        complete(1, 0, False)
        for size in (9, 6):
            np.testing.assert_array_equal(candidate.L_by_size[size].numpy(), held[size])
        mask = wp.array([False, True], dtype=wp.bool, device=self.device)
        for solver, pair in zip(solvers, states, strict=True):
            values = pair[0].joint_q.numpy().copy()
            values[23:32] += np.float32(0.001)
            pair[0].joint_q.assign(values)
            solver.reset(pair[0], mask)
            solver._step, solver._force_mass_update = 3, False
            solver._mass_update_requested.fill_(1)
        np.testing.assert_array_equal(candidate._franka_kinetic_state.geometry_valid.numpy(), [1, 0])
        complete(0, 1, True)
        model.body_mass.assign(model.body_mass.numpy() * np.float32(1.01))
        model.body_inertia.assign(model.body_inertia.numpy() * np.float32(1.02))
        com = model.body_com.numpy().copy()
        com[:, 0] += np.float32(0.001)
        model.body_com.assign(com)
        for solver, control in zip(solvers, controls, strict=True):
            solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
            solver._step, solver._force_mass_update = 5, False
            control.joint_f.assign(-control.joint_f.numpy())
        complete(1, 0, True)
        for retired in (
            "compute_composite_inertia",
            "eval_rigid_fk_id",
            "template_fk_kinematics",
            "finalize_body_dynamics",
            "integrate_generalized_joints",
            "compute_velocity_predictor",
        ):
            self.assertFalse(any(retired in key for key in seen), retired)
        self.assertTrue(any("franka_kinetic" in key for key in seen))
        contacts.rigid_contact_count.zero_()
        candidate._step = 8
        wp.synchronize_device(self.device)
        with wp.ScopedCapture(device=self.device) as captured:
            candidate.seed_double_buffer_events()
            candidate.step(states[1][0], states[1][1], controls[1], contacts, DT)
            candidate.step(states[1][1], states[1][0], controls[1], contacts, DT)
        for count in (0, 6, 0, 6):
            candidate.reset(states[1][0], mask)
            candidate._mass_update_requested.fill_(1)
            contacts.rigid_contact_count.fill_(count)
            wp.capture_launch(captured.graph)
            candidate.check_constraint_capacity()
            np.testing.assert_array_equal(candidate._mass_update_requested.numpy(), [0])
            current = actual_case(candidate, states[1][0], states[1][1])
            ref = physical_reference(current, current.state)
            ids = current.owner.host_plan["body_ids"][:, :13]
            for half in (slice(0, 3), slice(3, 6)):
                scaled(
                    self,
                    current.state.body_qd.numpy()[ids][..., half].reshape(-1, 3),
                    ref["public"][..., half].reshape(-1, 3),
                    name="graph public velocity",
                )
        report_errors(self)


if __name__ == "__main__":
    unittest.main()
