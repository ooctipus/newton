# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Actual original-chain and independent response controls for compact contacts."""

import inspect
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import compact_contact as compact
from newton._src.solvers.feather_pgs import kernels
from newton._src.solvers.feather_pgs import solver_feather_pgs as solver_module

OUTPUTS = (
    "row_type",
    "row_parent",
    "row_mu",
    "row_beta",
    "row_cfm",
    "phi",
    "target",
    "restitution",
    "J",
    "Y",
    "sparse_dof",
    "sparse_jy",
    "diag",
)


def fixture(device="cpu", *, prefix=(2, 3), kept=(7, 7), shared=0, friction_shared=1):
    """Include coupled/same-art/same-coordinate/static/prescribed contacts and permuted groups."""
    rng = np.random.default_rng(191)
    data = compact.ContactBoundaryData()

    def array(values, dtype):
        return wp.array(values, dtype=dtype, device=device)

    world = np.repeat(np.arange(2, dtype=np.int32), 8)
    local = np.tile(np.arange(8, dtype=np.int32), 2)
    pairs = np.array([[0, 2], [2, 5], [0, 1], [2, 3], [2, 4], [0, 4], [2, 2], [0, 2]])
    shapes0 = world * 6 + pairs[local, 0]
    shapes1 = world * 6 + pairs[local, 1]
    shapes1[1] = -1  # A truly missing shape, not just a bodyless shape.
    body_art = np.array([0, 0, 1, 1, 2, -1, 3, 3, 4, 4, 5, -1])
    art0 = body_art[shapes0]
    art1 = np.where(shapes1 >= 0, body_art[np.maximum(shapes1, 0)], -1)
    slots = np.array(
        [prefix[w] + 3 * c if c < kept[w] else -1 for w, c in zip(world, local, strict=True)], dtype=np.int32
    )
    data.count = array([16], int)
    data.point0 = array(rng.normal(0, 0.2, (16, 3)).astype(np.float32), wp.vec3)
    data.point1 = array(rng.normal(0, 0.2, (16, 3)).astype(np.float32), wp.vec3)
    normal = rng.normal(size=(16, 3))
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    data.normal = array(normal.astype(np.float32), wp.vec3)
    data.shape0 = array(shapes0, int)
    data.shape1 = array(shapes1, int)
    data.margin0 = array(np.full(16, 0.01, dtype=np.float32), float)
    data.margin1 = array(np.full(16, 0.02, dtype=np.float32), float)
    data.world = array(world, int)
    data.slot = array(slots, int)
    data.art0 = array(art0, int)
    data.art1 = array(art1, int)
    data.path = array(np.where(slots >= 0, 0, -1), int)
    data.slots_needed = array(np.full(16, 3, dtype=np.int32), int)
    data.shape_body = array([0, 1, 2, 3, 4, -1, 6, 7, 8, 9, 10, -1], int)
    poses = np.zeros((12, 7), dtype=np.float32)
    poses[:, :3] = rng.normal(size=(12, 3))
    theta = np.arange(12) / 30
    poses[:, 5] = np.sin(theta)
    poses[:, 6] = np.cos(theta)
    data.body_q = array(poses, wp.transform)
    data.body_v = array(rng.normal(size=(12, 6)).astype(np.float32), wp.spatial_vector)
    data.body_mask = array([63, 21, 0, 0, 0, 0, 63, 42, 0, 0, 0, 0], wp.uint32)
    data.body_single_dof = array([-1, -1, 6, 7, -1, -1, -1, -1, 126, 127, -1, -1], int)
    data.response_dofs = array([6, 108, 0, 6, 108, 0], int)
    data.dof_start = array([0, 6, 114, 120, 126, 234], int)
    data.dof_offset = array([0, 6, -1, 0, 6, -1], int)
    data.origin = array(rng.normal(size=(6, 3)).astype(np.float32), wp.vec3)
    data.motion = array(rng.normal(size=(240, 6)).astype(np.float32), wp.spatial_vector)
    data.prescribed = array([0, 0, 1, 0, 0, 1], int)
    data.material_mu = array(np.linspace(0.2, 0.8, 12, dtype=np.float32), float)
    data.material_restitution = array(np.linspace(0, 0.7, 12, dtype=np.float32), float)
    data.inverse_mass = array(np.linspace(0.1, 0.9, 240, dtype=np.float32), float)
    data.dense_group = array([1, 0], int)
    factor = np.tril(rng.normal(0, 0.1, (2, 6, 6))).astype(np.float32)
    for group in range(2):
        factor[group, np.arange(6), np.arange(6)] = np.linspace(0.8, 1.7, 6)
    data.factor = array(factor, float)
    data.sparse_size = 108
    data.shared_anchor = shared
    data.friction_shared_anchor = friction_shared
    data.friction_scale = 0.73
    data.beta = 0.17
    data.cfm = 0.001
    for name in OUTPUTS:
        dtype = int if name in ("row_type", "row_parent", "sparse_dof") else float
        shape = (2, 40)
        if name in ("J", "Y"):
            shape = (2, 40, 6)
        if name == "sparse_dof":
            shape = (2, 40, 2)
        if name == "sparse_jy":
            shape = (2, 40, 4)
        setattr(data, name, wp.full(shape, -777 if dtype is int else float("nan"), dtype=dtype, device=device))
    counts = array([prefix[w] + 3 * kept[w] for w in range(2)], int)
    bounds = array([[p, p] for p in prefix], int)
    return data, SimpleNamespace(counts=counts, bounds=bounds, device=device)


def install_prefix(data, aux):
    """Use the real bounded clear, then emulate only original unit-row limit stores."""
    wp.launch(
        compact.clear_limit_prefix, dim=(2, 12), inputs=[data.dense_group, data.J, data.sparse_dof], device=aux.device
    )
    J = data.J.numpy()
    types = data.row_type.numpy()
    cfm = data.row_cfm.numpy()
    for world, end in enumerate(aux.bounds.numpy()[:, 1]):
        for row in range(end):
            J[data.dense_group.numpy()[world], row, row % 6] = 1.0 if row < 6 else -1.0
            types[world, row] = 3
            cfm[world, row] = data.cfm
    data.J.assign(J)
    data.row_type.assign(types)
    data.row_cfm.assign(cfm)


def run_original(data, aux):
    """Execute all seven original producer/consumer kernels on exactly the same inputs."""
    data.J.zero_()
    install_prefix(data, aux)
    values = {
        "contact_count": data.count,
        "total_num_threads": 16,
        "total_num_workers": 16,
        "contact_point0": data.point0,
        "contact_point1": data.point1,
        "contact_normal": data.normal,
        "contact_shape0": data.shape0,
        "contact_shape1": data.shape1,
        "contact_thickness0": data.margin0,
        "contact_thickness1": data.margin1,
        "contact_world": data.world,
        "contact_slot": data.slot,
        "contact_art_a": data.art0,
        "contact_art_b": data.art1,
        "contact_path": data.path,
        "contact_slots_needed": data.slots_needed,
        "shape_body": data.shape_body,
        "body_q": data.body_q,
        "body_v_s": data.body_v,
        "prescribed_articulation": data.prescribed,
        "articulation_origin": data.origin,
        "shape_material_mu": data.material_mu,
        "shape_material_restitution": data.material_restitution,
        "enable_friction": 1,
        "contact_friction_gap_threshold": float("inf"),
        "contact_friction_shared_anchor": data.friction_shared_anchor,
        "contact_friction_anchor_limit": 0,
        "contact_friction_articulation_pairs_only": 0,
        "is_free_rigid": wp.zeros(6, dtype=int, device=aux.device),
        "contact_friction_scale": data.friction_scale,
        "contact_shared_anchor": data.shared_anchor,
        "pgs_beta": data.beta,
        "pgs_cfm": data.cfm,
        "world_row_type": data.row_type,
        "world_row_parent": data.row_parent,
        "world_row_mu": data.row_mu,
        "world_row_beta": data.row_beta,
        "world_row_cfm": data.row_cfm,
        "world_phi": data.phi,
        "world_target_velocity": data.target,
        "world_row_restitution": data.restitution,
        "target_size": 6,
        "articulation_response_dof_count": data.response_dofs,
        "art_group_idx": wp.array([1, 0, -1, 0, 1, -1], dtype=int, device=aux.device),
        "art_dof_start": data.dof_start,
        "body_response_dof_mask": data.body_mask,
        "joint_S_s": data.motion,
        "J_group": data.J,
        "articulation_dof_start": data.dof_start,
        "articulation_world_dof_offset": data.dof_offset,
        "body_single_response_dof": data.body_single_dof,
        "diagonal_inverse_mass": data.inverse_mass,
        "sparse_row_dof": data.sparse_dof,
        "sparse_row_jy": data.sparse_jy,
        "world_constraint_count": aux.counts,
        "max_constraints": 40,
        "L_group": data.factor,
        "group_to_art": wp.array([3, 0], dtype=int, device=aux.device),
        "art_to_world": wp.array([0, 0, 0, 1, 1, 1], dtype=int, device=aux.device),
        "n_dofs": 6,
        "n_arts": 2,
        "write_world": 0,
        "Y_group": data.Y,
        "J_world": data.J,
        "Y_world": data.Y,
        "world_diag": data.diag,
        "mf_constraint_count": wp.zeros(2, dtype=int, device=aux.device),
        "skip_rows_le": 0,
    }

    def launch(kernel, dim):
        args = [values[name] for name in inspect.signature(kernel.func).parameters]
        wp.launch(kernel, dim=dim, inputs=args, device=aux.device)

    launch(kernels.prepare_world_contact_rows, 16)
    launch(kernels.populate_world_J_for_compact_size, (16, 32))
    values["target_size"] = 108
    launch(kernels.populate_sparse_diagonal_contact_response, 16)
    launch(kernels.hinv_jt_par_row, 80)
    data.diag.zero_()
    launch(kernels.diag_from_JY_par_art, 80)
    launch(kernels.accumulate_sparse_diagonal_response_diag, 80)
    launch(kernels.finalize_world_diag_cfm, 2)


def run_candidate(data, aux):
    """Execute both complete owners, leaving original contact allocation untouched."""
    install_prefix(data, aux)
    wp.launch(compact.produce_contacts, dim=16, inputs=[data], device=aux.device)
    wp.launch(
        compact.produce_limit_response,
        dim=(2, 12),
        inputs=[
            aux.bounds,
            aux.counts,
            data.dense_group,
            data.factor,
            data.J,
            data.Y,
            data.row_cfm,
            data.diag,
        ],
        device=aux.device,
    )


def geometry_reference(data):
    """Independent FP64 point-velocity construction, without reading any produced row."""
    names = (
        "point0",
        "point1",
        "normal",
        "margin0",
        "margin1",
        "body_q",
        "body_v",
        "motion",
        "origin",
        "shape0",
        "shape1",
        "shape_body",
        "art0",
        "art1",
        "body_mask",
        "body_single_dof",
        "response_dofs",
        "dof_start",
        "dof_offset",
        "prescribed",
        "world",
        "slot",
        "path",
    )
    host = {name: getattr(data, name).numpy() for name in names}
    for name in ("point0", "point1", "normal", "margin0", "margin1", "body_q", "body_v", "motion", "origin"):
        host[name] = host[name].astype(np.float64)

    def rotate(quaternion, point):
        # Warp's quaternion action, independently evaluated in real-valued FP64 arithmetic.
        xyz, w = quaternion[:3], quaternion[3]
        return point + 2 * np.cross(xyz, np.cross(xyz, point) + w * point)

    results = []
    for contact in range(int(data.count.numpy()[0])):
        if host["path"][contact] != 0 or host["slot"][contact] < 0:
            continue
        bodies = [
            host["shape_body"][host[f"shape{i}"][contact]] if host[f"shape{i}"][contact] >= 0 else -1 for i in range(2)
        ]
        arts = [host[f"art{i}"][contact] for i in range(2)]
        n = -host["normal"][contact]
        points = [host[f"point{i}"][contact].copy() for i in range(2)]
        for endpoint, body in enumerate(bodies):
            if body >= 0:
                pose = host["body_q"][body]
                points[endpoint] = pose[:3] + rotate(pose[3:], points[endpoint])
            points[endpoint] += (-1 if endpoint == 0 else 1) * host[f"margin{endpoint}"][contact] * n
        tangent = np.cross(n, [1, 0, 0])
        if np.dot(tangent, tangent) < 1e-12:
            tangent = np.cross(n, [0, 1, 0])
        tangent /= np.linalg.norm(tangent)
        second = np.cross(n, tangent)
        second /= np.linalg.norm(second)
        for row, direction in enumerate((n, tangent, second)):
            anchor = data.shared_anchor or (row > 0 and data.friction_shared_anchor)
            used_points = [0.5 * (points[0] + points[1])] * 2 if anchor else points
            dense = np.zeros(6)
            sparse = np.zeros(114)
            target = 0.0
            for endpoint, (body, art) in enumerate(zip(bodies, arts, strict=True)):
                if body < 0 or art < 0:
                    continue
                sign = 1 if endpoint == 0 else -1
                lever = used_points[endpoint] - host["origin"][art]
                if host["prescribed"][art]:
                    v = host["body_v"][body]
                    target -= sign * np.dot(direction, v[:3] + np.cross(v[3:], lever))
                if host["response_dofs"][art] == 6:
                    for local in range(6):
                        if host["body_mask"][body] & (1 << local):
                            s = host["motion"][host["dof_start"][art] + local]
                            dense[local] += sign * np.dot(direction, s[:3] + np.cross(s[3:], lever))
                elif host["response_dofs"][art] == 108:
                    dof = host["body_single_dof"][body]
                    if dof >= 0:
                        s = host["motion"][dof]
                        coord = host["dof_offset"][art] + dof - host["dof_start"][art]
                        sparse[coord] += sign * np.dot(direction, s[:3] + np.cross(s[3:], lever))
            results.append(
                (
                    host["world"][contact],
                    host["slot"][contact] + row,
                    dense,
                    sparse,
                    target,
                    np.dot(n, points[0] - points[1]) if row == 0 else 0.0,
                )
            )
    return results


def build_solver_fixture(device):
    """Actual admitted two-world sparse108 + serial-six model with live contact shapes."""
    template = newton.ModelBuilder()
    inertia = wp.mat33(0.04, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.06)
    root = template.add_link(mass=2.0, inertia=inertia)
    joints = [template.add_joint_fixed(parent=-1, child=root)]
    for index in range(108):
        body = template.add_link(mass=0.2 + index / 300, inertia=inertia)
        joints.append(
            template.add_joint_prismatic(
                parent=root,
                child=body,
                axis=newton.Axis.Z,
                parent_xform=wp.transform(wp.vec3(index / 100, 0.0, 0.2), wp.quat_identity()),
                target_ke=12.0,
                target_kd=0.4,
                limit_lower=-0.05,
                limit_upper=0.05,
            )
        )
        if index < 2:
            template.add_shape_sphere(body=body, radius=0.01)
    template.add_articulation(joints)
    parent = -1
    joints = []
    for index in range(6):
        body = template.add_link(mass=0.4, com=wp.vec3(0.02, 0.01, 0.06), inertia=inertia)
        joints.append(
            template.add_joint_revolute(
                parent=parent,
                child=body,
                axis=newton.Axis.Y,
                parent_xform=wp.transform(wp.vec3(0.03, 0.0, 0.04), wp.quat_identity()),
                target_ke=2.0,
                target_kd=0.1,
                limit_lower=-0.04,
                limit_upper=0.04,
            )
        )
        if index in (0, 5):
            template.add_shape_sphere(body=body, radius=0.01)
        parent = body
    template.add_articulation(joints)
    builder = newton.ModelBuilder(gravity=(0.4, -0.9, -9.2))
    builder.add_world(template)
    builder.add_world(template)
    model = builder.finalize(device=device)
    model.rigid_contact_max = 8
    model.joint_q.assign(np.linspace(-0.06, 0.06, model.joint_coord_count, dtype=np.float32))
    model.joint_qd.assign(np.linspace(-0.1, 0.1, model.joint_dof_count, dtype=np.float32))
    return model


def install_contacts(model, state, contacts):
    """Six concrete same-input contacts; local witnesses are held during the short step test."""
    pairs = np.array([[0, 3], [0, 1], [2, 3], [4, 7], [4, 5], [6, 7]], dtype=np.int32)
    shape_body = model.shape_body.numpy()
    poses = state.body_q.numpy().astype(np.float64)
    for endpoint in range(2):
        ids = np.full(8, -1, dtype=np.int32)
        ids[:6] = pairs[:, endpoint]
        getattr(contacts, f"rigid_contact_shape{endpoint}").assign(ids)
        points = np.zeros((8, 3), dtype=np.float32)
        for contact, pair in enumerate(pairs):
            midpoint = np.mean(poses[shape_body[pair], :3], axis=0)
            pose = poses[shape_body[pair[endpoint]]]
            p = midpoint - pose[:3]
            q = pose[3:6]
            points[contact] = p + 2 * np.cross(-q, np.cross(-q, p) + pose[6] * p)
        getattr(contacts, f"rigid_contact_point{endpoint}").assign(points)
        getattr(contacts, f"rigid_contact_margin{endpoint}").zero_()
    contacts.rigid_contact_normal.assign(np.tile(np.array([0.0, 0.0, -1.0], np.float32), (8, 1)))
    contacts.rigid_contact_count.assign(np.array([6], dtype=np.int32))


class TestCompactContact(unittest.TestCase):
    def compare(self, original, candidate, aux):
        """Compare every active complete field, not only the diagonal or row count."""
        for world, count in enumerate(aux.counts.numpy()):
            prefix = aux.bounds.numpy()[world, 1]
            for name in OUTPUTS:
                a = getattr(original, name).numpy()
                b = getattr(candidate, name).numpy()
                index = candidate.dense_group.numpy()[world] if name in ("J", "Y") else world
                start = 0 if name in ("J", "Y", "diag", "row_type", "row_cfm", "sparse_dof") else prefix
                a = a[index, start:count]
                b = b[index, start:count]
                if a.dtype.kind in "iu":
                    np.testing.assert_array_equal(a, b, err_msg=name)
                else:
                    np.testing.assert_allclose(a, b, rtol=3e-6, atol=3e-6, err_msg=name)

    def test_complete_original_chain_and_prescribed_geometry(self):
        """Match same-art, sparse cancellation, prescribed and static laws with both anchor choices."""
        for shared, friction_shared in ((0, 0), (0, 1), (1, 0)):
            original, aux = fixture(shared=shared, friction_shared=friction_shared)
            candidate, _ = fixture(shared=shared, friction_shared=friction_shared)
            before = candidate.factor.numpy().copy()
            run_original(original, aux)
            run_candidate(candidate, aux)
            self.compare(original, candidate, aux)
            np.testing.assert_array_equal(candidate.factor.numpy(), before)
            # The prescribed body has genuine nonzero velocity, not a zero-target fixture.
            self.assertGreater(np.max(np.abs(candidate.target.numpy()[:, 14:20])), 0.01)

    def test_empty_and_growing_prefix_with_poison(self):
        """Overwrite every active row through empty, growing and shrinking prefix cases."""
        candidate, candidate_aux = fixture()
        owners = {name: getattr(candidate, name).ptr for name in OUTPUTS}
        for prefix, kept in (((0, 0), (0, 0)), ((12, 1), (2, 7)), ((1, 12), (7, 1)), ((3, 0), (0, 0))):
            original, aux = fixture(prefix=prefix, kept=kept)
            candidate.slot.assign(original.slot)
            candidate.path.assign(original.path)
            candidate_aux.counts.assign(aux.counts)
            candidate_aux.bounds.assign(aux.bounds)
            # Simulate the scheduler's previous negative linked-list sentinels.
            candidate.sparse_dof.fill_(-25)
            run_original(original, aux)
            run_candidate(candidate, candidate_aux)
            self.compare(original, candidate, aux)
            self.assertEqual(owners, {name: getattr(candidate, name).ptr for name in OUTPUTS})

    def test_held_factor_response_against_float64(self):
        """Check physical dense response and diagonal with an independent double-precision solve."""
        candidate, aux = fixture()
        run_candidate(candidate, aux)
        for world, count in enumerate(aux.counts.numpy()):
            group = candidate.dense_group.numpy()[world]
            L = candidate.factor.numpy()[group].astype(np.float64)
            J = candidate.J.numpy()[group, :count].astype(np.float64)
            Y = np.linalg.solve(L.T, np.linalg.solve(L, J.T)).T
            np.testing.assert_allclose(candidate.Y.numpy()[group, :count], Y, rtol=5e-6, atol=3e-6)
            expected = np.sum(J * Y, axis=1)
            sd = candidate.sparse_dof.numpy()[world, :count]
            sj = candidate.sparse_jy.numpy()[world, :count].astype(np.float64)
            for row in range(count):
                for slot in range(2):
                    if sd[row, slot] >= 0:
                        expected[row] += sj[row, 2 * slot] * sj[row, 2 * slot + 1]
            expected += candidate.cfm
            np.testing.assert_allclose(candidate.diag.numpy()[world, :count], expected, rtol=5e-6, atol=3e-6)

    def test_independent_geometry_action_and_prescribed_targets(self):
        """Compare Jv, physical H^-1 J^T action and prescribed targets, not row counts alone."""
        rng = np.random.default_rng(443)
        for shared, friction_shared in ((0, 0), (0, 1), (1, 0)):
            data, aux = fixture(shared=shared, friction_shared=friction_shared)
            reference = geometry_reference(data)
            run_candidate(data, aux)
            velocity = rng.normal(size=(2, 114))
            for world, row, dense, sparse, target, phi in reference:
                group = data.dense_group.numpy()[world]
                np.testing.assert_allclose(data.J.numpy()[group, row], dense, rtol=3e-6, atol=3e-6)
                self.assertAlmostEqual(data.target.numpy()[world, row], target, delta=3e-6)
                self.assertAlmostEqual(data.phi.numpy()[world, row], phi, delta=3e-6)
                actual = data.J.numpy()[group, row].astype(np.float64) @ velocity[world, :6]
                for slot, dof in enumerate(data.sparse_dof.numpy()[world, row]):
                    if dof >= 0:
                        actual += data.sparse_jy.numpy()[world, row, slot * 2] * velocity[world, dof]
                expected = dense @ velocity[world, :6] + sparse @ velocity[world]
                self.assertAlmostEqual(actual, expected, delta=1e-5)
                factor = data.factor.numpy()[group].astype(np.float64)
                response = np.linalg.solve(factor.T, np.linalg.solve(factor, dense))
                np.testing.assert_allclose(data.Y.numpy()[group, row], response, rtol=5e-6, atol=4e-6)

    def test_default_off_and_exact_mode_admission(self):
        """Reject incomplete mode compositions before any contact ownership changes."""
        valid = {
            "model": SimpleNamespace(device=SimpleNamespace(is_cuda=True)),
            "_sparse_diagonal_contact_solve": True,
            "_sparse_diagonal_dense_size": 6,
            "_sparse_diagonal_response_size": 108,
            "_sparse_diagonal_contact_triples": True,
            "dense_max_constraints": 40,
            "_fused_diagonal_joint_limits": True,
            "_dense_internal_max_rows": 12,
            "drive_mode": "augmented",
            "enable_joint_velocity_limits": False,
            "_mimic_count": 0,
            "_connect_count": 0,
            "_compact_diagonal_mass_size": 108,
            "grouped_dynamics": False,
            "_debug_buffers_enabled": False,
            "_preelim_active": False,
            "_simple_world_classifier": None,
        }
        self.assertTrue(compact.supported(SimpleNamespace(**valid)))
        for name, value in (
            ("_sparse_diagonal_contact_solve", False),
            ("_sparse_diagonal_dense_size", 7),
            ("_sparse_diagonal_contact_triples", False),
            ("dense_max_constraints", 11),
            ("_fused_diagonal_joint_limits", False),
            ("_dense_internal_max_rows", 13),
            ("drive_mode", "physx_pgs"),
            ("enable_joint_velocity_limits", True),
            ("_mimic_count", 1),
            ("_connect_count", 1),
            ("grouped_dynamics", True),
            ("_debug_buffers_enabled", True),
            ("_preelim_active", True),
            ("_compact_diagonal_mass_size", None),
            ("_simple_world_classifier", object()),
        ):
            self.assertFalse(compact.supported(SimpleNamespace(**{**valid, name: value})), name)

    def test_actual_fixture_layout_and_cpu_fallback(self):
        """Exercise the real model/Contacts field binding and exact unsupported CPU fallback."""
        model = build_solver_fixture("cpu")
        self.assertEqual(model.joint_dof_count, 228)
        self.assertEqual(model.shape_count, 8)
        state = model.state()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        contacts = newton.Contacts(model.rigid_contact_max, 0, device=model.device)
        install_contacts(model, state, contacts)
        self.assertEqual(contacts.rigid_contact_count.numpy()[0], 6)
        with mock.patch.object(solver_module, "_COMPACT_CONTACT_BOUNDARY", True):
            solver = newton.solvers.SolverFeatherPGS(model, pgs_mode="split", dense_max_constraints=64)
        self.assertFalse(solver._compact_contact_boundary)

    def test_cuda_actual_solver_step_reset_notify_and_graphs(self):
        """Use the actual allocator, scheduler, RHS and eight-sweep solve with loaded contacts."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("Actual sparse constructor and parallel-stream capture require CUDA")
        for device in devices:
            model = build_solver_fixture(device)
            solvers = []
            for enabled in (False, True):
                with (
                    mock.patch.object(solver_module, "_COMPACT_CONTACT_BOUNDARY", enabled),
                    mock.patch.object(solver_module, "_PRISMATIC_PUBLICATION", True),
                    mock.patch.object(solver_module, "_SPARSE_CONTACT_DIRECT", True),
                ):
                    solvers.append(
                        newton.solvers.SolverFeatherPGS(
                            model,
                            pgs_mode="matrix_free",
                            pgs_iterations=8,
                            update_mass_matrix_interval=2,
                            use_parallel_streams=True,
                            enable_joint_limits=True,
                            dense_max_constraints=64,
                            mf_max_constraints=64,
                        )
                    )
            self.assertFalse(solvers[0]._compact_contact_boundary)
            self.assertTrue(solvers[1]._compact_contact_boundary)
            self.assertLessEqual(solvers[1]._dense_internal_max_rows, 12)
            states = [[model.state(), model.state()] for _ in solvers]
            controls = [model.control() for _ in solvers]
            contacts = [newton.Contacts(model.rigid_contact_max, 0, device=device) for _ in solvers]
            for index in range(2):
                state = states[index][0]
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                install_contacts(model, state, contacts[index])

            def compare(states=states, solvers=solvers):
                for field in ("joint_q", "joint_qd", "body_q", "body_qd"):
                    np.testing.assert_allclose(
                        getattr(states[1][0], field).numpy(), getattr(states[0][0], field).numpy(), rtol=3e-5, atol=5e-6
                    )
                for solver in solvers:
                    solver.check_constraint_capacity()
                    self.assertGreater(int(np.sum(solver.constraint_count.numpy())), 0)
                    self.assertTrue(np.all(np.isfinite(solver.v_out.numpy())))

            for step in range(6):
                if step == 3:
                    model.body_mass.assign(model.body_mass.numpy() * 1.1)
                    for solver in solvers:
                        solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
                for index, solver in enumerate(solvers):
                    current, following = states[index]
                    if step == 2:
                        solver.reset(current, wp.array([True, False], dtype=bool, device=device))
                    solver.step(current, following, controls[index], contacts[index], 1 / 240)
                    states[index] = [following, current]
                compare()
            graphs = []
            for index, solver in enumerate(solvers):
                current, following = states[index]
                solver.reset(current)
                with wp.ScopedCapture(device=device) as capture:
                    solver.seed_double_buffer_events()
                    solver.step(current, following, controls[index], contacts[index], 1 / 240)
                    solver.step(following, current, controls[index], contacts[index], 1 / 240)
                graphs.append(capture.graph)
            for _ in range(2):
                for graph in graphs:
                    wp.capture_launch(graph)
                compare()

    def test_cuda_complete_owners_and_two_graphs(self):
        """Check real device owners and repeated graph outputs without launching CUDA on CPU hosts."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("CUDA graph coverage requires a CUDA device")
        for device in devices:
            original, aux = fixture(device)
            candidate, _ = fixture(device)
            run_original(original, aux)
            run_candidate(candidate, aux)
            self.compare(original, candidate, aux)
            graphs = []
            for _ in range(2):
                with wp.ScopedCapture(device=device) as capture:
                    wp.launch(compact.produce_contacts, dim=16, inputs=[candidate], device=device)
                    wp.launch(
                        compact.produce_limit_response,
                        dim=(2, 12),
                        inputs=[
                            aux.bounds,
                            aux.counts,
                            candidate.dense_group,
                            candidate.factor,
                            candidate.J,
                            candidate.Y,
                            candidate.row_cfm,
                            candidate.diag,
                        ],
                        device=device,
                    )
                graphs.append(capture.graph)
            for graph in graphs:
                candidate.Y.fill_(float("nan"))
                candidate.diag.fill_(float("nan"))
                wp.capture_launch(graph)
                self.compare(original, candidate, aux)


if __name__ == "__main__":
    unittest.main()
