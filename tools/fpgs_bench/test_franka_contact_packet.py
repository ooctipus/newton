# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent current-geometry controls for the bounded Franka packet producer."""

import inspect
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import franka_contact_packet as packet
from newton._src.solvers.feather_pgs import kernels

OUTPUTS = (
    "row_type",
    "row_parent",
    "row_mu",
    "row_beta",
    "row_cfm",
    "phi",
    "target",
    "restitution",
    "rhs",
    "diag",
    "row_w",
)


def fixture(device="cpu"):
    """Build mixed-articulation, prescribed, same-art and beyond-local-bound contacts."""
    data = packet.ContactInput()
    rng = np.random.default_rng(7134)

    def a(value, dtype=wp.float32):
        return wp.array(value, dtype=dtype, device=device)

    data.count = a([7], wp.int32)
    data.point0 = a(rng.normal(0, 0.08, (8, 3)), wp.vec3)
    data.point1 = a(rng.normal(0, 0.08, (8, 3)), wp.vec3)
    normals = rng.normal(size=(8, 3))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    data.normal = a(normals, wp.vec3)
    data.margin0, data.margin1 = a([0.01] * 8), a([0.02] * 8)
    data.shape0 = a([0, 0, 3, 4, 3, 4, 0, 999], wp.int32)
    data.shape1 = a([1, 2, 4, 5, 3, 5, -1, 999], wp.int32)
    data.world = a([0, 0, 1, 1, 1, 1, 0, -1], wp.int32)
    data.slot = a([2, 5, 38, 41, 44, 0, -1, -1], wp.int32)
    data.art0 = a([0, 0, 3, 4, 3, 4, 0, -1], wp.int32)
    data.art1 = a([1, 2, 4, 5, 3, 5, -1, -1], wp.int32)
    data.path = a([0, 0, 0, 0, 0, 1, -1, -1], wp.int32)
    data.slots_needed = a([3, 3, 3, 3, 3, 3, 0, 0], wp.int32)
    data.shape_body = a(np.arange(6), wp.int32)
    poses = np.zeros((6, 7), np.float32)
    poses[:, :3] = rng.normal(0, 0.1, (6, 3))
    poses[:, 3:] = rng.normal(size=(6, 4))
    poses[:, 3:] /= np.linalg.norm(poses[:, 3:], axis=1)[:, None]
    data.body_q = a(poses, wp.transform)
    data.body_v = a(rng.normal(0, 0.3, (6, 6)), wp.spatial_vector)
    data.body_mask = a([511, 63, 0, 341, 63, 63], wp.uint32)
    data.response_dofs = a([9, 6, 0, 9, 6, 6], wp.int32)
    data.dof_start = a([0, 9, 15, 15, 24, 30], wp.int32)
    data.origin = a(rng.normal(0, 0.1, (6, 3)), wp.vec3)
    data.motion = a(rng.normal(0, 0.5, (36, 6)), wp.spatial_vector)
    data.prescribed = a([0, 0, 1, 0, 0, 0], wp.int32)
    data.is_free = a([0, 1, 0, 0, 1, 1], wp.int32)
    data.material_mu = a([0.3, 0.7, 0.9, 0.2, 0.8, 0.5])
    data.material_restitution = a([0.5, 0.8, np.nan, 1.3, -0.1, 0.6])
    data.incident = a(rng.normal(0, 0.4, 36))
    data.shared_anchor, data.friction_shared_anchor = 0, 1
    data.friction_anchor_limit, data.friction_pairs_only = 0, 0
    data.friction_scale = 0.8
    data.beta, data.cfm, data.dt = 0.2, 0.01, 1 / 240
    data.bias_scale, data.speculative_scale = 0.7, 0.6
    data.restitution_threshold, data.contact_w = 0.01, 1.0
    for name in OUTPUTS:
        dtype = wp.int32 if name in ("row_type", "row_parent") else wp.float32
        setattr(data, name, wp.full((2, 64), -7, dtype=dtype, device=device))
    return data


def run_packet(data, device="cpu"):
    """Run the real cooperative kernel and return its bounded private storage."""
    result = packet.allocate_packet(2, device)
    result.jacobian.fill_(np.nan)
    wp.launch_tiled(packet.produce_contacts, dim=[2], inputs=[2, data, result], block_dim=32, device=device)
    return result


def geometry64(data, contact, row):
    """Contract current world-point geometry with stored motion columns in FP64."""
    fields = (
        "point0",
        "point1",
        "normal",
        "shape0",
        "shape1",
        "shape_body",
        "body_q",
        "margin0",
        "margin1",
        "art0",
        "art1",
        "response_dofs",
        "body_mask",
        "dof_start",
        "motion",
        "origin",
    )
    source = {name: getattr(data, name).numpy() for name in fields}
    normal = -source["normal"][contact].astype(np.float64)
    tangent = np.cross(normal, [1.0, 0.0, 0.0])
    if tangent @ tangent < 1e-12:
        tangent = np.cross(normal, [0.0, 1.0, 0.0])
    tangent /= np.linalg.norm(tangent)
    tangent1 = np.cross(normal, tangent)
    tangent1 /= np.linalg.norm(tangent1)
    direction = (normal, tangent, tangent1)[row]
    points, bodies, arts = [], [], []
    for side in (0, 1):
        shape = source[f"shape{side}"][contact]
        body = source["shape_body"][shape] if shape >= 0 else -1
        point = source[f"point{side}"][contact].astype(np.float64)
        if body >= 0:
            pose = source["body_q"][body].astype(np.float64)
            xyz, w = pose[3:6], pose[6]
            point = (2 * w * w - 1) * point + 2 * w * np.cross(xyz, point) + 2 * xyz * (xyz @ point) + pose[:3]
        point += (-1 if side == 0 else 1) * source[f"margin{side}"][contact] * normal
        points.append(point)
        bodies.append(body)
        arts.append(source[f"art{side}"][contact])
    if data.shared_anchor or (row and data.friction_shared_anchor):
        points = [(points[0] + points[1]) * 0.5] * 2
    result = np.zeros(15)
    for side in (0, 1):
        body, art = bodies[side], arts[side]
        if body < 0 or art < 0:
            continue
        size = source["response_dofs"][art]
        if size not in (9, 6):
            continue
        offset = 0 if size == 9 else 9
        for dof in range(size):
            if source["body_mask"][body] & (1 << dof):
                motion = source["motion"][source["dof_start"][art] + dof].astype(np.float64)
                velocity = motion[:3] + np.cross(motion[3:], points[side] - source["origin"][art])
                result[offset + dof] += (1 if side == 0 else -1) * (direction @ velocity)
    return result


def reference(data, *, enable_friction=1, device="cpu"):
    """Execute original metadata, both original J producers, bias and restitution."""
    common = {
        "contact_count": data.count,
        "total_num_threads": 2,
        "total_num_workers": 2,
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
        "enable_friction": enable_friction,
        "contact_friction_gap_threshold": np.inf,
        "contact_friction_shared_anchor": data.friction_shared_anchor,
        "contact_friction_anchor_limit": data.friction_anchor_limit,
        "contact_friction_articulation_pairs_only": data.friction_pairs_only,
        "is_free_rigid": data.is_free,
        "contact_friction_scale": data.friction_scale,
        "contact_shared_anchor": data.shared_anchor,
        "pgs_beta": data.beta,
        "pgs_cfm": data.cfm,
        "articulation_response_dof_count": data.response_dofs,
        "art_dof_start": data.dof_start,
        "body_response_dof_mask": data.body_mask,
        "joint_S_s": data.motion,
    }
    output_names = {
        "world_row_type": "row_type",
        "world_row_parent": "row_parent",
        "world_row_mu": "row_mu",
        "world_row_beta": "row_beta",
        "world_row_cfm": "row_cfm",
        "world_phi": "phi",
        "world_target_velocity": "target",
        "world_row_restitution": "restitution",
    }
    result = {}
    for argument, name in output_names.items():
        result[name] = wp.full(
            (2, 64), -7, dtype=wp.int32 if name in ("row_type", "row_parent") else wp.float32, device=device
        )
        common[argument] = result[name]
    wp.launch(
        kernels.prepare_world_contact_rows,
        dim=2,
        inputs=[common[n] for n in inspect.signature(kernels.prepare_world_contact_rows.func).parameters],
        device=device,
    )
    groups = wp.array([0, 0, -1, 1, 1, 2], dtype=wp.int32, device=device)
    for size in (9, 6):
        result[f"J{size}"] = wp.zeros((2 if size == 9 else 3, 64, size), dtype=wp.float32, device=device)
        common.update(target_size=size, art_group_idx=groups, J_group=result[f"J{size}"])
        wp.launch(
            kernels.populate_world_J_for_compact_size,
            dim=(2, 32),
            inputs=[common[n] for n in inspect.signature(kernels.populate_world_J_for_compact_size.func).parameters],
            device=device,
        )
    world_j = np.zeros((2, 64, 21), np.float32)
    world_j[:, :, :9] = result["J9"].numpy()
    world_j[:, :, 9:15] = result["J6"].numpy()[:2]
    world_j[1, :, 15:21] = result["J6"].numpy()[2]
    counts = np.zeros(2, np.int32)
    for world, slot, path, rows in zip(
        data.world.numpy()[:7], data.slot.numpy()[:7], data.path.numpy()[:7], data.slots_needed.numpy()[:7], strict=True
    ):
        if path == 0 and slot >= 0:
            counts[world] = max(counts[world], slot + rows)
    result["rhs"] = wp.full((2, 64), -7, dtype=wp.float32, device=device)
    result["row_w"] = wp.full((2, 64), -7, dtype=wp.float32, device=device)
    count_array = wp.array(counts, dtype=wp.int32, device=device)
    no_mf = wp.zeros(2, dtype=wp.int32, device=device)
    wp.launch(
        kernels.compute_world_contact_bias,
        dim=128,
        inputs=[
            count_array,
            64,
            result["phi"],
            result["row_beta"],
            result["row_type"],
            result["target"],
            data.dt,
            data.bias_scale,
            data.speculative_scale,
            1.0,
            data.contact_w,
            no_mf,
            0,
            result["rhs"],
            result["row_w"],
        ],
        device=device,
    )
    indices = np.full((2, 21), -1, np.int32)
    indices[0, :15] = np.arange(15)
    indices[1] = np.arange(15, 36)
    wp.launch(
        kernels.apply_world_contact_restitution_matrix_free,
        dim=128,
        inputs=[
            count_array,
            64,
            wp.array([15, 21], dtype=wp.int32, device=device),
            result["phi"],
            result["row_type"],
            result["target"],
            result["restitution"],
            data.incident,
            wp.array(indices, dtype=wp.int32, device=device),
            wp.array(world_j, dtype=wp.float32, device=device),
            data.dt,
            data.restitution_threshold,
            int(data.contact_w < 1),
            no_mf,
            0,
            result["rhs"],
            result["row_w"],
        ],
        device=device,
    )
    return result, groups


class TestFrankaContactPacket(unittest.TestCase):
    """Keep geometry, active rows, fallback publication and allocation bounds exact."""

    def test_original_metadata_jacobian_bias_and_restitution(self):
        """Compare actual original kernels for both anchor modes and regularization."""
        for shared, weight in ((0, 1.0), (1, 0.7)):
            with self.subTest(shared=shared, weight=weight):
                data = fixture()
                data.shared_anchor, data.contact_w = shared, weight
                original, _ = reference(data)
                produced = run_packet(data)
                for contact in range(5):
                    world, slot = data.world.numpy()[contact], data.slot.numpy()[contact]
                    for row in range(3):
                        index = slot + row
                        for name in OUTPUTS:
                            if name == "diag":
                                self.assertAlmostEqual(float(data.diag.numpy()[world, index]), data.cfm, places=7)
                            elif name != "row_w" or weight < 1:
                                np.testing.assert_allclose(
                                    getattr(data, name).numpy()[world, index],
                                    original[name].numpy()[world, index],
                                    rtol=3e-6,
                                    atol=3e-6,
                                    err_msg=name,
                                )
                        if index < 40:
                            expected = np.concatenate(
                                (original["J9"].numpy()[world, index], original["J6"].numpy()[world, index])
                            )
                            np.testing.assert_allclose(
                                produced.jacobian.numpy()[world, index], expected, rtol=3e-6, atol=3e-7
                            )
                            self.assertEqual(produced.row_contact.numpy()[world, index], contact * 3 + row)
                if weight == 1:
                    np.testing.assert_array_equal(data.row_w.numpy(), -7)

    def test_general_fallback_same_art_distinct_free_and_high_rows(self):
        """Restore current group J for general worlds without reading bounded packets."""
        data = fixture()
        original, groups = reference(data)
        owner = wp.array([1, 0], dtype=wp.int32, device="cpu")
        for size in (9, 6):
            result = wp.full(original[f"J{size}"].shape, -17, dtype=wp.float32, device="cpu")
            wp.launch_tiled(
                packet.materialize_fallback,
                dim=[2],
                inputs=[2, data, owner, size, groups, result],
                block_dim=32,
                device="cpu",
            )
            output = result.numpy()
            np.testing.assert_array_equal(output[0], -17)
            for contact in (2, 3, 4):
                slot = data.slot.numpy()[contact]
                for art in {int(data.art0.numpy()[contact]), int(data.art1.numpy()[contact])}:
                    if data.response_dofs.numpy()[art] == size:
                        group = groups.numpy()[art]
                        np.testing.assert_allclose(
                            output[group, slot : slot + 3],
                            original[f"J{size}"].numpy()[group, slot : slot + 3],
                            rtol=3e-6,
                            atol=3e-7,
                        )

    def test_independent_fp64_geometry_and_current_pose_motion(self):
        """Reject stale pose or motion assumptions using an independent physical contraction."""
        data = fixture()
        previous = None
        for epoch in range(2):
            if epoch:
                pose = data.body_q.numpy()
                pose[0, :3] += [0.2, -0.1, 0.3]
                data.body_q.assign(pose)
                motion = data.motion.numpy()
                motion[0, :3] += [0.3, -0.4, 0.2]
                data.motion.assign(motion)
            output = run_packet(data).jacobian.numpy()
            for contact in (0, 1, 2):
                world, slot = data.world.numpy()[contact], data.slot.numpy()[contact]
                for row in range(3):
                    if slot + row < 40:
                        np.testing.assert_allclose(
                            output[world, slot + row], geometry64(data, contact, row), rtol=3e-6, atol=3e-7
                        )
            if previous is not None:
                self.assertGreater(np.max(np.abs(output[0, 2:8] - previous[0, 2:8])), 0.01)
            previous = output

    def test_normal_only_anchor_scaling_and_typed_zero_blocks(self):
        """Preserve anchor-pair material scaling and authoritative one-row allocation."""
        data = fixture()
        shape1 = data.shape1.numpy()
        shape1[1] = shape1[0]
        data.shape1.assign(shape1)
        art1 = data.art1.numpy()
        art1[1] = art1[0]
        data.art1.assign(art1)
        data.friction_anchor_limit = 2
        original, _ = reference(data)
        result = run_packet(data)
        np.testing.assert_allclose(
            data.row_mu.numpy()[0, [3, 4, 6, 7]], original["row_mu"].numpy()[0, [3, 4, 6, 7]], rtol=1e-6
        )
        data = fixture()
        data.slots_needed.assign([1, 1, 1, 1, 1, 1, 0, 0])
        original, _ = reference(data, enable_friction=0)
        result = run_packet(data)
        np.testing.assert_array_equal(result.row_contact.numpy()[0, [3, 4, 6, 7]], -1)
        np.testing.assert_array_equal(result.jacobian.numpy()[0, 5, 9:], 0)
        np.testing.assert_allclose(
            data.rhs.numpy()[0, [2, 5]], original["rhs"].numpy()[0, [2, 5]], rtol=3e-6, atol=3e-6
        )

    def test_empty_epoch_and_bounded_allocation(self):
        """Keep inactive packet tails unread and allocate only the existing local bound."""
        data = fixture()
        data.count.assign([0])
        result = run_packet(data)
        self.assertTrue(np.all(np.isnan(result.jacobian.numpy())))
        np.testing.assert_array_equal(result.row_contact.numpy(), -1)
        self.assertEqual(result.jacobian.shape, (2, 40, 15))
        self.assertEqual(result.jacobian.capacity, 2 * 40 * 15 * 4)
        self.assertEqual(packet.allocate_packet(0, "cpu").jacobian.shape, (0, 40, 15))
        with self.assertRaises(ValueError):
            packet.allocate_packet(-1, "cpu")

    def test_cuda_geometry_and_empty_growing_graph(self):
        """Exercise actual warp broadcasts and publication across current graph epochs."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("cooperative CUDA graph control requires CUDA")
        device = devices[0]
        cpu = fixture()
        gpu = fixture(device)
        expected = run_packet(cpu).jacobian.numpy()
        result = run_packet(gpu, device)
        np.testing.assert_allclose(result.jacobian.numpy(), expected, rtol=3e-6, atol=3e-7, equal_nan=True)
        with wp.ScopedCapture(device=device) as capture:
            wp.launch_tiled(packet.produce_contacts, dim=[2], inputs=[2, gpu, result], block_dim=32, device=device)
        for count in (0, 7, 0, 7):
            gpu.count.assign([count])
            result.row_contact.fill_(-1)
            result.jacobian.fill_(np.nan)
            wp.capture_launch(capture.graph)
            if count:
                np.testing.assert_allclose(result.jacobian.numpy(), expected, rtol=3e-6, atol=3e-7, equal_nan=True)
            else:
                self.assertTrue(np.all(np.isnan(result.jacobian.numpy())))
                np.testing.assert_array_equal(result.row_contact.numpy(), -1)


if __name__ == "__main__":
    unittest.main()
