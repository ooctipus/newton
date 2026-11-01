# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise direct contact ownership without changing the sparse response law."""

import ast
import inspect
import textwrap
import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import dense_rows, field_group, kernels, response_builds_rows
from newton._src.solvers.feather_pgs import rows as implementation
from newton._src.solvers.feather_pgs.friction_patches import FrictionPatches
from newton.tests.feather_pgs_law_harness import contact_input, row_lines
from newton.tests.feather_pgs_patch_parity import row_law


def _launch_workers(capacity):
    """Evaluate the actual sparse launch's host-only worker selection."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(implementation.RowLaws.write)))
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and node.args
        and isinstance(node.args[0], ast.Attribute)
        and node.args[0].attr == "sparse_pair_contact_rows_kernel"
    )
    # A launch through the contact pool takes its dimension as the second positional argument.
    dimension = next((item.value for item in call.keywords if item.arg == "dim"), None) or call.args[1]
    arguments = next(item.value for item in call.keywords if item.arg == "inputs")
    if ast.dump(dimension) != ast.dump(arguments.elts[2]):
        raise AssertionError("launch dimension and device stride must agree")
    values = {
        "contacts": SimpleNamespace(rigid_contact_max=capacity),
        # The rows contact launches are sized for (ContactPool.launch_rows), eagerly the ready rows.
        "rows": capacity,
        "contact_jacobian_workers": min(capacity, implementation._CONTACT_JACOBIAN_WORKER_CAP),
        "contact_build_threads": min(capacity, implementation._CONTACT_BUILD_THREAD_CAP),
    }
    if dimension.id != "contact_jacobian_workers":
        assignment = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == dimension.id for target in node.targets)
        )
        expression = ast.Expression(assignment.value)
        values[dimension.id] = eval(
            compile(expression, "<actual sparse worker selection>", "eval"), vars(implementation), values
        )
    return values[dimension.id]


def _fixture(device, capacity=4353):
    """Create unique row slots and mixed endpoint/route ownership above4096 contacts."""
    indices = np.arange(capacity, dtype=np.int32)
    world_count = (capacity + 4) // 5
    world = indices // 5
    slot = 3 * (indices % 5)
    path = np.where(indices % 11 == 0, -1, 0).astype(np.int32)
    slot[indices % 13 == 0] = -1
    shape1 = (indices % 4 - 1).astype(np.int32)
    art1 = np.where(shape1 < 0, -1, np.where(shape1 == 2, 1, 0)).astype(np.int32)
    body_q = np.array(
        ((0.1, 0.2, 0.3, 0, 0, 0, 1), (0.2, -0.1, 0.4, 0, 0, 0, 1), (0, 0, 0, 0, 0, 0, 1)), dtype=np.float32
    )
    motion = np.zeros((114, 6), dtype=np.float32)
    motion[0] = (0.2, 0.3, 1.0, 0.1, -0.2, 0.3)
    motion[1] = (-0.4, 0.5, 0.7, -0.2, 0.1, 0.4)
    normal = np.tile(np.array((0.0, 0.0, -1.0), dtype=np.float32), (capacity, 1))
    point0 = np.tile(np.array((0.1, -0.2, 0.3), dtype=np.float32), (capacity, 1))
    point1 = np.tile(np.array((-0.2, 0.1, 0.4), dtype=np.float32), (capacity, 1))

    def array(value, dtype):
        return wp.array(value, dtype=dtype, device=device)

    values = {
        "contact_count": array([capacity], int),
        "contact_point0": array(point0, wp.vec3),
        "contact_point1": array(point1, wp.vec3),
        "contact_normal": array(normal, wp.vec3),
        "contact_shape0": array(np.zeros(capacity, dtype=np.int32), int),
        "contact_shape1": array(shape1, int),
        "contact_thickness0": array(np.full(capacity, 0.01, dtype=np.float32), float),
        "contact_thickness1": array(np.full(capacity, 0.02, dtype=np.float32), float),
        "contact_world": array(world, int),
        "contact_slot": array(slot, int),
        "contact_art_a": array(np.zeros(capacity, dtype=np.int32), int),
        "contact_art_b": array(art1, int),
        "contact_path": array(path, int),
        "contact_slots_needed": array(1 + indices % 3, int),
        "target_size": 108,
        "articulation_response_dof_count": array([108, 6], int),
        "art_dof_start": array([0, 108], int),
        "articulation_world_dof_offset": array([6, 0], int),
        "articulation_origin": array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], wp.vec3),
        "body_single_response_dof": array([0, 1, -1], int),
        "diagonal_inverse_mass": array(np.linspace(0.25, 1.0, 114, dtype=np.float32), float),
        "joint_S_s": array(motion, wp.spatial_vector),
        "shape_body": array([0, 1, 2], int),
        "body_q": array(body_q, wp.transform),
        "friction_patches": FrictionPatches(),
        "sparse_row_dof": wp.full((world_count, 15, 2), -777, dtype=int, device=device),
        "sparse_row_z": wp.full((world_count, 15, 2), np.nan, dtype=float, device=device),
    }
    return values


def _launch(values, workers, device):
    """Launch the unchanged source kernel with the selected complete-prefix owner."""
    data, _arrays = contact_input(values, device)
    values = {
        **values,
        "dense_first": row_lines(device),
        "data": data,
        "total_num_workers": workers,
        "law": row_law(values["friction_patches"]),
    }
    kernel = response_builds_rows.sparse_pair_contact_rows_kernel
    arguments = [values[name] for name in inspect.signature(kernel.func).parameters]
    wp.launch(kernel, dim=workers, inputs=arguments, device=device)


def _expected(values, active_count):
    """Evaluate the whitened sparse slots independently in FP64: ``Z_s = J_s / sqrt(m_s)``."""
    host = {key: value.numpy() if isinstance(value, wp.array) else value for key, value in values.items()}
    dofs = np.full(host["sparse_row_dof"].shape, -777, dtype=np.int32)
    whitened = np.full(host["sparse_row_z"].shape, np.nan, dtype=np.float32)
    count = min(active_count, len(host["contact_slot"]))
    for contact in range(count):
        slot = host["contact_slot"][contact]
        if slot < 0 or host["contact_path"][contact] != 0:
            continue
        normal = -host["contact_normal"][contact].astype(np.float64)
        tangent0 = np.cross(normal, (1.0, 0.0, 0.0))
        tangent0 /= np.linalg.norm(tangent0)
        tangent1 = np.cross(normal, tangent0)
        tangent1 /= np.linalg.norm(tangent1)
        bodies = (host["contact_shape0"][contact], host["contact_shape1"][contact])
        points = [host[f"contact_point{side}"][contact].astype(np.float64).copy() for side in range(2)]
        for side in range(2):
            if bodies[side] >= 0:
                points[side] += host["body_q"][bodies[side], :3]
            points[side] += (-1.0 if side == 0 else 1.0) * host[f"contact_thickness{side}"][contact] * normal
        anchor = 0.5 * (points[0] + points[1])
        for row, direction in enumerate((normal, tangent0, tangent1)):
            if row >= host["contact_slots_needed"][contact]:
                continue
            coordinates = [-1, -1]
            coefficients = [0.0, 0.0]
            for side in range(2):
                body = bodies[side]
                art = host["contact_art_a" if side == 0 else "contact_art_b"][contact]
                if body < 0 or art < 0 or host["articulation_response_dof_count"][art] != 108:
                    continue
                dof = host["body_single_response_dof"][body]
                if dof < 0:
                    continue
                point = anchor
                if row > 0 and values["friction_patches"].enabled:
                    point = values["friction_patches"].point.numpy()[contact].astype(np.float64)
                motion = host["joint_S_s"][dof].astype(np.float64)
                velocity = motion[:3] + np.cross(motion[3:], point)
                coefficients[side] = (
                    (1.0 if side == 0 else -1.0)
                    * np.dot(direction, velocity)
                    * np.sqrt(np.float64(host["diagonal_inverse_mass"][dof]))
                )
                coordinates[side] = 6 + dof
            if coordinates[0] >= 0 and coordinates[0] == coordinates[1]:
                coefficients[0] += coefficients[1]
                coordinates[1], coefficients[1] = -1, 0.0
            world = host["contact_world"][contact]
            dofs[world, slot + row] = coordinates
            whitened[world, slot + row] = coefficients
    return dofs, whitened


class TestSparseDiagonalWhitenedRows(unittest.TestCase):
    def test_incident_velocity_is_the_jacobian_s(self):
        """The rows' build forms each row's incident velocity ``J v``, its dense part's ``J`` while in hand and its
        slots' ``Z_s (v / s)``, the one the physical Jacobian gives, and restitution bounces from it."""
        rng = np.random.default_rng(7)
        worlds, rows, dense, diagonal = 3, 5, 4, 3
        dofs = dense + diagonal
        offsets = np.array((0, diagonal, 1), dtype=np.int32)  # the dense block's first world coordinate
        inverse = rng.uniform(0.3, 2.0, worlds * dofs).astype(np.float32)
        velocity = rng.normal(size=worlds * dofs).astype(np.float32)
        factors = np.zeros((worlds, dense, dense), np.float32)
        jacobian = np.zeros((worlds, rows, dense), np.float32)
        sparse_dof = np.full((worlds, rows, 2), -1, np.int32)
        sparse_z = np.zeros((worlds, rows, 2), np.float32)
        jv = np.zeros((worlds, rows))
        for w in range(worlds):
            a = rng.normal(size=(dense, dense))
            mass = a @ a.T + dense * np.eye(dense)
            factors[w] = np.linalg.cholesky(mass[::-1, ::-1])  # the leaf-first factor: P H P^T = L L^T
            block = [offsets[w] + k for k in range(dense)]
            others = [d for d in range(dofs) if d not in block]
            v = velocity[w * dofs : (w + 1) * dofs].astype(np.float64)
            for r in range(rows):
                jacobian[w, r] = rng.normal(size=dense)
                jv[w, r] = jacobian[w, r] @ v[block]
                for slot, coordinate in enumerate(rng.choice(others, size=2, replace=False)[: 1 + r % 2]):
                    value = rng.normal()
                    sparse_dof[w, r, slot] = coordinate
                    sparse_z[w, r, slot] = value * np.sqrt(inverse[w * dofs + coordinate])
                    jv[w, r] += value * v[coordinate]
        rhs = np.full((worlds, rows), -7.0, np.float32)
        counts = wp.full(worlds, rows, dtype=wp.int32, device="cpu")
        Z = wp.array(jacobian, device="cpu")
        incident = wp.zeros((worlds, rows), dtype=wp.float32, device="cpu")
        wp.launch(
            response_builds_rows.sparse_pair_whiten_kernel(),
            dim=worlds * rows,
            inputs=[
                wp.array(factors, device="cpu"),
                wp.array(np.arange(worlds, dtype=np.int32), device="cpu"),
                wp.array(offsets, device="cpu"),
                wp.array(np.arange(worlds * dofs, dtype=np.int32).reshape(worlds, dofs), device="cpu"),
                counts,
                row_lines("cpu"),
                wp.zeros((worlds, 2), dtype=wp.int32, device="cpu"),
                0,
                rows,
                rows,
                dense,
                wp.array(sparse_dof, device="cpu"),
                wp.array(sparse_z, device="cpu"),
                wp.zeros((worlds, rows), dtype=wp.float32, device="cpu"),
                wp.array(inverse, device="cpu"),
                wp.array(velocity, dtype=wp.float32, device="cpu"),
            ],
            outputs=[Z, wp.zeros((worlds, rows), dtype=wp.float32, device="cpu"), incident],
            device="cpu",
        )
        np.testing.assert_allclose(incident.numpy(), jv, rtol=2e-5, atol=2e-6)
        records = field_group.upload(
            dense_rows.DenseRows,
            "cpu",
            row_type=np.zeros((worlds, rows), np.int32),
            phi=np.full((worlds, rows), -0.01, np.float32),
            row_beta=np.zeros((worlds, rows), np.float32),
            target=np.zeros((worlds, rows), np.float32),
            restitution=np.full((worlds, rows), 0.5, np.float32),
            rhs=rhs,
            row_w=np.ones((worlds, rows), np.float32),
            first=row_lines("cpu"),
        )
        wp.launch(
            kernels.compute_world_contact_bias,
            dim=(worlds, rows),
            inputs=[records, incident, counts, rows, rows, 0.01, 1.0, 1.0, 0.0, 0, 0],
            device="cpu",
        )
        # A restitutive impact: rhs = e * J v (target 0, the threshold 0 admits every approaching row).
        approaching = jv < 0.0
        self.assertTrue(approaching.any())
        actual = records.rhs.numpy()
        np.testing.assert_allclose(actual[approaching], 0.5 * jv[approaching], rtol=2e-5, atol=2e-6)


class TestSparseResponseOwner(unittest.TestCase):
    def test_actual_host_launch_selection(self):
        """Expose scalar parallelism while bounding work for mostly unused capacity."""
        for capacity in (0, 1, 4096, 4353, 65536, 65569, 147456, 25231360):
            self.assertEqual(_launch_workers(capacity), min(capacity, 65536))

    def test_cpu_mixed_complete_prefix(self):
        """Preserve every sparse response and inactive output above the old worker cap."""
        values = _fixture("cpu")
        count = len(values["contact_slot"])
        expected = _expected(values, count)
        # The selected workers and the full capacity: the device stride covers either.
        for workers in (_launch_workers(count), count):
            values["sparse_row_dof"].fill_(-777)
            values["sparse_row_z"].fill_(np.nan)
            _launch(values, workers, "cpu")
            np.testing.assert_array_equal(values["sparse_row_dof"].numpy(), expected[0])
            np.testing.assert_allclose(values["sparse_row_z"].numpy(), expected[1], rtol=2e-6, atol=2e-7)

    def test_cpu_empty_and_partial_prefix(self):
        """Respond to empty, partial and full prefixes; a collision pass clamps its count to the rows."""
        values = _fixture("cpu", capacity=17)
        for raw_count in (0, 9, 17):
            values["contact_count"].assign(np.array([raw_count], dtype=np.int32))
            values["sparse_row_dof"].fill_(-777)
            values["sparse_row_z"].fill_(np.nan)
            _launch(values, 17, "cpu")
            expected = _expected(values, raw_count)
            np.testing.assert_array_equal(values["sparse_row_dof"].numpy(), expected[0])
            np.testing.assert_allclose(values["sparse_row_z"].numpy(), expected[1], rtol=2e-6, atol=2e-7)
        empty = _fixture("cpu", capacity=0)
        _launch(empty, 0, "cpu")
        self.assertEqual(empty["sparse_row_dof"].numpy().size, 0)

    def test_cpu_persistent_tangent_anchors(self):
        """Changing worker ownership must retain patch-specific tangent geometry."""
        values = _fixture("cpu", capacity=73)
        patches = values["friction_patches"]
        patches.enabled = 1
        patches.point = wp.array(np.tile((0.23, -0.17, 0.41), (73, 1)), dtype=wp.vec3, device="cpu")
        expected = _expected(values, 73)
        _launch(values, _launch_workers(73), "cpu")
        np.testing.assert_array_equal(values["sparse_row_dof"].numpy(), expected[0])
        np.testing.assert_allclose(values["sparse_row_z"].numpy(), expected[1], rtol=2e-6, atol=2e-7)

    @unittest.skipUnless(wp.is_cuda_available(), "actual CUDA graph replay requires CUDA")
    def test_cuda_graph_actual_prefix_transitions(self):
        """Replay both ownership graphs over full, empty and truncated active prefixes."""
        device = wp.get_device("cuda:0")
        values = _fixture(device, capacity=65569)
        capacity = len(values["contact_slot"])
        owners = (values["sparse_row_dof"].ptr, values["sparse_row_z"].ptr)
        for workers in (_launch_workers(capacity), capacity):
            _launch(values, workers, device)
            with wp.ScopedCapture(device=device) as capture:
                _launch(values, workers, device)
            for raw_count in (capacity, 0, 65537, capacity - 1, capacity):
                values["contact_count"].assign(np.array([raw_count], dtype=np.int32))
                expected = _expected(values, raw_count)
                for graph in (False, True, True):
                    values["sparse_row_dof"].fill_(-777)
                    values["sparse_row_z"].fill_(np.nan)
                    if graph:
                        wp.capture_launch(capture.graph)
                    else:
                        _launch(values, workers, device)
                    np.testing.assert_array_equal(values["sparse_row_dof"].numpy(), expected[0])
                    np.testing.assert_allclose(values["sparse_row_z"].numpy(), expected[1], rtol=2e-6, atol=2e-7)
                self.assertEqual(owners, (values["sparse_row_dof"].ptr, values["sparse_row_z"].ptr))


if __name__ == "__main__":
    unittest.main()
