# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Launch the solver's contact-capacity consumers over about one wave of blocks."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.sim.contacts import CAPACITY_BLOCKS_PER_SM, contact_capacity_blocks
from newton._src.solvers.feather_pgs import field_group, kernels, keyed_rows
from newton._src.solvers.feather_pgs.contact_input import ContactInput, ContactRoutes
from newton.solvers import SolverFeatherPGS
from newton.tests.feather_pgs_patch_parity import point_row_law

# Every launch over the contact capacity that grid-strides over the capped grid.
_CONSUMERS = (
    kernels.compute_contact_linear_force_from_impulses,
    kernels.pack_contact_linear_force_as_spatial,
    kernels.snapshot_contact_warmstart,
    kernels.gather_dense_warmstart,
    kernels.gather_mf_warmstart,
)


def _devices():
    return ["cpu", *(["cuda:0"] if wp.is_cuda_available() else [])]


def _press(device):
    """A prismatic box pressed onto the ground: one dense contact patch that warm starts."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.add_ground_plane()
    body = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.30), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    joint = builder.add_joint_prismatic(
        parent=-1,
        child=body,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.30), wp.quat_identity()),
        target_ke=2.0e3,
        target_kd=50.0,
    )
    builder.add_articulation([joint])
    free = builder.add_body(xform=wp.transform(wp.vec3(0.5, 0.0, 0.11), wp.quat_identity()))
    builder.add_shape_box(free, hx=0.1, hy=0.1, hz=0.1)
    return builder.finalize(device=device)


def _export_inputs(device, capacity, count, rng):
    """Random contact routes and solved rows: about a third of the contacts carry friction rows."""
    worlds, rows = 7, 48
    slot = rng.integers(0, rows - 3, capacity).astype(np.int32)
    world = rng.integers(0, worlds, capacity).astype(np.int32)
    path = rng.integers(-1, 2, capacity).astype(np.int32)
    row_type = np.full((worlds, rows), kernels.PGS_CONSTRAINT_TYPE_CONTACT, np.int32)
    row_parent = np.full((worlds, rows), -1, np.int32)
    for c in np.flatnonzero(rng.random(capacity) < 0.35):
        row_type[world[c], slot[c] + 1 : slot[c] + 3] = kernels.PGS_CONSTRAINT_TYPE_FRICTION
        row_parent[world[c], slot[c] + 1 : slot[c] + 3] = slot[c]
    normal = rng.normal(size=(capacity, 3))
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)

    def floats(shape):
        return wp.array(rng.normal(size=shape).astype(np.float32), dtype=float, device=device)

    def ints(values, dtype=wp.int32):
        return wp.array(values, dtype=dtype, device=device)

    counts = ints(np.full(worlds, rows, np.int32))
    routes = field_group.upload(
        ContactRoutes,
        device,
        count=[count],
        normal=wp.array(normal.astype(np.float32), dtype=wp.vec3, device=device),
        world=world,
        slot=slot,
        path=path,
    )
    return [
        routes,
        floats((worlds, rows)),
        floats((worlds, rows)),
        counts,
        counts,
        *(ints(values) for values in (row_type, row_parent) * 2),
        1,
        240.0,
    ]


_SCATTER_CAPACITY, _SCATTER_WORLDS = 5000, 16


def _scatter(anchors, device, count, path, max_blocks):
    """Launch the keyed-row key scatter over unique per-contact slots under one friction law; return its outputs."""
    contact = np.arange(_SCATTER_CAPACITY)
    rows = 3 * (-(-_SCATTER_CAPACITY // _SCATTER_WORLDS)) + 3
    data = ContactInput()
    data.count = wp.array([count], dtype=int, device=device)
    data.point0 = wp.zeros(_SCATTER_CAPACITY, dtype=wp.vec3, device=device)
    data.world = wp.array(contact % _SCATTER_WORLDS, dtype=int, device=device)
    data.slot = wp.array(3 * (contact // _SCATTER_WORLDS), dtype=int, device=device)
    data.path = wp.array(path, dtype=int, device=device)
    data.slots_needed = wp.array(1 + 2 * (contact % 3 != 0), dtype=int, device=device)
    data.shape0 = wp.array(contact % 5 - 1, dtype=int, device=device)
    data.shape1 = wp.array(contact % 7 - 1, dtype=int, device=device)
    data.material_mu = wp.array(np.linspace(0.2, 1.1, 6, dtype=np.float32), dtype=float, device=device)
    data.friction_scale = 0.5
    data.row_type = wp.full((_SCATTER_WORLDS, rows), -9, dtype=int, device=device)
    data.row_parent = wp.full((_SCATTER_WORLDS, rows), -9, dtype=int, device=device)
    data.row_mu = wp.full((_SCATTER_WORLDS, rows), -9.0, dtype=float, device=device)
    z = keyed_rows.KineticRows()
    z.rowkeys = wp.full((_SCATTER_WORLDS, rows), -9, dtype=int, device=device)
    law = point_row_law(device, _SCATTER_CAPACITY)
    law.patches.enabled = int(anchors)
    wp.launch(
        keyed_rows.scatter_contact_keys,
        dim=_SCATTER_CAPACITY,
        inputs=[data, z, law],
        device=device,
        max_blocks=max_blocks,
    )
    return [z.rowkeys.numpy(), data.row_type.numpy(), data.row_parent.numpy(), data.row_mu.numpy()]


class TestFeatherPGSCapacityLaunches(unittest.TestCase):
    def test_force_export_matches_on_a_capped_grid(self):
        """Export bitwise equal forces on a capped grid at every live count, the dead tail untouched.

        A capped grid loops grid-stride, and each early ``return`` past the live count becomes a
        ``continue`` in the loop, so every slot is visited exactly once whatever the grid. CPU
        launches ignore ``max_blocks``; CUDA runs one block, three blocks and the full grid.
        """
        capacity = 5000
        for device in _devices():
            for count in (0, 1, 255, 2100, capacity):
                inputs = _export_inputs(device, capacity, count, np.random.default_rng(count))
                results = []
                for max_blocks in (0, 1, 3):
                    force = wp.full(capacity, wp.vec3(7.0), dtype=wp.vec3, device=device)
                    spatial = wp.full(capacity, wp.spatial_vector(7.0), dtype=wp.spatial_vector, device=device)
                    wp.launch(
                        kernels.compute_contact_linear_force_from_impulses,
                        dim=capacity,
                        inputs=inputs,
                        outputs=[force],
                        device=device,
                        max_blocks=max_blocks,
                    )
                    wp.launch(
                        kernels.pack_contact_linear_force_as_spatial,
                        dim=capacity,
                        inputs=[inputs[0].count, force],
                        outputs=[spatial],
                        device=device,
                        max_blocks=max_blocks,
                    )
                    results.append((force.numpy(), spatial.numpy()))
                with self.subTest(device=device, count=count):
                    for capped in results[1:]:
                        np.testing.assert_array_equal(capped[0], results[0][0])
                        np.testing.assert_array_equal(capped[1], results[0][1])
                    np.testing.assert_array_equal(results[0][1][count:], 7.0)
                    if count:
                        self.assertTrue(np.isfinite(results[0][0][:count]).all())

    def test_key_scatters_match_on_a_capped_grid(self):
        """Scatter bitwise equal row keys, types, parents and coefficients on a capped grid.

        Each contact owns its rows, as the allocator guarantees, so every write has one writer
        whatever the grid. Point friction scatters keys only; patch friction also writes the rows'
        type, parent and undivided coefficient.
        """
        contact = np.arange(_SCATTER_CAPACITY)
        slots_needed = 1 + 2 * (contact % 3 != 0)
        for device in _devices():
            for count in (0, 1, 300, 4000, _SCATTER_CAPACITY):
                path = np.random.default_rng(count).integers(-1, 3, _SCATTER_CAPACITY)
                rows = int(slots_needed[(contact < count) & (path == 0)].sum())
                for anchors in (False, True):
                    results = [_scatter(anchors, device, count, path, max_blocks) for max_blocks in (0, 1, 3)]
                    with self.subTest(device=device, count=count, anchors=anchors):
                        for capped in results[1:]:
                            for full, other in zip(results[0], capped, strict=True):
                                np.testing.assert_array_equal(other, full)
                        self.assertEqual(int((results[0][0] != -9).sum()), rows)

    def test_capacity_launches_take_the_recorded_blocks(self):
        """Give every capacity-sized consumer launch the route record's block count.

        The record's count is about one wave on CUDA and 0 (uncapped) on CPU; a count set on the
        solver reaches the force export, the warm-start snapshot and every warm-start gather the
        configuration launches.
        """
        for device in _devices():
            configurations = [{"pgs_warmstart": True}]
            if wp.get_device(device).is_cuda:
                configurations.append({"pgs_warmstart": True})
            for options in configurations:
                with self.subTest(device=device, **options):
                    model = _press(device)
                    model.request_contact_attributes("force")
                    solver = SolverFeatherPGS(model, pgs_iterations=8, friction_anchor_beta=0.0, **options)
                    record = solver.get_path_selection()["routes"]["contact_consumers"]
                    self.assertEqual(record["max_blocks"], contact_capacity_blocks(model.device))
                    self.assertEqual(record["max_blocks"], solver._contact_capacity_blocks)
                    solver._contact_capacity_blocks = 3
                    self.assertEqual(solver.get_path_selection()["routes"]["contact_consumers"]["max_blocks"], 3)
                    pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
                    contacts = pipeline.contacts()
                    states, control = [model.state(), model.state()], model.control()
                    launched, launch = [], wp.launch

                    def record_launch(kernel, *args, launched=launched, launch=launch, **kwargs):
                        if any(kernel is consumer for consumer in _CONSUMERS):
                            launched.append((kernel.key, kwargs.get("max_blocks", 0)))
                        return launch(kernel, *args, **kwargs)

                    with patch.object(wp, "launch", record_launch):
                        for _ in range(3):
                            pipeline.collide(states[0], contacts)
                            solver.step(states[0], states[1], control, contacts, 1.0 / 240.0)
                            states.reverse()
                        solver.update_contacts(contacts)
                    self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
                    keys = {key for key, _ in launched}
                    for required in (
                        kernels.compute_contact_linear_force_from_impulses,
                        kernels.pack_contact_linear_force_as_spatial,
                        kernels.snapshot_contact_warmstart,
                    ):
                        self.assertIn(required.key, keys)
                    self.assertTrue(keys & {kernel.key for kernel in _CONSUMERS[3:]}, "a warm-start gather must run")
                    self.assertEqual({blocks for _, blocks in launched}, {3}, launched)

    def test_capacity_blocks_fill_every_sm(self):
        """Size the capped grid as 8 blocks of 256 threads per SM, about one wave, and leave CPU uncapped."""
        for name in _devices():
            with self.subTest(device=name):
                device = wp.get_device(name)
                expected = CAPACITY_BLOCKS_PER_SM * device.sm_count if device.is_cuda else 0
                self.assertEqual(contact_capacity_blocks(device), expected)
        self.assertEqual(CAPACITY_BLOCKS_PER_SM, 8)

    def test_lean_kernels_keep_the_full_grid(self):
        """Leave capacity launches uncapped while Warp compiles kernels without the grid-stride loop.

        A lean kernel raises on ``max_blocks > 0``, so with ``wp.config.default_grid_stride`` off the
        block count every consumer reads at construction is 0, the full grid, on CUDA as on CPU.
        """
        device = SimpleNamespace(is_cuda=True, sm_count=188)
        self.assertEqual(contact_capacity_blocks(device), CAPACITY_BLOCKS_PER_SM * 188)
        with patch.object(wp.config, "default_grid_stride", False):
            self.assertEqual(contact_capacity_blocks(device), 0)
            self.assertEqual(contact_capacity_blocks(wp.get_device("cpu")), 0)


if __name__ == "__main__":
    unittest.main()
