# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The general row-packet producer builds a world's prefix and contact packets in place."""

import inspect
import os
import unittest
from types import SimpleNamespace
from unittest import mock
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import compact_contact, contact_laws, field_group, friction_patches, kernels
from newton._src.solvers.feather_pgs import row_packets as rp
from newton._src.solvers.feather_pgs.dense_rows import DenseRows
from newton._src.solvers.feather_pgs.field_group import SolverArrays
from newton._src.solvers.feather_pgs.friction_patches import FrictionPatches, link_patch_routes, patch_routes
from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS
from newton.tests.feather_pgs_law_harness import position_pass, world_restitution_layout
from newton.tests.feather_pgs_patch_parity import (
    assert_contact_impulses_close,
    assert_history_close,
    assert_rows_close,
    contact_rows,
    point_row_law,
)
from newton.tests.feather_pgs_row_builders import assert_row_builders_named, launched_kernels
from newton.tests.test_feather_pgs_compact_contact import assert_patch_tangent_law
from newton.tests.test_feather_pgs_mixed_tree_free_body import DT, _build

WORLDS, PRIMARY, SECONDARY, ROWS, PACKETS = 3, 3, 6, 48, 20
# Per world, in contact order: shape0 and shape1 (the arm's links 0 and 1, the box 2, a static support 3,
# or -1 for none), the rows allocation reserved (3 for an anchor's normal and tangents, 1 for a
# normal-only unit) and the route (0 dense, 1 matrix-free, None rejected by allocation).
_CONTACTS = (
    (1, 2, 3, 0),  # the arm on the box
    (2, 3, 3, 0),  # the box on its support: two anchors and a normal-only unit in one ring
    (2, 3, 1, 0),
    (2, 3, 3, 0),
    (0, 1, 3, 0),  # the arm on itself
    (3, 2, 1, 0),  # the support first
    (2, 1, 3, 0),  # the box on the arm
    (1, -1, 3, 0),  # no second shape
    (2, 3, 3, 1),
    (0, 3, 3, None),
)
# Patch rings by contact within a world; the rejected contact shares the support's ring.
_RINGS = ((1, 2, 3, 9), (0, 6), (4,))
# Normal-only units that hold an anchor nonetheless, as allocation leaves one whose gap is above the
# friction threshold. The second world ends with one, so rows written past it reach no other contact.
_ANCHORS_WITHOUT_ROWS = (5,)
# Limit rows before the contacts, and contacts kept, per world. The first and last worlds have more
# rows than the packets hold.
_PREFIX = (2, 0, 4)
_KEPT = (10, 6, 10)
# Row outputs of a contact producer, with their poison.
_ROW_OUTPUTS = {
    "row_type": (int, -777),
    "row_parent": (int, -777),
    "row_mu": (float, float("nan")),
    "row_beta": (float, float("nan")),
    "row_cfm": (float, float("nan")),
    "phi": (float, float("nan")),
    "target": (float, float("nan")),
    "restitution": (float, float("nan")),
    "rhs": (float, float("nan")),
    "diag": (float, float("nan")),
    "row_w": (float, float("nan")),
}


def _poisoned_rows(device):
    return {
        name: wp.full((WORLDS, ROWS), poison, dtype=dtype, device=device)
        for name, (dtype, poison) in _ROW_OUTPUTS.items()
    }


def _patch_fixture(device="cpu", *, shared=0, friction_shared=0):
    """Contacts of three arm-and-box worlds, their enabled patch view and poisoned row outputs.

    Each world holds a primary arm (three coordinates over two links) and a secondary box (six).
    The arm and the box touch each other in both shape orders, the arm touches itself, static
    supports sit on either side, one contact has no second shape, and a matrix-free and a rejected
    contact are skipped. Groups are permuted per size, the second world's box is prescribed and the
    first world's box bounces. Anchors hold every three-row contact and one normal-only unit per
    world. Returns the producer input and the layout.
    """
    rng = np.random.default_rng(1017)

    def array(values, dtype):
        return wp.array(values, dtype=dtype, device=device)

    records = [
        (world, local, *contact) for world, kept in enumerate(_KEPT) for local, contact in enumerate(_CONTACTS[:kept])
    ]
    count = len(records)
    first = np.cumsum((0, *_KEPT))
    world = np.array([record[0] for record in records], dtype=np.int32)
    shape0 = np.array([4 * w + s0 for w, _, s0, _, _, _ in records], dtype=np.int32)
    shape1 = np.array([4 * w + s1 if s1 >= 0 else -1 for w, _, _, s1, _, _ in records], dtype=np.int32)
    shape_body = np.array([3 * w + s if s < 3 else -1 for w in range(WORLDS) for s in range(4)], dtype=np.int32)
    body_art = np.array([2 * w + (k == 2) for w in range(WORLDS) for k in range(3)], dtype=np.int32)

    def art(shape):
        return int(body_art[shape_body[shape]]) if shape >= 0 and shape_body[shape] >= 0 else -1

    needed = np.array([record[4] for record in records], dtype=np.int32)
    anchored = (needed == 3) | np.isin([record[1] for record in records], _ANCHORS_WITHOUT_ROWS)
    slot = np.full(count, -1, dtype=np.int32)
    path = np.zeros(count, dtype=np.int32)
    dense_rows = list(_PREFIX)
    free_rows = [0] * WORLDS
    for c, (w, _, _, _, rows, route) in enumerate(records):
        if route == 0:
            slot[c], dense_rows[w] = dense_rows[w], dense_rows[w] + rows
        elif route == 1:
            slot[c], free_rows[w], path[c] = free_rows[w], free_rows[w] + rows, 1
    next_contact = np.full(count, -1, dtype=np.int32)
    for w, kept in enumerate(_KEPT):
        for ring in _RINGS:
            members = [first[w] + local for local in ring if local < kept]
            for index, member in enumerate(members):
                next_contact[member] = members[(index + 1) % len(members)]
    view = FrictionPatches()
    view.enabled = 1
    view.weight = array(np.where(anchored, 0.5, 0.0).astype(np.float32), float)
    view.next_contact = array(next_contact, int)
    view.point_a = array(rng.normal(0.0, 0.3, (count, 3)).astype(np.float32), wp.vec3)
    view.point_b = array(rng.normal(0.0, 0.3, (count, 3)).astype(np.float32), wp.vec3)
    view.phi = array(rng.normal(0.0, 0.02, (count, 2)).astype(np.float32), wp.vec2)

    data = rp.ContactInput()
    data.count = array([count], int)
    data.point0 = array(rng.normal(0.0, 0.1, (count, 3)).astype(np.float32), wp.vec3)
    data.point1 = array(rng.normal(0.0, 0.1, (count, 3)).astype(np.float32), wp.vec3)
    normal = rng.normal(size=(count, 3))
    data.normal = array((normal / np.linalg.norm(normal, axis=1)[:, None]).astype(np.float32), wp.vec3)
    data.shape0, data.shape1 = array(shape0, int), array(shape1, int)
    data.margin0 = array(rng.uniform(0.0, 0.01, count).astype(np.float32), float)
    data.margin1 = array(rng.uniform(0.0, 0.01, count).astype(np.float32), float)
    data.world, data.slot, data.path = array(world, int), array(slot, int), array(path, int)
    data.art0 = array([art(s) for s in shape0], int)
    data.art1 = array([art(s) for s in shape1], int)
    data.slots_needed = array(needed, int)
    data.shape_body = array(shape_body, int)
    poses = np.zeros((3 * WORLDS, 7), dtype=np.float32)
    poses[:, :3] = rng.normal(size=(3 * WORLDS, 3))
    quaternions = rng.normal(size=(3 * WORLDS, 4))
    poses[:, 3:] = quaternions / np.linalg.norm(quaternions, axis=1)[:, None]
    data.body_q = array(poses, wp.transform)
    data.body_v = array(rng.normal(size=(3 * WORLDS, 6)).astype(np.float32), wp.spatial_vector)
    data.body_mask = array(np.tile(np.array([0b1, 0b111, 0b111111]), WORLDS), wp.uint32)
    data.response_dofs = array(np.tile(np.array([PRIMARY, SECONDARY]), WORLDS), int)
    data.dof_start = array([9 * w + (3 if a else 0) for w in range(WORLDS) for a in range(2)], int)
    data.origin = array(rng.normal(size=(2 * WORLDS, 3)).astype(np.float32), wp.vec3)
    data.motion = array(rng.normal(size=(9 * WORLDS, 6)).astype(np.float32), wp.spatial_vector)
    data.prescribed = array([int(a == 3) for a in range(2 * WORLDS)], int)
    data.material_mu = array(rng.uniform(0.2, 1.2, 4 * WORLDS).astype(np.float32), float)
    data.material_restitution = array([0.8 if s == 2 else 0.0 for s in range(4 * WORLDS)], float)
    data.incident = array(rng.normal(0.0, 3.0, 9 * WORLDS).astype(np.float32), float)
    data.shared_anchor, data.friction_shared_anchor = shared, friction_shared
    data.friction_scale, data.beta, data.cfm, data.dt = 0.85, 0.23, 2.0e-4, 1.0 / 240.0
    data.bias_scale, data.speculative_scale, data.restitution_threshold, data.contact_w = 1.0, 0.6, 0.05, 1.0
    for name, values in _poisoned_rows(device).items():
        setattr(data, name, values)
    layout = SimpleNamespace(
        device=device,
        view=view,
        counts=array(dense_rows, int),
        # Each size's groups are permuted against the worlds.
        groups=array([group for w in range(WORLDS) for group in ((2, 0, 1)[w], (1, 2, 0)[w])], int),
    )
    return data, layout


def _packet(device):
    packet = rp.ContactPacketData()
    packet.jacobian = wp.full((WORLDS, PACKETS, PRIMARY + SECONDARY), float("nan"), dtype=float, device=device)
    packet.row_contact = wp.full((WORLDS, PACKETS), -1, dtype=int, device=device)
    return packet


def _law(view, beta=0.35):
    law = rp.PatchRowLaw()
    law.patches = view
    law.beta = beta
    return law


def _run_general(data, layout, beta):
    """Run the general patch route's row chain on the fixture's inputs, into fresh outputs.

    As Franka's compact-Jacobian route: prepare_world_contact_rows writes the metadata,
    populate_world_J_for_compact_size each size's Jacobian, compute_world_contact_bias the bias and
    apply_world_contact_restitution_matrix_free the bounces. Returns the rows, the Jacobians by size
    and the right-hand sides before the bounces.
    """
    device, rows = layout.device, SimpleNamespace(**_poisoned_rows(layout.device))
    general = field_group.replace(data, **vars(rows))
    law = _law(layout.view, beta)
    wp.launch(kernels.prepare_world_contact_rows, dim=16, inputs=[general, 16, law], device=device)
    jacobians = {}
    for size in (PRIMARY, SECONDARY):
        jacobians[size] = wp.zeros((WORLDS, ROWS, size), dtype=float, device=device)
        wp.launch(
            kernels.populate_world_J_for_compact_size,
            dim=(4, 32),
            inputs=[general, 4, size, layout.groups, law],
            outputs=[jacobians[size]],
            device=device,
        )
    # The world Jacobian the general response gathers, which only the bounce reads.
    groups = layout.groups.numpy()
    world_jacobian = np.zeros((WORLDS, ROWS, PRIMARY + SECONDARY), dtype=np.float32)
    for w in range(WORLDS):
        world_jacobian[w, :, :PRIMARY] = jacobians[PRIMARY].numpy()[groups[2 * w]]
        world_jacobian[w, :, PRIMARY:] = jacobians[SECONDARY].numpy()[groups[2 * w + 1]]
    records = field_group.upload(DenseRows, device, **vars(rows))
    wp.launch(
        kernels.compute_world_contact_bias,
        dim=WORLDS * ROWS,
        inputs=[
            *(records, layout.counts, ROWS, data.dt, data.bias_scale, data.speculative_scale, 1.0, data.contact_w),
            *position_pass(device),
        ],
        device=device,
    )
    bias = rows.rhs.numpy().copy()
    wp.launch(
        kernels.apply_world_contact_restitution_matrix_free,
        dim=WORLDS * ROWS,
        inputs=[
            *(records, layout.counts, ROWS, wp.full(WORLDS, PRIMARY + SECONDARY, dtype=int, device=device)),
            data.incident,
            wp.array(np.arange(9 * WORLDS, dtype=np.int32).reshape(WORLDS, 9), dtype=int, device=device),
            wp.array(world_jacobian, dtype=float, device=device),
            *world_restitution_layout(device),
            *(data.dt, data.restitution_threshold, 0),
        ],
        device=device,
    )
    return rows, jacobians, bias


def _launch(kernel, data, layout, *inputs):
    """Launch a packet producer on the fixture as the owner does (one 32-lane worker per block)."""
    wp.launch_tiled(kernel, dim=[4], inputs=[4, data, *inputs], block_dim=32, device=layout.device)


def _contact_rows(data):
    """Yield ``(contact, world, first row, row count)`` of every allocated dense-route contact."""
    slots, paths = data.slot.numpy(), data.path.numpy()
    for c, (w, rows) in enumerate(zip(data.world.numpy(), data.slots_needed.numpy(), strict=True)):
        if paths[c] == 0 and slots[c] >= 0:
            yield c, int(w), int(slots[c]), int(rows)


def _owned_rows(data):
    """Return the mask of the rows that allocated dense-route contacts own, by world."""
    owned = np.zeros((WORLDS, ROWS), dtype=bool)
    for _c, w, first, count in _contact_rows(data):
        owned[w, first : first + count] = True
    return owned


def _called_functions(kernel):
    """Return the user functions a kernel calls, directly or through the functions it calls."""
    called, pending = set(), [kernel.adj]
    while pending:
        _, _, functions = pending.pop().get_references()
        for function in functions:
            if not function.is_builtin() and function not in called:
                called.add(function)
                overloads = {**function.user_overloads, **function.user_templates}.values()
                pending.extend(overload.adj for overload in overloads if overload.generic_parent is None)
    return called


def _link(data, layout, rows):
    """Link the rows' patch rings and divide their coefficients, as the patch link does after the producer."""
    contacts = data.point0.shape[0]
    live = min(int(data.count.numpy()[0]), contacts)
    routing = (data.world, data.slot, data.path, data.slots_needed)
    # The dense route's rows; linking reads no impulses.
    routes = patch_routes(((0, rows.row_parent, rows.row_mu, None, 0.0),))
    wp.launch(
        link_patch_routes,
        dim=contacts,
        inputs=[wp.array([live, live], dtype=int, device=layout.device), layout.view, *routing, routes, contacts],
        device=layout.device,
    )


class TestRowPacketPlan(unittest.TestCase):
    def test_prefix_topology_validation(self):
        """Every prefix entry is checked, including disabled mimic entries."""
        groups = np.array([0, 2], dtype=np.int32)  # primaries of two worlds
        worlds = np.array([0, 0, 1, 1], dtype=np.int32)
        starts = np.array([0, 3, 9, 12, 18], dtype=np.int32)  # 3-DOF arms and 6-DOF boxes
        mimic_start = np.array([0, 1, 1, 2, 2], dtype=np.int32)
        mimic_list = np.array([0, 1], dtype=np.int32)
        dof0 = np.array([1, 10], dtype=np.int32)
        dof1 = np.array([2, 11], dtype=np.int32)
        mimic_world = np.array([0, 1], dtype=np.int32)
        ok = rp.validate_prefix_topology(groups, worlds, starts, mimic_start, mimic_list, dof0, dof1, 3, 1, mimic_world)
        self.assertTrue(ok)
        # A mimic coefficient outside its primary's coordinates is rejected.
        bad = np.array([1, 3], dtype=np.int32)
        self.assertFalse(
            rp.validate_prefix_topology(groups, worlds, starts, mimic_start, mimic_list, dof0, bad, 3, 1, mimic_world)
        )
        # More mimic entries than the prefix admits are rejected.
        self.assertFalse(
            rp.validate_prefix_topology(groups, worlds, starts, mimic_start, mimic_list, dof0, dof1, 3, 0, mimic_world)
        )
        # The primary's coordinate count must match the layout.
        self.assertFalse(
            rp.validate_prefix_topology(groups, worlds, starts, mimic_start, mimic_list, dof0, dof1, 4, 1, mimic_world)
        )

    def test_prefix_kernel_sizes_come_from_the_layout(self):
        """Any primary whose candidates fit the packet capacity; nothing names a size."""
        for primary, mimics, packets in ((3, 0, 40), (9, 2, 40), (19, 2, 40), (7, 0, 14)):
            self.assertTrue(rp.get_prefix_kernel(primary, mimics, packets))
        for primary, mimics, packets in ((20, 1, 40), (0, 0, 40), (9, 3, 20)):
            with self.subTest(primary=primary), self.assertRaises(ValueError):
                rp.get_prefix_kernel(primary, mimics, packets)

    def test_module_names_no_robot(self):
        source = inspect.getsource(rp).lower()
        for token in ("franka", "kuka", "allegro", "[6, 9]", "size_groups) != [", "== 9"):
            self.assertNotIn(token, source)


def _owner_solver(device, *, anchors, beta=0.0):
    """A solver namespace holding what the packet owner reads, over small arrays."""

    def ints(*shape):
        return wp.zeros(shape, dtype=int, device=device)

    worlds = 2
    view = FrictionPatches()
    view.enabled = int(anchors)
    for name, dtype in (("weight", float), ("next_contact", int), ("point_a", wp.vec3), ("point_b", wp.vec3)):
        setattr(view, name, wp.zeros(4, dtype=dtype, device=device))
    view.phi = wp.zeros(4, dtype=wp.vec2, device=device)
    queue = {
        name: ints(worlds)
        for name in (
            "_local_primary_articulation",
            "_local_pair_articulation",
            "_local_residual_pair_articulation",
            "mf_constraint_count",
            "constraint_count",
            "_local_solve_owner",
            "_local_general_worlds",
        )
    }
    active = {
        name: {PRIMARY: ints(worlds)}
        for name in (
            "_local_pair_active_counts",
            "_local_pair_active_candidates",
            "_local_pair_active_secondaries",
            "_local_residual_active_counts",
            "_local_residual_active_candidates",
            "_local_residual_active_secondaries",
        )
    }
    return SimpleNamespace(
        model=SimpleNamespace(device=wp.get_device(device), articulation_count=2 * worlds),
        world_count=worlds,
        size_groups=[PRIMARY, SECONDARY],
        _local_solve_max_rows=PACKETS // 2,
        _local_residual_max_rows=PACKETS,
        _local_residual_mf_max_rows=4,
        **queue,
        **active,
        mf_body_a=ints(worlds, 4),
        mf_body_b=ints(worlds, 4),
        mf_row_type=ints(worlds, 4),
        body_to_articulation=ints(3 * worlds),
        _local_general_world_count=ints(1),
        world_response_group_art_start={size: ints(worlds + 1) for size in (PRIMARY, SECONDARY)},
        world_response_group_to_art={size: ints(worlds) for size in (PRIMARY, SECONDARY)},
        art_group_idx=ints(2 * worlds),
        J_by_size={size: wp.zeros((worlds, ROWS, size), dtype=float, device=device) for size in (PRIMARY, SECONDARY)},
        _friction_anchors_enabled=anchors,
        _friction_patches=SimpleNamespace(view=view),
        friction_anchor_beta=beta,
        # The owner's placeholders come from the solver's allocator.
        _arrays=SolverArrays(None, wp.get_device(device)),
    )


class _PatchRowTwins:
    """Twins: the packet producers write the persistent patch rows of the general route.

    On the CPU one lane writes each contact's rows; on CUDA a warp's lanes split its coefficients.
    """

    device = "cpu"

    def assert_same(self, actual, expected, err_msg=""):
        """Require equal arrays: bitwise on the CPU, and floats to 1e-5 on CUDA, where FMA contraction may differ."""
        if self.device == "cpu" or np.asarray(expected).dtype.kind != "f":
            np.testing.assert_array_equal(actual, expected, err_msg=err_msg)
        else:
            np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=1.0e-5, err_msg=err_msg)

    def test_patch_rows_match_the_general_route(self):
        """Write every patch row field and Jacobian of the general route, then link the same rings.

        Rows equal the general route's bitwise, except a bouncing normal row's right-hand side: its
        incident velocity sums per body instead of per world coordinate, which agrees to rounding
        because every incident velocity here lies far from the bounce thresholds. Whole buffers are
        compared, so a row written past a contact's allocation shows even where a later contact
        rewrites it.
        """
        beta = 0.35
        for shared, friction_shared in ((0, 0), (0, 1), (1, 0), (1, 1)):
            with self.subTest(shared=shared, friction_shared=friction_shared):
                data, layout = _patch_fixture(self.device, shared=shared, friction_shared=friction_shared)
                general, jacobians, bias = _run_general(data, layout, beta)
                packet = _packet(layout.device)
                _launch(rp.produce_contacts, data, layout, _law(layout.view, beta), packet, PRIMARY, SECONDARY)
                owned = _owned_rows(data)
                # The general route writes the metadata of allocated rows alone; the producer also seeds their
                # diagonals with cfm and, with contact_w 1, leaves row_w alone. Other rows keep their poison.
                expected = {name: getattr(general, name).numpy().copy() for name in _ROW_OUTPUTS}
                for name, (_, poison) in _ROW_OUTPUTS.items():
                    expected[name][~owned] = poison
                expected["diag"][owned] = data.cfm
                # A bouncing normal row agrees to rounding; compare it apart from the bitwise rest.
                rhs, reference = data.rhs.numpy(), expected["rhs"]
                bounced = owned & (reference != bias)
                np.testing.assert_allclose(rhs[bounced], reference[bounced], rtol=1.0e-5, atol=1.0e-5)
                self.assertTrue(np.all(rhs[bounced] != bias[bounced]), "Bouncing rows must leave the bias")
                reference[bounced] = rhs[bounced]
                for name in _ROW_OUTPUTS:
                    self.assert_same(getattr(data, name).numpy(), expected[name], err_msg=name)
                groups = layout.groups.numpy()
                jacobian = np.full((WORLDS, PACKETS, PRIMARY + SECONDARY), np.nan, dtype=np.float32)
                row_contact = np.full((WORLDS, PACKETS), -1, dtype=np.int32)
                for c, w, first, count in _contact_rows(data):
                    for row in range(first, min(first + count, PACKETS)):
                        row_contact[w, row] = 3 * c + row - first
                        jacobian[w, row, :PRIMARY] = jacobians[PRIMARY].numpy()[groups[2 * w], row]
                        jacobian[w, row, PRIMARY:] = jacobians[SECONDARY].numpy()[groups[2 * w + 1], row]
                self.assert_same(packet.row_contact.numpy(), row_contact)
                self.assert_same(packet.jacobian.numpy(), jacobian)
                # The fixture bounces, holds anchors with and without tangent rows and overflows the packets.
                weight = layout.view.weight.numpy()
                contacts = list(_contact_rows(data))
                self.assertTrue(bounced.any())
                self.assertTrue(any(count == 3 for *_, count in contacts))
                self.assertTrue(any(weight[c] > 0.0 and count == 1 for c, *_, count in contacts))
                self.assertTrue(owned[:, PACKETS:].any())
                # The patch link runs after the producer and writes the same ring parents and coefficients.
                _link(data, layout, data)
                _link(data, layout, general)
                for name in ("row_parent", "row_mu"):
                    self.assert_same(getattr(data, name).numpy(), getattr(general, name).numpy(), name)
                parents = data.row_parent.numpy()
                linked = [parents[w, first] for c, w, first, count in contacts if count == 1]
                self.assertTrue(any(parent >= 0 for parent in linked), "A normal-only unit must join a ring")

    def test_patch_fallback_matches_the_general_jacobians(self):
        """Materialize only general worlds' patch rows, with the general route's Jacobians.

        Whole buffers are compared: a general world's rows past its contacts' allocations keep their
        cleared zeros, and a locally owned world's rows stay stale.
        """
        for shared, friction_shared in ((0, 0), (1, 1)):
            data, layout = _patch_fixture(self.device, shared=shared, friction_shared=friction_shared)
            _, jacobians, _ = _run_general(data, layout, 0.35)
            law, groups = _law(layout.view), layout.groups.numpy()
            # The second world is locally owned, or every world falls back to the general response.
            for owners in ((0, 2, 0), (0, 0, 0)):
                with self.subTest(shared=shared, friction_shared=friction_shared, owners=owners):
                    owner = wp.array(owners, dtype=int, device=layout.device)
                    J, expected = {}, {}
                    for size in (PRIMARY, SECONDARY):
                        # materialize_general_prefix has cleared the general worlds' rows; a local world's are stale.
                        values = np.zeros((WORLDS, ROWS, size), dtype=np.float32)
                        expected[size] = jacobians[size].numpy().copy()
                        for w in np.flatnonzero(owners):
                            group = groups[2 * w + int(size == SECONDARY)]
                            values[group] = expected[size][group] = np.nan
                        J[size] = wp.array(values, dtype=float, device=layout.device)
                    wp.launch(
                        rp.materialize_fallback,
                        dim=128,
                        inputs=[128, data, law, owner, PRIMARY, SECONDARY, layout.groups, J[PRIMARY], J[SECONDARY]],
                        device=layout.device,
                    )
                    for size in (PRIMARY, SECONDARY):
                        self.assert_same(J[size].numpy(), expected[size], err_msg=f"size {size}")

    def test_patch_tangent_rows_replace_the_point_law(self):
        """Keep the point law's normal rows and move only the tangent rows to the patch law.

        Tangent rows carry the anchor's phi and ``friction_anchor_beta``, keep the undivided
        coefficient and act at the patch location; the point producer's act at the contact points
        with no phi or beta. Normal rows differ only in the bias rounding, ``x * (1 / dt)`` against
        the point producer's ``x / dt``.
        """
        beta = 0.35
        data, layout = _patch_fixture(self.device)
        point, _ = _patch_fixture(self.device)
        packet, point_packet = _packet(layout.device), _packet(layout.device)
        _launch(rp.produce_contacts, data, layout, _law(layout.view, beta), packet, PRIMARY, SECONDARY)
        point_law = point_row_law(layout.device, layout.view.phi.shape[0])
        _launch(rp.produce_contacts, point, layout, point_law, point_packet, PRIMARY, SECONDARY)
        phi = layout.view.phi.numpy()
        moved = 0.0
        for c, w, first, count in _contact_rows(data):
            for name in ("row_type", "row_parent", "row_mu", "row_beta", "row_cfm", "phi", "target", "restitution"):
                self.assert_same(
                    getattr(data, name).numpy()[w, first], getattr(point, name).numpy()[w, first], f"contact {c}"
                )
            np.testing.assert_allclose(
                data.rhs.numpy()[w, first], point.rhs.numpy()[w, first], rtol=1.0e-6, atol=1.0e-6
            )
            if first < PACKETS:
                self.assert_same(packet.jacobian.numpy()[w, first], point_packet.jacobian.numpy()[w, first])
            if count != 3:
                continue
            tangents = slice(first + 1, first + 3)
            np.testing.assert_array_equal(data.row_type.numpy()[w, tangents], kernels.PGS_CONSTRAINT_TYPE_FRICTION)
            np.testing.assert_array_equal(data.row_parent.numpy()[w, tangents], first)
            np.testing.assert_array_equal(data.phi.numpy()[w, tangents], phi[c])
            np.testing.assert_array_equal(data.row_beta.numpy()[w, tangents], np.float32(beta))
            np.testing.assert_array_equal(data.row_mu.numpy()[w, tangents], point.row_mu.numpy()[w, tangents])
            np.testing.assert_array_equal(point.phi.numpy()[w, tangents], 0.0)
            np.testing.assert_array_equal(point.row_beta.numpy()[w, tangents], 0.0)
            moved = max(
                moved, float(np.max(np.abs(data.target.numpy()[w, tangents] - point.target.numpy()[w, tangents])))
            )
            if first + 2 < PACKETS:
                difference = packet.jacobian.numpy()[w, tangents] - point_packet.jacobian.numpy()[w, tangents]
                moved = max(moved, float(np.max(np.abs(difference))))
        self.assertGreater(moved, 1.0e-3, "Tangent rows must act at the patch friction location")


class TestRowPacketPatchRows(_PatchRowTwins, unittest.TestCase):
    """CPU twins, the owner's choice of the friction law, and the shared row law."""

    def test_patch_producers_share_one_row_law(self):
        """Write patch rows through the contact row frame law and the patch tangent row in the packet and compact producers."""
        producers = (
            rp.produce_contacts,
            rp.materialize_fallback,
            compact_contact.produce_contacts,
        )
        for kernel in producers:
            name = f"{kernel.func.__module__}.{kernel.key}"
            called = _called_functions(kernel)
            self.assertIn(contact_laws.contact_row_frame, called, name)
            # The fallback writes Jacobians alone; the producers also write the tangent rows' metadata.
            if kernel is not rp.materialize_fallback:
                self.assertIn(friction_patches.patch_tangent_row, called, name)

    def test_owner_launches_the_friction_law_it_was_built_with(self):
        """Launch the producer and fallback with the solver's patch view and beta under either friction law.

        Point friction is the patch law with beta 0 and a disabled view. ``friction_anchor_beta`` is read at
        each launch, as the general route reads it, so a change after construction reaches the next step's rows.
        """
        device = "cpu"
        for anchors in (False, True):
            with self.subTest(anchors=anchors):
                solver = _owner_solver(device, anchors=anchors, beta=0.375 if anchors else 0.0)
                with mock.patch.object(rp.TopologyPin, "read", return_value=None):
                    owner = rp.LocalRowPackets(solver, PRIMARY, SECONDARY, 0)
                for beta in (0.5, 0.125) if anchors else (0.0,):
                    solver.friction_anchor_beta = beta
                    contact_input = rp.ContactInput()
                    with (
                        mock.patch.object(rp.wp, "launch_tiled") as tiled,
                        mock.patch.object(rp.wp, "launch") as launch,
                    ):
                        owner.produce_contacts(contact_input)
                        owner.finish_rows()
                    fallback = rp.materialize_fallback
                    self.assertEqual(
                        [call.args[0] for call in launch.call_args_list], [rp.finalize_and_queue, fallback]
                    )
                    self.assertEqual(
                        [call.args[0] for call in tiled.call_args_list],
                        [rp.produce_contacts, rp.materialize_general_prefix, rp.materialize_general_prefix],
                    )
                    laws = []
                    for call in (tiled.call_args_list[0], launch.call_args_list[1]):
                        inputs = call.kwargs["inputs"]
                        self.assertIs(inputs[1], contact_input)
                        # The row law follows the contact input.
                        self.assertIsInstance(inputs[2], rp.PatchRowLaw.cls)
                        laws.append(inputs[2])
                    self.assertIs(tiled.call_args_list[0].kwargs["inputs"][3], owner.packet)
                    view = solver._friction_patches.view
                    for law in laws:
                        self.assertEqual(law.beta, beta)
                        self.assertEqual(law.patches.enabled, int(anchors))
                        for name in ("weight", "next_contact", "point_a", "point_b", "phi"):
                            self.assertIs(getattr(law.patches, name), getattr(view, name), name)


@unittest.skipUnless(wp.is_cuda_available(), "The packet producers split each contact over a warp on CUDA")
class TestRowPacketPatchRowsCUDA(_PatchRowTwins, unittest.TestCase):
    """The twins on CUDA, where a contact's coefficients are split over a warp's lanes."""

    device = "cuda:0"


# Drive each arm's capsule into the side of its box from contact. The arm-box rows are the dense patch
# rows (the box rests on matrix-free rows): they carry friction from the first step, and on the mimic
# arm their anchors carry phi after about 30 steps.
_PRESS = {"swing": 0.0, "start": -0.16, "target": -0.3}
# Press a row of sphere pads (``_build``'s ``pads``) into each box's side from contact while the box
# settles onto the ground. Per world, one dense patch region slides at its pooled bound from the first
# step: its end pads own two anchors that share the friction coefficient, its inner pads are
# normal-only units. On the CPU split route, a start change of up to 1e-5 rad keeps the rings and the
# poses of _PAD_STEPS steps inside the twin's checks.
_PADS = {"swing": 0.0, "start": -0.16, "target": -0.26}
_PAD_STEPS = 10


def _arm(model, *, packets, swing_stride=3, friction_anchor_beta=0.0, swing=1.5, start=0.0, target=0.0):
    """Build one route with its own collision and states.

    Every ``swing_stride``-th revolute joint starts at angle ``start`` and speed ``swing`` and is
    driven to ``target``; by default the arms swing over the boxes from rest.
    """
    env = {
        "FEATHER_PGS_MIXED_TREE_KINETIC_STATE": "1",
        "FEATHER_PGS_TREE_KINETIC_STATE": "0",
        "FEATHER_PGS_RESOLVED_WORLDS": "0",
        "FEATHER_PGS_ROW_PACKETS": str(int(packets)),
    }
    with patch.dict(os.environ, env):
        solver = SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            pgs_iterations=8,
            enable_joint_limits=True,
            update_mass_matrix_interval=2,
            friction_anchor_beta=friction_anchor_beta,
        )
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    states = [model.state() for _ in range(2)]
    types, q_start, qd_start = model.joint_type.numpy(), model.joint_q_start.numpy(), model.joint_qd_start.numpy()
    driven = np.flatnonzero(types == int(newton.JointType.REVOLUTE))[::swing_stride]
    q = states[0].joint_q.numpy()
    q[q_start[driven]] = start
    states[0].joint_q.assign(q)
    newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
    control = model.control()
    targets = np.zeros_like(model.joint_target_q.numpy())
    targets[qd_start[driven]] = target
    control.joint_target_q.assign(targets)
    qd = np.zeros(model.joint_dof_count, dtype=np.float32)
    qd[qd_start[driven]] = swing
    states[0].joint_qd.assign(qd)
    return SimpleNamespace(solver=solver, pipeline=pipeline, contacts=contacts, states=states, control=control)


def _step(arm, contacts=None):
    """Advance one route by a step on its own collision, or on ``contacts``; return the contacts it used."""
    if contacts is None:
        arm.pipeline.collide(arm.states[0], arm.contacts)
        contacts = arm.contacts
    arm.states[0].clear_forces()
    arm.solver.step(arm.states[0], arm.states[1], arm.control, contacts, DT)
    arm.states.reverse()
    return contacts


def _rings(records):
    """Return each patch ring of :func:`contact_rows` records, keyed by its lowest contact.

    A ring holds its member contacts, its anchors (the members with tangent rows) and its impulses:
    the pooled normal impulse and the anchors' summed tangent impulses, in the shared basis of the
    members' common normal.
    """
    rings = {}
    for first in records:
        members, contact = [], first
        while contact >= 0 and contact not in members:
            members.append(contact)
            contact = int(records[contact]["parents"][0])
        key = min(members)
        if key in rings:
            continue
        anchors = [c for c in sorted(members) if len(records[c]["types"]) == 3]
        tangent = sum((records[c]["impulses"][1:].astype(np.float64) for c in anchors), np.zeros(2))
        load = sum(float(records[c]["impulses"][0]) for c in members)
        rings[key] = SimpleNamespace(members=sorted(members), anchors=anchors, impulses=np.array([load, *tangent]))
    return rings


def _rollout(model, *, packets, steps=120, **kwargs):
    """Swing the arms over the boxes; return the solver and the body poses."""
    arm = _arm(model, packets=packets, **kwargs)
    poses = []
    for _ in range(steps):
        _step(arm)
        poses.append(arm.states[0].body_q.numpy().copy())
    wp.synchronize_device(model.device)
    return arm.solver, np.array(poses)


@unittest.skipUnless(wp.is_cuda_available(), "The local solve owners are a CUDA path")
class TestRowPacketsCUDA(unittest.TestCase):
    def _compare(self, model, **kwargs):
        plain_solver, plain = _rollout(model, packets=False, **kwargs)
        self.assertIsNone(plain_solver._row_packets)
        solver, poses = _rollout(model, packets=True, **kwargs)
        self.assertIsInstance(solver._row_packets, rp.LocalRowPackets)
        self.assertTrue(np.isfinite(poses).all())
        np.testing.assert_allclose(poses[:, :, :3], plain[:, :, :3], atol=1.0e-4, rtol=0.0)
        np.testing.assert_allclose(poses[:, :, 3:], plain[:, :, 3:], atol=1.0e-4, rtol=0.0)
        if solver._friction_anchors_enabled:
            # Each rollout collides on its own state, so anchor history is compared on shared contacts only.
            self.assertTrue(plain_solver._friction_patches.previous.valid.numpy().any(), "The boxes hold anchors")
            self.assertTrue(solver._friction_patches.previous.valid.numpy().any(), "The boxes hold anchors")
        return solver

    def test_arm_and_box_match_the_ordinary_producers(self):
        """A fixed-base arm over a free box: the packet owner tracks the ordinary row producers."""
        model = _build(wp.get_device("cuda:0"), worlds=6)
        solver = self._compare(model)
        self.assertEqual((solver._row_packets.primary, solver._row_packets.secondary), (3, 6))
        self.assertEqual(solver._row_packets.mimics, 0)

    def test_mimic_rows_join_the_prefix(self):
        """A following finger joint: the mimic row is built in the prefix with the limits."""
        model = _build(wp.get_device("cuda:0"), worlds=6, mimic=True)
        solver = self._compare(model)
        self.assertEqual(solver._row_packets.mimics, 1)
        self.assertEqual(int(solver._mimic_count), 6)

    def test_longer_arm_takes_the_same_owner(self):
        """Nothing in the owner is sized for one arm: a five-joint arm is admitted the same way."""
        model = _build(wp.get_device("cuda:0"), worlds=4, links=5)
        solver = self._compare(model)
        self.assertEqual(solver._row_packets.primary, 5)

    def test_patch_friction_matches_the_ordinary_producers(self):
        """Write the ordinary producers' persistent patch rows and track their short rollout.

        Both routes run the local owners under patch friction; only the row producer differs, and
        both step on the same contacts. The arms press their boxes, so the dense patch rows carry
        friction from the first step, when rows and impulses by contact agree to rounding. Every
        step's anchors carry their patch phi and friction_anchor_beta, and the routes end with the
        same poses, rows and anchor history.
        """
        model = _build(wp.get_device("cuda:0"), worlds=6, mimic=True)
        plain, packets = (_arm(model, packets=enabled, friction_anchor_beta=0.2, **_PRESS) for enabled in (False, True))
        self.assertIsNone(plain.solver._row_packets)
        self.assertIsInstance(packets.solver._row_packets, rp.LocalRowPackets)
        self.assertTrue(packets.solver._friction_anchors_enabled)
        routes = packets.solver.get_path_selection()["routes"]
        self.assertEqual(routes["friction"]["law"], "patch")
        self.assertEqual(routes["contact_rows"]["producer"], "local_packets")
        self.assertEqual(routes["sweep"]["local"]["factory"], "row_packets")
        # The packets' patch producer and the fallback materialization of the general worlds, for every size.
        builder = {
            "method": "local_packets",
            "kernels": ["produce_contacts", "materialize_fallback"],
            "law": "patch",
        }
        builders = routes["contact_rows"]["builder_by_dof"]
        self.assertEqual(builders, dict.fromkeys(builders, builder))
        anchors, carried, poses = 0, 0, []
        for step in range(40):
            with launched_kernels() as plain_keys:
                contacts = _step(plain)
            with launched_kernels() as packet_keys:
                _step(packets, contacts)
            if step == 0:
                for arm, keys in ((plain, plain_keys), (packets, packet_keys)):
                    assert_row_builders_named(self, arm.solver.get_path_selection()["routes"]["contact_rows"], keys)
                first = [contact_rows(arm.solver, contacts) for arm in (plain, packets)]
                self.assertTrue(
                    any(len(record["types"]) == 3 and np.any(record["impulses"][1:]) for record in first[0].values()),
                    "The arms must press their boxes with friction from the first step",
                )
                assert_rows_close(*first, atol=1.0e-6, rtol=1.0e-5)
                assert_contact_impulses_close(*first)
            anchors += assert_patch_tangent_law(packets.solver, contacts)
            records = contact_rows(packets.solver, contacts).values()
            carried += sum(len(record["types"]) == 3 and np.any(record["patch"][1:3]) for record in records)
            poses.append([arm.states[0].body_q.numpy() for arm in (plain, packets)])
        poses = np.array(poses)
        self.assertTrue(np.isfinite(poses).all())
        np.testing.assert_allclose(poses[:, 1], poses[:, 0], atol=1.0e-4, rtol=0.0)
        self.assertGreater(anchors, 0)
        self.assertGreater(carried, 0, "Anchors must carry tangential phi")
        assert_rows_close(*(contact_rows(arm.solver, contacts) for arm in (plain, packets)))
        self.assertTrue(plain.solver._friction_patches.previous.valid.numpy().any(), "Pressed boxes hold anchors")
        assert_history_close(plain.solver, packets.solver)

    def test_patch_rings_match_the_ordinary_producers(self):
        """Pool loaded patch rings of two anchors and normal-only units from the packet rows.

        Each box slides down the pads that press its side, so the pads' ring carries friction at its
        pooled bound from the first step. There the rows agree by contact, and each ring's pooled
        normal and summed tangent impulses agree. The pads' own normal impulses split the load in the
        sweep's row order, which slot allocation leaves free, so they are compared through their ring.
        Both routes step on the same contacts and keep the same rings, the tangent law and close poses.
        """
        model = _build(wp.get_device("cuda:0"), worlds=6, pads=_PADS["start"])
        plain, packets = (_arm(model, packets=enabled, friction_anchor_beta=0.2, **_PADS) for enabled in (False, True))
        self.assertIsNone(plain.solver._row_packets)
        self.assertIsInstance(packets.solver._row_packets, rp.LocalRowPackets)
        self.assertTrue(packets.solver._friction_anchors_enabled)
        self._step_pad_rings(plain, packets)

    def _step_pad_rings(self, plain, packets):
        """Step both routes on the plain route's contacts and compare their patch rings."""
        scale, poses = plain.solver.contact_friction_scale, []
        for step in range(_PAD_STEPS):
            contacts = _step(plain)
            _step(packets, contacts)
            records = [contact_rows(arm.solver, contacts) for arm in (plain, packets)]
            self.assertEqual(sorted(records[1]), sorted(records[0]))
            for c, record in records[0].items():
                for name in ("types", "parents"):
                    np.testing.assert_array_equal(records[1][c][name], record[name], f"step {step} contact {c} {name}")
            if step == 0:
                rings = [_rings(record) for record in records]
                loaded = [ring for ring in rings[0].values() if len(ring.anchors) == 2 and len(ring.members) > 2]
                loaded = [ring for ring in loaded if ring.impulses[0] > 0.0]
                self.assertTrue(loaded, "The pads must load a ring of two anchors and normal-only units")
                bound = 0.0
                for ring in loaded:
                    for c in ring.anchors:
                        mu, impulses = records[0][c]["mu"], records[0][c]["impulses"]
                        # The link divides the coefficient between the ring's two anchors.
                        np.testing.assert_allclose(mu[1:], mu[0] * scale / 2.0, rtol=1.0e-6)
                        bound = max(bound, float(np.linalg.norm(impulses[1:])) / (float(mu[1]) * ring.impulses[0]))
                self.assertGreater(bound, 0.98, "The pads' friction must reach the ring's pooled bound")
                assert_rows_close(*records, atol=1.0e-6, rtol=1.0e-5)
                for key, ring in rings[0].items():
                    np.testing.assert_allclose(
                        rings[1][key].impulses, ring.impulses, atol=2.0e-5, rtol=7.0e-4, err_msg=f"ring {key}"
                    )
            assert_patch_tangent_law(packets.solver, contacts)
            poses.append([arm.states[0].body_q.numpy() for arm in (plain, packets)])
        poses = np.array(poses)
        self.assertTrue(np.isfinite(poses).all())
        np.testing.assert_allclose(poses[:, 1], poses[:, 0], atol=1.0e-4, rtol=0.0)

    def test_patch_friction_keeps_every_packet_layout(self):
        """Admit every packet layout under patch friction and track the ordinary producers' rollout."""
        for kwargs, layout in (
            ({"worlds": 6}, (3, 0)),
            ({"worlds": 6, "mimic": True}, (3, 1)),
            ({"worlds": 4, "links": 5}, (5, 0)),
        ):
            with self.subTest(**kwargs):
                solver = self._compare(_build(wp.get_device("cuda:0"), **kwargs), friction_anchor_beta=0.2)
                self.assertTrue(solver._friction_anchors_enabled)
                self.assertEqual((solver._row_packets.primary, solver._row_packets.mimics), layout)


if __name__ == "__main__":
    unittest.main()
