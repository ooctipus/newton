# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The contact row allocation and the Jacobian's ancestor walk against the contact laws they compose.

Contact endpoints and admission, and Jacobian entries: each kernel's output equals the laws' composition written out
in this module. The gap's precision far from the origin is tested with its consumers (the resolved-world classifier in
``test_feather_pgs_resolved_worlds``, the rows in ``test_feather_pgs_regimes`` Q20), the torsion oracle in
``test_feather_pgs_torsion_device``.
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import contact_filters, contact_laws, friction_patches, row_kernels, row_slots
from newton.tests.feather_pgs_law_harness import (
    assert_reproduces,
    call,
    register_device_tests,
    transforms,
    unit_vectors,
)

# ---------------------------------------------------------------------------------------------------------------------
# Contact row allocation and the ancestor walk against the laws
# ---------------------------------------------------------------------------------------------------------------------


def _contact_scene(rng, contacts, *, worlds=4, shapes=12, bodies=8, arts=4, friction_fraction=0.7):
    """Contacts between shapes of bodies of articulations, in body frames, with materials and prescribed motion."""
    shape_body = rng.integers(-1, bodies, shapes).astype(np.int32)
    shape_body[:2] = -1  # static shapes
    body_to_art = (np.arange(bodies) % arts).astype(np.int32)
    art_to_world = (np.arange(arts) % worlds).astype(np.int32)
    shape0 = rng.integers(-1, shapes, contacts).astype(np.int32)
    shape1 = rng.integers(0, shapes, contacts).astype(np.int32)
    body0 = np.where(shape0 >= 0, shape_body[np.maximum(shape0, 0)], -1)
    body1 = shape_body[shape1]
    art_a = np.where(body0 >= 0, body_to_art[np.maximum(body0, 0)], -1).astype(np.int32)
    art_b = np.where(body1 >= 0, body_to_art[np.maximum(body1, 0)], -1).astype(np.int32)
    world = np.where(art_a >= 0, art_to_world[np.maximum(art_a, 0)], art_to_world[np.maximum(art_b, 0)])
    restitution = rng.choice(np.array([0.0, 0.3, 1.5, -0.2, np.nan], dtype=np.float32), shapes)
    return {
        "contact_count": np.array([contacts], dtype=np.int32),
        "contact_point0": rng.normal(scale=0.2, size=(contacts, 3)).astype(np.float32),
        "contact_point1": rng.normal(scale=0.2, size=(contacts, 3)).astype(np.float32),
        "contact_normal": unit_vectors(rng, contacts),
        "contact_shape0": shape0,
        "contact_shape1": shape1,
        "contact_thickness0": np.abs(rng.normal(scale=0.01, size=contacts)).astype(np.float32),
        "contact_thickness1": np.abs(rng.normal(scale=0.01, size=contacts)).astype(np.float32),
        "contact_world": world.astype(np.int32),
        "contact_art_a": art_a,
        "contact_art_b": art_b,
        "contact_slots_needed": np.where(rng.uniform(size=contacts) < friction_fraction, 3, 1).astype(np.int32),
        "shape_body": shape_body,
        "body_q": transforms(rng, bodies),
        "body_v_s": rng.normal(size=(bodies, 6)).astype(np.float32),
        "prescribed_articulation": (rng.uniform(size=arts) < 0.5).astype(np.int32),
        "articulation_origin": rng.normal(size=(arts, 3)).astype(np.float32),
        "shape_material_mu": rng.uniform(0.0, 1.2, shapes).astype(np.float32),
        "shape_material_restitution": restitution,
        "rigid_body_max_depenetration_velocity": np.full(bodies, np.inf, dtype=np.float32),
        "body_to_articulation": body_to_art,
        "art_to_world": art_to_world,
    }


def _patches(rng, contacts, device, enabled):
    patches = friction_patches.FrictionPatches()
    patches.enabled = int(enabled)
    patches.weight = wp.array(rng.uniform(0.0, 1.0, contacts).astype(np.float32), device=device)
    patches.next_contact = wp.full(contacts, -1, dtype=int, device=device)
    patches.point = wp.array(rng.normal(size=(contacts, 3)).astype(np.float32), dtype=wp.vec3, device=device)
    patches.phi = wp.array(rng.normal(scale=0.01, size=(contacts, 2)).astype(np.float32), dtype=wp.vec2, device=device)
    patches.spin_phi = wp.zeros(contacts, dtype=float, device=device)
    patches.spin_arm = wp.zeros(contacts, dtype=float, device=device)
    return patches


@wp.kernel
def _law_contact_admission(
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_thickness0: wp.array[float],
    contact_thickness1: wp.array[float],
    body_q: wp.array[wp.transform],
    shape_body: wp.array[int],
    body_to_articulation: wp.array[int],
    art_to_world: wp.array[int],
    articulation_response_dof_count: wp.array[int],
    body_flags: wp.array[wp.int32],
    body_has_response_dofs: wp.array[int],
    is_free_rigid: wp.array[int],
    contact_gap_gate: float,
    contact_friction_gap_threshold: float,
    world_out: wp.array[int],
    art_a_out: wp.array[int],
    art_b_out: wp.array[int],
    slots_out: wp.array[int],
):
    c = wp.tid()
    world_out[c] = -1
    body_a, body_b, art_a, art_b, world = contact_laws.contact_endpoints(
        contact_shape0[c], contact_shape1[c], shape_body, body_to_articulation, art_to_world
    )
    responds = contact_laws.contact_responds(
        art_a, body_a, articulation_response_dof_count, body_has_response_dofs, body_flags
    ) or contact_laws.contact_responds(
        art_b, body_b, articulation_response_dof_count, body_has_response_dofs, body_flags
    )
    if not responds or world < 0:
        return
    _normal, _translation_a, _offset_a, _translation_b, _offset_b, _separation, phi = contact_laws.contact_points(
        contact_normal[c],
        contact_point0[c],
        contact_point1[c],
        contact_thickness0[c],
        contact_thickness1[c],
        body_a,
        body_b,
        body_q,
    )
    a_non_free = art_a >= 0 and is_free_rigid[art_a] == 0
    b_non_free = art_b >= 0 and is_free_rigid[art_b] == 0
    limit = contact_filters.contact_normal_gap_limit(a_non_free, b_non_free, art_a == art_b, contact_gap_gate, 0.0, 0.0)
    if phi > limit:
        return
    world_out[c] = world
    art_a_out[c] = art_a
    art_b_out[c] = art_b
    slots_out[c] = 1
    if phi <= contact_friction_gap_threshold:
        slots_out[c] = 3


def test_contact_endpoints_match_row_allocation(test, device):
    """Admit, place and size the contacts ``allocate_world_contact_slots`` admits (gap gate, worlds, responders)."""
    from newton._src.sim import BodyFlags  # noqa: PLC0415

    rng = np.random.default_rng(34)
    contacts = 1024
    scene = _contact_scene(rng, contacts, worlds=3, arts=6, bodies=10, shapes=16)
    scene["contact_point0"] *= 5.0  # wide gaps: the gate rejects some contacts
    arts, bodies = 6, 10
    common = {
        "contact_shape0": scene["contact_shape0"],
        "contact_shape1": scene["contact_shape1"],
        "contact_point0": scene["contact_point0"],
        "contact_point1": scene["contact_point1"],
        "contact_normal": scene["contact_normal"],
        "contact_thickness0": scene["contact_thickness0"],
        "contact_thickness1": scene["contact_thickness1"],
        "body_q": scene["body_q"],
        "shape_body": scene["shape_body"],
        "body_to_articulation": np.where(np.arange(bodies) == 9, -1, np.arange(bodies) % arts).astype(np.int32),
        "art_to_world": np.array([0, 1, 2, 0, 1, -1], dtype=np.int32),
        "articulation_response_dof_count": np.array([6, 0, 3, 6, 6, 1], dtype=np.int32),
        "body_flags": np.where(np.arange(bodies) == 4, int(BodyFlags.KINEMATIC), 0).astype(np.int32),
        "body_has_response_dofs": (np.arange(bodies) != 7).astype(np.int32),
        "is_free_rigid": np.array([0, 0, 1, 0, 1, 0], dtype=np.int32),
        "contact_gap_gate": 0.5,
        "contact_friction_gap_threshold": 0.05,
    }
    big = 1 << 20
    out = call(
        row_slots.allocate_world_contact_slots,
        contacts,
        device,
        contact_count=np.array([contacts], dtype=np.int32),
        total_num_threads=contacts,
        shape_transform=transforms(rng, 16),
        has_free_rigid=0,
        same_articulation_contact_gap_gate=0.0,
        articulation_pair_contact_gap_gate=0.0,
        max_constraints=big,
        mf_max_constraints=big,
        enable_friction=1,
        row_capacity_telemetry=0,
        friction_patches=_patches(rng, contacts, device, 0),
        resolved_worlds=np.zeros(0, dtype=np.int32),
        units=np.zeros((0, 0), dtype=np.int32),
        contact_world=np.full(contacts, -7, dtype=np.int32),
        contact_slot=np.full(contacts, -7, dtype=np.int32),
        contact_art_a=np.full(contacts, -7, dtype=np.int32),
        contact_art_b=np.full(contacts, -7, dtype=np.int32),
        world_slot_counter=np.zeros(3, dtype=np.int32),
        contact_path=np.full(contacts, -7, dtype=np.int32),
        mf_slot_counter=np.zeros(3, dtype=np.int32),
        dense_contact_world_flag=np.zeros(3, dtype=np.int32),
        contact_slots_needed=np.zeros(contacts, dtype=np.int32),
        dense_dropped_contact_rows=np.zeros(3, dtype=np.int32),
        mf_dropped_contact_rows=np.zeros(3, dtype=np.int32),
        dense_first_rejected_slot=np.full(3, big, dtype=np.int32),
        mf_first_rejected_slot=np.full(3, big, dtype=np.int32),
        capacity_status=np.zeros(4, dtype=np.int32),
        **common,
    )
    law = call(
        _law_contact_admission,
        contacts,
        device,
        world_out=np.zeros(contacts, dtype=np.int32),
        art_a_out=np.zeros(contacts, dtype=np.int32),
        art_b_out=np.zeros(contacts, dtype=np.int32),
        slots_out=np.zeros(contacts, dtype=np.int32),
        **common,
    )
    admitted = out["contact_path"].numpy() >= 0
    np.testing.assert_array_equal(admitted, law["world_out"].numpy() >= 0)
    test.assertTrue(admitted.any() and not admitted.all())
    for name, law_name in (
        ("contact_world", "world_out"),
        ("contact_art_a", "art_a_out"),
        ("contact_art_b", "art_b_out"),
        ("contact_slots_needed", "slots_out"),
    ):
        np.testing.assert_array_equal(out[name].numpy()[admitted], law[law_name].numpy()[admitted], name)


@wp.kernel
def _kernel_ancestor_rows(
    body: wp.array[int],
    sign: wp.array[float],
    point: wp.array[wp.vec3],
    origin: wp.array[wp.vec3],
    direction: wp.array[wp.vec3],
    body_to_joint: wp.array[int],
    joint_ancestor: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_S_s: wp.array[wp.spatial_vector],
    n_dofs: int,
    J_group: wp.array3d[float],
):
    row = wp.tid()
    row_kernels.accumulate_jacobian_row_world(
        body[row],
        sign[row],
        point[row],
        origin[row],
        direction[row],
        False,
        body_to_joint,
        joint_ancestor,
        joint_qd_start,
        joint_S_s,
        0,
        n_dofs,
        wp.vec2i(0, 0),
        row,
        J_group,
    )


@wp.kernel
def _law_ancestor_rows(
    body: wp.array[int],
    sign: wp.array[float],
    point: wp.array[wp.vec3],
    origin: wp.array[wp.vec3],
    direction: wp.array[wp.vec3],
    body_to_joint: wp.array[int],
    joint_ancestor: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_S_s: wp.array[wp.spatial_vector],
    n_dofs: int,
    J_group: wp.array3d[float],
):
    row = wp.tid()
    if body[row] < 0:
        return
    joint = body_to_joint[body[row]]
    while joint >= 0:
        for dof in range(joint_qd_start[joint], joint_qd_start[joint + 1]):
            if dof < n_dofs:
                J_group[0, row, dof] += sign[row] * contact_laws.jacobian_entry(
                    joint_S_s[dof], direction[row], point[row], origin[row]
                )
        joint = joint_ancestor[joint]


def test_jacobian_entry_matches_ancestor_walk(test, device):
    """Equal ``accumulate_jacobian_row_world``'s walk up the tree bitwise, multi-coordinate joints included."""
    rng = np.random.default_rng(35)
    joints, rows = 7, 512
    ancestor = np.array([-1, 0, 1, 1, 3, 0, 5], dtype=np.int32)
    qd_start = np.array([0, 6, 7, 8, 11, 12, 13, 14], dtype=np.int32)  # a free root, then 1- and 3-coordinate joints
    dofs = int(qd_start[-1])
    values = {
        "body": rng.integers(-1, joints, rows).astype(np.int32),
        "sign": rng.choice(np.array([1.0, -1.0], dtype=np.float32), rows),
        "point": rng.normal(size=(rows, 3)).astype(np.float32),
        "origin": rng.normal(size=(rows, 3)).astype(np.float32),
        "direction": unit_vectors(rng, rows),
        "body_to_joint": np.arange(joints, dtype=np.int32),
        "joint_ancestor": ancestor,
        "joint_qd_start": qd_start,
        "joint_S_s": rng.normal(size=(dofs, 6)).astype(np.float32),
        "n_dofs": dofs,
    }
    out = call(_kernel_ancestor_rows, rows, device, J_group=np.zeros((1, rows, dofs), dtype=np.float32), **values)
    law = call(_law_ancestor_rows, rows, device, J_group=np.zeros((1, rows, dofs), dtype=np.float32), **values)
    arm = np.abs(values["point"] - values["origin"]).sum(axis=1)
    entry = np.abs(values["joint_S_s"]).sum(axis=1).max() * (1.0 + arm)
    assert_reproduces(test, device, out["J_group"].numpy()[0], law["J_group"].numpy()[0], entry, "ancestor walk J")


class TestFeatherPGSContactLaws(unittest.TestCase):
    """The contact row allocation and the ancestor walk against the laws."""


register_device_tests(TestFeatherPGSContactLaws, globals())


if __name__ == "__main__":
    unittest.main(verbosity=2)
