# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The contact-row producers against the contact laws they compose.

Geometry, materials, row frames, prescribed targets and Jacobian entries: each
producer's rows equal the laws' composition written out in this module. The two
arithmetic contracts kept as modes are tested with their consumers (endpoint-relative
in ``test_feather_pgs_resolved_worlds``, the torsion oracle in
``test_feather_pgs_torsion_device``).
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import contact_filters, contact_laws, kernels
from newton.tests.feather_pgs_law_harness import (
    assert_reproduces,
    call,
    register_device_tests,
    transforms,
    unit_vectors,
)
from newton.tests.feather_pgs_patch_parity import point_row_law

# ---------------------------------------------------------------------------------------------------------------------
# Contact front end against the producers it replaces
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
        "body_to_articulation": body_to_art,
        "art_to_world": art_to_world,
    }


def _term_scales(scene, patches=None):
    """The sizes of the terms of a scene's world-frame values, which a comparison on a CUDA device measures in.

    Returns ``(length, speed)``: ``length`` bounds a world point or lever arm (body translation, body-frame
    point, margin, articulation origin, patch anchor), ``speed`` a prescribed velocity at such a point.
    """
    length = (
        np.abs(scene["body_q"][:, :3]).sum(axis=1).max()
        + max(np.abs(scene["contact_point0"]).sum(axis=1).max(), np.abs(scene["contact_point1"]).sum(axis=1).max())
        + max(scene["contact_thickness0"].max(), scene["contact_thickness1"].max())
        + np.abs(scene["articulation_origin"]).sum(axis=1).max()
    )
    if patches is not None and patches.enabled:
        length += max(
            np.abs(patches.point_a.numpy()).sum(axis=1).max(), np.abs(patches.point_b.numpy()).sum(axis=1).max()
        )
    speed = 2.0 * np.abs(scene["body_v_s"]).sum(axis=1).max() * (1.0 + length)
    return 2.0 * length, speed


def _patches(rng, contacts, device, enabled):
    patches = kernels.FrictionPatches()
    patches.enabled = int(enabled)
    patches.weight = wp.array(rng.uniform(0.0, 1.0, contacts).astype(np.float32), device=device)
    patches.next_contact = wp.full(contacts, -1, dtype=int, device=device)
    patches.point_a = wp.array(rng.normal(size=(contacts, 3)).astype(np.float32), dtype=wp.vec3, device=device)
    patches.point_b = wp.array(rng.normal(size=(contacts, 3)).astype(np.float32), dtype=wp.vec3, device=device)
    patches.phi = wp.array(rng.normal(scale=0.01, size=(contacts, 2)).astype(np.float32), dtype=wp.vec2, device=device)
    return patches


@wp.kernel
def _law_contact_rows(
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_thickness0: wp.array[float],
    contact_thickness1: wp.array[float],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    body_v_s: wp.array[wp.spatial_vector],
    prescribed_articulation: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    shape_material_mu: wp.array[float],
    shape_material_restitution: wp.array[float],
    contact_friction_shared_anchor: int,
    contact_friction_scale: float,
    contact_shared_anchor: int,
    friction_patches: kernels.FrictionPatches,
    phi_out: wp.array[float],
    mu_out: wp.array[wp.vec2],
    restitution_out: wp.array[float],
    target_out: wp.array[wp.vec3],
):
    c = wp.tid()
    body_a, body_b, normal, point_a, point_b, phi = contact_laws.contact_geometry(
        c,
        contact_shape0,
        contact_shape1,
        contact_point0,
        contact_point1,
        contact_normal,
        contact_thickness0,
        contact_thickness1,
        shape_body,
        body_q,
    )
    mu, restitution = contact_laws.contact_materials(
        contact_shape0[c], contact_shape1[c], shape_material_mu, shape_material_restitution
    )
    tangent0, tangent1 = contact_laws.contact_tangent_basis(normal)
    friction_a, friction_b = contact_laws.contact_friction_points(
        point_a, point_b, contact_shared_anchor, contact_friction_shared_anchor
    )
    if friction_patches.enabled != 0:
        friction_a = friction_patches.point_a[c]
        friction_b = friction_patches.point_b[c]
    targets = wp.vec3(0.0)
    for row in range(3):
        direction, p0, p1 = contact_laws.contact_row_frame(
            row, contact_shared_anchor, normal, tangent0, tangent1, point_a, point_b, friction_a, friction_b
        )
        targets[row] = contact_laws.prescribed_target(
            body_a,
            contact_art_a[c],
            body_b,
            contact_art_b[c],
            p0,
            p1,
            direction,
            prescribed_articulation,
            articulation_origin,
            body_v_s,
        )
    phi_out[c] = phi
    mu_out[c] = wp.vec2(mu, mu * contact_friction_scale)
    restitution_out[c] = restitution
    target_out[c] = targets


def test_contact_front_end_matches_world_contact_rows(test, device):
    """Equal ``prepare_world_contact_rows`` bitwise: gap, materials, friction coefficient and every row's target.

    The geometry, materials, row frame (point law with each anchor option, and the
    patch law) and prescribed targets come from the laws.
    """
    rng = np.random.default_rng(31)
    contacts = 512
    scene = _contact_scene(rng, contacts)
    slots = (3 * np.arange(contacts)).astype(np.int32)
    rows = 3 * contacts
    for shared, friction_shared, patched in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)):
        with test.subTest(shared_anchor=shared, friction_shared_anchor=friction_shared, patches=patched):
            patches = _patches(rng, contacts, device, patched)
            out = call(
                kernels.prepare_world_contact_rows,
                contacts,
                device,
                total_num_threads=contacts,
                contact_slot=slots,
                contact_world=np.zeros(contacts, dtype=np.int32),
                contact_path=np.zeros(contacts, dtype=np.int32),
                articulation_response_dof_count=np.full(4, 6, dtype=np.int32),
                contact_friction_shared_anchor=friction_shared,
                contact_friction_scale=0.8,
                contact_shared_anchor=shared,
                pgs_beta=0.2,
                pgs_cfm=1.0e-6,
                friction_patches=patches,
                friction_anchor_beta=0.1,
                world_row_type=np.zeros((1, rows), dtype=np.int32),
                world_row_parent=np.zeros((1, rows), dtype=np.int32),
                world_row_mu=np.zeros((1, rows), dtype=np.float32),
                world_row_beta=np.zeros((1, rows), dtype=np.float32),
                world_row_cfm=np.zeros((1, rows), dtype=np.float32),
                world_phi=np.zeros((1, rows), dtype=np.float32),
                world_target_velocity=np.zeros((1, rows), dtype=np.float32),
                world_row_restitution=np.zeros((1, rows), dtype=np.float32),
                **{
                    k: v for k, v in scene.items() if k not in ("body_to_articulation", "art_to_world", "contact_world")
                },
            )
            law = call(
                _law_contact_rows,
                contacts,
                device,
                contact_friction_shared_anchor=friction_shared,
                contact_friction_scale=0.8,
                contact_shared_anchor=shared,
                friction_patches=patches,
                phi_out=np.zeros(contacts, dtype=np.float32),
                mu_out=np.zeros((contacts, 2), dtype=np.float32),
                restitution_out=np.zeros(contacts, dtype=np.float32),
                target_out=np.zeros((contacts, 3), dtype=np.float32),
                **{
                    k: v
                    for k, v in scene.items()
                    if k.startswith(("contact_", "shape_", "body_", "prescribed", "articulation"))
                    and k not in ("contact_count", "contact_world", "contact_slots_needed", "body_to_articulation")
                },
            )
            phi = out["world_phi"].numpy().reshape(contacts, 3)
            mu = out["world_row_mu"].numpy().reshape(contacts, 3)
            target = out["world_target_velocity"].numpy().reshape(contacts, 3)
            restitution = out["world_row_restitution"].numpy().reshape(contacts, 3)
            friction = scene["contact_slots_needed"] >= 3
            length, speed = _term_scales(scene, patches)
            mu_law = law["mu_out"].numpy()
            target_law = law["target_out"].numpy()
            assert_reproduces(test, device, phi[:, 0], law["phi_out"].numpy(), length, "world rows: phi")
            assert_reproduces(test, device, mu[:, 0], mu_law[:, 0], name="world rows: mu")
            assert_reproduces(test, device, mu[friction, 1], mu_law[friction, 1], name="world rows: friction mu")
            assert_reproduces(test, device, restitution[:, 0], law["restitution_out"].numpy(), name="world rows: e")
            assert_reproduces(test, device, target[:, 0], target_law[:, 0], speed, "world rows: normal target")
            assert_reproduces(
                test, device, target[friction, 1:], target_law[friction, 1:], speed, "world rows: targets"
            )


def test_contact_front_end_matches_masked_row_prelude(test, device):
    """Equal the masked builder's prelude (``row_build.contact_row_prelude``) bitwise: gap, coefficients, targets.

    The point law's prelude writes the same row metadata as ``prepare_world_contact_rows``
    from its own copy of the geometry, materials and row frames.
    """
    from newton._src.solvers.feather_pgs import row_build  # noqa: PLC0415

    rng = np.random.default_rng(38)
    contacts = 512
    scene = _contact_scene(rng, contacts)
    slots = (3 * np.arange(contacts)).astype(np.int32)
    rows = 3 * contacts
    for shared, friction_shared in ((0, 0), (1, 0), (0, 1)):
        with test.subTest(shared_anchor=shared, friction_shared_anchor=friction_shared):
            out = call(
                row_build.contact_row_prelude,
                3 * contacts,
                device,
                total_threads=3 * contacts,
                world_constraint_count=np.full(1, rows, dtype=np.int32),
                mf_constraint_count=np.zeros(1, dtype=np.int32),
                owned_rows_le=0,
                contact_slot=slots,
                contact_world=np.zeros(contacts, dtype=np.int32),
                contact_path=np.zeros(contacts, dtype=np.int32),
                size_a=6,
                size_b=6,
                law=point_row_law(device, contacts),
                articulation_response_dof_count=np.full(4, 6, dtype=np.int32),
                contact_friction_shared_anchor=friction_shared,
                contact_friction_articulation_pairs_only=0,
                is_free_rigid=np.zeros(4, dtype=np.int32),
                contact_friction_scale=0.8,
                contact_shared_anchor=shared,
                pgs_beta=0.2,
                pgs_cfm=1.0e-6,
                pre_flags=np.zeros(contacts, dtype=np.int32),
                world_row_type=np.zeros((1, rows), dtype=np.int32),
                world_row_parent=np.zeros((1, rows), dtype=np.int32),
                world_row_mu=np.zeros((1, rows), dtype=np.float32),
                world_row_beta=np.zeros((1, rows), dtype=np.float32),
                world_row_cfm=np.zeros((1, rows), dtype=np.float32),
                world_phi=np.zeros((1, rows), dtype=np.float32),
                world_target_velocity=np.zeros((1, rows), dtype=np.float32),
                world_row_restitution=np.zeros((1, rows), dtype=np.float32),
                **{
                    k: v for k, v in scene.items() if k not in ("body_to_articulation", "art_to_world", "contact_world")
                },
            )
            law = call(
                _law_contact_rows,
                contacts,
                device,
                contact_friction_shared_anchor=friction_shared,
                contact_friction_scale=0.8,
                contact_shared_anchor=shared,
                friction_patches=_patches(rng, contacts, device, 0),
                phi_out=np.zeros(contacts, dtype=np.float32),
                mu_out=np.zeros((contacts, 2), dtype=np.float32),
                restitution_out=np.zeros(contacts, dtype=np.float32),
                target_out=np.zeros((contacts, 3), dtype=np.float32),
                **{
                    k: v
                    for k, v in scene.items()
                    if k.startswith(("contact_", "shape_", "body_", "prescribed", "articulation"))
                    and k not in ("contact_count", "contact_world", "contact_slots_needed", "body_to_articulation")
                },
            )
            phi = out["world_phi"].numpy().reshape(contacts, 3)
            mu = out["world_row_mu"].numpy().reshape(contacts, 3)
            target = out["world_target_velocity"].numpy().reshape(contacts, 3)
            restitution = out["world_row_restitution"].numpy().reshape(contacts, 3)
            # The prelude writes the contacts with an endpoint articulation of its sizes (flag bit 0).
            active = (out["pre_flags"].numpy() & 1) != 0
            friction = active & (scene["contact_slots_needed"] >= 3)
            test.assertGreater(active.sum(), contacts // 2)
            length, speed = _term_scales(scene)
            mu_law = law["mu_out"].numpy()
            target_law = law["target_out"].numpy()
            restitution_law = law["restitution_out"].numpy()
            assert_reproduces(test, device, phi[active, 0], law["phi_out"].numpy()[active], length, "prelude: phi")
            assert_reproduces(test, device, mu[active, 0], mu_law[active, 0], name="prelude: mu")
            assert_reproduces(test, device, mu[friction, 1], mu_law[friction, 1], name="prelude: friction mu")
            assert_reproduces(test, device, restitution[active, 0], restitution_law[active], name="prelude: e")
            assert_reproduces(test, device, target[active, 0], target_law[active, 0], speed, "prelude: normal target")
            assert_reproduces(test, device, target[friction, 1:], target_law[friction, 1:], speed, "prelude: targets")


@wp.kernel
def _law_compact_jacobian(
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_thickness0: wp.array[float],
    contact_thickness1: wp.array[float],
    contact_slot: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    contact_slots_needed: wp.array[int],
    target_size: int,
    articulation_response_dof_count: wp.array[int],
    art_group_idx: wp.array[int],
    art_dof_start: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    body_response_dof_mask: wp.array[wp.uint32],
    joint_S_s: wp.array[wp.spatial_vector],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    contact_friction_shared_anchor: int,
    friction_patches: kernels.FrictionPatches,
    contact_shared_anchor: int,
    J_group: wp.array3d[float],
):
    c, row, local_dof = wp.tid()
    if row >= contact_slots_needed[c]:
        return
    body_a, body_b = contact_laws.contact_bodies(contact_shape0[c], contact_shape1[c], shape_body)
    normal, point_a, point_b, _phi = contact_laws.contact_points(
        contact_normal[c],
        contact_point0[c],
        contact_point1[c],
        contact_thickness0[c],
        contact_thickness1[c],
        body_a,
        body_b,
        body_q,
    )
    tangent0, tangent1 = contact_laws.contact_tangent_basis(normal)
    friction_a, friction_b = contact_laws.contact_friction_points(
        point_a, point_b, contact_shared_anchor, contact_friction_shared_anchor
    )
    if friction_patches.enabled != 0:
        friction_a = friction_patches.point_a[c]
        friction_b = friction_patches.point_b[c]
    direction, p0, p1 = contact_laws.contact_row_frame(
        row, contact_shared_anchor, normal, tangent0, tangent1, point_a, point_b, friction_a, friction_b
    )
    bit = wp.uint32(1) << wp.uint32(local_dof)
    art_a = contact_art_a[c]
    art_b = contact_art_b[c]
    group_a = int(-1)
    group_b = int(-1)
    value_a = float(0.0)
    value_b = float(0.0)
    if art_a >= 0 and articulation_response_dof_count[art_a] == target_size:
        group_a = art_group_idx[art_a]
        if body_a >= 0 and (body_response_dof_mask[body_a] & bit) != wp.uint32(0):
            motion = joint_S_s[art_dof_start[art_a] + local_dof]
            value_a = contact_laws.jacobian_entry(motion, direction, p0, articulation_origin[art_a])
    if art_b >= 0 and articulation_response_dof_count[art_b] == target_size:
        group_b = art_group_idx[art_b]
        if body_b >= 0 and (body_response_dof_mask[body_b] & bit) != wp.uint32(0):
            motion = joint_S_s[art_dof_start[art_b] + local_dof]
            value_b = -contact_laws.jacobian_entry(motion, direction, p1, articulation_origin[art_b])
    slot = contact_slot[c] + row
    if group_a >= 0:
        if group_b == group_a:
            J_group[group_a, slot, local_dof] = value_a + value_b
        else:
            J_group[group_a, slot, local_dof] = value_a
    if group_b >= 0 and group_b != group_a:
        J_group[group_b, slot, local_dof] = value_b


def test_jacobian_entry_matches_compact_jacobian(test, device):
    """Equal ``populate_world_J_for_compact_size`` bitwise: every row frame and Jacobian entry of both endpoints."""
    rng = np.random.default_rng(32)
    contacts, size = 256, 6
    scene = _contact_scene(rng, contacts, worlds=1, arts=4)
    arts = 4
    bodies = scene["body_q"].shape[0]
    slots = (3 * np.arange(contacts)).astype(np.int32)
    common = {
        "contact_point0": scene["contact_point0"],
        "contact_point1": scene["contact_point1"],
        "contact_normal": scene["contact_normal"],
        "contact_shape0": scene["contact_shape0"],
        "contact_shape1": scene["contact_shape1"],
        "contact_thickness0": scene["contact_thickness0"],
        "contact_thickness1": scene["contact_thickness1"],
        "contact_slot": slots,
        "contact_art_a": scene["contact_art_a"],
        "contact_art_b": scene["contact_art_b"],
        "contact_slots_needed": scene["contact_slots_needed"],
        "target_size": size,
        "articulation_response_dof_count": np.array([size, size, 3, size], dtype=np.int32),
        "art_group_idx": np.array([0, 1, -1, 0], dtype=np.int32),
        "art_dof_start": (size * np.arange(arts)).astype(np.int32),
        "articulation_origin": scene["articulation_origin"],
        "body_response_dof_mask": rng.integers(0, 2**size, bodies).astype(np.uint32),
        "joint_S_s": rng.normal(size=(arts * size, 6)).astype(np.float32),
        "shape_body": scene["shape_body"],
        "body_q": scene["body_q"],
    }
    rows = 3 * contacts
    for shared, friction_shared, patched in ((0, 0, 0), (1, 1, 0), (0, 0, 1)):
        with test.subTest(shared_anchor=shared, friction_shared_anchor=friction_shared, patches=patched):
            patches = _patches(rng, contacts, device, patched)
            flags = {
                "contact_friction_shared_anchor": friction_shared,
                "friction_patches": patches,
                "contact_shared_anchor": shared,
            }
            out = call(
                kernels.populate_world_J_for_compact_size,
                (contacts, 3 * size),
                device,
                contact_count=np.array([contacts], dtype=np.int32),
                total_num_workers=contacts,
                contact_path=np.zeros(contacts, dtype=np.int32),
                J_group=np.zeros((2, rows, size), dtype=np.float32),
                **common,
                **flags,
            )
            law = call(
                _law_compact_jacobian,
                (contacts, 3, size),
                device,
                J_group=np.zeros((2, rows, size), dtype=np.float32),
                **common,
                **flags,
            )
            length, _speed = _term_scales(scene, patches)
            entry = np.abs(common["joint_S_s"]).sum(axis=1).max() * (1.0 + length)
            assert_reproduces(test, device, out["J_group"].numpy(), law["J_group"].numpy(), entry, "compact J")
            test.assertTrue(np.any(out["J_group"].numpy() != 0.0))


@wp.kernel
def _law_mf_jacobian(
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_thickness0: wp.array[float],
    contact_thickness1: wp.array[float],
    contact_slot: wp.array[int],
    contact_slots_needed: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    articulation_response_dof_count: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    body_v_s: wp.array[wp.spatial_vector],
    prescribed_articulation: wp.array[int],
    shape_material_mu: wp.array[float],
    shape_material_restitution: wp.array[float],
    contact_friction_shared_anchor: int,
    contact_friction_scale: float,
    contact_shared_anchor: int,
    friction_patches: kernels.FrictionPatches,
    friction_anchor_beta: float,
    mf_Z_a: wp.array3d[float],
    mf_Z_b: wp.array3d[float],
    mf_phi: wp.array2d[float],
    mf_row_mu: wp.array2d[float],
    mf_target_velocity: wp.array2d[float],
    mf_row_restitution: wp.array2d[float],
):
    c, row = wp.tid()
    if row >= contact_slots_needed[c]:
        return
    body_a, body_b = contact_laws.contact_bodies(contact_shape0[c], contact_shape1[c], shape_body)
    normal, point_a, point_b, phi = contact_laws.contact_points(
        contact_normal[c],
        contact_point0[c],
        contact_point1[c],
        contact_thickness0[c],
        contact_thickness1[c],
        body_a,
        body_b,
        body_q,
    )
    mu, restitution = contact_laws.contact_materials(
        contact_shape0[c], contact_shape1[c], shape_material_mu, shape_material_restitution
    )
    tangent0, tangent1 = contact_laws.contact_tangent_basis(normal)
    friction_a, friction_b = contact_laws.contact_friction_points(
        point_a, point_b, contact_shared_anchor, contact_friction_shared_anchor
    )
    if friction_patches.enabled != 0:
        friction_a = friction_patches.point_a[c]
        friction_b = friction_patches.point_b[c]
    direction, p0, p1 = contact_laws.contact_row_frame(
        row, contact_shared_anchor, normal, tangent0, tangent1, point_a, point_b, friction_a, friction_b
    )
    slot = contact_slot[c] + row
    art_a = contact_art_a[c]
    art_b = contact_art_b[c]
    if body_a >= 0 and art_a >= 0 and articulation_response_dof_count[art_a] > 0:
        J = contact_laws.free_body_jacobian(direction, p0, articulation_origin[art_a])
        for k in range(6):
            mf_Z_a[0, slot, k] = J[k]
    if body_b >= 0 and art_b >= 0 and articulation_response_dof_count[art_b] > 0:
        J = contact_laws.free_body_jacobian(direction, p1, articulation_origin[art_b])
        for k in range(6):
            mf_Z_b[0, slot, k] = -J[k]
    if row == 0:
        mf_phi[0, slot] = phi
        mf_row_mu[0, slot] = mu
        mf_row_restitution[0, slot] = restitution
    else:
        mf_phi[0, slot] = friction_anchor_beta * friction_patches.phi[c][row - 1]
        mf_row_mu[0, slot] = mu * contact_friction_scale
        mf_row_restitution[0, slot] = 0.0
    mf_target_velocity[0, slot] = contact_laws.prescribed_target(
        body_a, art_a, body_b, art_b, p0, p1, direction, prescribed_articulation, articulation_origin, body_v_s
    )


def test_free_body_jacobian_matches_mf_contact_rows(test, device):
    """Equal ``build_mf_contact_rows`` bitwise: free-body Jacobians, gaps, coefficients and targets of every row."""
    rng = np.random.default_rng(33)
    contacts = 256
    scene = _contact_scene(rng, contacts, worlds=1)
    slots = (3 * np.arange(contacts)).astype(np.int32)
    rows = 3 * contacts
    common = {
        k: v
        for k, v in scene.items()
        if k.startswith(("contact_", "shape_", "body_", "prescribed", "articulation"))
        and k not in ("contact_count", "contact_world", "body_to_articulation")
    }
    common.update(
        contact_slot=slots,
        articulation_response_dof_count=np.array([6, 6, 0, 6], dtype=np.int32),
        contact_friction_scale=0.7,
        friction_anchor_beta=0.15,
    )
    outputs = ("mf_Z_a", "mf_Z_b", "mf_phi", "mf_row_mu", "mf_target_velocity", "mf_row_restitution")
    for shared, friction_shared, patched in ((0, 0, 0), (1, 0, 0), (0, 1, 1)):
        with test.subTest(shared_anchor=shared, friction_shared_anchor=friction_shared, patches=patched):
            patches = _patches(rng, contacts, device, patched)
            flags = {
                "contact_friction_shared_anchor": friction_shared,
                "contact_shared_anchor": shared,
                "friction_patches": patches,
            }
            buffers = lambda: {  # noqa: E731
                "mf_Z_a": np.zeros((1, rows, 6), dtype=np.float32),
                "mf_Z_b": np.zeros((1, rows, 6), dtype=np.float32),
                "mf_phi": np.zeros((1, rows), dtype=np.float32),
                "mf_row_mu": np.zeros((1, rows), dtype=np.float32),
                "mf_target_velocity": np.zeros((1, rows), dtype=np.float32),
                "mf_row_restitution": np.zeros((1, rows), dtype=np.float32),
            }
            out = call(
                kernels.build_mf_contact_rows,
                contacts,
                device,
                contact_count=np.array([contacts], dtype=np.int32),
                total_num_threads=contacts,
                contact_world=np.zeros(contacts, dtype=np.int32),
                contact_path=np.ones(contacts, dtype=np.int32),
                has_target_velocity=1,
                mf_body_a=np.zeros((1, rows), dtype=np.int32),
                mf_body_b=np.zeros((1, rows), dtype=np.int32),
                mf_row_type=np.zeros((1, rows), dtype=np.int32),
                mf_row_parent=np.zeros((1, rows), dtype=np.int32),
                **common,
                **flags,
                **buffers(),
            )
            law = call(_law_mf_jacobian, (contacts, 3), device, **common, **flags, **buffers())
            length, speed = _term_scales(scene, patches)
            magnitudes = {"mf_Z_a": 1.0 + length, "mf_Z_b": 1.0 + length, "mf_phi": length, "mf_target_velocity": speed}
            for name in outputs:
                assert_reproduces(
                    test,
                    device,
                    out[name].numpy().reshape(rows, -1),
                    law[name].numpy().reshape(rows, -1),
                    magnitudes.get(name),
                    f"mf rows: {name}",
                )


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
    _normal, _point_a, _point_b, phi = contact_laws.contact_points(
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
    if contact_filters.contact_friction_eligible(phi, a_non_free, b_non_free, 0, contact_friction_gap_threshold):
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
        kernels.allocate_world_contact_slots,
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
        contact_friction_articulation_pairs_only=0,
        row_capacity_telemetry=0,
        friction_patches=_patches(rng, contacts, device, 0),
        resolved_worlds=np.zeros(0, dtype=np.int32),
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
    kernels.accumulate_jacobian_row_world(
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
    """The contact front end against the producers it replaces."""


register_device_tests(TestFeatherPGSContactLaws, globals())


if __name__ == "__main__":
    unittest.main(verbosity=2)
