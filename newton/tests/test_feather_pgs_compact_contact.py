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
from newton._src.core.contact_pool import ContactPool
from newton._src.solvers.feather_pgs import compact_contact as compact
from newton._src.solvers.feather_pgs import (
    contact_input,
    field_group,
    kernels,
    response_builds_dense,
    response_builds_rows,
)
from newton._src.solvers.feather_pgs import solver_feather_pgs as solver_module
from newton._src.solvers.feather_pgs.contact_input import CONTACT_INPUT, ContactInput
from newton._src.solvers.feather_pgs.friction_patches import FrictionPatches
from newton.tests.feather_pgs_patch_parity import (
    assert_contact_impulses_close,
    assert_history_close,
    assert_ring_coulomb,
    assert_rows_close,
    contact_rows,
    point_row_law,
    row_law,
)
from newton.tests.feather_pgs_row_builders import assert_row_builders_named, launched_kernels

_FIELDS = (*CONTACT_INPUT, *response_builds_rows.COMPACT_BOUNDARY)
"""The fields of the compact producer's two structs."""


def structs(data):
    """The producer's contact input and compact boundary, bound from the fixture's fields."""
    return tuple(
        field_group.bind(struct, **{field.name: getattr(data, field.name) for field in field_group.fields_of(struct)})
        for struct in (ContactInput, response_builds_rows.CompactBoundary)
    )


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


def fixture(device="cpu", *, prefix=(2, 3), kept=(7, 7), shared=0, friction_shared=1, rows=(3,) * 8):
    """Include coupled/same-art/same-coordinate/static/prescribed contacts and permuted groups.

    ``rows`` gives each world's contacts their reserved rows: 3 for a normal and two tangents, or 1
    for a patch contact without an anchor.
    """
    rng = np.random.default_rng(191)
    # Every field of the producer's two structs (contact input and compact boundary), unused ones empty.
    data = SimpleNamespace(**{field.name: None if field.dims else 0 for field in _FIELDS})

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
    rows = np.asarray(rows, dtype=np.int32)
    slots = np.array(
        [prefix[w] + int(np.sum(rows[:c])) if c < kept[w] else -1 for w, c in zip(world, local, strict=True)],
        dtype=np.int32,
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
    data.slots_needed = array(rows[local], int)
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
    counts = array([prefix[w] + int(np.sum(rows[: kept[w]])) for w in range(2)], int)
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


def patch_view(device, count=16, *, seed=29):
    """An enabled patch view: friction locations away from the contact points and nonzero tangential phi."""
    rng = np.random.default_rng(seed)
    patches = FrictionPatches()
    patches.enabled = 1
    patches.weight = wp.array(rng.uniform(0.0, 1.0, count).astype(np.float32), dtype=float, device=device)
    patches.next_contact = wp.full(count, -1, dtype=int, device=device)
    patches.point_a = wp.array(rng.normal(0, 0.2, (count, 3)).astype(np.float32), dtype=wp.vec3, device=device)
    patches.point_b = wp.array(rng.normal(0, 0.2, (count, 3)).astype(np.float32), dtype=wp.vec3, device=device)
    patches.phi = wp.array(rng.normal(0, 0.01, (count, 2)).astype(np.float32), dtype=wp.vec2, device=device)
    return patches


def run_original(data, aux, patches=None, friction_anchor_beta=0.0):
    """Execute all seven original producer/consumer kernels on exactly the same inputs.

    Without ``patches`` the producers see point friction's zero-phi view.
    """
    data.J.zero_()
    install_prefix(data, aux)
    if patches is None:
        patches = FrictionPatches()
        patches.phi = wp.zeros(data.point0.shape[0], dtype=wp.vec2, device=aux.device)
    values = {
        "data": structs(data)[0],
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
        "friction_patches": patches,
        "friction_anchor_beta": friction_anchor_beta,
        "law": row_law(patches, friction_anchor_beta),
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
    launch(response_builds_rows.sparse_pair_contact_rows_kernel, 16)
    launch(response_builds_dense.response_rows_kernel(), 80)
    data.diag.zero_()
    launch(response_builds_dense.diagonal_rows_kernel(), 80)
    launch(response_builds_rows.sparse_pair_diagonal_kernel, 80)
    launch(response_builds_dense.cfm_worlds_kernel(), 2)


def run_candidate(data, aux, law=None):
    """Execute complete owners and the contact clear, leaving original allocation untouched.

    ``law`` is the friction row law; point friction's (beta 0, a disabled view) by default.
    """
    install_prefix(data, aux)
    wp.launch(
        compact.clear_contact_response,
        dim=(2, 256),
        inputs=[aux.counts, aux.bounds, data.dense_group, data.J, data.Y],
        block_dim=256,
        device=aux.device,
    )
    if law is None:
        law = point_row_law(aux.device, data.point0.shape[0])
    # One thread per row: three per contact.
    wp.launch(compact.produce_contacts, dim=(16, 3), inputs=[*structs(data), law], device=aux.device)
    wp.launch(
        response_builds_rows.sparse_pair_limit_kernel,
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


# Solver attributes that launch_contacts binds, by the struct field they fill.
_LAUNCH_FIELDS = {field.attr or field.name: field.name for field in _FIELDS if field.owner == "solver"}


def launch_solver(data, aux, *, anchors, beta=0.0):
    """A solver namespace holding what ``launch_contacts`` binds, over the fixture's arrays.

    ``anchors`` turns on persistent patch friction with an enabled patch view and ``friction_anchor_beta``
    ``beta``.
    """
    return SimpleNamespace(
        **{name: getattr(data, field) for name, field in _LAUNCH_FIELDS.items()},
        world_count=2,
        slot_counter=aux.counts,
        dense_phase_bounds=aux.bounds,
        model=SimpleNamespace(device=aux.device, shape_body=data.shape_body),
        L_by_size={6: data.factor},
        J_by_size={6: data.J},
        Y_by_size={6: data.Y},
        _mark_independent_sparse_contact_candidates_kernel=object(),
        _friction_anchors_enabled=anchors,
        _friction_patches=SimpleNamespace(view=patch_view(aux.device) if anchors else FrictionPatches()),
        friction_anchor_beta=beta,
        # Launches sized by contact rows go through the pool, eagerly here.
        _contact_pool=ContactPool(device=aux.device, rows=16),
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


def _unsupported(solver) -> bool:
    return False


def compact_boundary(enabled: bool):
    """Admit the compact contact boundary where the data supports it, or withhold it from the sparse-diagonal owner.

    Withheld, the sparse-diagonal producers write the rows, as for a dense articulation other than six coordinates.
    """
    return mock.patch.object(compact, "supported", compact.supported if enabled else _unsupported)


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
    # The chain stands on a fixed base: the held tree, which alone evaluates the compact diagonal, takes fixed or
    # free roots.
    parent = template.add_link(mass=0.4, inertia=inertia)
    joints = [template.add_joint_fixed(parent=-1, child=parent)]
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


def install_patch_contacts(model, state, contacts):
    """Contacts that the patch builder links into rings, held during short steps.

    Per world, each star sphere rests on a static support at three contacts in a row: the outer two
    become the ring's anchors (three-row units) and the middle one a normal-only unit. A two-contact
    ring couples the first star sphere to the chain's last sphere, which keeps that sphere's
    coordinate out of the independent schedule.

    The stars slide along z only, so a level support's tangent rows have no Jacobian. The first star's
    support is tilted by 30 degrees and its contacts start 1 mm deep: the depth's correction slides the
    star, its friction reaches the pooled load of a ring whose normal-only unit carries load, and its
    anchors gain tangential phi. The second star rests on a level support at zero depth, where gravity
    keeps its anchors loaded and their history held. Deeper contacts would lift the stars off within a
    step. Only the second world's stars start inside their limits and stay supported.
    """
    shape_body = model.shape_body.numpy()
    poses = state.body_q.numpy().astype(np.float64)
    radius = 0.01
    down = np.array((0.0, 0.0, -1.0))
    tilted = np.array((-0.5, 0.0, -np.sqrt(0.75)))

    def local(shape, point):
        if shape < 0:
            return point
        pose = poses[shape_body[shape]]
        q = -pose[3:6]
        p = point - pose[:3]
        return p + 2 * np.cross(q, np.cross(q, p) + pose[6] * p)

    rows = []
    for world in range(2):
        star0, star1, _, chain = 4 * world, 4 * world + 1, 4 * world + 2, 4 * world + 3
        for shape, normal, depth in ((star0, tilted, 1.0e-3), (star1, down, 0.0)):
            bottom = poses[shape_body[shape], :3] - np.array((0.0, 0.0, radius))
            # The row runs along y, which both supports keep tangent.
            rows += [
                (shape, -1, bottom + np.array((0.0, offset, 0.0)), normal, depth) for offset in (-0.004, 0.0, 0.004)
            ]
        middle = 0.5 * (poses[shape_body[star0], :3] + poses[shape_body[chain], :3])
        rows += [(star0, chain, middle + np.array((0.0, offset, 0.0)), down, 0.0) for offset in (-0.004, 0.004)]
    capacity = contacts.rigid_contact_shape0.shape[0]
    shapes = np.full((2, capacity), -1, dtype=np.int32)
    points = np.zeros((2, capacity, 3), dtype=np.float32)
    normals = np.tile(down, (capacity, 1)).astype(np.float32)
    depths = np.zeros(capacity, dtype=np.float32)
    for contact, (shape0, shape1, point, normal, depth) in enumerate(rows):
        shapes[:, contact] = (shape0, shape1)
        points[0, contact] = local(shape0, point)
        points[1, contact] = local(shape1, point)
        normals[contact] = normal
        depths[contact] = depth
    for endpoint in range(2):
        getattr(contacts, f"rigid_contact_shape{endpoint}").assign(shapes[endpoint])
        getattr(contacts, f"rigid_contact_point{endpoint}").assign(points[endpoint])
    contacts.rigid_contact_margin0.assign(depths)
    contacts.rigid_contact_margin1.zero_()
    contacts.rigid_contact_normal.assign(normals)
    contacts.rigid_contact_count.assign(np.array([len(rows)], dtype=np.int32))


def assert_patch_tangent_law(solver, contacts):
    """Require each anchor's tangent rows to carry its patch phi and ``friction_anchor_beta``.

    Returns the number of anchors checked. ``contact_rows`` keeps phi for the normal row only.
    """
    count = min(int(contacts.rigid_contact_count.numpy()[0]), solver.contact_slot.shape[0])
    world, slot = solver.contact_world.numpy()[:count], solver.contact_slot.numpy()[:count]
    path, needed = solver.contact_path.numpy()[:count], solver.contact_slots_needed.numpy()[:count]
    phi, beta, view = solver.phi.numpy(), solver.row_beta.numpy(), solver._friction_patches.view.phi.numpy()
    anchors = 0
    for c in range(count):
        if path[c] != 0 or slot[c] < 0 or needed[c] != 3:
            continue
        tangents = slice(int(slot[c]) + 1, int(slot[c]) + 3)
        np.testing.assert_array_equal(phi[world[c], tangents], view[c], err_msg=f"contact {c} phi")
        np.testing.assert_array_equal(
            beta[world[c], tangents], np.float32(solver.friction_anchor_beta), err_msg=f"contact {c} beta"
        )
        anchors += 1
    return anchors


# The two solver arms of the CUDA tests: the sparse-diagonal owner, then the compact boundary.
_ARMS = ("sparse_diagonal", "compact")


def check_compact_records(test, solvers, law):
    """Check and return the contact-row records of the two arms of ``build_solver_fixture`` under ``law``.

    Each names the builders of the diagonal-mass size (108) and the dense six with the friction law's
    variants. The marker encodes schedule links into the sparse rows: of point friction's triples, or of
    patch friction's units.
    """
    records = []
    for solver, arm in zip(solvers, _ARMS, strict=True):
        record = solver.get_path_selection()["routes"]["contact_rows"]
        marker = solver._mark_independent_sparse_contact_candidates_kernel.key
        if arm == "sparse_diagonal":
            metadata = "prepare_world_contact_rows"
            expected = {
                "108": ("sparse_diagonal", [metadata, "sparse_pair_contact_rows_kernel", marker]),
                "6": ("compact_jacobian", [metadata, "populate_world_J_for_compact_size"]),
            }
        else:
            produce = ["produce_contacts"]
            expected = {
                "108": ("compact_contact_boundary", [*produce, marker]),
                "6": ("compact_contact_boundary", ["clear_contact_response", *produce]),
            }
        test.assertEqual(
            record["producer"], "sparse_diagonal" if arm == "sparse_diagonal" else "compact_contact_boundary"
        )
        test.assertEqual(record["law"], law)
        test.assertEqual(
            {size: (entry["method"], entry["kernels"]) for size, entry in record["builder_by_dof"].items()}, expected
        )
        records.append(record)
    return records


def ring_carries_friction(records, first):
    """Whether contact ``first``'s patch ring, walked through ``contact_rows`` parents, holds friction."""
    contact = first
    for _ in range(len(records)):
        record = records[contact]
        if len(record["types"]) == 3 and np.any(record["impulses"][1:] != 0.0):
            return True
        contact = int(record["parents"][0])
        if contact < 0 or contact == first:
            return False
    return False


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

    def test_patch_rows_match_original_chain(self):
        """Write patch rows of mixed units as the original producers do, at the patch friction location."""
        # The prescribed contacts (4 and 5) own anchors; the last allocated contact is normal-only.
        units = (3, 1, 1, 3, 3, 3, 1, 3)
        beta = 0.35
        for shared, friction_shared in ((0, 0), (0, 1), (1, 0)):
            original, aux = fixture(shared=shared, friction_shared=friction_shared, rows=units)
            candidate, _ = fixture(shared=shared, friction_shared=friction_shared, rows=units)
            point, _ = fixture(shared=shared, friction_shared=friction_shared, rows=units)
            patches = patch_view(aux.device)
            law = compact.PatchRowLaw()
            law.patches = patches
            law.beta = beta
            run_original(original, aux, patches=patches, friction_anchor_beta=beta)
            run_candidate(candidate, aux, law=law)
            self.compare(original, candidate, aux)
            run_candidate(point, aux)
            # Tangent rows carry the anchor's phi and beta and the undivided coefficient, and they act
            # at the patch location: the point producer's tangent rows differ.
            phi = patches.phi.numpy()
            for contact, (world, slot) in enumerate(zip(candidate.world.numpy(), candidate.slot.numpy(), strict=True)):
                if slot < 0 or candidate.slots_needed.numpy()[contact] != 3:
                    continue
                tangents = slice(slot + 1, slot + 3)
                np.testing.assert_array_equal(candidate.row_type.numpy()[world, tangents], 2)
                np.testing.assert_array_equal(candidate.row_parent.numpy()[world, tangents], slot)
                np.testing.assert_array_equal(candidate.phi.numpy()[world, tangents], phi[contact])
                np.testing.assert_array_equal(candidate.row_beta.numpy()[world, tangents], np.float32(beta))
                np.testing.assert_array_equal(
                    candidate.row_mu.numpy()[world, tangents], point.row_mu.numpy()[world, tangents]
                )
            friction = candidate.row_type.numpy() == 2
            self.assertGreater(np.max(np.abs(candidate.target.numpy() - point.target.numpy())[friction]), 1.0e-3)
            self.assertGreater(np.max(np.abs(candidate.sparse_jy.numpy() - point.sparse_jy.numpy())[friction]), 1.0e-3)
            # Normal-only units hold one row; normal rows match the point producer.
            normal = candidate.row_type.numpy() == 0
            np.testing.assert_array_equal(candidate.diag.numpy()[normal], point.diag.numpy()[normal])

    def test_launch_binds_the_friction_law_of_the_solver(self):
        """Launch the patch producer exactly when anchors are on, with the solver's patch view and beta.

        The tests above check the producers' rows; this checks the selection in ``launch_contacts``.
        """
        data, aux = fixture(rows=(3, 1, 1, 3, 3, 3, 1, 3))
        contacts = SimpleNamespace(
            rigid_contact_max=16,
            **{
                "rigid_contact_" + name: getattr(data, name)
                for name in ("count", "point0", "point1", "normal", "shape0", "shape1", "margin0", "margin1")
            },
        )
        for anchors in (False, True):
            with self.subTest(anchors=anchors):
                solver = launch_solver(data, aux, anchors=anchors, beta=0.375 if anchors else 0.0)
                with mock.patch.object(compact.wp, "launch") as launch:
                    bound = contact_input.bind(
                        solver,
                        SimpleNamespace(body_q=data.body_q),
                        SimpleNamespace(body_v_s=data.body_v, joint_S_s=data.motion),
                        contacts,
                        0.0,
                        1.0,
                    )
                    compact.launch_contacts(solver, bound, 16)
                marker = solver._mark_independent_sparse_contact_candidates_kernel
                self.assertEqual(
                    [call.args[0] for call in launch.call_args_list],
                    [compact.clear_contact_response, compact.produce_contacts, marker],
                )
                produce = launch.call_args_list[-2]
                # A thread per row over the contact capacity.
                self.assertEqual(produce.kwargs["dim"], (16, 3))
                inputs = produce.kwargs["inputs"]
                # The contact input, the compact boundary and the row law: point friction is the patch law
                # with beta 0 and a disabled view.
                self.assertEqual(len(inputs), 3)
                law, view = inputs[2], solver._friction_patches.view
                self.assertEqual(law.beta, 0.375 if anchors else 0.0)
                self.assertEqual(law.patches.enabled, int(anchors))
                for name in ("weight", "next_contact", "point_a", "point_b", "phi"):
                    self.assertIs(getattr(law.patches, name), getattr(view, name), name)

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

    def test_exact_mode_admission(self):
        """Reject incomplete mode compositions before any contact ownership changes.

        The sparse-diagonal owner's own admission (its fused limits: augmented drives, no velocity-limit, mimic or
        connect rows, no debug rows or pre-elimination) is the owner's, which ``_sparse_diagonal_contact_solve``
        stands for here.
        """
        valid = {
            "model": SimpleNamespace(device=SimpleNamespace(is_cuda=True)),
            "_sparse_diagonal_contact_solve": True,
            "_sparse_diagonal_dense_size": 6,
            "_sparse_diagonal_response_size": 108,
            "_sparse_diagonal_contact_triples": True,
            "_sparse_diagonal_contact_units": False,
            "dense_max_constraints": 40,
            "_dense_internal_max_rows": 12,
            "_compact_diagonal_mass_size": 108,
        }
        self.assertTrue(compact.supported(SimpleNamespace(**valid)))
        # Patch friction schedules units instead of triples.
        units = {**valid, "_sparse_diagonal_contact_triples": False, "_sparse_diagonal_contact_units": True}
        self.assertTrue(compact.supported(SimpleNamespace(**units)))
        for name, value in (
            ("_sparse_diagonal_contact_solve", False),
            ("_sparse_diagonal_dense_size", 7),
            ("_sparse_diagonal_contact_triples", False),
            ("dense_max_constraints", 11),
            ("_dense_internal_max_rows", 13),
            ("_compact_diagonal_mass_size", None),
        ):
            self.assertFalse(compact.supported(SimpleNamespace(**{**valid, name: value})), name)

    def test_actual_fixture_layout_and_cpu_fallback(self):
        """Exercise the real model/Contacts field binding and exact unsupported CPU fallback."""
        model = build_solver_fixture("cpu")
        self.assertEqual(model.joint_dof_count, 228)
        self.assertEqual(model.shape_count, 8)
        state = model.state()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        contacts = newton.Contacts(8, 0, device=model.device)
        install_contacts(model, state, contacts)
        self.assertEqual(contacts.rigid_contact_count.numpy()[0], 6)
        solver = newton.solvers.SolverFeatherPGS(model, pgs_mode="split", dense_max_constraints=64)
        self.assertFalse(solver._compact_contact_boundary)

    def test_cuda_actual_solver_step_reset_notify_and_graphs(self):
        """Use the actual allocator, scheduler, RHS and eight-sweep solve with loaded contacts.

        Point friction runs the triple schedule; each arm's route record names exactly the row
        builders its first step launches.
        """
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("Actual sparse constructor and parallel-stream capture require CUDA")
        for device in devices:
            model = build_solver_fixture(device)
            solvers = []
            for enabled in (False, True):
                with (
                    compact_boundary(enabled),
                    mock.patch.object(solver_module, "_SPARSE_CONTACT_DIRECT", True),
                ):
                    solvers.append(
                        newton.solvers.SolverFeatherPGS(
                            model,
                            pgs_mode="matrix_free",
                            pgs_iterations=8,
                            friction_anchor_beta=0.0,
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
            for solver in solvers:
                self.assertTrue(solver._sparse_diagonal_contact_triples)
            records = check_compact_records(self, solvers, "point")
            states = [[model.state(), model.state()] for _ in solvers]
            controls = [model.control() for _ in solvers]
            contacts = [newton.Contacts(8, 0, device=device) for _ in solvers]
            for index in range(len(solvers)):
                state = states[index][0]
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                install_contacts(model, state, contacts[index])

            def compare(states=states, solvers=solvers):
                for pair in states[1:]:
                    for field in ("joint_q", "joint_qd", "body_q", "body_qd"):
                        np.testing.assert_allclose(
                            getattr(pair[0], field).numpy(),
                            getattr(states[0][0], field).numpy(),
                            rtol=3e-5,
                            atol=5e-6,
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
                    with launched_kernels() as keys:
                        solver.step(current, following, controls[index], contacts[index], 1 / 240)
                    if step == 0:
                        assert_row_builders_named(self, records[index], keys)
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

    def test_cuda_patch_rows_match_the_original_producers(self):
        """Write persistent patch rows from the compact boundary as the original producers do.

        All arms run the sparse unit schedule under default patch friction; the compact arms replace only
        the row producer, so rows by contact, impulses, states and anchor history must agree.
        """
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("Actual sparse constructor requires CUDA")
        for device in devices:
            model = build_solver_fixture(device)
            solvers = []
            for enabled in (False, True):
                with (
                    compact_boundary(enabled),
                    mock.patch.object(solver_module, "_SPARSE_CONTACT_DIRECT", True),
                ):
                    solvers.append(
                        newton.solvers.SolverFeatherPGS(
                            model,
                            pgs_mode="matrix_free",
                            pgs_iterations=8,
                            use_parallel_streams=True,
                            enable_joint_limits=True,
                            dense_max_constraints=64,
                            mf_max_constraints=64,
                        )
                    )
            for solver, enabled in zip(solvers, (False, True), strict=True):
                self.assertTrue(solver._friction_anchors_enabled)
                self.assertTrue(solver._sparse_diagonal_contact_units)
                self.assertEqual(solver._compact_contact_boundary, enabled)
            records = check_compact_records(self, solvers, "patch")
            states = [[model.state(), model.state()] for _ in solvers]
            controls = [model.control() for _ in solvers]
            contacts = [newton.Contacts(24, 0, device=device) for _ in solvers]
            for index in range(len(solvers)):
                state = states[index][0]
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                install_patch_contacts(model, state, contacts[index])
            linked = pooled = False
            independent = [False] * len(solvers)
            serial = [False] * len(solvers)
            for step in range(6):
                for index, solver in enumerate(solvers):
                    current, following = states[index]
                    with launched_kernels() as keys:
                        solver.step(current, following, controls[index], contacts[index], 1 / 240)
                    if step == 0:
                        assert_row_builders_named(self, records[index], keys)
                    states[index] = [following, current]
                reference = contact_rows(solvers[0], contacts[0])
                for index in range(1, len(solvers)):
                    for field in ("joint_q", "joint_qd", "body_q", "body_qd"):
                        np.testing.assert_allclose(
                            getattr(states[index][0], field).numpy(),
                            getattr(states[0][0], field).numpy(),
                            rtol=3e-5,
                            atol=5e-6,
                        )
                    rows = contact_rows(solvers[index], contacts[index])
                    assert_rows_close(reference, rows)
                    assert_contact_impulses_close(reference, rows)
                    assert_history_close(solvers[0], solvers[index])
                for index, solver in enumerate(solvers):
                    solver.check_constraint_capacity()
                    assert_ring_coulomb(solver)
                    self.assertGreater(assert_patch_tangent_law(solver, contacts[index]), 0)
                    independent[index] |= bool(np.any(solver._sparse_contact_group_count.numpy() > 0))
                    serial[index] |= bool(np.any(solver._sparse_contact_serial_count.numpy() > 0))
                linked |= any(len(record["types"]) == 1 and record["parents"][0] >= 0 for record in reference.values())
                pooled |= any(
                    ring_carries_friction(reference, c)
                    for c, record in reference.items()
                    if len(record["types"]) == 1 and record["impulses"][0] > 0.0
                )
            self.assertTrue(linked, "A normal-only unit must link into a patch ring")
            self.assertTrue(pooled, "A loaded normal-only unit must share a ring that carries friction")
            # In every arm the level support's ring schedules independently; the coupled ring keeps the tilted
            # support's ring serial, in speculative batches.
            self.assertEqual(independent, [True] * len(solvers), "A patch ring must schedule independently")
            self.assertEqual(serial, [True] * len(solvers), "Patch rings on a coupled coordinate must stay serial")
            self.assertTrue(solvers[0]._friction_patches.previous.valid.numpy().any(), "Anchors must carry history")

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
                    wp.launch(
                        compact.clear_contact_response,
                        dim=(2, 256),
                        inputs=[aux.counts, aux.bounds, candidate.dense_group, candidate.J, candidate.Y],
                        block_dim=256,
                        device=device,
                    )
                    wp.launch(
                        compact.produce_contacts,
                        dim=(16, 3),
                        inputs=[*structs(candidate), point_row_law(device, candidate.point0.shape[0])],
                        device=device,
                    )
                    wp.launch(
                        response_builds_rows.sparse_pair_limit_kernel,
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
