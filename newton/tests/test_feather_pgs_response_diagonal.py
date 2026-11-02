# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import itertools
import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import field_group
from newton._src.solvers.feather_pgs.contact_input import ContactInput
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
    PGS_LOCAL_SOLVE_OWNER_GENERAL,
    PGS_LOCAL_SOLVE_OWNER_PAIR,
    PGS_LOCAL_SOLVE_OWNER_PAIR_RESIDUAL,
)
from newton._src.solvers.feather_pgs.projection_laws import friction_pair
from newton._src.solvers.feather_pgs.solver_feather_pgs import _FeatherPGSExecutionPlan
from newton._src.solvers.feather_pgs.sweep_levels import (
    _get_build_independent_sparse_contact_groups_kernel,
    _get_mark_independent_sparse_contact_candidates_kernel,
)
from newton._src.solvers.feather_pgs.sweep_routes_diag import sparse_diagonal_kernel
from newton._src.solvers.feather_pgs.sweep_routes_local import paired_factor_kernel, paired_factor_source
from newton._src.solvers.feather_pgs.tree_factor import (
    WALK_COORDINATE,
    WALK_LINK,
    WalkClass,
    WalkRelease,
    WalkTrees,
    walk_leave_kernel,
)
from newton.solvers import SolverFeatherPGS
from newton.tests.feather_pgs_law_harness import row_lines, row_velocity


def _build_mixed_response_model(
    device,
    world_count=1,
    *,
    dof_count=13,
    friction=0.0,
    restitution=0.0,
    static_plane=False,
    static_support=False,
    free_body_first=False,
    free_body_velocity_limit=None,
):
    """Build one serial articulation contacting one free rigid body.

    ``static_support`` rests the free body on a static ledge that clears the arm, so the free body also
    produces matrix-free contact rows. ``free_body_first`` builds the free body before the articulation,
    which packs the world DOFs free-body first. ``free_body_velocity_limit`` caps the free body's linear
    and angular velocity [m/s, rad/s], producing matrix-free velocity-limit rows once exceeded.
    ``static_plane`` adds a ground plane under the free body.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    if free_body_velocity_limit is not None:
        SolverFeatherPGS.register_custom_attributes(builder)
    builder.default_shape_cfg.density = 1000.0
    builder.default_shape_cfg.ke = 1.0e5
    builder.default_shape_cfg.kd = 1.0e3
    builder.default_shape_cfg.mu = friction
    builder.default_shape_cfg.restitution = restitution
    builder.default_shape_cfg.margin = 0.0
    builder.default_shape_cfg.gap = 0.0

    def add_free_body():
        custom_attributes = None
        if free_body_velocity_limit is not None:
            custom_attributes = {
                "rigid_body_max_linear_velocity": free_body_velocity_limit,
                "rigid_body_max_angular_velocity": free_body_velocity_limit,
            }
        box = builder.add_link(
            xform=wp.transform(wp.vec3(0.7, 0.0, 0.5695), wp.quat_identity()), custom_attributes=custom_attributes
        )
        builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.05)
        builder.add_articulation([builder.add_joint_free(parent=-1, child=box)])
        if static_support:
            builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(0.7, 0.11, 0.47), wp.quat_identity()),
                hx=0.05,
                hy=0.05,
                hz=0.05,
            )

    if free_body_first:
        add_free_body()
    arm = builder.add_link()
    builder.add_shape_box(arm, hx=0.4, hy=0.05, hz=0.02)
    joints = [
        builder.add_joint_revolute(
            parent=-1,
            child=arm,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.4, 0.0, 0.0), wp.quat_identity()),
        )
    ]
    parent = arm
    for index in range(dof_count - 1):
        child = builder.add_link(
            mass=0.05,
            inertia=wp.mat33(np.eye(3, dtype=np.float32) * 1.0e-3),
            lock_inertia=True,
        )
        joints.append(
            builder.add_joint_revolute(
                parent=parent,
                child=child,
                axis=(newton.Axis.X, newton.Axis.Y, newton.Axis.Z)[index % 3],
            )
        )
        parent = child
    builder.add_articulation(joints)
    builder.add_constraint_mimic(joint0=joints[1], joint1=joints[0], coef0=0.0, coef1=1.0)

    if not free_body_first:
        add_free_body()
    if static_plane:
        builder.add_shape_plane(plane=(0.0, 0.0, 1.0, -0.53))
    if world_count == 1:
        return builder.finalize(device=device)
    replicated = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    replicated.replicate(builder, world_count, spacing=(3.0, 0.0, 0.0))
    return replicated.finalize(device=device)


def _run_mixed_response(
    kernel,
    *,
    warmstart,
    preelimination,
    dof_count=13,
    dense_max_constraints=32,
    inactive_joint_limit_capacity=False,
    friction=0.0,
    restitution=0.0,
    tangential_velocity=0.0,
    static_plane=False,
    contact_regularization=0.0,
    friction_anchor_beta=None,
    model_kwargs=None,
):
    """Run a short mixed-contact trajectory with one H-inverse implementation."""
    model = _build_mixed_response_model(
        "cuda:0",
        dof_count=dof_count,
        friction=friction,
        restitution=restitution,
        static_plane=static_plane,
        **(model_kwargs or {}),
    )
    # The arms compare serial owners: the pin keeps the Jacobi tiers (case K4, a tree beside a free body) off the
    # worlds no other owner claims.
    with mock.patch.object(SolverFeatherPGS, "_route_pins", {"hinv_jt_kernel": kernel, "tiers": 0}):
        solver = SolverFeatherPGS(
            model,
            pgs_warmstart=warmstart,
            enable_bilateral_preelimination=preelimination,
            enable_contact_friction=friction > 0.0,
            enable_joint_limits=inactive_joint_limit_capacity,
            joint_limit_activation_gap=0.0,
            pgs_iterations=8,
            pgs_contact_regularization=contact_regularization,
            friction_anchor_beta=friction_anchor_beta,
            dense_max_constraints=dense_max_constraints,
            mf_max_constraints=32,
        )
    state_in, state_out = model.state(), model.state()
    joint_qd = state_in.joint_qd.numpy()
    free_articulation = int(np.flatnonzero(solver.forest.is_free_rigid_host)[0])
    free_dof_start = int(solver.forest.articulation_dof_start_host[free_articulation])
    joint_qd[free_dof_start] = tangential_velocity
    joint_qd[free_dof_start + 2] = -3.0
    state_in.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)

    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        reduce_contacts=False,
        contact_matching="latest",
    )
    contacts = pipeline.contacts()
    control = model.control()
    samples = []
    for _ in range(4):
        state_in.clear_forces()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, 1.0 / 240.0)
        constraint_count = int(solver.placement.constraint_count.numpy()[0])
        mf_count = int(solver.placement.mf_constraint_count.numpy()[0])
        samples.append(
            (
                constraint_count,
                solver.placement.diag.numpy()[0, :constraint_count].copy(),
                solver.placement.impulses.numpy()[0, :constraint_count].copy(),
                state_out.joint_q.numpy().copy(),
                state_out.joint_qd.numpy().copy(),
                int(solver.placement.local_solve_owner.numpy()[0]),
                solver.placement.row_type.numpy()[0, :constraint_count].copy(),
                mf_count,
                solver.placement.mf_impulses.numpy()[0, :mf_count].copy(),
                solver.placement.mf_row_type.numpy()[0, :mf_count].copy(),
                solver.placement.mf_row_parent.numpy()[0, :mf_count].copy(),
                solver.placement.mf_Z_a.numpy()[0, :mf_count].copy(),
                solver.placement.mf_Z_b.numpy()[0, :mf_count].copy(),
            )
        )
        state_in, state_out = state_out, state_in
    return solver, samples


def _assert_owner_parity(test, general, local):
    """Assert the local owner trajectory matches the general owner row for row."""
    test.assertEqual(len(general), len(local))
    test.assertGreater(general[0][0], 0, "mixed scene generated no dense constraint rows")
    for step, (expected, actual) in enumerate(zip(general, local, strict=True)):
        test.assertEqual(actual[0], expected[0], f"constraint count differed at step {step}")
        test.assertEqual(actual[7], expected[7], f"matrix-free row count differed at step {step}")
        np.testing.assert_array_equal(actual[6], expected[6], err_msg=f"row types differed at step {step}")
        np.testing.assert_array_equal(actual[9], expected[9], err_msg=f"matrix-free row types differed at step {step}")
        for label, expected_value, actual_value in zip(
            ("diagonal", "impulses", "joint_q", "joint_qd", "mf_impulses"),
            (*expected[1:5], expected[8]),
            (*actual[1:5], actual[8]),
            strict=True,
        ):
            np.testing.assert_allclose(
                actual_value,
                expected_value,
                rtol=5.0e-4,
                atol=2.0e-6,
                err_msg=f"{label} differed at step {step}",
            )


# Body pairs of the patch-unit fixture: (kind, diagonal coordinates). Scalar pairs touch one coordinate
# (two pairs share coordinate 2); the coupled pairs add the dense articulation or a second diagonal
# coordinate, which keeps coordinates 1, 3 and 4 in the serial chain; the free pair touches none.
_UNIT_PAIRS = (
    ("scalar", (0,)),
    ("scalar", (1,)),
    ("scalar", (2,)),
    ("scalar", (2,)),
    ("coupled", (1, 3)),
    ("dense", (4,)),
    ("dense", ()),
    ("scalar", (5,)),
    ("free", ()),
)


# Serial units whose results hinge on the unit tests in the solve's skips, in row order: (ring, kind,
# coordinates, rows, normal right-hand side, initial impulses). Every fourth unit heads a speculative batch
# of either width (four units of eight lanes, or two of sixteen). A loaded anchor comes first; then, at
# batch heads, an anchor whose normal and tangents are zero but whose ring mate is loaded, and an unloaded
# anchor that carries tangent impulses: neither is a no-op. Last, a one-row unit without load precedes a
# loaded unit, whose rows the inactive-friction skip must not pass over. Free units are separating no-ops.
_UNIT_EDGES = (
    (0, "coupled", (1, 3), 3, -2.0, (0.2, 0.0, 0.0)),
    *((1, "free", (), 1, 2.0, (0.0,)),) * 3,
    (0, "coupled", (1, 3), 3, 2.0, (0.0, 0.0, 0.0)),
    *((2, "free", (), 1, 2.0, (0.0,)),) * 3,
    (3, "dense", (4,), 3, 2.0, (0.0, 0.03, -0.02)),
    *((4, "free", (), 1, 2.0, (0.0,)),) * 3,
    (5, "free", (), 1, 2.0, (0.0,)),
    (6, "dense", (4,), 3, -2.0, (0.0, 0.0, 0.0)),
)


def _diagonal_walk(inverse, device, dofs=None, offsets=None):
    """The walk of each world's dense articulation whose mass matrix is diagonal, ``H = diag(1 / inverse[w])``, as the
    step's one leave walks it (:func:`.tree_factor.walk_leave_kernel`, :func:`_leave`): world ``w``'s articulation
    ``w`` at its world's ``offsets[w]`` of ``dofs`` coordinates (global coordinate ``w dofs + d``), a chain of one
    coordinate a body whose columns and lifts are zero, so ``WALK_DOWN`` leaves ``u_c / ell_c = u_c sqrt(m_c^-1)``.
    Returns its :class:`.tree_factor.WalkTrees`."""
    inverse = np.asarray(inverse, np.float32)
    worlds, n = inverse.shape
    dofs = n if dofs is None else int(dofs)
    offsets = np.zeros(worlds, np.int32) if offsets is None else np.asarray(offsets, np.int32)
    first = (np.arange(worlds) * dofs + offsets).astype(np.int32)
    walk = WalkTrees()
    data = np.zeros((worlds * dofs, WALK_COORDINATE), np.float32)
    for world in range(worlds):
        data[first[world] : first[world] + n, 12] = np.sqrt(inverse[world])
    walk.data = wp.array(data, device=device)
    walk.link = wp.zeros((worlds * n, WALK_LINK), dtype=wp.float32, device=device)
    path = np.stack([np.arange(n) - 1, np.arange(n)], axis=1).astype(np.int32)
    walk.path = wp.array(np.tile(path, (worlds, 1)), dtype=wp.vec2i, device=device)
    walk.scratch = wp.zeros(worlds * n, dtype=wp.spatial_vector, device=device)
    starts = (np.arange(worlds) * n).astype(np.int32)
    walk.joint_start = wp.array(starts, device=device)
    walk.joint_end = wp.array(starts + n, device=device)
    walk.dof_start = wp.array(first, device=device)
    walk.dofs = wp.full(worlds, n, dtype=wp.int32, device=device)
    release = WalkRelease()
    release.count = wp.zeros(worlds, dtype=wp.int32, device=device)
    release.dof = wp.zeros((1, 1), dtype=wp.int32, device=device)
    release.coordinate = wp.full((1, 1, 1), -1, dtype=wp.int32, device=device)
    release.b, release.a, release.g = (wp.zeros((1, 1, 1), dtype=wp.float32, device=device) for _ in range(3))
    walk.release = release
    walk.du = wp.zeros(worlds * dofs, dtype=wp.float32, device=device)
    return walk


def _leave(walk, velocity, device):
    """The step's one leave of :func:`_diagonal_walk`'s chains: the sweep's change in ``walk.du`` walked onto
    ``velocity``."""
    worlds, n = walk.dofs.shape[0], walk.dofs.numpy()[0]
    chain = WalkClass(tuple(range(-1, n - 1)), tuple(range(n + 1)))
    lanes = 4 if wp.get_device(device).is_cuda else 1
    wp.launch(
        walk_leave_kernel((chain,), False),
        dim=worlds * lanes,
        inputs=[
            wp.array(np.arange(worlds, dtype=np.int32), device=device),
            lanes,
            walk,
            wp.zeros(worlds, dtype=wp.int32, device=device),
            walk.du,
            walk.dof_start,
            velocity,
        ],
        device=device,
    )
    walk.du.zero_()


def _whiten_slots(sparse_dof, jacobian, inverse):
    """A sparse-diagonal row's slots whitened, ``Z_s = J_s sqrt(m_s^-1)`` where a slot holds a coordinate."""
    present = sparse_dof >= 0
    scale = np.sqrt(np.asarray(inverse, np.float32)[np.where(present, sparse_dof, 0)])
    return np.where(present, jacobian * scale, 0.0).astype(np.float32)


def _patch_unit_rows(seed, *, worlds=3, capacity=40, prefix=2, contacts=10, diagonal=8, dense=6, layout=None):
    """Build sparse-diagonal rows of patch contact units with linked normal rings.

    Each world holds ``prefix`` dense limit rows, then contacts in a random allocation order. A contact
    is a unit: its normal row, followed by two tangent rows when it owns an anchor (at most two per
    ring). The contacts of a body pair split into rings whose normal rows link circularly through their
    parents, as ``link_patch_routes`` leaves them. Coordinates ``[0, diagonal)`` belong to the diagonal
    articulation (index 0) and the following ``dense`` ones to the dense articulation (index 1).
    ``layout`` replaces every world's random contacts with given units, as in :data:`_UNIT_EDGES`.
    """
    rng = np.random.default_rng(seed)
    dofs = diagonal + dense
    rows = {
        "worlds": worlds,
        "capacity": capacity,
        "dofs": dofs,
        "dense": dense,
        "diagonal": diagonal,
        "cfm": 1.0e-6,
        "row_count": np.zeros(worlds, np.int32),
        "bounds": np.zeros((worlds, 2), np.int32),
        # Rows past each world's count hold a type no row family uses.
        "row_type": np.full((worlds, capacity), 7, np.int32),
        "row_parent": np.full((worlds, capacity), -1, np.int32),
        "row_mu": np.zeros((worlds, capacity), np.float32),
        "rhs": np.zeros((worlds, capacity), np.float32),
        "diag": np.zeros((worlds, capacity), np.float32),
        "impulses": np.zeros((worlds, capacity), np.float32),
        "dense_J": np.zeros((worlds, capacity, dense), np.float32),
        "dense_Y": np.zeros((worlds, capacity, dense), np.float32),
        "sparse_dof": np.full((worlds, capacity, 2), -1, np.int32),
        "sparse_jy": np.zeros((worlds, capacity, 4), np.float32),
        "velocity": rng.uniform(-0.3, 0.3, worlds * dofs).astype(np.float32),
        "inverse_mass": rng.uniform(0.5, 1.5, worlds * dofs).astype(np.float32),
        "limit_active": np.zeros((worlds, dofs), np.int32),
        "limit_lower_rhs": np.zeros((worlds, dofs), np.float32),
        "limit_upper_rhs": np.zeros((worlds, dofs), np.float32),
        "dense_offsets": np.full(worlds, diagonal, np.int32),
        "dense_groups": np.arange(worlds, dtype=np.int32),
        "rings": [],
    }
    contact = {name: [] for name in ("world", "slot", "art_a", "art_b", "slots_needed")}

    def add_unit(w, row, kind, coords, count):
        """Write a unit's rows and contact, drawing its Jacobians, coefficients and right-hand sides."""
        inverse = rows["inverse_mass"][w * dofs : (w + 1) * dofs]
        slots = rng.permutation(2)
        values = rng.uniform(0.3, 1.0, (count, 2)) * rng.choice((-1.0, 1.0), (count, 2))
        for k in range(count):
            index = row + k
            rows["row_type"][w, index] = PGS_CONSTRAINT_TYPE_CONTACT if k == 0 else PGS_CONSTRAINT_TYPE_FRICTION
            if k == 0:
                # Separating normals stay unloaded, so speculative batches find no-op units.
                rows["rhs"][w, index] = rng.uniform(-0.3, 0.0) if rng.random() < 0.5 else rng.uniform(0.0, 0.3)
            else:
                rows["row_parent"][w, index] = row
                rows["row_mu"][w, index] = rng.uniform(0.2, 0.9)
                rows["rhs"][w, index] = rng.uniform(-0.15, 0.15)
            if kind == "dense":
                rows["dense_J"][w, index] = rng.normal(0.0, 0.5, dense)
            for coord, slot, value in zip(coords, slots, values[k], strict=False):
                rows["sparse_dof"][w, index, slot] = coord
                rows["sparse_jy"][w, index, 2 * slot : 2 * slot + 2] = (value, value * inverse[coord])
        contact["world"].append(w)
        contact["slot"].append(row)
        contact["art_a"].append(-1 if kind == "free" else 0)
        contact["art_b"].append(1 if kind == "dense" else -1)
        contact["slots_needed"].append(count)

    for w in range(worlds):
        inverse = rows["inverse_mass"][w * dofs : (w + 1) * dofs]
        rows["row_type"][w, :prefix] = PGS_CONSTRAINT_TYPE_JOINT_LIMIT
        rows["dense_J"][w, :prefix] = rng.normal(0.0, 0.5, (prefix, dense))
        rows["rhs"][w, :prefix] = rng.uniform(-0.2, 0.1, prefix)
        rows["bounds"][w] = prefix
        row = prefix
        if layout is None:
            rings = []
            for kind, coords in _UNIT_PAIRS:
                members = np.arange(int(rng.integers(1, 5)))
                cuts = rng.choice(np.arange(1, len(members)), size=int(rng.integers(0, len(members))), replace=False)
                for ring in np.split(members, np.sort(cuts)):
                    anchors = rng.choice(len(ring), size=min(len(ring), int(rng.integers(0, 3))), replace=False)
                    rings.append([(kind, coords, 3 if k in anchors else 1) for k in range(len(ring))])
            units = [(index, unit) for index, ring in enumerate(rings) for unit in ring]
            heads = [[] for _ in rings]
            for position in rng.permutation(len(units))[:contacts]:
                ring_index, (kind, coords, count) = units[position]
                if row + count > capacity:
                    continue
                heads[ring_index].append(row)
                add_unit(w, row, kind, coords, count)
                if rng.random() < 0.4:
                    rows["impulses"][w, row] = rng.uniform(0.0, 0.2)
                    if count == 3 and rng.random() < 0.6:
                        rows["impulses"][w, row + 1 : row + 3] = rng.uniform(-0.02, 0.02, 2)
                row += count
        else:
            heads = [[] for _ in range(1 + max(unit[0] for unit in layout))]
            for ring_index, kind, coords, count, normal_rhs, impulses in layout:
                heads[ring_index].append(row)
                add_unit(w, row, kind, coords, count)
                rows["rhs"][w, row] = normal_rhs
                rows["impulses"][w, row : row + count] = impulses
                row += count
        for members in heads:
            linked = rng.permutation(members)
            for k, head in enumerate(linked):
                rows["row_parent"][w, head] = linked[(k + 1) % len(linked)]
            if members:
                rows["rings"].append((w, sorted(members)))
        rows["row_count"][w] = row
        rows["dense_Y"][w, :row] = rows["dense_J"][w, :row] * inverse[diagonal:]
        diag = np.sum(rows["dense_J"][w, :row] * rows["dense_Y"][w, :row], axis=1)
        for slot in range(2):
            present = rows["sparse_dof"][w, :row, slot] >= 0
            diag += np.where(
                present, rows["sparse_jy"][w, :row, 2 * slot] * rows["sparse_jy"][w, :row, 2 * slot + 1], 0
            )
        rows["diag"][w, :row] = diag + np.float32(rows["cfm"])
        limited = rng.choice(diagonal, size=3, replace=False)
        rows["limit_active"][w, limited] = rng.integers(1, 4, 3)
        rows["limit_lower_rhs"][w, limited] = rng.uniform(-0.1, 0.05, 3)
        rows["limit_upper_rhs"][w, limited] = rng.uniform(-0.1, 0.05, 3)
    # The sweep's whitened rows: the dense articulation's walk (its mass matrix is diagonal here) and Z = L^-1 P J^T,
    # and the slots' J / sqrt(m).
    inverse = rows["inverse_mass"].reshape(worlds, dofs)
    rows["dense_inverse"] = inverse[:, diagonal:].copy()
    rows["dense_Z"] = (rows["dense_J"] * np.sqrt(inverse[:, None, diagonal:])).astype(np.float32)
    # Each row's incident velocity J v over the fixture's velocity, and the limits' on their coordinates.
    velocity = rows["velocity"].reshape(worlds, dofs)
    incident = np.zeros((worlds, capacity))
    for w in range(worlds):
        offset = rows["dense_offsets"][w]
        incident[w] = rows["dense_J"][w] @ velocity[w, offset : offset + dense].astype(np.float64)
        for row, slot in np.argwhere(rows["sparse_dof"][w] >= 0):
            incident[w, row] += rows["sparse_jy"][w, row, 2 * slot] * velocity[w, rows["sparse_dof"][w, row, slot]]
    rows["row_incident"] = incident.astype(np.float32)
    rows["sparse_z"] = np.stack(
        [
            np.stack(
                [
                    _whiten_slots(rows["sparse_dof"][w, :, s], rows["sparse_jy"][w, :, 2 * s], inverse[w])
                    for s in (0, 1)
                ],
                axis=-1,
            )
            for w in range(worlds)
        ]
    )
    rows["contacts"] = {name: np.array(values, np.int32) for name, values in contact.items()}
    return rows


def _patch_unit_cases():
    """Fixtures of the CUDA unit twins, labeled ``(dense, seed, capacity)``: random layouts, then the edges.

    Six dense coordinates batch four units in eight lanes; nine, as a Franka arm, batch two in sixteen.
    """
    for dense in (6, 9):
        for seed, (capacity, contacts, prefix) in itertools.product(range(4), ((40, 10, 2), (32, 30, 0), (20, 40, 1))):
            rows = _patch_unit_rows(seed, capacity=capacity, contacts=contacts, prefix=prefix, dense=dense)
            yield (dense, seed, capacity), rows
        yield (dense, "edges", 32), _patch_unit_rows(0, capacity=32, dense=dense, layout=_UNIT_EDGES)


def _mark_patch_units(rows, device):
    """Run the unit marker over the fixture's contacts and return the marked sparse coordinates."""
    contacts = rows["contacts"]
    count = len(contacts["slot"])
    sparse_dof = wp.array(rows["sparse_dof"], dtype=wp.int32, device=device)
    marker = _get_mark_independent_sparse_contact_candidates_kernel(rows["diagonal"], str(device.arch), units=True)
    if count:
        wp.launch(
            marker,
            dim=count,
            inputs=[
                row_lines(device),
                field_group.upload(
                    ContactInput,
                    device,
                    count=wp.array((count,), dtype=wp.int32, device=device),
                    world=wp.array(contacts["world"], dtype=wp.int32, device=device),
                    slot=wp.array(contacts["slot"], dtype=wp.int32, device=device),
                    art0=wp.array(contacts["art_a"], dtype=wp.int32, device=device),
                    art1=wp.array(contacts["art_b"], dtype=wp.int32, device=device),
                    path=wp.zeros(count, dtype=wp.int32, device=device),
                    slots_needed=wp.array(contacts["slots_needed"], dtype=wp.int32, device=device),
                    response_dofs=wp.array((rows["diagonal"], rows["dense"]), dtype=wp.int32, device=device),
                    first=row_lines(device),
                ),
                count,
            ],
            outputs=[sparse_dof],
            device=device,
        )
    return sparse_dof.numpy()


def _unit_schedule_oracle(rows, marked):
    """Return the unit schedule the builder must produce from marked unit heads.

    Every row other than a tangent row heads a unit. A marked head joins its coordinate's chain unless a
    coupled unit uses that coordinate; chains link in ascending row order (``-(next + 3)``, ``-2`` at the
    end), group heads list each chain's first unit by coordinate, and the serial list holds the remaining
    heads in row order.
    """
    links = marked.copy()
    group_count = np.zeros(rows["worlds"], np.int32)
    group_heads = np.full((rows["worlds"], rows["dofs"]), -1, np.int32)
    serial = []
    for w in range(rows["worlds"]):
        start, end = int(rows["bounds"][w, 1]), int(rows["row_count"][w])
        heads = [row for row in range(start, end) if rows["row_type"][w, row] != PGS_CONSTRAINT_TYPE_FRICTION]
        candidates = [row for row in heads if np.any(marked[w, row] < -1)]
        reserved = {int(c) for row in heads if row not in candidates for c in marked[w, row] if c >= 0}
        chains = {}
        for row in candidates:
            coordinate, slot = int(np.max(marked[w, row])), int(np.argmin(marked[w, row]))
            if coordinate in reserved:
                links[w, row, slot] = -1
            else:
                chains.setdefault(coordinate, []).append((row, slot))
        for chain in chains.values():
            for (row, slot), following in zip(chain, [*chain[1:], None], strict=True):
                links[w, row, slot] = -2 if following is None else -(following[0] + 3)
        firsts = [chain[0][0] for _, chain in sorted(chains.items())]
        group_count[w] = len(firsts)
        group_heads[w, : len(firsts)] = firsts
        serial.append([row for row in heads if not np.any(links[w, row] < -1)])
    return {"sparse_row_dof": links, "group_count": group_count, "group_heads": group_heads, "serial": serial}


def _unit_rows(rows, world, head):
    """The rows of the unit ``head`` heads: its normal row, then the two tangent rows it owns, if any."""
    owns_tangents = (
        head + 2 < rows["row_count"][world]
        and rows["row_type"][world, head + 1] == PGS_CONSTRAINT_TYPE_FRICTION
        and rows["row_parent"][world, head + 1] == head
    )
    return [head, head + 1, head + 2] if owns_tangents else [head]


def _unit_schedule_order(rows, schedule, world):
    """Rows in the unit schedule's order: independent chains by coordinate, the prefix, then serial units."""
    order = []
    for first in schedule["group_heads"][world, : schedule["group_count"][world]]:
        head = int(first)
        while head >= 0:
            order += _unit_rows(rows, world, head)
            link = int(np.min(schedule["sparse_row_dof"][world, head]))
            head = -1 if link == -2 else -link - 3
    order += list(range(int(rows["bounds"][world, 1])))
    for head in schedule["serial"][world]:
        order += _unit_rows(rows, world, head)
    return order


def _ring_pooled_serial_reference(rows, *, iterations, schedule=None, omega=1.0):
    """Sweep the sparse-diagonal rows in float64 with the general owner's row law.

    Each iteration projects the fused diagonal limits, then visits rows in order (or in the unit
    schedule's order): a normal row clamps at zero, a unit's first tangent row solves both tangents on
    the friction disk of radius mu times the pooled normal load of its patch ring, and the second
    tangent row is a no-op.
    """
    dofs, dense, cfm = rows["dofs"], rows["dense"], rows["cfm"]
    velocity = rows["velocity"].astype(np.float64)
    impulses = rows["impulses"].astype(np.float64)
    lower = np.zeros((rows["worlds"], dofs))
    upper = np.zeros((rows["worlds"], dofs))
    for w in range(rows["worlds"]):
        v, lam = velocity[w * dofs : (w + 1) * dofs], impulses[w]
        count, offset = int(rows["row_count"][w]), int(rows["dense_offsets"][w])
        J, Y = np.zeros((count, dofs)), np.zeros((count, dofs))
        J[:, offset : offset + dense] = rows["dense_J"][w, :count]
        Y[:, offset : offset + dense] = rows["dense_Y"][w, :count]
        for row, slot in np.argwhere(rows["sparse_dof"][w, :count] >= 0):
            coord = rows["sparse_dof"][w, row, slot]
            J[row, coord] += rows["sparse_jy"][w, row, 2 * slot]
            Y[row, coord] += rows["sparse_jy"][w, row, 2 * slot + 1]
        inverse = rows["inverse_mass"][w * dofs : (w + 1) * dofs].astype(np.float64)
        parents = rows["row_parent"][w, :count]
        order = range(count) if schedule is None else _unit_schedule_order(rows, schedule, w)
        for _ in range(iterations):
            changed = False
            for coord in range(dofs):
                for bit, sign, limit_rhs, lambdas in (
                    (1, 1.0, "limit_lower_rhs", lower),
                    (2, -1.0, "limit_upper_rhs", upper),
                ):
                    if rows["limit_active"][w, coord] & bit:
                        old = lambdas[w, coord]
                        residual = sign * v[coord] + rows[limit_rhs][w, coord]
                        lambdas[w, coord] = max(0.0, old - omega * residual / (inverse[coord] + cfm))
                        v[coord] += sign * inverse[coord] * (lambdas[w, coord] - old)
                        changed |= lambdas[w, coord] != old
            for row in order:
                kind, parent, denominator = rows["row_type"][w, row], parents[row], rows["diag"][w, row]
                if kind == PGS_CONSTRAINT_TYPE_FRICTION and row != parent + 1:
                    continue
                if denominator <= 0.0 and kind != PGS_CONSTRAINT_TYPE_FRICTION:
                    continue
                residual = J[row] @ v + rows["rhs"][w, row]
                old = lam[row]
                new = old - omega * residual / denominator if denominator > 0.0 else old
                if kind != PGS_CONSTRAINT_TYPE_FRICTION:
                    new = max(new, 0.0)
                else:
                    sibling = parent + 2
                    load, member = lam[parent], parents[parent]
                    while member >= 0 and member != parent:
                        load, member = load + lam[member], parents[member]
                    radius = max(rows["row_mu"][w, row] * load, 0.0)
                    pair = (0.0, 0.0)
                    if radius > 0.0:
                        pair = friction_pair(
                            float(denominator),
                            float(J[row] @ Y[sibling]),
                            float(rows["diag"][w, sibling]),
                            wp.vec2(float(residual), float(J[sibling] @ v + rows["rhs"][w, sibling])),
                            wp.vec2(float(old), float(lam[sibling])),
                            float(radius),
                            float(omega),
                        )
                    magnitude = np.hypot(pair[0], pair[1])
                    scale = radius / magnitude if magnitude > radius else 1.0
                    new, sibling_new = pair[0] * scale, pair[1] * scale
                    changed |= sibling_new != lam[sibling]
                    v += Y[sibling] * (sibling_new - lam[sibling])
                    lam[sibling] = sibling_new
                changed |= new != old
                v += Y[row] * (new - old)
                lam[row] = new
            if not changed:
                break
    return {"impulses": impulses, "velocity": velocity, "lower": lower, "upper": upper}


def _paired_unit_rows(seed, *, worlds=6, capacity=64, primary=23, secondary=6):
    """Build paired factor-coordinate rows of patch contact units with linked normal rings.

    Each world holds a prefix of position-limit rows (one world adds a velocity-limit row), then
    contacts in a random allocation order. A contact is a unit: its normal row, followed by two
    tangent rows naming it as parent when it owns an anchor (at most two per ring). The normal rows
    of a ring link circularly through their parents, as ``link_patch_routes`` leaves them; a ring of
    one ends with -1 or links to itself. Factor rows, right-hand sides and impulses are random:
    normals separate or close, and some unloaded anchors start with tangent impulses, signed zeros
    included.
    """
    rng = np.random.default_rng(seed)
    dofs = primary + secondary
    rows = {
        "row_count": np.zeros(worlds, np.int32),
        "bounds": np.zeros((worlds, 2), np.int32),
        # Rows past each world's count hold a type no row family uses.
        "row_type": np.full((worlds, capacity), 7, np.int32),
        "row_parent": np.full((worlds, capacity), -1, np.int32),
        "row_mu": np.zeros((worlds, capacity), np.float32),
        "rhs": np.zeros((worlds, capacity), np.float32),
        "diag": np.zeros((worlds, capacity), np.float32),
        "impulses": np.zeros((worlds, capacity), np.float32),
        "factor_rows": np.zeros((worlds, capacity, dofs), np.float32),
        "anchors": 0,
        "loaded_rings": 0,
    }
    for w in range(worlds):
        prefix = int(rng.integers(0, 4))
        rows["row_type"][w, :prefix] = PGS_CONSTRAINT_TYPE_JOINT_LIMIT
        if w == 1:
            rows["row_type"][w, prefix] = PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT
            prefix += 1
        rows["rhs"][w, :prefix] = rng.uniform(-0.2, 0.1, prefix)
        rows["bounds"][w] = prefix
        rings = []
        for _ in range(int(rng.integers(3, 8))):
            members = int(rng.integers(1, 5))
            anchors = rng.choice(members, size=min(members, int(rng.integers(0, 3))), replace=False)
            rings.append([3 if k in anchors else 1 for k in range(members)])
        units = [(ring, count) for ring, counts in enumerate(rings) for count in counts]
        heads = [[] for _ in rings]
        row = prefix
        for position in rng.permutation(len(units)):
            ring, count = units[position]
            if row + count > capacity:
                continue
            heads[ring].append(row)
            rows["row_type"][w, row] = PGS_CONSTRAINT_TYPE_CONTACT
            closing = rng.random() < 0.5
            rows["rhs"][w, row] = rng.uniform(-0.3, 0.0) if closing else rng.uniform(0.0, 0.3)
            if rng.random() < 0.4:
                rows["impulses"][w, row] = rng.uniform(0.0, 0.2)
            if count == 3:
                rows["anchors"] += 1
                rows["row_type"][w, row + 1 : row + 3] = PGS_CONSTRAINT_TYPE_FRICTION
                rows["row_parent"][w, row + 1 : row + 3] = row
                rows["row_mu"][w, row + 1 : row + 3] = rng.uniform(0.2, 0.9)
                rows["rhs"][w, row + 1 : row + 3] = rng.uniform(-0.15, 0.15, 2)
                draw = rng.random()
                if draw < 0.3:
                    rows["impulses"][w, row + 1 : row + 3] = rng.uniform(-0.02, 0.02, 2)
                elif draw < 0.45:
                    rows["impulses"][w, row + 1 : row + 3] = -0.0
            row += count
        for members in heads:
            if len(members) == 1:
                rows["row_parent"][w, members[0]] = members[0] if rng.random() < 0.5 else -1
                continue
            linked = rng.permutation(members)
            for k, head in enumerate(linked):
                rows["row_parent"][w, head] = linked[(k + 1) % len(linked)]
            rows["loaded_rings"] += int(np.any(rows["impulses"][w, members] > 0.0))
        rows["row_count"][w] = row
        rows["factor_rows"][w, :row] = rng.normal(0.0, 0.3, (row, dofs))
        rows["diag"][w, :row] = np.sum(rows["factor_rows"][w, :row] ** 2, axis=1) + np.float32(0.01)
    # A normal row without response is skipped by both schedules.
    first_normal = int(np.flatnonzero(rows["row_type"][0] == PGS_CONSTRAINT_TYPE_CONTACT)[0])
    rows["diag"][0, first_normal] = 0.0
    return rows


def _launch_paired_factor_solve(kernel, rows, device, *, primary, components, iterations, omega):
    """Run one paired factor-coordinate sweep over the rows; return its impulses and velocities."""
    worlds, _capacity, dofs = rows["factor_rows"].shape
    secondary = dofs - primary
    rng = np.random.default_rng(99)

    def factor(size):
        lower = np.tril(rng.normal(0.0, 0.2, (worlds, size, size))) + 1.5 * np.eye(size)
        return lower.astype(np.float32)

    primary_L = factor(primary)
    secondary_L = factor(secondary)
    odd = np.arange(worlds) % 2 == 1
    selector = np.zeros(worlds, np.int32)
    if components:
        selector[2] = -1

    def array(value, dtype):
        return wp.array(np.ascontiguousarray(value), dtype=dtype, device=device)

    impulses = array(rows["impulses"], wp.float32)
    velocity = array(rng.normal(0.0, 0.3, worlds * dofs).astype(np.float32), wp.float32)
    wp.launch_tiled(
        kernel,
        dim=[(worlds + 1) // 2],
        inputs=[
            worlds,
            array(rows["row_count"], wp.int32),
            row_lines(device),
            array(rows["bounds"], wp.int32),
            array(np.arange(worlds * dofs, dtype=np.int32).reshape(worlds, dofs), wp.int32),
            array(rows["rhs"], wp.float32),
            array(rng.normal(0.0, 0.3, rows["rhs"].shape).astype(np.float32), wp.float32),
            array(rows["diag"], wp.float32),
            impulses,
            array(rows["factor_rows"], wp.float32),
            array(rows["row_type"], wp.int32),
            array(rows["row_parent"], wp.int32),
            array(rows["row_mu"], wp.float32),
            array(selector, wp.int32),
            array(primary_L, wp.float32),
            array(secondary_L, wp.float32),
            array(np.arange(worlds, dtype=np.int32), wp.int32),
            array(np.arange(worlds, dtype=np.int32), wp.int32),
            array(np.where(odd, secondary, 0).astype(np.int32), wp.int32),
            array(np.where(odd, 0, primary).astype(np.int32), wp.int32),
            iterations,
            omega,
            velocity,
        ],
        block_dim=64,
        device=device,
    )
    return impulses.numpy(), velocity.numpy()


class TestPairedFactorUnitSource(unittest.TestCase):
    def test_unit_walk_replaces_the_runtime_layout_test(self):
        """Walk units from each row's type and parent; the triple schedule tests its layout, the general none."""
        units = paired_factor_source(192, 29, 23, 6, contact_units=True, independent_components=True)[0]
        general = paired_factor_source(192, 29, 23, 6, independent_components=True)[0]
        triples = paired_factor_source(192, 29, 23, 6, contact_triples=True, independent_components=True)[0]
        # A unit owns tangent rows only when the next row is a tangent naming it as parent; three
        # normal-only units must not read as one triple, so the walk needs no stride-3 layout test.
        self.assertIn("world_row_parent.data[off + tangent1] == i", units)
        self.assertIn("world_row_mu.data[off + tangent1]", units)
        for text in ("triple_layout", "s_contact_mu", "% 3"):
            self.assertNotIn(text, units)
        self.assertNotIn("triple_layout", general)
        self.assertIn("const bool triple_layout = dense_phase_bounds.data[world * 2] == contact_start", triples)
        with self.assertRaises(ValueError):
            paired_factor_source(192, 29, 23, 6, contact_triples=True, contact_units=True)[0]
        factory = paired_factor_kernel.__wrapped__
        self.assertEqual(
            factory(192, 29, 23, 6, "120", False, True, contact_units=True).key,
            "sweep_paired_factor_192_29_23_6_units_components",
        )
        self.assertEqual(
            factory(192, 29, 23, 6, "120", True, True).key, "sweep_paired_factor_192_29_23_6_contact3_components"
        )


class TestFeatherPGSResponseDiagonal(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "paired factor sweeps require CUDA")
    def test_paired_factor_units_match_the_general_sweep(self):
        """Walk patch units on the unit's Gram block: the general sweep's impulses and velocity to rounding.

        A loaded unit reduces its three rows' velocity terms and the pair's couplings with the normal in one ladder
        and moves the pair's residuals by the normal's change (``Z_t^T (u + Z_n dn) = Z_t^T u + G_tn dn``): in exact
        arithmetic the general sweep's row order, so the two differ by rounding only. An unloaded unit visits its
        rows with the general sweep's reductions, so a sweep from zero impulses, which reaches every unit unloaded,
        reproduces the general sweep's bits.
        """
        device = wp.get_device("cuda:0")
        for (capacity, primary, secondary), components, (iterations, omega) in itertools.product(
            ((64, 23, 6), (48, 10, 6)), (False, True), ((8, 1.0), (8, 0.9), (5, 1.0))
        ):
            rows = _paired_unit_rows(capacity + primary, capacity=capacity, primary=primary, secondary=secondary)
            self.assertGreater(rows["anchors"], 4)
            self.assertGreater(rows["loaded_rings"], 0)
            dofs = primary + secondary
            kernels = [
                paired_factor_kernel(
                    capacity, dofs, primary, secondary, str(device.arch), independent_components=components, **kwargs
                )
                for kwargs in ({}, {"contact_units": True})
            ]
            general, units = (
                _launch_paired_factor_solve(
                    kernel,
                    rows,
                    device,
                    primary=primary,
                    components=components,
                    iterations=iterations,
                    omega=omega,
                )
                for kernel in kernels
            )
            with self.subTest(capacity=capacity, components=components, iterations=iterations, omega=omega):
                self.assertFalse(np.array_equal(general[0], rows["impulses"]), "the sweep changed no impulse")
                tangents = rows["row_type"] == PGS_CONSTRAINT_TYPE_FRICTION
                self.assertTrue(np.any(general[0][tangents] != 0.0), "no tangent row was loaded")
                for unit_values, general_values in zip(units, general, strict=True):
                    scale = float(np.max(np.abs(general_values)))
                    gap = float(np.max(np.abs(unit_values.astype(np.float64) - general_values)))
                    self.assertLessEqual(gap, 1.0e-5 * scale, f"units differ from the general sweep by {gap:.3g}")
            cold = dict(rows, impulses=np.zeros_like(rows["impulses"]))
            general, units = (
                _launch_paired_factor_solve(
                    kernel,
                    cold,
                    device,
                    primary=primary,
                    components=components,
                    iterations=1,
                    omega=omega,
                )
                for kernel in kernels
            )
            with self.subTest("cold", capacity=capacity, components=components, omega=omega):
                tangents = cold["row_type"] == PGS_CONSTRAINT_TYPE_FRICTION
                self.assertTrue(np.any(general[0][tangents] != 0.0), "no tangent row was loaded")
                for unit_values, general_values in zip(units, general, strict=True):
                    np.testing.assert_array_equal(unit_values.view(np.uint32), general_values.view(np.uint32))

    @unittest.skipUnless(wp.is_cuda_available(), "sparse diagonal GS requires CUDA")
    def test_sparse_diagonal_gs_matches_scalar_reference(self):
        """Match a scalar PGS reference with coupled limits and friction."""
        device = wp.get_device("cuda:0")
        max_constraints, world_dofs, dense_dofs = 8, 4, 2
        cfm, omega, iterations = 1.0e-6, 1.0, 8
        inverse_mass = np.array((0.5, 0.25, 0.75, 1.0), dtype=np.float32)
        initial_velocity = np.array((-0.4, 0.3, -0.2, 0.1), dtype=np.float32)
        jacobian = np.array(
            ((0.6, -0.4, 0.2, 0.0), (0.1, 0.5, -0.3, 0.4), (-0.2, 0.25, 0.1, -0.35)),
            dtype=np.float32,
        )
        response = jacobian * inverse_mass
        rhs = np.zeros((1, max_constraints), dtype=np.float32)
        rhs[0, :3] = (-0.1, 0.0, 0.0)
        diag = np.zeros_like(rhs)
        diag[0, :3] = np.sum(jacobian * response, axis=1) + cfm
        row_type = np.zeros((1, max_constraints), dtype=np.int32)
        row_type[0, :3] = (PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION, PGS_CONSTRAINT_TYPE_FRICTION)
        row_parent = np.full((1, max_constraints), -1, dtype=np.int32)
        row_parent[0, 1:3] = 0
        row_mu = np.zeros((1, max_constraints), dtype=np.float32)
        row_mu[0, 1:3] = 0.7
        dense_j = np.zeros((1, max_constraints, dense_dofs), dtype=np.float32)
        dense_j[0, :3] = jacobian[:, :dense_dofs]
        sparse_dof = np.full((1, max_constraints, 2), -1, dtype=np.int32)
        sparse_jy = np.zeros((1, max_constraints, 4), dtype=np.float32)
        sparse_dof[0, :3] = (2, 3)
        sparse_jy[0, :3, 0] = jacobian[:, 2]
        sparse_jy[0, :3, 1] = response[:, 2]
        sparse_jy[0, :3, 2] = jacobian[:, 3]
        sparse_jy[0, :3, 3] = response[:, 3]
        limit_active = np.zeros((1, world_dofs), dtype=np.int32)
        limit_active[0, 2] = 1
        limit_lower_rhs = np.zeros((1, world_dofs), dtype=np.float32)
        limit_lower_rhs[0, 2] = -0.15

        expected_velocity = initial_velocity.copy()
        expected_impulses = np.zeros(3, dtype=np.float32)
        expected_limit_lambda = np.float32(0.0)
        for _ in range(iterations):
            old_limit = expected_limit_lambda
            residual = expected_velocity[2] + limit_lower_rhs[0, 2]
            expected_limit_lambda = np.maximum(0.0, old_limit - omega * residual / (inverse_mass[2] + cfm))
            expected_velocity[2] += inverse_mass[2] * (expected_limit_lambda - old_limit)
            old_impulse = expected_impulses[0]
            new_impulse = old_impulse - omega * (jacobian[0] @ expected_velocity + rhs[0, 0]) / diag[0, 0]
            new_impulse = max(new_impulse, 0.0)
            expected_impulses[0] = new_impulse
            expected_velocity += response[0] * (new_impulse - old_impulse)
            # The general owner solves both tangent rows together on the friction disk at the current normal
            # load (projection_laws.friction_pair); the second tangent visit is a no-op.
            radius = max(row_mu[0, 1] * expected_impulses[0], 0.0)
            pair = friction_pair(
                float(diag[0, 1]),
                float(jacobian[1] @ response[2]),
                float(diag[0, 2]),
                wp.vec2(
                    float(jacobian[1] @ expected_velocity + rhs[0, 1]),
                    float(jacobian[2] @ expected_velocity + rhs[0, 2]),
                ),
                wp.vec2(float(expected_impulses[1]), float(expected_impulses[2])),
                float(radius),
                omega,
            )
            magnitude = np.sqrt(pair[0] * pair[0] + pair[1] * pair[1])
            scale = radius / magnitude if magnitude > radius else 1.0
            for tangent, value in ((1, pair[0] * scale), (2, pair[1] * scale)):
                expected_velocity += response[tangent] * (value - expected_impulses[tangent])
                expected_impulses[tangent] = np.float32(value)

        impulses = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
        lower_lambda = wp.zeros((1, world_dofs), dtype=wp.float32, device=device)
        upper_lambda = wp.zeros_like(lower_lambda)
        velocity = wp.array(initial_velocity, dtype=wp.float32, device=device)
        kernel = sparse_diagonal_kernel(max_constraints, world_dofs, dense_dofs, str(device.arch), contact_triples=True)
        walk = _diagonal_walk(inverse_mass[None, :dense_dofs], device, world_dofs)
        wp.launch_tiled(
            kernel,
            dim=[1],
            inputs=[
                wp.array((3,), dtype=wp.int32, device=device),
                row_lines(device),
                wp.array(np.arange(world_dofs, dtype=np.int32)[None, :], device=device),
                wp.array(rhs, device=device),
                wp.array(np.pad(row_velocity(jacobian[None], initial_velocity), ((0, 0), (0, 5))), device=device),
                wp.array(diag, device=device),
                impulses,
                wp.array(row_type, device=device),
                wp.array(row_parent, device=device),
                wp.array(row_mu, device=device),
                wp.zeros((1, 2), dtype=wp.int32, device=device),
                wp.zeros(1, dtype=wp.int32, device=device),
                wp.empty((1, world_dofs), dtype=wp.int32, device=device),
                wp.zeros(1, dtype=wp.int32, device=device),
                wp.empty((1, (max_constraints + 2) // 3), dtype=wp.int32, device=device),
                wp.array((0,), dtype=wp.int32, device=device),
                wp.array((0,), dtype=wp.int32, device=device),
                walk,
                wp.array(dense_j * np.sqrt(inverse_mass[:dense_dofs]), device=device),
                wp.array(sparse_dof, device=device),
                wp.array(
                    np.stack(
                        [_whiten_slots(sparse_dof[..., s], sparse_jy[..., 2 * s], inverse_mass) for s in (0, 1)], -1
                    ),
                    device=device,
                ),
                wp.array(limit_active, device=device),
                wp.array(limit_lower_rhs + initial_velocity[None], device=device),
                wp.zeros((1, world_dofs), dtype=wp.float32, device=device),
                wp.array(inverse_mass, device=device),
                cfm,
                iterations,
                omega,
            ],
            outputs=[lower_lambda, upper_lambda, velocity],
            block_dim=32,
            device=device,
        )
        _leave(walk, velocity, device)

        np.testing.assert_allclose(velocity.numpy(), expected_velocity, rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_allclose(impulses.numpy()[0, :3], expected_impulses, rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_allclose(lower_lambda.numpy()[0, 2], expected_limit_lambda, rtol=2.0e-5, atol=2.0e-6)
        self.assertEqual(float(upper_lambda.numpy()[0, 2]), 0.0)

    @unittest.skipUnless(wp.is_cuda_available(), "speculative contact batches require CUDA")
    def test_speculative_sparse_contact_batches_match_serial_reference(self):
        """Skip exact no-op prefixes without changing later serial contact updates."""
        device = wp.get_device("cuda:0")
        max_constraints, world_dofs, dense_dofs = 12, 8, 6
        cfm, omega, iterations = 1.0e-6, 1.0, 2
        inverse_mass = np.array((0.5, 0.25, 0.75, 1.0, 0.4, 0.6, 0.75, 0.8), dtype=np.float32)
        initial_velocity = np.array((0.3, -0.2, 0.1, 0.4, -0.1, 0.2, 0.5, 0.25), dtype=np.float32)
        jacobian = np.zeros((max_constraints, world_dofs), dtype=np.float32)
        jacobian[0, 6] = 1.0
        jacobian[3, 7] = 1.0
        jacobian[6, (0, 6)] = (-0.2, -1.0)
        jacobian[9, (0, 6)] = (0.3, 1.0)
        response = jacobian * inverse_mass
        rhs = np.zeros((1, max_constraints), dtype=np.float32)
        rhs[0, 6] = -0.1
        rhs[0, 9] = -0.2
        diag = np.zeros_like(rhs)
        normal_rows = np.array((0, 3, 6, 9), dtype=np.int32)
        diag[0, normal_rows] = np.sum(jacobian[normal_rows] * response[normal_rows], axis=1) + cfm
        row_type = np.full((1, max_constraints), PGS_CONSTRAINT_TYPE_FRICTION, dtype=np.int32)
        row_type[0, normal_rows] = PGS_CONSTRAINT_TYPE_CONTACT
        row_parent = np.full((1, max_constraints), -1, dtype=np.int32)
        for normal in normal_rows:
            row_parent[0, normal + 1 : normal + 3] = normal
        row_mu = np.zeros((1, max_constraints), dtype=np.float32)
        dense_j = jacobian[:, :dense_dofs][None, ...]
        sparse_dof = np.empty((1, max_constraints, 2), dtype=np.int32)
        sparse_dof[:, :, 0] = 6
        sparse_dof[:, :, 1] = 7
        sparse_jy = np.zeros((1, max_constraints, 4), dtype=np.float32)
        sparse_jy[0, :, 0] = jacobian[:, 6]
        sparse_jy[0, :, 1] = response[:, 6]
        sparse_jy[0, :, 2] = jacobian[:, 7]
        sparse_jy[0, :, 3] = response[:, 7]

        expected_velocity = initial_velocity.copy()
        expected_impulses = np.zeros(max_constraints, dtype=np.float32)
        for _ in range(iterations):
            for row in normal_rows:
                old_impulse = expected_impulses[row]
                residual = jacobian[row] @ expected_velocity + rhs[0, row]
                new_impulse = max(old_impulse - omega * residual / diag[0, row], 0.0)
                expected_impulses[row] = new_impulse
                expected_velocity += response[row] * (new_impulse - old_impulse)

        impulses = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
        velocity = wp.array(initial_velocity, dtype=wp.float32, device=device)
        kernel = sparse_diagonal_kernel(
            max_constraints,
            world_dofs,
            dense_dofs,
            str(device.arch),
            contact_triples=True,
            speculative_contact_batches=True,
        )
        walk = _diagonal_walk(inverse_mass[None, :dense_dofs], device, world_dofs)
        wp.launch_tiled(
            kernel,
            dim=[1],
            inputs=[
                wp.array((max_constraints,), dtype=wp.int32, device=device),
                row_lines(device),
                wp.array(np.arange(world_dofs, dtype=np.int32)[None, :], device=device),
                wp.array(rhs, device=device),
                wp.array(row_velocity(jacobian[None], initial_velocity), device=device),
                wp.array(diag, device=device),
                impulses,
                wp.array(row_type, device=device),
                wp.array(row_parent, device=device),
                wp.array(row_mu, device=device),
                wp.zeros((1, 2), dtype=wp.int32, device=device),
                wp.zeros(1, dtype=wp.int32, device=device),
                wp.empty((1, world_dofs), dtype=wp.int32, device=device),
                wp.array((len(normal_rows),), dtype=wp.int32, device=device),
                wp.array(normal_rows[None, :], device=device),
                wp.array((0,), dtype=wp.int32, device=device),
                wp.array((0,), dtype=wp.int32, device=device),
                walk,
                wp.array(dense_j * np.sqrt(inverse_mass[:dense_dofs]), device=device),
                wp.array(sparse_dof, device=device),
                wp.array(
                    np.stack(
                        [_whiten_slots(sparse_dof[..., s], sparse_jy[..., 2 * s], inverse_mass) for s in (0, 1)], -1
                    ),
                    device=device,
                ),
                wp.zeros((1, world_dofs), dtype=wp.int32, device=device),
                wp.zeros((1, world_dofs), dtype=wp.float32, device=device),
                wp.zeros((1, world_dofs), dtype=wp.float32, device=device),
                wp.array(inverse_mass, device=device),
                cfm,
                iterations,
                omega,
            ],
            outputs=[
                wp.zeros((1, world_dofs), dtype=wp.float32, device=device),
                wp.zeros((1, world_dofs), dtype=wp.float32, device=device),
                velocity,
            ],
            block_dim=32,
            device=device,
        )
        _leave(walk, velocity, device)

        np.testing.assert_allclose(velocity.numpy(), expected_velocity, rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_allclose(impulses.numpy()[0], expected_impulses, rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_array_equal(impulses.numpy()[0, :6], np.zeros(6, dtype=np.float32))

    @unittest.skipUnless(wp.is_cuda_available(), "independent contact groups require CUDA")
    def test_independent_sparse_contact_groups_exclude_coupled_coordinates(self):
        """Keep scalar contacts serial when a coupled contact shares their coordinate."""
        device = wp.get_device("cuda:0")
        max_constraints, max_world_dofs, contact_count = 12, 4, 4
        sparse_response_dofs = 108
        sparse_row_dof_np = np.full((1, max_constraints, 2), -1, dtype=np.int32)
        sparse_row_dof_np[0, 0:3, 0] = 2
        sparse_row_dof_np[0, 3:6, 0] = 3
        sparse_row_dof_np[0, 6:9, 0] = 3
        sparse_row_dof_np[0, 9:12, 0] = 2
        sparse_row_dof = wp.array(sparse_row_dof_np, device=device)
        group_count = wp.zeros(1, dtype=wp.int32, device=device)
        group_heads = wp.empty((1, max_world_dofs), dtype=wp.int32, device=device)
        serial_count = wp.zeros(1, dtype=wp.int32, device=device)
        serial_normals = wp.empty((1, (max_constraints + 2) // 3), dtype=wp.int32, device=device)

        marker = _get_mark_independent_sparse_contact_candidates_kernel(sparse_response_dofs, str(device.arch))
        wp.launch(
            marker,
            dim=contact_count,
            inputs=[
                row_lines(device),
                field_group.upload(
                    ContactInput,
                    device,
                    count=wp.array((contact_count,), dtype=wp.int32, device=device),
                    world=wp.zeros(contact_count, dtype=wp.int32, device=device),
                    slot=wp.array((0, 3, 6, 9), dtype=wp.int32, device=device),
                    art0=wp.zeros(contact_count, dtype=wp.int32, device=device),
                    art1=wp.array((-1, 1, -1, -1), dtype=wp.int32, device=device),
                    path=wp.zeros(contact_count, dtype=wp.int32, device=device),
                    slots_needed=wp.full(contact_count, 3, dtype=wp.int32, device=device),
                    response_dofs=wp.array((sparse_response_dofs, 6), dtype=wp.int32, device=device),
                    first=row_lines(device),
                ),
                contact_count,
            ],
            outputs=[sparse_row_dof],
            device=device,
        )
        schedule = _get_build_independent_sparse_contact_groups_kernel(
            max_constraints, max_world_dofs, str(device.arch), build_serial_contacts=True
        )
        wp.launch(
            schedule,
            dim=32,
            inputs=[
                wp.array((max_constraints,), dtype=wp.int32, device=device),
                row_lines(device),
                wp.zeros((1, 2), dtype=wp.int32, device=device),
            ],
            outputs=[sparse_row_dof, group_count, group_heads, serial_count, serial_normals],
            device=device,
        )
        wp.synchronize_device(device)

        self.assertEqual(int(group_count.numpy()[0]), 1)
        self.assertEqual(int(group_heads.numpy()[0, 0]), 0)
        self.assertEqual(int(serial_count.numpy()[0]), 2)
        np.testing.assert_array_equal(serial_normals.numpy()[0, :2], np.array((3, 6), dtype=np.int32))
        normal_links = sparse_row_dof.numpy()[0, ::3, 1]
        np.testing.assert_array_equal(normal_links, np.array((-12, -1, -1, -2), dtype=np.int32))

    def test_unit_marker_admits_one_and_three_row_units(self):
        """Mark patch units of one or three rows confined to one diagonal coordinate, as triples are marked."""
        device = wp.get_device("cpu")
        # Contacts: a 3-row and a 1-row scalar unit, a scalar contact of another route, an unallocated
        # one, one with a stale slot count, a dense-coupled unit and a two-coordinate unit.
        slots = np.array((0, 3, 4, -1, 5, 6, 7), dtype=np.int32)
        sparse_dof = np.full((1, 12, 2), -1, dtype=np.int32)
        sparse_dof[0, (0, 4), 0] = 2
        sparse_dof[0, 3, 1] = 5
        sparse_dof[0, 5, 0] = 1
        sparse_dof[0, 6, 1] = 4
        sparse_dof[0, 7] = (2, 3)
        launches = {}
        for units in (False, True):
            marked = wp.array(sparse_dof, device=device)
            marker = _get_mark_independent_sparse_contact_candidates_kernel(108, "cpu", units=units)
            wp.launch(
                marker,
                dim=3,
                inputs=[
                    row_lines(device),
                    field_group.upload(
                        ContactInput,
                        device,
                        count=wp.array((7,), dtype=wp.int32, device=device),
                        world=wp.zeros(7, dtype=wp.int32, device=device),
                        slot=wp.array(slots, device=device),
                        art0=wp.zeros(7, dtype=wp.int32, device=device),
                        art1=wp.array((-1, -1, -1, -1, -1, 1, -1), dtype=wp.int32, device=device),
                        path=wp.array((0, 0, 1, 0, 0, 0, 0), dtype=wp.int32, device=device),
                        slots_needed=wp.array((3, 1, 3, 3, 2, 3, 1), dtype=wp.int32, device=device),
                        response_dofs=wp.array((108, 6), dtype=wp.int32, device=device),
                        first=row_lines(device),
                    ),
                    3,
                ],
                outputs=[marked],
                device=device,
            )
            launches[units] = marked.numpy()[0]
            self.assertEqual(marker.key.endswith("_units"), units)
        expected = sparse_dof[0].copy()
        expected[0, 1] = -2
        # Triples skip the 1-row unit; units mark it in its empty slot.
        np.testing.assert_array_equal(launches[False], expected)
        expected[3, 0] = -2
        np.testing.assert_array_equal(launches[True], expected)

    def test_unit_schedule_factories_name_and_reject_layouts(self):
        """Name the unit variants and keep speculative batches within sixteen response lanes."""
        solve = sparse_diagonal_kernel(24, 12, 9, "120", contact_units=True)
        batched = sparse_diagonal_kernel(24, 12, 9, "120", contact_units=True, speculative_contact_batches=True)
        builder = _get_build_independent_sparse_contact_groups_kernel(
            24, 12, "120", build_serial_contacts=True, units=True
        )
        self.assertEqual(solve.key, "sweep_sparse_diagonal_24_12_9_units")
        self.assertEqual(batched.key, "sweep_sparse_diagonal_24_12_9_speculative_batches_units")
        self.assertEqual(builder.key, "build_independent_sparse_contact_groups_24_12_serial_contacts_units")
        for kwargs in (
            {"contact_triples": True, "contact_units": True},
            {"contact_units": True, "speculative_contact_batches": True, "dense_dofs": 15},
        ):
            dense_dofs = kwargs.pop("dense_dofs", 9)
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                sparse_diagonal_kernel(24, dense_dofs + 3, dense_dofs, "120", **kwargs)

    def test_unit_schedule_reorders_only_commuting_rows(self):
        """Keep GS order on every coordinate when sweeping in the unit schedule's order.

        The schedule oracle is the reference the CUDA unit builder must reproduce. Sweeping the
        rows in its order (independent chains first, then the prefix and the serial units) must give
        exactly the impulses and velocities of the original row order, including on independent patch
        rings that mix one- and three-row units. The members of a fixture ring share one coordinate, so
        the oracle cannot split a ring's independence marks; the CUDA builder twin checks that property
        on the kernel.
        """
        device = wp.get_device("cpu")
        independent = serial = mixed_rings = 0
        for seed, (capacity, contacts, prefix) in itertools.product(
            range(6), ((40, 10, 2), (64, 24, 2), (24, 12, 2), (32, 30, 0), (20, 40, 1))
        ):
            with self.subTest(seed=seed, capacity=capacity):
                rows = _patch_unit_rows(seed, capacity=capacity, contacts=contacts, prefix=prefix)
                schedule = _unit_schedule_oracle(rows, _mark_patch_units(rows, device))
                for world, members in rows["rings"]:
                    lengths = {len(_unit_rows(rows, world, head)) for head in members}
                    if lengths == {1, 3} and np.any(schedule["sparse_row_dof"][world, members] < -1):
                        mixed_rings += 1
                independent += sum(int(count) for count in schedule["group_count"])
                serial += sum(len(heads) for heads in schedule["serial"])
                for iterations in (6, 5):
                    original = _ring_pooled_serial_reference(rows, iterations=iterations)
                    scheduled = _ring_pooled_serial_reference(rows, iterations=iterations, schedule=schedule)
                    for name, values in original.items():
                        np.testing.assert_array_equal(scheduled[name], values, err_msg=name)
        # The fixtures exercise both phases, and independent rings whose loads cross unit lengths.
        self.assertGreater(independent, 50)
        self.assertGreater(serial, 50)
        self.assertGreater(mixed_rings, 20)

    @unittest.skipUnless(wp.is_cuda_available(), "independent contact units require CUDA")
    def test_independent_sparse_contact_units_keep_rings_together(self):
        """Schedule mixed one- and three-row units; every ring member carries the same independence mark."""
        device = wp.get_device("cuda:0")
        for (dense, seed, capacity), rows in _patch_unit_cases():
            with self.subTest(dense=dense, seed=seed, capacity=capacity):
                marked = _mark_patch_units(rows, device)
                expected = _unit_schedule_oracle(rows, marked)
                worlds, dofs = rows["worlds"], rows["dofs"]
                for serial_contacts in (False, True):
                    sparse_row_dof = wp.array(marked, device=device)
                    group_count = wp.full(worlds, -7, dtype=wp.int32, device=device)
                    group_heads = wp.full((worlds, dofs), -7, dtype=wp.int32, device=device)
                    serial_count = wp.full(worlds, -7, dtype=wp.int32, device=device)
                    serial_heads = wp.full((worlds, capacity), -7, dtype=wp.int32, device=device)
                    schedule = _get_build_independent_sparse_contact_groups_kernel(
                        capacity, dofs, str(device.arch), build_serial_contacts=serial_contacts, units=True
                    )
                    wp.launch(
                        schedule,
                        dim=worlds * 32,
                        inputs=[
                            wp.array(rows["row_count"], device=device),
                            row_lines(device),
                            wp.array(rows["bounds"], device=device),
                            wp.array(rows["row_type"], device=device),
                            wp.array(rows["row_parent"], device=device),
                        ],
                        outputs=[sparse_row_dof, group_count, group_heads, serial_count, serial_heads],
                        device=device,
                    )
                    np.testing.assert_array_equal(sparse_row_dof.numpy(), expected["sparse_row_dof"])
                    np.testing.assert_array_equal(group_count.numpy(), expected["group_count"])
                    for world in range(worlds):
                        count = int(expected["group_count"][world])
                        np.testing.assert_array_equal(
                            group_heads.numpy()[world, :count], expected["group_heads"][world, :count]
                        )
                        if serial_contacts:
                            heads = expected["serial"][world]
                            self.assertEqual(int(serial_count.numpy()[world]), len(heads))
                            np.testing.assert_array_equal(serial_heads.numpy()[world, : len(heads)], heads)
                    for world, members in rows["rings"]:
                        marks = {bool(np.any(sparse_row_dof.numpy()[world, member] < -1)) for member in members}
                        self.assertEqual(len(marks), 1, f"ring {members} of world {world} split its marks")

    @unittest.skipUnless(wp.is_cuda_available(), "speculative contact units require CUDA")
    def test_speculative_sparse_contact_units_match_serial_loop(self):
        """Solve mixed one- and three-row units on linked rings bitwise like the serial loop.

        The edge layout makes the ring load, the carried tangents and the unit length decide the
        speculative batches' no-op test and the inactive-friction skip.
        """
        device = wp.get_device("cuda:0")
        for (dense, seed, capacity), rows in _patch_unit_cases():
            worlds, dofs = rows["worlds"], rows["dofs"]
            marked = _mark_patch_units(rows, device)
            schedule = _unit_schedule_oracle(rows, marked)
            serial_heads = np.zeros((worlds, capacity), np.int32)
            for world, heads in enumerate(schedule["serial"]):
                serial_heads[world, : len(heads)] = heads
            for iterations in (8, 5):
                results = {}
                for label, units, batches in (
                    ("serial", False, False),
                    ("units", True, False),
                    ("batches", True, True),
                ):
                    impulses = wp.array(rows["impulses"], device=device)
                    lower = wp.zeros((worlds, dofs), dtype=wp.float32, device=device)
                    upper = wp.zeros((worlds, dofs), dtype=wp.float32, device=device)
                    velocity = wp.array(rows["velocity"], device=device)
                    kernel = sparse_diagonal_kernel(
                        capacity,
                        dofs,
                        dense,
                        str(device.arch),
                        contact_units=units,
                        speculative_contact_batches=batches,
                    )
                    walk = _diagonal_walk(rows["dense_inverse"], device, dofs, rows["dense_offsets"])
                    wp.launch_tiled(
                        kernel,
                        dim=[worlds],
                        inputs=[
                            wp.array(rows["row_count"], device=device),
                            row_lines(device),
                            wp.array(np.arange(worlds * dofs, dtype=np.int32).reshape(worlds, dofs), device=device),
                            wp.array(rows["rhs"], device=device),
                            wp.array(rows["row_incident"], device=device),
                            wp.array(rows["diag"], device=device),
                            impulses,
                            wp.array(rows["row_type"], device=device),
                            wp.array(rows["row_parent"], device=device),
                            wp.array(rows["row_mu"], device=device),
                            wp.array(rows["bounds"], device=device),
                            wp.array(schedule["group_count"], device=device),
                            wp.array(schedule["group_heads"], device=device),
                            wp.array([len(heads) for heads in schedule["serial"]], dtype=wp.int32, device=device),
                            wp.array(serial_heads, device=device),
                            wp.array(rows["dense_offsets"], device=device),
                            wp.array(rows["dense_groups"], device=device),
                            walk,
                            wp.array(rows["dense_Z"], device=device),
                            wp.array(schedule["sparse_row_dof"], device=device),
                            wp.array(rows["sparse_z"], device=device),
                            wp.array(rows["limit_active"], device=device),
                            wp.array(rows["limit_lower_rhs"] + rows["velocity"].reshape(worlds, dofs), device=device),
                            wp.array(rows["limit_upper_rhs"] - rows["velocity"].reshape(worlds, dofs), device=device),
                            wp.array(rows["inverse_mass"], device=device),
                            rows["cfm"],
                            iterations,
                            1.0,
                        ],
                        outputs=[lower, upper, velocity],
                        block_dim=32,
                        device=device,
                    )
                    _leave(walk, velocity, device)
                    results[label] = {
                        "impulses": impulses.numpy(),
                        "velocity": velocity.numpy(),
                        "lower": lower.numpy(),
                        "upper": upper.numpy(),
                    }
                reference = _ring_pooled_serial_reference(rows, iterations=iterations)
                for label in ("units", "batches"):
                    for name, values in results["serial"].items():
                        with self.subTest(
                            dense=dense, seed=seed, capacity=capacity, schedule=label, iterations=iterations, value=name
                        ):
                            np.testing.assert_array_equal(results[label][name], values)
                            np.testing.assert_allclose(results[label][name], reference[name], rtol=1.0e-4, atol=2.0e-5)

    def test_mixed_world_diagonal_map_includes_free_rigid_response(self):
        """Include free rigid articulations in every forced-tiled diagonal group."""
        solver = SolverFeatherPGS(_build_mixed_response_model("cpu", world_count=2), dense_max_constraints=32)
        self.assertEqual(set(solver.forest.size_groups), {6, 13})

        response_dof_count = solver.forest.response_dof_count
        articulation_world = solver.forest.articulation_world
        free_mask = solver.forest.is_free_rigid_host != 0
        for size in solver.forest.size_groups:
            size_arts = np.flatnonzero(response_dof_count == size)
            response_order = np.asarray(
                sorted(size_arts, key=lambda art: (int(articulation_world[art]), int(art))), dtype=np.int32
            )
            response_counts = np.bincount(articulation_world[response_order], minlength=solver.forest.world_count)
            response_starts = np.pad(np.cumsum(response_counts), (1, 0)).astype(np.int32)

            np.testing.assert_array_equal(solver.forest.world_response_group_art_start[size].numpy(), response_starts)
            np.testing.assert_array_equal(solver.forest.world_response_group_to_art[size].numpy(), response_order)

        free_articulations = np.flatnonzero(free_mask).astype(np.int32)
        np.testing.assert_array_equal(
            solver.forest.world_response_group_art_start[6].numpy(), np.array((0, 1, 2), dtype=np.int32)
        )
        np.testing.assert_array_equal(solver.forest.world_response_group_to_art[6].numpy(), free_articulations)

        forced_tiled = _FeatherPGSExecutionPlan.build(
            solver.forest.size_groups,
            max_constraints=solver.placement.dense_max_constraints,
            max_shared_memory=101376,
            hinv_jt_kernel="tiled",
            small_dof_threshold=12,
            tile_threads=64,
        )
        self.assertEqual(forced_tiled.hinv_jt_tiled_sizes, frozenset((6, 13)))

    @unittest.skipUnless(wp.is_cuda_available(), "matrix-free response diagonal parity requires CUDA")
    def test_forced_tiled_mixed_world_matches_par_row_trajectory(self):
        """Match diagonal, impulses, and state when every response group is tiled."""
        for warmstart in (False, True):
            for preelimination in (False, True):
                with self.subTest(warmstart=warmstart, preelimination=preelimination):
                    reference_solver, reference = _run_mixed_response(
                        "par_row", warmstart=warmstart, preelimination=preelimination
                    )
                    tiled_solver, tiled = _run_mixed_response(
                        "tiled", warmstart=warmstart, preelimination=preelimination
                    )

                    self.assertEqual(reference_solver.factor.eliminates, preelimination)
                    self.assertEqual(tiled_solver.factor.eliminates, preelimination)
                    self.assertEqual(len(reference), len(tiled))
                    self.assertGreater(reference[0][0], 0, "mixed scene generated no dense constraint rows")
                    # Without warm start or pre-elimination the par_row arm's world takes the local owners, which
                    # solve the free body in their held rows, against the tiled arm's packed rows (factor
                    # coordinates): a slow coordinate differs by up to 4e-6 where the arms' velocities reach 3.
                    for step, (expected, actual) in enumerate(zip(reference, tiled, strict=True)):
                        self.assertEqual(actual[0], expected[0], f"constraint count differed at step {step}")
                        for label, expected_value, actual_value in zip(
                            ("diagonal", "impulses", "joint_q", "joint_qd"),
                            expected[1:5],
                            actual[1:5],
                            strict=True,
                        ):
                            np.testing.assert_allclose(
                                actual_value,
                                expected_value,
                                rtol=5.0e-4,
                                atol=1.0e-5,
                                err_msg=f"{label} differed at step {step}",
                            )

    @unittest.skipUnless(wp.is_cuda_available(), "paired response ownership requires CUDA")
    def test_paired_response_matches_general_23_dof_trajectory(self):
        """Match the general response when one warp owns a robot/free-body pair, with patch and point friction."""
        run_kwargs = {
            "warmstart": False,
            "preelimination": False,
            "dof_count": 23,
            "dense_max_constraints": 96,
            "inactive_joint_limit_capacity": True,
            "friction": 0.7,
            "restitution": 0.3,
            "tangential_velocity": 2.0,
        }
        # Patch friction rows are not uniform triples: the factor sweep walks patch contact units, pooling
        # the patch load as the generic path does; point friction keeps the contact-triple layout.
        for label, friction_anchor_beta, triples in (("patch", None, False), ("point", 0.0, True)):
            with self.subTest(friction=label):
                reference_solver, reference = _run_mixed_response(
                    "par_row", friction_anchor_beta=friction_anchor_beta, **run_kwargs
                )
                paired_solver, paired = _run_mixed_response(
                    "auto", friction_anchor_beta=friction_anchor_beta, **run_kwargs
                )

                self.assertIsNone(reference_solver.sweep.paired_response_primary_size)
                self.assertEqual(paired_solver.laws.friction_anchors_enabled, label == "patch")
                self.assertEqual(paired_solver.sweep.paired_response_primary_size, 23)
                self.assertEqual(paired_solver.sweep.paired_response_secondary_size, 6)
                self.assertIsNotNone(paired_solver.sweep.paired_factor_kernel)
                self.assertTrue(paired_solver.sweep.paired_factor_coordinates)
                self.assertEqual(paired_solver.sweep.factor_coordinate_contact_triples, triples)
                self.assertEqual(paired_solver.sweep.factor_coordinate_contact_units, label == "patch")
                self.assertEqual("_units" in paired_solver.sweep.paired_factor_kernel.key, label == "patch")
                self.assertGreater(reference[0][0], 0, "mixed scene generated no dense constraint rows")
                self.assertTrue(
                    any(np.any(sample[6] == PGS_CONSTRAINT_TYPE_FRICTION) for sample in paired),
                    "mixed scene generated no friction rows",
                )
                for step, (expected, actual) in enumerate(zip(reference, paired, strict=True)):
                    self.assertEqual(actual[0], expected[0], f"constraint count differed at step {step}")
                    np.testing.assert_array_equal(actual[6], expected[6], err_msg=f"row types differed at step {step}")
                    for label_value, expected_value, actual_value in zip(
                        ("diagonal", "impulses", "joint_q", "joint_qd"), expected[1:5], actual[1:5], strict=True
                    ):
                        np.testing.assert_allclose(
                            actual_value,
                            expected_value,
                            rtol=5.0e-4,
                            atol=1.0e-5,
                            err_msg=f"{label_value} differed at step {step}",
                        )

    @unittest.skipUnless(wp.is_cuda_available(), "paired response fallback requires CUDA")
    def test_paired_response_falls_back_for_matrix_free_rows(self):
        """Keep mixed dense/matrix-free worlds in physical coordinates."""
        run_kwargs = {
            "warmstart": False,
            "preelimination": False,
            "dof_count": 23,
            "dense_max_constraints": 96,
            "friction": 0.7,
            "restitution": 0.3,
            "tangential_velocity": 2.0,
            "static_plane": True,
        }
        reference_solver, reference = _run_mixed_response("par_row", **run_kwargs)
        paired_solver, paired = _run_mixed_response("auto", **run_kwargs)

        self.assertIsNone(reference_solver.sweep.paired_response_primary_size)
        self.assertTrue(paired_solver.sweep.paired_factor_coordinates)
        self.assertTrue(any(sample[7] > 0 for sample in paired), "static contact generated no matrix-free rows")
        for step, (expected, actual) in enumerate(zip(reference, paired, strict=True)):
            self.assertEqual(actual[0], expected[0], f"constraint count differed at step {step}")
            np.testing.assert_array_equal(actual[6], expected[6], err_msg=f"row types differed at step {step}")
            for label, expected_value, actual_value in zip(
                ("diagonal", "impulses", "joint_q", "joint_qd"), expected[1:5], actual[1:5], strict=True
            ):
                np.testing.assert_allclose(
                    actual_value,
                    expected_value,
                    rtol=5.0e-4,
                    atol=1.0e-5,
                    err_msg=f"{label} differed at step {step}",
                )

    @unittest.skipUnless(wp.is_cuda_available(), "paired response ownership requires CUDA")
    def test_paired_factor_coordinates_follow_patch_friction_and_torsion(self):
        """Keep factor coordinates with default patch friction, and with contact torsion: the factor sweep's units
        hold the spin rows."""
        model = _build_mixed_response_model("cuda:0", dof_count=23, friction=0.7)
        solvers = {}
        with mock.patch.object(SolverFeatherPGS, "_route_pins", {"hinv_jt_kernel": "auto"}):
            for torsion in (False, True):
                solvers[torsion] = SolverFeatherPGS(
                    model,
                    enable_joint_limits=True,
                    joint_limit_activation_gap=0.0,
                    pgs_iterations=8,
                    dense_max_constraints=96,
                    mf_max_constraints=32,
                    contact_torsion=torsion,
                )
            point = SolverFeatherPGS(
                model,
                enable_joint_limits=True,
                joint_limit_activation_gap=0.0,
                pgs_iterations=8,
                dense_max_constraints=96,
                mf_max_constraints=32,
                friction_anchor_beta=0.0,
            )
        self.assertTrue(solvers[False].laws.friction_anchors_enabled)
        self.assertTrue(solvers[False].sweep.paired_factor_coordinates)
        # Patch rows are not uniform triples; the factor sweep walks patch contact units instead.
        self.assertFalse(solvers[False].sweep.factor_coordinate_contact_triples)
        self.assertTrue(solvers[False].sweep.factor_coordinate_contact_units)
        self.assertIn("_units", solvers[False].sweep.paired_factor_kernel.key)
        self.assertTrue(solvers[False].get_path_selection()["routes"]["paired_response"]["contact_units"])
        self.assertFalse(point.laws.friction_anchors_enabled)
        self.assertTrue(point.sweep.paired_factor_coordinates)
        self.assertTrue(point.sweep.factor_coordinate_contact_triples)
        self.assertFalse(point.sweep.factor_coordinate_contact_units)
        self.assertTrue(solvers[True].laws.contact_torsion_enabled)
        self.assertIsNotNone(solvers[True].sweep.paired_response_primary_size)
        self.assertTrue(solvers[True].sweep.paired_factor_coordinates)
        self.assertFalse(solvers[True].sweep.factor_coordinate_contact_triples)
        self.assertTrue(solvers[True].sweep.factor_coordinate_contact_units)
        self.assertIn("_torsion", solvers[True].sweep.paired_factor_kernel.key)

    @unittest.skipUnless(wp.is_cuda_available(), "articulation-local mixed-world parity requires CUDA")
    def test_local_internal_mixed_world_matches_general_response(self):
        """Match the general response when an articulation contacts a free body."""
        general_solver, general = _run_mixed_response(
            "tiled", warmstart=False, preelimination=False, inactive_joint_limit_capacity=True
        )
        local_solver, local = _run_mixed_response(
            "par_row", warmstart=False, preelimination=False, inactive_joint_limit_capacity=True
        )

        self.assertFalse(general_solver.placement.local_internal_fast_path)
        self.assertTrue(local_solver.placement.local_internal_fast_path)
        self.assertEqual(len(general), len(local))
        self.assertGreater(general[0][0], 0, "mixed scene generated no dense constraint rows")
        self.assertTrue(
            any(sample[5] == PGS_LOCAL_SOLVE_OWNER_PAIR for sample in local),
            "mixed contact never selected the paired local owner",
        )

        free_articulation = int(np.flatnonzero(general_solver.forest.is_free_rigid_host)[0])
        free_dof_start = int(general_solver.forest.articulation_dof_start_host[free_articulation])
        self.assertGreater(
            abs(float(general[0][4][free_dof_start + 2]) + 3.0),
            1.0e-4,
            "general response did not couple the dense contact to the free body",
        )

        for step, (expected, actual) in enumerate(zip(general, local, strict=True)):
            self.assertEqual(actual[0], expected[0], f"constraint count differed at step {step}")
            for label, expected_value, actual_value in zip(
                ("diagonal", "impulses", "joint_q", "joint_qd"), expected[1:5], actual[1:5], strict=True
            ):
                np.testing.assert_allclose(
                    actual_value,
                    expected_value,
                    rtol=5.0e-4,
                    atol=2.0e-6,
                    err_msg=f"{label} differed at step {step}",
                )

    @unittest.skipUnless(wp.is_cuda_available(), "articulation-local mixed-world parity requires CUDA")
    def test_local_pair_matches_general_friction_and_restitution(self):
        """Preserve paired friction projection and restitution response."""
        run_kwargs = {
            "warmstart": False,
            "preelimination": False,
            "inactive_joint_limit_capacity": True,
            "friction": 0.7,
            "restitution": 0.3,
            "tangential_velocity": 2.0,
        }
        general_solver, general = _run_mixed_response("tiled", **run_kwargs)
        local_solver, local = _run_mixed_response("par_row", **run_kwargs)

        self.assertFalse(general_solver.placement.local_internal_fast_path)
        self.assertTrue(local_solver.placement.local_internal_fast_path)
        pair_samples = [sample for sample in local if sample[5] == PGS_LOCAL_SOLVE_OWNER_PAIR]
        self.assertTrue(pair_samples, "mixed friction contact never selected the paired local owner")
        self.assertTrue(
            any(
                np.any(np.abs(sample[2][sample[6] == PGS_CONSTRAINT_TYPE_FRICTION]) > 1.0e-5) for sample in pair_samples
            ),
            "paired solve produced no nonzero friction impulse",
        )

        for step, (expected, actual) in enumerate(zip(general, local, strict=True)):
            self.assertEqual(actual[0], expected[0], f"constraint count differed at step {step}")
            np.testing.assert_array_equal(actual[6], expected[6], err_msg=f"row types differed at step {step}")
            for label, expected_value, actual_value in zip(
                ("diagonal", "impulses", "joint_q", "joint_qd"), expected[1:5], actual[1:5], strict=True
            ):
                np.testing.assert_allclose(
                    actual_value,
                    expected_value,
                    rtol=5.0e-4,
                    atol=2.0e-6,
                    err_msg=f"{label} differed at step {step}",
                )

    @unittest.skipUnless(wp.is_cuda_available(), "articulation-local mixed-world parity requires CUDA")
    def test_local_owners_match_general_contact_regularization(self):
        """Apply the general owner's per-row regularization weights in the local owners."""
        run_kwargs = {
            "warmstart": False,
            "preelimination": False,
            "inactive_joint_limit_capacity": True,
            "friction": 0.7,
            "restitution": 0.3,
            "tangential_velocity": 2.0,
            "contact_regularization": 1.0,
            "model_kwargs": {"static_support": True},
        }
        general_solver, general = _run_mixed_response("tiled", **run_kwargs)
        local_solver, local = _run_mixed_response("par_row", **run_kwargs)

        self.assertTrue(general_solver.laws.regularization_enabled)
        self.assertFalse(general_solver.placement.local_internal_fast_path)
        self.assertTrue(local_solver.placement.local_internal_fast_path)
        owners = [sample[5] for sample in local]
        self.assertIn(
            PGS_LOCAL_SOLVE_OWNER_PAIR_RESIDUAL, owners, "regularized mixed contact never selected the residual owner"
        )
        self.assertTrue(
            any(sample[7] > 0 and sample[5] != PGS_LOCAL_SOLVE_OWNER_GENERAL for sample in local),
            "no local owner solved matrix-free rows",
        )
        _assert_owner_parity(self, general, local)

    @unittest.skipUnless(wp.is_cuda_available(), "articulation-local mixed-world parity requires CUDA")
    def test_local_owner_matches_general_persistent_patch_friction(self):
        """Preserve pooled normal load and coupled tangent response under local ownership."""
        run_kwargs = {
            "warmstart": False,
            "preelimination": False,
            "inactive_joint_limit_capacity": True,
            "friction": 0.7,
            "restitution": 0.3,
            "tangential_velocity": 2.0,
            "model_kwargs": {"static_support": True},
        }
        general_solver, general = _run_mixed_response("tiled", **run_kwargs)
        local_solver, local = _run_mixed_response("par_row", **run_kwargs)

        self.assertTrue(general_solver.laws.friction_anchors_enabled)
        self.assertTrue(local_solver.laws.friction_anchors_enabled)
        self.assertFalse(general_solver.placement.local_internal_fast_path)
        self.assertTrue(local_solver.placement.local_internal_fast_path)
        self.assertIn(PGS_LOCAL_SOLVE_OWNER_PAIR_RESIDUAL, [sample[5] for sample in local])

        self.assertTrue(
            any(
                sample[10][row] >= 0 and sample[10][row] != row
                for sample in local
                for row in np.flatnonzero(sample[9] == PGS_CONSTRAINT_TYPE_CONTACT)
            ),
            "scene generated no linked multi-normal patch",
        )

        tangent_cross_terms = []
        for sample in local:
            row_type, row_parent = sample[9], sample[10]
            # The whitened rows: the pair's coupling J_i H^-1 J_s^T is Z_i . Z_s.
            whitened_a, whitened_b = sample[11], sample[12]
            for row in np.flatnonzero(row_type == PGS_CONSTRAINT_TYPE_FRICTION):
                parent = row_parent[row]
                if row != parent + 1:
                    continue
                sibling = parent + 2
                tangent_cross_terms.append(
                    float(whitened_a[row] @ whitened_a[sibling] + whitened_b[row] @ whitened_b[sibling])
                )
        self.assertTrue(tangent_cross_terms, "scene generated no tangent pair")
        self.assertGreater(max(abs(value) for value in tangent_cross_terms), 1.0e-3)
        _assert_owner_parity(self, general, local)

    @unittest.skipUnless(wp.is_cuda_available(), "articulation-local mixed-world parity requires CUDA")
    def test_free_body_velocity_limit_rows_stay_on_general_owner(self):
        """Keep worlds with free-body velocity-limit rows on the general owner.

        The rows exist from the first step (activation fraction 0) and never act: the arm sets the body moving
        below the limit, so both owners solve the same problem. An acting speed row couples the struck body's
        velocity to the arm's contact, where the two response kernels' rounding leaves near-zero impulses apart.
        """
        run_kwargs = {
            "warmstart": False,
            "preelimination": False,
            "inactive_joint_limit_capacity": True,
            "model_kwargs": {"free_body_velocity_limit": 5.0},
        }
        general_solver, general = _run_mixed_response("tiled", **run_kwargs)
        local_solver, local = _run_mixed_response("par_row", **run_kwargs)

        self.assertFalse(general_solver.placement.local_internal_fast_path)
        self.assertTrue(local_solver.placement.local_internal_fast_path)
        limited = [sample for sample in local if np.any(sample[9] == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)]
        self.assertTrue(limited, "free body never produced a velocity-limit row")
        for sample in general + local:
            np.testing.assert_array_equal(sample[8][sample[9] == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT], 0.0)
        # The residual owner is eligible for this world; only the velocity-limit rows keep it general.
        self.assertGreaterEqual(int(local_solver.sweep.local_residual_pair_articulation.numpy()[0]), 0)
        for sample in limited:
            self.assertEqual(sample[5], PGS_LOCAL_SOLVE_OWNER_GENERAL, "velocity-limit rows left the general owner")
        _assert_owner_parity(self, general, local)

    @unittest.skipUnless(wp.is_cuda_available(), "articulation-local mixed-world parity requires CUDA")
    def test_free_body_first_world_layout_matches_general(self):
        """Reject the residual owner when the world packs the free body before the articulation."""
        run_kwargs = {
            "warmstart": False,
            "preelimination": False,
            "inactive_joint_limit_capacity": True,
            "friction": 0.7,
            "restitution": 0.3,
            "tangential_velocity": 2.0,
            "model_kwargs": {"static_support": True, "free_body_first": True},
        }
        general_solver, general = _run_mixed_response("tiled", **run_kwargs)
        local_solver, local = _run_mixed_response("par_row", **run_kwargs)

        free_articulation = int(np.flatnonzero(local_solver.forest.is_free_rigid_host)[0])
        self.assertEqual(int(local_solver.forest.articulation_dof_start_host[free_articulation]), 0)
        self.assertFalse(general_solver.placement.local_internal_fast_path)
        self.assertTrue(local_solver.placement.local_internal_fast_path)
        np.testing.assert_array_equal(local_solver.sweep.local_residual_pair_articulation.numpy(), [-1])
        self.assertGreaterEqual(int(local_solver.sweep.local_pair_articulation.numpy()[0]), 0)
        self.assertTrue(any(sample[7] > 0 for sample in local), "free body generated no matrix-free rows")
        for sample in local:
            if sample[7] > 0:
                self.assertEqual(sample[5], PGS_LOCAL_SOLVE_OWNER_GENERAL, "matrix-free rows left the general owner")
        _assert_owner_parity(self, general, local)

    @unittest.skipUnless(wp.is_cuda_available(), "articulation-local mixed-world parity requires CUDA")
    def test_local_owners_take_contact_torsion(self):
        """Keep the local owners for a model with contact torsion: their sweeps solve its spin rows."""
        model = _build_mixed_response_model("cuda:0", friction=0.7)
        solvers = {}
        with mock.patch.object(SolverFeatherPGS, "_route_pins", {"hinv_jt_kernel": "par_row"}):
            for torsion in (False, True):
                solvers[torsion] = SolverFeatherPGS(
                    model,
                    enable_joint_limits=True,
                    joint_limit_activation_gap=0.0,
                    pgs_iterations=8,
                    dense_max_constraints=32,
                    mf_max_constraints=32,
                    contact_torsion=torsion,
                )
        self.assertTrue(solvers[False].placement.local_internal_fast_path)
        self.assertTrue(solvers[True].laws.contact_torsion_enabled)
        self.assertTrue(solvers[True].placement.local_internal_fast_path)
        kernels = (
            *solvers[True].sweep.local_pair.kernels.values(),
            *solvers[True].sweep.local_residual.kernels.values(),
        )
        self.assertTrue(kernels)
        self.assertTrue(all("_torsion" in kernel.key for kernel in kernels))


if __name__ == "__main__":
    unittest.main()
