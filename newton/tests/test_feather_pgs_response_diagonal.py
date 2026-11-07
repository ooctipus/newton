# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import itertools
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import field_group
from newton._src.solvers.feather_pgs.contact_input import ContactInput
from newton._src.solvers.feather_pgs.projection_laws import friction_pair
from newton._src.solvers.feather_pgs.row_laws import (
    ROW_CONTACT,
    ROW_FRICTION,
    ROW_JOINT_LIMIT,
)
from newton._src.solvers.feather_pgs.sweep import EXACT, LevelLaw, sweep_kernel
from newton._src.solvers.feather_pgs.sweep_levels import (
    _get_mark_independent_sparse_contact_candidates_kernel,
    _get_sparse_contact_levels_kernel,
    levels_launch,
)
from newton._src.solvers.feather_pgs.sweep_routes_diag import DiagonalSweep
from newton.solvers import SolverFeatherPGS
from newton.tests.feather_pgs_law_harness import diagonal_walk, leave_walk, row_lines, row_velocity


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
            rows["row_type"][w, index] = ROW_CONTACT if k == 0 else ROW_FRICTION
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
        rows["row_type"][w, :prefix] = ROW_JOINT_LIMIT
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
    marker = _get_mark_independent_sparse_contact_candidates_kernel(rows["diagonal"], str(device.arch))
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

    Every row other than a tangent row heads a unit. A marked head is a scalar contact on its coordinate unless a
    coupled unit uses that coordinate (its mark then becomes ``-1``); a coordinate's ``k``-th scalar contact in row
    order is level ``k``, and each level lists its contacts in row order (``level_count`` levels, level ``l`` the
    heads ``level_units[level_offsets[l]:level_offsets[l + 1]]``). The serial list holds the remaining heads in row
    order.
    """
    links = marked.copy()
    capacity = marked.shape[1]
    level_count = np.zeros(rows["worlds"], np.int32)
    level_offsets = np.zeros((rows["worlds"], capacity + 1), np.int32)
    level_units = np.zeros((rows["worlds"], capacity), np.int32)
    serial = []
    for w in range(rows["worlds"]):
        start, end = int(rows["bounds"][w, 1]), int(rows["row_count"][w])
        heads = [row for row in range(start, end) if rows["row_type"][w, row] != ROW_FRICTION]
        candidates = [row for row in heads if np.any(marked[w, row] < -1)]
        reserved = {int(c) for row in heads if row not in candidates for c in marked[w, row] if c >= 0}
        levels, placed = [], {}
        for row in candidates:
            coordinate, slot = int(np.max(marked[w, row])), int(np.argmin(marked[w, row]))
            if coordinate in reserved:
                links[w, row, slot] = -1
            else:
                level = placed.get(coordinate, 0)
                placed[coordinate] = level + 1
                levels += [[]] * (level + 1 - len(levels))
                levels[level] = [*levels[level], row]
        level_count[w] = len(levels)
        units = [row for level in levels for row in level]
        level_units[w, : len(units)] = units
        level_offsets[w, : len(levels) + 1] = np.cumsum([0, *(len(level) for level in levels)])
        serial.append([row for row in heads if not np.any(links[w, row] < -1)])
    return {
        "sparse_row_dof": links,
        "level_count": level_count,
        "level_offsets": level_offsets,
        "level_units": level_units,
        "serial": serial,
    }


def _unit_rows(rows, world, head):
    """The rows of the unit ``head`` heads: its normal row, then the two tangent rows it owns, if any."""
    owns_tangents = (
        head + 2 < rows["row_count"][world]
        and rows["row_type"][world, head + 1] == ROW_FRICTION
        and rows["row_parent"][world, head + 1] == head
    )
    return [head, head + 1, head + 2] if owns_tangents else [head]


def _unit_schedule_order(rows, schedule, world):
    """Rows in the unit schedule's order: the scalar contacts level by level, the prefix, then serial units."""
    order = []
    for head in schedule["level_units"][world, : schedule["level_offsets"][world, schedule["level_count"][world]]]:
        order += _unit_rows(rows, world, int(head))
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
                if kind == ROW_FRICTION and row != parent + 1:
                    continue
                if denominator <= 0.0 and kind != ROW_FRICTION:
                    continue
                residual = J[row] @ v + rows["rhs"][w, row]
                old = lam[row]
                new = old - omega * residual / denominator if denominator > 0.0 else old
                if kind != ROW_FRICTION:
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


class TestFeatherPGSResponseDiagonal(unittest.TestCase):
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
        row_type[0, :3] = (ROW_CONTACT, ROW_FRICTION, ROW_FRICTION)
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
        kernel = sweep_kernel(
            DiagonalSweep(max_constraints, world_dofs, dense_dofs, str(device.arch), levels=LevelLaw(EXACT))
        )
        walk = diagonal_walk(inverse_mass[None, :dense_dofs], device, world_dofs)
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
                wp.zeros((1, max_constraints + 1), dtype=wp.int32, device=device),
                wp.empty((1, max_constraints), dtype=wp.int32, device=device),
                wp.zeros(1, dtype=wp.int32, device=device),
                wp.empty((1, max_constraints), dtype=wp.int32, device=device),
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
        leave_walk(walk, velocity, device)

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
        row_type = np.full((1, max_constraints), ROW_FRICTION, dtype=np.int32)
        row_type[0, normal_rows] = ROW_CONTACT
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
        kernel = sweep_kernel(
            DiagonalSweep(
                max_constraints,
                world_dofs,
                dense_dofs,
                str(device.arch),
                levels=LevelLaw(EXACT),
                speculative_contact_batches=True,
            )
        )
        walk = diagonal_walk(inverse_mass[None, :dense_dofs], device, world_dofs)
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
                wp.zeros((1, max_constraints + 1), dtype=wp.int32, device=device),
                wp.empty((1, max_constraints), dtype=wp.int32, device=device),
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
        leave_walk(walk, velocity, device)

        np.testing.assert_allclose(velocity.numpy(), expected_velocity, rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_allclose(impulses.numpy()[0], expected_impulses, rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_array_equal(impulses.numpy()[0, :6], np.zeros(6, dtype=np.float32))

    @unittest.skipUnless(wp.is_cuda_available(), "independent contact groups require CUDA")
    def test_independent_sparse_contact_groups_exclude_coupled_coordinates(self):
        """Keep scalar contact units serial when a coupled unit shares their coordinate."""
        device = wp.get_device("cuda:0")
        max_constraints, max_world_dofs, contact_count = 12, 4, 4
        sparse_response_dofs = 108
        sparse_row_dof_np = np.full((1, max_constraints, 2), -1, dtype=np.int32)
        sparse_row_dof_np[0, 0:3, 0] = 2
        sparse_row_dof_np[0, 3:6, 0] = 3
        sparse_row_dof_np[0, 6:9, 0] = 3
        sparse_row_dof_np[0, 9:12, 0] = 2
        sparse_row_dof = wp.array(sparse_row_dof_np, device=device)
        level_count = wp.zeros(1, dtype=wp.int32, device=device)
        level_offsets = wp.empty((1, max_constraints + 1), dtype=wp.int32, device=device)
        level_units = wp.empty((1, max_constraints), dtype=wp.int32, device=device)
        serial_count = wp.zeros(1, dtype=wp.int32, device=device)
        serial_normals = wp.empty((1, max_constraints), dtype=wp.int32, device=device)

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
        schedule = _get_sparse_contact_levels_kernel(
            max_constraints, max_world_dofs, str(device.arch), build_serial_contacts=True
        )
        # Four three-row contact units: a normal heading its tangent pair, which names it as parent.
        heads = np.arange(0, max_constraints, 3)
        row_type = np.full((1, max_constraints), ROW_FRICTION, dtype=np.int32)
        row_type[0, heads] = ROW_CONTACT
        dim, block = levels_launch(1, max_world_dofs, device.arch)
        wp.launch(
            schedule,
            dim=dim,
            inputs=[
                wp.array((max_constraints,), dtype=wp.int32, device=device),
                row_lines(device),
                wp.zeros((1, 2), dtype=wp.int32, device=device),
                wp.array(row_type, device=device),
            ],
            outputs=[sparse_row_dof, level_count, level_offsets, level_units, serial_count, serial_normals],
            block_dim=block,
            device=device,
        )
        wp.synchronize_device(device)

        # Rows 0 and 9 share coordinate 2: levels 0 and 1. Row 6's coordinate 3 is row 3's, a coupled unit's.
        self.assertEqual(int(level_count.numpy()[0]), 2)
        np.testing.assert_array_equal(level_offsets.numpy()[0, :3], np.array((0, 1, 2), dtype=np.int32))
        np.testing.assert_array_equal(level_units.numpy()[0, :2], np.array((0, 9), dtype=np.int32))
        self.assertEqual(int(serial_count.numpy()[0]), 2)
        np.testing.assert_array_equal(serial_normals.numpy()[0, :2], np.array((3, 6), dtype=np.int32))
        normal_marks = sparse_row_dof.numpy()[0, ::3, 1]
        np.testing.assert_array_equal(normal_marks, np.array((-2, -1, -1, -2), dtype=np.int32))

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
        for units in (True,):
            marked = wp.array(sparse_dof, device=device)
            marker = _get_mark_independent_sparse_contact_candidates_kernel(108, "cpu")
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
        # Units mark the 1-row unit in its empty slot.
        expected[3, 0] = -2
        np.testing.assert_array_equal(launches[True], expected)

    def test_unit_schedule_factories_name_and_reject_layouts(self):
        """Name the unit variants and keep speculative batches within sixteen response lanes."""
        solve = sweep_kernel(DiagonalSweep(24, 12, 9, "120", levels=LevelLaw(EXACT)))
        batched = sweep_kernel(
            DiagonalSweep(24, 12, 9, "120", levels=LevelLaw(EXACT), speculative_contact_batches=True)
        )
        builder = _get_sparse_contact_levels_kernel(24, 12, "120", build_serial_contacts=True)
        self.assertEqual(solve.key, "sweep_sparse_diagonal_24_12_9_units")
        self.assertEqual(batched.key, "sweep_sparse_diagonal_24_12_9_speculative_batches_units")
        self.assertEqual(builder.key, "sparse_contact_levels_24_12_serial_contacts_units")
        for kwargs in ({"levels": LevelLaw(EXACT), "speculative_contact_batches": True, "dense_dofs": 15},):
            dense_dofs = kwargs.pop("dense_dofs", 9)
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                sweep_kernel(DiagonalSweep(24, dense_dofs + 3, dense_dofs, "120", **kwargs))

    def test_unit_schedule_reorders_only_commuting_rows(self):
        """Keep GS order on every coordinate when sweeping in the unit schedule's order.

        The schedule oracle is the reference the CUDA unit builder must reproduce. Sweeping the
        rows in its order (the scalar contacts level by level first, then the prefix and the serial units) must give
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
                independent += int(
                    np.sum(schedule["level_offsets"][np.arange(rows["worlds"]), schedule["level_count"]])
                )
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
                    level_count = wp.full(worlds, -7, dtype=wp.int32, device=device)
                    level_offsets = wp.full((worlds, capacity + 1), -7, dtype=wp.int32, device=device)
                    level_units = wp.full((worlds, capacity), -7, dtype=wp.int32, device=device)
                    serial_count = wp.full(worlds, -7, dtype=wp.int32, device=device)
                    serial_heads = wp.full((worlds, capacity), -7, dtype=wp.int32, device=device)
                    schedule = _get_sparse_contact_levels_kernel(
                        capacity, dofs, str(device.arch), build_serial_contacts=serial_contacts
                    )
                    dim, block = levels_launch(worlds, dofs, device.arch)
                    wp.launch(
                        schedule,
                        dim=dim,
                        inputs=[
                            wp.array(rows["row_count"], device=device),
                            row_lines(device),
                            wp.array(rows["bounds"], device=device),
                            wp.array(rows["row_type"], device=device),
                        ],
                        outputs=[sparse_row_dof, level_count, level_offsets, level_units, serial_count, serial_heads],
                        block_dim=block,
                        device=device,
                    )
                    np.testing.assert_array_equal(sparse_row_dof.numpy(), expected["sparse_row_dof"])
                    np.testing.assert_array_equal(level_count.numpy(), expected["level_count"])
                    for world in range(worlds):
                        levels = int(expected["level_count"][world])
                        offsets = expected["level_offsets"][world, : levels + 1]
                        np.testing.assert_array_equal(level_offsets.numpy()[world, : levels + 1], offsets)
                        np.testing.assert_array_equal(
                            level_units.numpy()[world, : offsets[-1]], expected["level_units"][world, : offsets[-1]]
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
                    kernel = sweep_kernel(
                        DiagonalSweep(
                            capacity,
                            dofs,
                            dense,
                            str(device.arch),
                            levels=LevelLaw(EXACT) if units else LevelLaw(),
                            speculative_contact_batches=batches,
                        )
                    )
                    walk = diagonal_walk(rows["dense_inverse"], device, dofs, rows["dense_offsets"])
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
                            wp.array(schedule["level_count"], device=device),
                            wp.array(schedule["level_offsets"], device=device),
                            wp.array(schedule["level_units"], device=device),
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
                    leave_walk(walk, velocity, device)
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
        """Include free rigid articulations in every world response group."""
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


if __name__ == "__main__":
    unittest.main()
